//! The peer mesh over TCP loopback: four meshes in one process, as four rank
//! daemons would run them.
//!
//! - Every rank's reduced partition equals the in-process protocol
//!   (`reduce_scatter::outgoing` / `place_frames` / `Exchange::reduce`) bit
//!   for bit, for both exchange dtypes and row counts from 16 to 1,024.
//! - Two exchanges in flight (the coordinator's two prefill lanes): a rank that
//!   collects its first exchange late, with the second exchange's frames
//!   already arrived, still sums each exchange's own frames.
//! - A peer that never sends: the others fail at the timeout, not later.
//! - A peer that dies while its frame is awaited: the others fail at once.
//! - A peer that is gone before the exchange: the others fail at the timeout.
//! - A peer list that points a rank at the wrong rank: the hello refuses the
//!   link, and the exchange fails at the timeout.

use std::net::TcpListener;
use std::time::{Duration, Instant};

use glm53f_rank::consts::{HIDDEN, WORLD};
use glm53f_rank::mesh::{Mesh, MeshConfig};
use glm53f_rank::reduce_scatter::{outgoing, place_frames, Exchange, ExchangeDtype};
use glm53f_rank::testkit::Rng;
use glm53f_wire::row_shard::ExchangeView;
use glm53f_wire::WireNaive;

/// Four meshes on loopback ports; `swap` exchanges two entries of rank 0's
/// peer list (a misconfiguration).
fn meshes(timeout: Duration, swap: Option<(usize, usize)>) -> Vec<Mesh> {
    let listeners: Vec<TcpListener> = (0..WORLD).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs: Vec<String> = listeners.iter().map(|l| l.local_addr().unwrap().to_string()).collect();
    listeners
        .into_iter()
        .enumerate()
        .map(|(r, l)| {
            let mut peers = addrs.clone();
            if let (0, Some((a, b))) = (r, swap) {
                peers.swap(a, b);
            }
            let cfg = MeshConfig::new(r, &peers.join(","), timeout, false).unwrap();
            Mesh::with_listener(&cfg, l).unwrap()
        })
        .collect()
}

fn partials(seed: u64, rows: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..WORLD).map(|_| (0..rows * HIDDEN).map(|i| (rng.normal() * if i % 331 == 5 { 40.0 } else { 1.0 }) as f32).collect()).collect()
}

/// What rank `q` must return: the in-process protocol on the same partials.
fn reference(p: &[Vec<f32>], x: Exchange) -> Vec<u16> {
    let frames: Vec<Vec<u8>> = (0..WORLD)
        .filter(|&r| r != x.rank)
        .flat_map(|r| outgoing(&Exchange { rank: r, ..x }, &p[r], 0).unwrap())
        .filter(|f| ExchangeView::parse(f, WireNaive::NONE).unwrap().header.dst == x.rank)
        .collect();
    let peers = place_frames(&x, &frames).unwrap();
    let mut out = vec![0u16; x.own().1 * HIDDEN];
    x.reduce(&p[x.rank], &peers, &mut out).unwrap();
    out
}

fn exchange(request_id: u64, rows: usize, dtype: ExchangeDtype, rank: usize) -> Exchange {
    Exchange { request_id, layer: 3 + request_id as u32 % 42, rows, dtype, rank }
}

#[test]
fn four_meshes_reduce_scatter_every_partition() {
    let mut m = meshes(Duration::from_secs(20), None);
    let cases: Vec<(usize, ExchangeDtype)> =
        [16usize, 37, 1024].iter().flat_map(|&r| [(r, ExchangeDtype::Bf16), (r, ExchangeDtype::Fp8RowScaled)]).collect();
    let data: Vec<Vec<Vec<f32>>> = cases.iter().enumerate().map(|(i, &(rows, _))| partials(0x3E5_0000 + i as u64, rows)).collect();
    let results: Vec<Vec<(Vec<u16>, f64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = m
            .iter_mut()
            .enumerate()
            .map(|(rank, mesh)| {
                let (cases, data) = (&cases, &data);
                s.spawn(move || {
                    cases
                        .iter()
                        .enumerate()
                        .map(|(i, &(rows, dtype))| {
                            let x = exchange(100 + i as u64, rows, dtype, rank);
                            let mut out = vec![0u16; x.own().1 * HIDDEN];
                            let t = Instant::now();
                            mesh.exchange(&x, &data[i][rank], &mut out).unwrap();
                            (out, t.elapsed().as_secs_f64() * 1e3)
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, &(rows, dtype)) in cases.iter().enumerate() {
        let mut ms = 0f64;
        for (rank, per_rank) in results.iter().enumerate() {
            let want = reference(&data[i], exchange(100 + i as u64, rows, dtype, rank));
            assert!(per_rank[i].0 == want, "{rows} rows, {dtype:?}, rank {rank}: differs from the in-process protocol");
            ms = ms.max(per_rank[i].1);
        }
        eprintln!("{rows} rows, {}: all four partitions exact, slowest rank {ms:.2} ms over loopback TCP", dtype.name());
    }
}

#[test]
fn two_exchanges_in_flight_are_kept_apart() {
    let mut m = meshes(Duration::from_secs(20), None);
    let rows = 64;
    let (px, py) = (partials(0xAB_0001, rows), partials(0xAB_0002, rows));
    let got: Vec<(Vec<u16>, Vec<u16>)> = std::thread::scope(|s| {
        let hs: Vec<_> = m
            .iter_mut()
            .enumerate()
            .map(|(rank, mesh)| {
                let (px, py) = (&px, &py);
                s.spawn(move || {
                    let (x, y) = (exchange(7, rows, ExchangeDtype::Bf16, rank), exchange(8, rows, ExchangeDtype::Bf16, rank));
                    let (mut ox, mut oy) = (vec![0u16; x.own().1 * HIDDEN], vec![0u16; y.own().1 * HIDDEN]);
                    if rank == 1 {
                        // Rank 1 sends its first exchange, then collects it late: by then the
                        // others have finished it and sent the second one's frames too.
                        mesh.send(&x, &px[rank]).unwrap();
                        std::thread::sleep(Duration::from_millis(300));
                        mesh.finish(&x, &px[rank], &mut ox).unwrap();
                    } else {
                        mesh.exchange(&x, &px[rank], &mut ox).unwrap();
                    }
                    mesh.exchange(&y, &py[rank], &mut oy).unwrap();
                    (ox, oy)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (rank, (ox, oy)) in got.iter().enumerate() {
        assert!(*ox == reference(&px, exchange(7, rows, ExchangeDtype::Bf16, rank)), "rank {rank}, first exchange");
        assert!(*oy == reference(&py, exchange(8, rows, ExchangeDtype::Bf16, rank)), "rank {rank}, second exchange");
    }
}

/// Ranks 0-2 run one exchange while rank 3 does `rank3`; returns each one's
/// result and time.
fn three_ranks(m: &mut [Mesh], rank3: impl FnOnce(&mut Mesh) + Send) -> Vec<(Result<(), String>, Duration)> {
    let rows = 32;
    let p = partials(0xDEAD, rows);
    let (first, last) = m.split_at_mut(3);
    std::thread::scope(|s| {
        let hs: Vec<_> = first
            .iter_mut()
            .enumerate()
            .map(|(rank, mesh)| {
                let p = &p;
                s.spawn(move || {
                    let x = exchange(55, rows, ExchangeDtype::Bf16, rank);
                    let mut out = vec![0u16; x.own().1 * HIDDEN];
                    let t = Instant::now();
                    let r = mesh.exchange(&x, &p[rank], &mut out).map(|_| ());
                    (r, t.elapsed())
                })
            })
            .collect();
        s.spawn(|| rank3(&mut last[0]));
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[test]
fn a_peer_that_never_sends_fails_the_exchange_at_the_timeout() {
    let timeout = Duration::from_millis(600);
    let mut m = meshes(timeout, None);
    for mesh in m.iter_mut() {
        mesh.wait_ready().unwrap();
    }
    for (r, t) in three_ranks(&mut m, |_| {}) {
        let e = r.unwrap_err();
        assert!(e.contains("no frame from rank(s) [3]"), "{e}");
        assert!(t >= timeout && t < timeout + Duration::from_secs(2), "{t:?}");
    }
}

#[test]
fn a_peer_that_dies_mid_exchange_fails_it_at_once() {
    let timeout = Duration::from_secs(20);
    let mut m = meshes(timeout, None);
    for mesh in m.iter_mut() {
        mesh.wait_ready().unwrap();
    }
    let dead = m.pop().unwrap();
    let mut three: Vec<Mesh> = m;
    // Rank 3 dies 300 ms into the others' exchange, without sending.
    let killer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        drop(dead);
    });
    let rows = 32;
    let p = partials(0xDEAD, rows);
    let res: Vec<(Result<(), String>, Duration)> = std::thread::scope(|s| {
        let hs: Vec<_> = three
            .iter_mut()
            .enumerate()
            .map(|(rank, mesh)| {
                let p = &p;
                s.spawn(move || {
                    let x = exchange(56, rows, ExchangeDtype::Bf16, rank);
                    let mut out = vec![0u16; x.own().1 * HIDDEN];
                    let t = Instant::now();
                    (mesh.exchange(&x, &p[rank], &mut out).map(|_| ()), t.elapsed())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    killer.join().unwrap();
    for (r, t) in res {
        let e = r.unwrap_err();
        assert!(e.contains("link to rank 3 was lost"), "{e}");
        assert!(t < Duration::from_secs(3), "failed after {t:?}, not at once");
    }
}

#[test]
fn a_peer_gone_before_the_exchange_fails_it_at_the_timeout() {
    let timeout = Duration::from_millis(800);
    let mut m = meshes(timeout, None);
    for mesh in m.iter_mut() {
        mesh.wait_ready().unwrap();
    }
    drop(m.pop());
    std::thread::sleep(Duration::from_millis(100));
    let rows = 32;
    let p = partials(0xBEEF, rows);
    let res: Vec<(Result<(), String>, Duration)> = std::thread::scope(|s| {
        let hs: Vec<_> = m
            .iter_mut()
            .enumerate()
            .map(|(rank, mesh)| {
                let p = &p;
                s.spawn(move || {
                    let x = exchange(57, rows, ExchangeDtype::Bf16, rank);
                    let mut out = vec![0u16; x.own().1 * HIDDEN];
                    let t = Instant::now();
                    (mesh.exchange(&x, &p[rank], &mut out).map(|_| ()), t.elapsed())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r, t) in res {
        let e = r.unwrap_err();
        assert!(e.contains("rank 3") || e.contains("[3]"), "{e}");
        assert!(t < timeout + Duration::from_secs(2), "{t:?}");
    }
}

#[test]
fn a_rank_pointed_at_the_wrong_peer_gets_no_link() {
    // Rank 0 has ranks 2 and 3 swapped: its hello to "rank 2" reaches rank 3, which refuses it.
    let timeout = Duration::from_millis(800);
    let mut m = meshes(timeout, Some((2, 3)));
    let t = Instant::now();
    let e = m[0].wait_ready().unwrap_err();
    assert!(e.contains("no link to rank(s)"), "{e}");
    assert!(t.elapsed() < timeout + Duration::from_secs(2));
    // The other ranks' links to each other are fine.
    let three = three_ranks(&mut m, |_| {});
    assert!(three.iter().all(|(r, _)| r.is_err()), "rank 0 cannot complete, so nobody can");
}
