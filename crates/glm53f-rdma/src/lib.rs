//! RDMA RC transport for the expert exchange (perf reset R2 part 2).
//!
//! One [`Endpoint`] per peer: an RC queue pair on the RoCE device that owns the
//! TCP connection's local IPv4, registered send/receive/header buffers, SEND into
//! pre-posted RECV slots and busy-polled completions (the upstream DS41RT v15
//! design). Connection setup is exchanged by the caller over its TCP socket with
//! [`HANDSHAKE_MAGIC`] + [`Info::to_bytes`]; TCP stays the fallback transport.
//!
//! **Receive slots in the handshake.** A side names the receive slots it keeps
//! posted ([`Info::recv_slots`], in the `Info` word that was reserved and zero), so
//! its peer knows how many messages it may send ahead: a rank queues that many
//! requests. A peer built before the field sends zero and posts two
//! ([`LEGACY_RECV_SLOTS`], [`Info::peer_slots`]); it ignores the word, so the
//! handshake is unchanged for it.
//!
//! Without the `rdma` feature the native shim is not built and [`Endpoint::open`]
//! returns an error, so callers keep a TCP path.

#![cfg_attr(not(feature = "rdma"), allow(dead_code, unused_mut))]

use std::time::Duration;

/// First 8 bytes a coordinator writes on the TCP socket to request RDMA mode.
pub const HANDSHAKE_MAGIC: &[u8; 8] = b"M26RDMA1";
/// Handshake length: magic + one [`Info`].
pub const HANDSHAKE_LEN: usize = 8 + INFO_LEN;
/// Serialized [`Info`] length.
pub const INFO_LEN: usize = 32;
/// Receive slots a peer posts when its handshake does not say ([`Info::recv_slots`] 0): the
/// builds before the field posted two.
pub const LEGACY_RECV_SLOTS: u32 = 2;

/// One side's queue-pair coordinates (the C `m26r_info`) and the receive slots it keeps posted.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Info {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub mtu: u32,
    /// Receive slots this side keeps posted: messages its peer may send ahead of it (0: not
    /// said). The C shim's `reserved` word, which `m26r_local_info` leaves at zero: the caller
    /// sets it.
    pub recv_slots: u32,
}

impl Info {
    pub fn to_bytes(&self) -> [u8; INFO_LEN] {
        let mut b = [0u8; INFO_LEN];
        b[0..4].copy_from_slice(&self.qpn.to_le_bytes());
        b[4..8].copy_from_slice(&self.psn.to_le_bytes());
        b[8..24].copy_from_slice(&self.gid);
        b[24..28].copy_from_slice(&self.mtu.to_le_bytes());
        b[28..32].copy_from_slice(&self.recv_slots.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < INFO_LEN {
            return None;
        }
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&b[8..24]);
        Some(Self {
            qpn: u32::from_le_bytes(b[0..4].try_into().ok()?),
            psn: u32::from_le_bytes(b[4..8].try_into().ok()?),
            gid,
            mtu: u32::from_le_bytes(b[24..28].try_into().ok()?),
            recv_slots: u32::from_le_bytes(b[28..32].try_into().ok()?),
        })
    }

    /// The receive slots the peer that sent this posts: [`Info::recv_slots`], or
    /// [`LEGACY_RECV_SLOTS`] when it does not say.
    pub fn peer_slots(&self) -> u32 {
        if self.recv_slots == 0 {
            LEGACY_RECV_SLOTS
        } else {
            self.recv_slots
        }
    }
}

/// A page-aligned, zeroed heap buffer, registerable with `ibv_reg_mr`.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain owned memory.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(4096), 4096).expect("layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "AlignedBuf: out of memory");
        Self { ptr, len: len.max(4096) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: owned, initialized (zeroed) allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, 4096).expect("layout");
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

/// Find the RoCE v2 GID whose value is the IPv4-mapped `ip`: `(device, port,
/// gid_index)`, scanning `/sys/class/infiniband/*/ports/*/gids`.
pub fn find_roce_v2(ip: std::net::Ipv4Addr) -> Option<(String, u8, i32)> {
    find_roce_v2_in(std::path::Path::new("/sys/class/infiniband"), ip)
}

/// [`find_roce_v2`] under `root`, a directory laid out as `/sys/class/infiniband`. A GID table can
/// have empty slots below used ones (an address removed, or GIDs added out of order): an empty
/// slot reads as the zero GID and its type cannot be read, so the scan skips every slot it cannot
/// read instead of stopping there.
pub fn find_roce_v2_in(root: &std::path::Path, ip: std::net::Ipv4Addr) -> Option<(String, u8, i32)> {
    let mut want = [0u8; 16];
    want[10] = 0xff;
    want[11] = 0xff;
    want[12..16].copy_from_slice(&ip.octets());
    let devs = std::fs::read_dir(root).ok()?;
    for dev in devs.flatten() {
        let name = dev.file_name().to_string_lossy().into_owned();
        let Ok(ports) = std::fs::read_dir(dev.path().join("ports")) else { continue };
        for port in ports.flatten() {
            let Ok(pn) = port.file_name().to_string_lossy().parse::<u8>() else { continue };
            for idx in 0..256i32 {
                let gid = std::fs::read_to_string(port.path().join(format!("gids/{idx}")));
                let ty = std::fs::read_to_string(port.path().join(format!("gid_attrs/types/{idx}")));
                let (Ok(gid), Ok(ty)) = (gid, ty) else { continue };
                if !ty.trim().eq_ignore_ascii_case("RoCE v2") {
                    continue;
                }
                if parse_gid(gid.trim()) == Some(want) {
                    return Some((name, pn, idx));
                }
            }
        }
    }
    None
}

/// The link rate of `dev` port `port` in Gb/s (sysfs `rate`, e.g. "200 Gb/sec (4X HDR)").
pub fn port_rate_gbps(dev: &str, port: u8) -> Option<u32> {
    let r = std::fs::read_to_string(format!("/sys/class/infiniband/{dev}/ports/{port}/rate")).ok()?;
    r.split_whitespace().next()?.parse::<f64>().ok().map(|g| g as u32)
}

/// The inference-fabric rule: expert traffic (the coordinator <-> Spark MoE wire)
/// runs only on an RDMA-capable link of at least `GLM53F_WIRE_MIN_GBPS` (default
/// 100) Gb/s, never on the LAN/10G path, which serves the API only.
/// The check is by the connection's LOCAL address, so it holds on any
/// Spark/5090/switch recipe without naming interfaces or subnets. Loopback is
/// exempt (single-host tests); `GLM53F_WIRE_ALLOW_LAN=1` is the explicit test
/// override. Returns `(dev, port, gid index, Gb/s)`, or `None` when exempt.
pub fn fabric_port(local: std::net::IpAddr) -> Result<Option<(String, u8, i32, u32)>, String> {
    if local.is_loopback() || std::env::var("GLM53F_WIRE_ALLOW_LAN").map(|v| v == "1").unwrap_or(false) {
        return Ok(None);
    }
    let std::net::IpAddr::V4(ip) = local else {
        return Err(format!("wire: local address {local} is not IPv4; the RDMA fabric is RoCE v2 over IPv4"));
    };
    let (dev, port, gid) = find_roce_v2(ip).ok_or_else(|| {
        format!("wire: local address {ip} has no RoCE v2 device: that is a LAN path, and inference runs only on \
                 the RDMA fabric (point GLM53F_SPARK_ADDRS at the Sparks' fabric addresses)")
    })?;
    let min: u32 = std::env::var("GLM53F_WIRE_MIN_GBPS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let rate = port_rate_gbps(&dev, port).ok_or_else(|| format!("wire: no link rate for {dev} port {port}"))?;
    if rate < min {
        return Err(format!("wire: {ip} is on {dev} port {port} at {rate} Gb/s, below the {min} Gb/s inference floor"));
    }
    Ok(Some((dev, port, gid, rate)))
}

fn parse_gid(s: &str) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    let groups: Vec<&str> = s.split(':').collect();
    if groups.len() != 8 {
        return None;
    }
    for (i, g) in groups.iter().enumerate() {
        let v = u16::from_str_radix(g, 16).ok()?;
        out[2 * i] = (v >> 8) as u8;
        out[2 * i + 1] = v as u8;
    }
    Some(out)
}

#[cfg(feature = "rdma")]
mod ffi {
    use super::Info;
    use core::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        pub fn m26r_open(
            dev: *const c_char,
            port: u8,
            gid_index: c_int,
            send_buf: *mut u8,
            send_len: usize,
            recv_buf: *mut u8,
            recv_len: usize,
            recv_slots: u32,
            hdr_buf: *mut u8,
            hdr_len: usize,
            out: *mut *mut c_void,
            err: *mut c_char,
            errlen: usize,
        ) -> c_int;
        pub fn m26r_close(ep: *mut c_void);
        pub fn m26r_local_info(ep: *const c_void, out: *mut Info);
        pub fn m26r_connect(ep: *mut c_void, remote: *const Info, err: *mut c_char, errlen: usize) -> c_int;
        pub fn m26r_post_recv(ep: *mut c_void, slot: u32) -> c_int;
        pub fn m26r_post_send(ep: *mut c_void, hdr_off: usize, hdr_len: usize, body_off: usize, body_len: usize) -> c_int;
        pub fn m26r_wait_send(ep: *mut c_void, timeout_us: i64) -> c_int;
        pub fn m26r_wait_recv(ep: *mut c_void, spin_us: i64, timeout_us: i64, slot: *mut u32, len: *mut u32) -> c_int;
    }
}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

fn us(d: Option<Duration>) -> i64 {
    d.map_or(-1, |d| d.as_micros().min(i64::MAX as u128) as i64)
}

/// One RC queue pair plus its registered buffers (the buffers are owned by the
/// caller and must outlive the endpoint).
pub struct Endpoint {
    ep: *mut core::ffi::c_void,
    slots: u32,
    slot_len: usize,
}

// SAFETY: the endpoint is used by one thread at a time (moved between threads).
unsafe impl Send for Endpoint {}

impl Endpoint {
    /// Open `dev`/`port` with `gid_index`; register `recv` (split into `slots`
    /// equal slots) and the optional `send` body and `hdr` buffers.
    #[allow(unused_variables)]
    pub fn open(
        dev: &str,
        port: u8,
        gid_index: i32,
        send: Option<&mut AlignedBuf>,
        recv: &mut AlignedBuf,
        slots: u32,
        hdr: Option<&mut AlignedBuf>,
    ) -> Result<Self, String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("glm53f-rdma built without the `rdma` feature".into())
        }
        #[cfg(feature = "rdma")]
        {
            let cdev = std::ffi::CString::new(dev).map_err(|e| e.to_string())?;
            let mut err = [0u8; 256];
            let mut ep = core::ptr::null_mut();
            let (sp, sl) = send.map_or((core::ptr::null_mut(), 0), |b| (b.as_ptr(), b.len()));
            let (hp, hl) = hdr.map_or((core::ptr::null_mut(), 0), |b| (b.as_ptr(), b.len()));
            let slot_len = recv.len() / slots.max(1) as usize;
            let recv_len = slot_len * slots.max(1) as usize;
            // SAFETY: every buffer is a live allocation of the stated length that the
            // caller keeps alive for the endpoint's lifetime.
            let rc = unsafe {
                ffi::m26r_open(cdev.as_ptr(), port, gid_index, sp, sl, recv.as_ptr(), recv_len, slots, hp, hl, &mut ep,
                    err.as_mut_ptr() as *mut core::ffi::c_char, err.len())
            };
            if rc != 0 {
                return Err(err_string(&err));
            }
            Ok(Self { ep, slots, slot_len })
        }
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Bytes per receive slot.
    pub fn slot_len(&self) -> usize {
        self.slot_len
    }

    pub fn local_info(&self) -> Info {
        let mut info = Info::default();
        #[cfg(feature = "rdma")]
        // SAFETY: live endpoint; out is a valid Info.
        unsafe {
            ffi::m26r_local_info(self.ep, &mut info)
        };
        info
    }

    #[allow(unused_variables)]
    pub fn connect(&mut self, remote: &Info) -> Result<(), String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("no rdma".into())
        }
        #[cfg(feature = "rdma")]
        {
            let mut err = [0u8; 256];
            // SAFETY: live endpoint, valid remote info.
            let rc = unsafe { ffi::m26r_connect(self.ep, remote, err.as_mut_ptr() as *mut core::ffi::c_char, err.len()) };
            if rc != 0 {
                return Err(err_string(&err));
            }
            Ok(())
        }
    }

    #[allow(unused_variables)]
    pub fn post_recv(&mut self, slot: u32) -> Result<(), String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("no rdma".into())
        }
        #[cfg(feature = "rdma")]
        {
            // SAFETY: live endpoint.
            let rc = unsafe { ffi::m26r_post_recv(self.ep, slot) };
            if rc != 0 {
                return Err(format!("ibv_post_recv slot {slot}: errno {rc}"));
            }
            Ok(())
        }
    }

    /// SEND `hdr` `(offset, len)` from the header buffer then `body` from the send
    /// buffer (either may be absent), signaled.
    #[allow(unused_variables)]
    pub fn post_send(&mut self, hdr: Option<(usize, usize)>, body: Option<(usize, usize)>) -> Result<(), String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("no rdma".into())
        }
        #[cfg(feature = "rdma")]
        {
            let (ho, hl) = hdr.unwrap_or((0, 0));
            let (bo, bl) = body.unwrap_or((0, 0));
            // SAFETY: live endpoint; the shim bounds-checks against the registrations.
            let rc = unsafe { ffi::m26r_post_send(self.ep, ho, hl, bo, bl) };
            if rc != 0 {
                return Err(format!("ibv_post_send: errno {rc}"));
            }
            Ok(())
        }
    }

    /// Wait for the SEND completion (`None`: forever).
    #[allow(unused_variables)]
    pub fn wait_send(&mut self, timeout: Option<Duration>) -> Result<(), String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("no rdma".into())
        }
        #[cfg(feature = "rdma")]
        {
            // SAFETY: live endpoint.
            match unsafe { ffi::m26r_wait_send(self.ep, us(timeout)) } {
                0 => Ok(()),
                1 => Err("rdma send completion timed out".into()),
                s => Err(format!("rdma send completion status {}", -s)),
            }
        }
    }

    /// Wait for one RECV completion: `Some((slot, len))`, or `None` on timeout.
    /// Busy-polls for `spin`, then backs off with 20 us sleeps.
    #[allow(unused_variables)]
    pub fn wait_recv(&mut self, spin: Duration, timeout: Option<Duration>) -> Result<Option<(u32, usize)>, String> {
        #[cfg(not(feature = "rdma"))]
        {
            Err("no rdma".into())
        }
        #[cfg(feature = "rdma")]
        {
            let (mut slot, mut len) = (0u32, 0u32);
            // SAFETY: live endpoint; out pointers valid.
            match unsafe { ffi::m26r_wait_recv(self.ep, us(Some(spin)), us(timeout), &mut slot, &mut len) } {
                0 => Ok(Some((slot, len as usize))),
                1 => Ok(None),
                s => Err(format!("rdma recv completion status {}", -s)),
            }
        }
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        #[cfg(feature = "rdma")]
        // SAFETY: opened by m26r_open, closed once.
        unsafe {
            ffi::m26r_close(self.ep)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_round_trips() {
        let i = Info { qpn: 0x1234, psn: 0xabcdef, gid: [7; 16], mtu: 5, recv_slots: 4 };
        assert_eq!(Info::from_bytes(&i.to_bytes()), Some(i));
        assert_eq!(i.peer_slots(), 4);
    }

    /// A peer built before the receive-slot word sent zero there (its `to_bytes` wrote bytes
    /// 0..28 of a zeroed buffer) and posts two slots; a peer that says, what it says.
    #[test]
    fn a_handshake_without_receive_slots_means_two() {
        let mut old = [0u8; INFO_LEN];
        old[0..4].copy_from_slice(&7u32.to_le_bytes());
        old[24..28].copy_from_slice(&5u32.to_le_bytes());
        let i = Info::from_bytes(&old).unwrap();
        assert_eq!((i.qpn, i.mtu, i.recv_slots, i.peer_slots()), (7, 5, 0, LEGACY_RECV_SLOTS));
        let new = Info { recv_slots: 3, ..i }.to_bytes();
        assert_eq!(new[..28], old[..28], "the other fields are where they were");
        assert_eq!(Info::from_bytes(&new).unwrap().peer_slots(), 3);
    }

    #[test]
    fn parses_ipv4_mapped_gid() {
        // A documentation address (192.0.2.0/24), IPv4-mapped.
        let g = parse_gid("0000:0000:0000:0000:0000:ffff:c000:0203").unwrap();
        assert_eq!(&g[10..16], &[0xff, 0xff, 192, 0, 2, 3]);
    }

    /// A GID table slot: `(gid, type)`, or `None` for an empty slot.
    type Slot<'a> = Option<(&'a str, &'a str)>;

    /// A tree laid out as `/sys/class/infiniband` in a temporary directory: per device and port,
    /// `gids/N` and `gid_attrs/types/N` for each slot of its table. An empty slot reads as the
    /// zero GID and has no type to read, as the kernel shows one.
    fn sysfs(name: &str, ports: &[(&str, u8, &[Slot])]) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("glm53f-rdma-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for &(dev, port, table) in ports {
            let p = root.join(dev).join("ports").join(port.to_string());
            std::fs::create_dir_all(p.join("gids")).unwrap();
            std::fs::create_dir_all(p.join("gid_attrs/types")).unwrap();
            for (i, slot) in table.iter().enumerate() {
                let gid = slot.map_or("0000:0000:0000:0000:0000:0000:0000:0000", |s| s.0);
                std::fs::write(p.join(format!("gids/{i}")), format!("{gid}\n")).unwrap();
                if let Some((_, ty)) = slot {
                    std::fs::write(p.join(format!("gid_attrs/types/{i}")), format!("{ty}\n")).unwrap();
                }
            }
        }
        root
    }

    #[test]
    fn find_roce_v2_skips_empty_gid_slots() {
        let (v1, v2) = ("IB/RoCE v1", "RoCE v2");
        let link = "fe80:0000:0000:0000:0000:0000:0000:0001";
        // Documentation addresses 192.0.2.10 and 192.0.2.11, IPv4-mapped.
        let (a, b) = ("0000:0000:0000:0000:0000:ffff:c000:020a", "0000:0000:0000:0000:0000:ffff:c000:020b");
        // A contiguous table, as a port lists its GIDs: link-local then the address, v1 and v2.
        let full: [Slot; 4] = [Some((link, v1)), Some((link, v2)), Some((a, v1)), Some((a, v2))];
        // A table with a hole: an address removed from slots 2 and 3, another added after it.
        let holed: [Slot; 6] = [Some((link, v1)), Some((link, v2)), None, None, Some((b, v1)), Some((b, v2))];
        let root = sysfs("gids", &[("mlx5_0", 1, &full[..]), ("mlx5_1", 1, &holed[..])]);
        let ip = |d| std::net::Ipv4Addr::new(192, 0, 2, d);
        assert_eq!(find_roce_v2_in(&root, ip(10)), Some(("mlx5_0".to_string(), 1, 3)));
        // Past the empty slots.
        assert_eq!(find_roce_v2_in(&root, ip(11)), Some(("mlx5_1".to_string(), 1, 5)));
        assert_eq!(find_roce_v2_in(&root, ip(12)), None);
        // Only RoCE v2 GIDs count: the address's v1 GID alone is not found.
        let v1_only = sysfs("gids-v1", &[("mlx5_0", 1, &full[..3])]);
        assert_eq!(find_roce_v2_in(&v1_only, ip(10)), None);
        assert_eq!(find_roce_v2_in(&root.join("absent"), ip(10)), None);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&v1_only);
    }
}
