//! SHA-256 vectors — the manifest is worthless if the hash is only
//! self-consistent.
//!
//! Classification: **BOTH-RUNS** (no naive flag).
//!
//! Kills: a wrong round count, a wrong padding rule, a wrong endianness, a
//! wrong initial state, and a hex encoder that is not lowercase.

use glm53f_rank::sha256::{hex, sha256, unhex, Sha256};

#[test]
fn empty_string_vector() {
    // FIPS 180-4 / NIST CAVP: SHA-256("") =
    // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn abc_vector() {
    // SHA-256("abc") =
    // ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn two_block_vector() {
    // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") =
    // 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
    let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    assert_eq!(
        hex(&sha256(msg)),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn million_a_vector() {
    // SHA-256(1,000,000 x 'a') =
    // cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0
    let mut h = Sha256::new();
    let chunk = vec![b'a'; 1000];
    for _ in 0..1000 {
        h.update(&chunk);
    }
    assert_eq!(
        hex(&h.finalize()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn streaming_matches_one_shot_at_every_split() {
    // The slicer hashes 0.9 GB layer images in chunks; a streaming bug at a
    // block boundary would silently produce a wrong manifest. Split the message at every
    // offset around the 64-byte block boundary.
    let msg: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let want = sha256(&msg);
    for split in 0..msg.len() {
        let mut h = Sha256::new();
        h.update(&msg[..split]);
        h.update(&msg[split..]);
        assert_eq!(h.finalize(), want, "split at {split}");
    }
    // And byte-at-a-time.
    let mut h = Sha256::new();
    for b in &msg {
        h.update(&[*b]);
    }
    assert_eq!(h.finalize(), want);
}

#[test]
fn hex_round_trip_and_rejects() {
    let d = sha256(b"glm53f-rank");
    let s = hex(&d);
    assert_eq!(s.len(), 64);
    assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(unhex(&s).unwrap(), d);
    // Uppercase is accepted on read (a manifest written by another tool).
    assert_eq!(unhex(&s.to_uppercase()).unwrap(), d);
    // Malformed shas must be rejected, not compared unequal by luck.
    assert!(unhex("").is_err());
    assert!(unhex(&s[..63]).is_err());
    assert!(unhex(&format!("{s}0")).is_err());
    assert!(unhex(&"z".repeat(64)).is_err());
}
