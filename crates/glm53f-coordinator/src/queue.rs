//! The bounded request queue (mimo26f-afd perf reset V3; DS41RT v15's `--http-queue-depth` and
//! `--http-queue-wait-ms`): at most `depth` jobs sent to the scheduler and not yet taken
//! (`GLM53F_QUEUE_DEPTH`, default the slot count, 16); at most `depth` more callers wait up to
//! `wait` (`GLM53F_QUEUE_WAIT_MS`, default 25,000) for a place; any other caller is refused, and
//! the API answers 429 with `Retry-After` before the response starts.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use glm53f_api::engine::QueuePlace;

pub struct Queue {
    queued: Mutex<usize>,
    freed: Condvar,
    waiters: AtomicUsize,
    depth: usize,
    wait: Duration,
}

impl Queue {
    pub fn new(depth: usize, wait: Duration) -> Arc<Queue> {
        Arc::new(Queue { queued: Mutex::new(0), freed: Condvar::new(), waiters: AtomicUsize::new(0), depth: depth.max(1), wait })
    }

    /// `GLM53F_QUEUE_DEPTH` (default `slots`) and `GLM53F_QUEUE_WAIT_MS` (default 25,000).
    pub fn from_env(slots: usize) -> Arc<Queue> {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Queue::new(env("GLM53F_QUEUE_DEPTH", slots), Duration::from_millis(env("GLM53F_QUEUE_WAIT_MS", 25_000) as u64))
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Places taken and not yet given back.
    pub fn queued(&self) -> usize {
        *self.queued.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A place in the queue, waiting up to the wait budget while it is full. `Err` when the queue
    /// and its waiters are full or the wait expired. Dropping the place gives it back.
    pub fn admit(self: &Arc<Self>) -> Result<QueuePlace, String> {
        let q = self.clone();
        let busy = |why: &str| {
            eprintln!("[coordinator] 429: the queue (depth {}) is full and {why}", q.depth);
            Err("request queue is full or its wait budget expired".to_string())
        };
        let mut n = q.queued.lock().unwrap_or_else(|p| p.into_inner());
        if *n >= q.depth {
            if q.waiters.fetch_add(1, Ordering::Relaxed) >= q.depth {
                q.waiters.fetch_sub(1, Ordering::Relaxed);
                drop(n);
                return busy("so are its waiters");
            }
            let deadline = Instant::now() + q.wait;
            while *n >= q.depth {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    q.waiters.fetch_sub(1, Ordering::Relaxed);
                    drop(n);
                    return busy("the wait expired");
                }
                n = q.freed.wait_timeout(n, left).unwrap_or_else(|p| p.into_inner()).0;
            }
            q.waiters.fetch_sub(1, Ordering::Relaxed);
        }
        *n += 1;
        drop(n);
        let q2 = q.clone();
        Ok(QueuePlace::new(move || {
            *q2.queued.lock().unwrap_or_else(|p| p.into_inner()) -= 1;
            q2.freed.notify_one();
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_queue_refuses_after_its_waiters_and_the_wait() {
        let q = Queue::new(2, Duration::from_millis(50));
        let a = q.admit().expect("first place");
        let _b = q.admit().expect("second place");
        assert_eq!(q.queued(), 2);
        // Full: this caller waits 50 ms, then is refused.
        let t0 = Instant::now();
        assert!(q.admit().is_err());
        assert!(t0.elapsed() >= Duration::from_millis(45));
        // A place given back while a caller waits goes to it.
        let q2 = q.clone();
        let waiter = std::thread::spawn(move || q2.admit().map(|_| ()));
        std::thread::sleep(Duration::from_millis(10));
        drop(a);
        assert!(waiter.join().unwrap().is_ok());
        assert_eq!(q.queued(), 1, "the waiter's place was dropped when its thread ended");
    }

    #[test]
    fn waiters_beyond_the_depth_are_refused_at_once() {
        let q = Queue::new(1, Duration::from_secs(5));
        let held = q.admit().unwrap();
        let q1 = q.clone();
        let w = std::thread::spawn(move || q1.admit().map(|_| ()));
        // Let the first waiter take its waiting place.
        while q.waiters.load(Ordering::Relaxed) == 0 {
            std::thread::yield_now();
        }
        let t0 = Instant::now();
        assert!(q.admit().is_err(), "a second waiter over depth 1 is refused");
        assert!(t0.elapsed() < Duration::from_secs(1), "without waiting");
        drop(held);
        assert!(w.join().unwrap().is_ok());
    }
}
