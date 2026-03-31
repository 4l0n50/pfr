//! Operation counting for theoretical cost analysis.
//!
//! Provides a thread-local [`OpCounter`] that can be activated before running
//! a PFR phase and read afterwards.  Every FFT, MSM, batch-inversion, pairing,
//! and G2 scalar-multiplication site in the prover/verifier calls the `record_*`
//! helpers here, which are no-ops when counting is disabled.
//!
//! # Usage
//! ```rust,ignore
//! use pfr::counting::OpCounter;
//!
//! OpCounter::reset();
//! commit_statement(&pk, &row, &col, rng);
//! let counts = OpCounter::take();
//! println!("{counts:#?}");
//! ```

use std::cell::RefCell;

// ---------------------------------------------------------------------------
// The counter type
// ---------------------------------------------------------------------------

/// Recorded operation counts for one PFR phase.
///
/// Each field is a list of `(size, count)` pairs: `count` operations of the
/// given `size` were recorded.  Sizes are accumulated in insertion order.
#[derive(Debug, Clone, Default)]
pub struct OpCounts {
    /// Forward FFTs: (domain size, count).
    pub ffts: Vec<(usize, usize)>,
    /// Inverse FFTs: (domain size, count).
    pub iffts: Vec<(usize, usize)>,
    /// G1 MSMs: (number of points, count).
    pub msm_g1: Vec<(usize, usize)>,
    /// G2 scalar multiplications.
    pub g2_scalarmul: usize,
    /// Full pairings (Miller loop + final exp).
    pub pairings: usize,
    /// Batch inversions: (number of elements inverted, count).
    pub batch_inv: Vec<(usize, usize)>,
    /// Individual field inversions (outside batch).
    pub field_inv: usize,
}

impl OpCounts {
    fn record_fft(&mut self, size: usize) {
        accumulate(&mut self.ffts, size);
    }
    fn record_ifft(&mut self, size: usize) {
        accumulate(&mut self.iffts, size);
    }
    fn record_msm_g1(&mut self, n_points: usize) {
        accumulate(&mut self.msm_g1, n_points);
    }
    fn record_batch_inv(&mut self, n_elems: usize) {
        accumulate(&mut self.batch_inv, n_elems);
    }
    fn record_field_inv(&mut self) {
        self.field_inv += 1;
    }
    fn record_pairing(&mut self) {
        self.pairings += 1;
    }
    fn record_g2_scalarmul(&mut self) {
        self.g2_scalarmul += 1;
    }

    /// Total number of FFT operations (forward + inverse).
    pub fn total_ffts(&self) -> usize {
        self.ffts.iter().map(|(_, c)| c).sum::<usize>()
            + self.iffts.iter().map(|(_, c)| c).sum::<usize>()
    }

    /// Total number of G1 MSMs.
    pub fn total_msm_g1(&self) -> usize {
        self.msm_g1.iter().map(|(_, c)| c).sum()
    }

    /// Print a one-line summary.
    pub fn print_summary(&self, label: &str) {
        println!(
            "{label:25}  FFT={:2}  IFFT={:2}  MSM_G1={:2}  batch_inv={:2}  \
             field_inv={:3}  pairings={:1}  g2_smul={:1}",
            self.ffts.iter().map(|(_, c)| c).sum::<usize>(),
            self.iffts.iter().map(|(_, c)| c).sum::<usize>(),
            self.total_msm_g1(),
            self.batch_inv.iter().map(|(_, c)| c).sum::<usize>(),
            self.field_inv,
            self.pairings,
            self.g2_scalarmul,
        );
        if !self.ffts.is_empty() {
            let detail: Vec<_> = self.ffts.iter()
                .map(|(s, c)| format!("{c}×FFT({s})"))
                .collect();
            println!("  {:25}  fft detail : {}", "", detail.join(", "));
        }
        if !self.iffts.is_empty() {
            let detail: Vec<_> = self.iffts.iter()
                .map(|(s, c)| format!("{c}×IFFT({s})"))
                .collect();
            println!("  {:25}  ifft detail: {}", "", detail.join(", "));
        }
        if !self.msm_g1.is_empty() {
            let detail: Vec<_> = self.msm_g1.iter()
                .map(|(s, c)| format!("{c}×MSM({s})"))
                .collect();
            println!("  {:25}  msm detail : {}", "", detail.join(", "));
        }
        if !self.batch_inv.is_empty() {
            let detail: Vec<_> = self.batch_inv.iter()
                .map(|(s, c)| format!("{c}×batch_inv({s})"))
                .collect();
            println!("  {:25}  inv detail : {}", "", detail.join(", "));
        }
    }
}

/// Add one occurrence of `size` to the accumulator, merging with the last
/// entry if it has the same size (so repeated same-size calls are grouped).
fn accumulate(v: &mut Vec<(usize, usize)>, size: usize) {
    if let Some(last) = v.last_mut() {
        if last.0 == size {
            last.1 += 1;
            return;
        }
    }
    v.push((size, 1));
}

// ---------------------------------------------------------------------------
// Thread-local counter
// ---------------------------------------------------------------------------

thread_local! {
    static COUNTER: RefCell<Option<OpCounts>> = RefCell::new(None);
}

/// Activate counting on the current thread, resetting any previous state.
pub fn reset() {
    COUNTER.with(|c| *c.borrow_mut() = Some(OpCounts::default()));
}

/// Deactivate counting and return the accumulated counts.
/// Returns `None` if counting was not active.
pub fn take() -> Option<OpCounts> {
    COUNTER.with(|c| c.borrow_mut().take())
}

/// Returns `true` if counting is currently active on this thread.
pub fn is_active() -> bool {
    COUNTER.with(|c| c.borrow().is_some())
}

// ---------------------------------------------------------------------------
// Record helpers — called from prover.rs / verifier.rs
// ---------------------------------------------------------------------------

#[inline]
pub fn record_fft(size: usize) {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_fft(size);
        }
    });
}

#[inline]
pub fn record_ifft(size: usize) {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_ifft(size);
        }
    });
}

#[inline]
pub fn record_msm_g1(n_points: usize) {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_msm_g1(n_points);
        }
    });
}

#[inline]
pub fn record_batch_inv(n_elems: usize) {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_batch_inv(n_elems);
        }
    });
}

#[inline]
pub fn record_field_inv() {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_field_inv();
        }
    });
}

#[inline]
pub fn record_pairing() {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_pairing();
        }
    });
}

#[inline]
pub fn record_g2_scalarmul() {
    COUNTER.with(|c| {
        if let Some(ref mut counts) = *c.borrow_mut() {
            counts.record_g2_scalarmul();
        }
    });
}
