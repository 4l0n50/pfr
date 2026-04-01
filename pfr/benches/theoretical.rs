// Theoretical cost estimation for the PFR protocol on Bn254.
//
// Operation counts are produced by running the real prover/verifier code with
// the thread-local `pfr::counting` instrumentation active.  Every FFT, MSM,
// batch-inversion, pairing, and G2 scalar-mul call site in the prover and
// verifier calls one of the `counting::record_*` helpers, so the counts are
// definitionally correct — they cannot drift from the implementation.
//
// Structure:
//   Part 1  – Micro-benchmarks for each primitive operation (MSM_G1, FFT,
//             IFFT, Finv, batch_inv, pairing, G2 scalar mul).
//   Part 2  – Calibration: quick wall-clock measurements of each primitive
//             at several sizes.
//   Part 3  – Theoretical table: for each (n, m) size, run commit_statement /
//             prove / verify with counting on, look up each recorded operation
//             in the calibration table, and print estimated + (where available)
//             measured times.
//
// Run with:
//   cargo bench --bench theoretical --features std

use ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine};
use ark_ec::{msm::VariableBaseMSM, AffineCurve, PairingEngine, ProjectiveCurve};
use ark_ff::{Field, PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pfr::{commit_statement, counting, prove, verify, PfrPublicKey};
use std::time::Duration;

type E = Bn254;
type F = Fr;

// ---------------------------------------------------------------------------
// Helpers: random data for micro-benchmarks
// ---------------------------------------------------------------------------

fn rand_g1_vec(n: usize) -> Vec<G1Affine> {
    let rng = &mut ark_std::test_rng();
    (0..n).map(|_| G1Projective::rand(rng).into_affine()).collect()
}

fn rand_g2() -> G2Affine {
    ark_bn254::G2Projective::rand(&mut ark_std::test_rng()).into_affine()
}

fn rand_fr_repr_vec(n: usize) -> Vec<<F as PrimeField>::BigInt> {
    let rng = &mut ark_std::test_rng();
    (0..n).map(|_| F::rand(rng).into_repr()).collect()
}

fn rand_fr_vec(n: usize) -> Vec<F> {
    let rng = &mut ark_std::test_rng();
    (0..n).map(|_| F::rand(rng)).collect()
}

// ---------------------------------------------------------------------------
// Benchmark sizes
// ---------------------------------------------------------------------------

const MSM_SIZES: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
const FFT_SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192];

const PFR_SIZES: &[(usize, usize)] = &[
    (64, 64),
    (64, 256),
    (128, 128),
    (128, 512),
    (256, 256),
    (256, 1024),
];
const T: usize = 2;

// ---------------------------------------------------------------------------
// Part 1: Criterion micro-benchmarks
// ---------------------------------------------------------------------------

fn bench_msm_g1(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/msm_g1");
    group.sample_size(20);
    for &k in MSM_SIZES {
        let bases = rand_g1_vec(k);
        let scalars = rand_fr_repr_vec(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| VariableBaseMSM::multi_scalar_mul(&bases, &scalars))
        });
    }
    group.finish();
}

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/fft");
    group.sample_size(20);
    for &d in FFT_SIZES {
        let domain = GeneralEvaluationDomain::<F>::new(d).unwrap();
        let v = rand_fr_vec(domain.size());
        group.bench_with_input(BenchmarkId::from_parameter(domain.size()), &d, |b, _| {
            b.iter(|| domain.fft(&v))
        });
    }
    group.finish();
}

fn bench_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/ifft");
    group.sample_size(20);
    for &d in FFT_SIZES {
        let domain = GeneralEvaluationDomain::<F>::new(d).unwrap();
        let v = rand_fr_vec(domain.size());
        group.bench_with_input(BenchmarkId::from_parameter(domain.size()), &d, |b, _| {
            b.iter(|| domain.ifft(&v))
        });
    }
    group.finish();
}

fn bench_finv(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/finv");
    group.sample_size(200);
    let x = F::rand(&mut ark_std::test_rng());
    group.bench_function("single", |b| b.iter(|| x.inverse().unwrap()));
    group.finish();
}

fn bench_batch_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/batch_inv");
    group.sample_size(20);
    for &k in &[64usize, 256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter_batched(
                || rand_fr_vec(k),
                |mut v| ark_ff::batch_inversion(&mut v),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_pairing(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/pairing");
    group.sample_size(50);
    let p = G1Affine::prime_subgroup_generator();
    let q = G2Affine::prime_subgroup_generator();
    group.bench_function("single", |b| b.iter(|| E::pairing(p, q)));
    group.bench_function("two", |b| {
        b.iter(|| {
            let a = E::pairing(p, q);
            let b2 = E::pairing(p, q);
            a == b2
        })
    });
    group.finish();
}

fn bench_g2_scalarmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive/g2_scalarmul");
    group.sample_size(50);
    let q = rand_g2();
    let s = F::rand(&mut ark_std::test_rng());
    group.bench_function("single", |b| b.iter(|| q.mul(s.into_repr())));
    group.finish();
}

// ---------------------------------------------------------------------------
// Part 2: Calibration — quick wall-clock timing of each primitive
// ---------------------------------------------------------------------------

fn time_ns(mut f: impl FnMut(), iters: usize) -> f64 {
    for _ in 0..3 { f(); } // warmup
    let t = std::time::Instant::now();
    for _ in 0..iters { f(); }
    t.elapsed().as_nanos() as f64 / iters as f64
}

struct Calibration {
    msm_g1: Vec<(usize, f64)>,   // (size, ns)
    fft:    Vec<(usize, f64)>,
    ifft:   Vec<(usize, f64)>,
    finv_ns:          f64,
    batch_inv_ns_per: f64,       // ns per element inverted (amortised)
    pairing_ns:       f64,
    g2_scalarmul_ns:  f64,
}

impl Calibration {
    fn run() -> Self {
        println!("\n[theoretical] Calibrating primitives on Bn254…");

        let msm_g1 = MSM_SIZES.iter().map(|&k| {
            let bases   = rand_g1_vec(k);
            let scalars = rand_fr_repr_vec(k);
            let iters = if k <= 256 { 200 } else if k <= 1024 { 50 } else { 20 };
            let ns = time_ns(|| { let _ = VariableBaseMSM::multi_scalar_mul(&bases, &scalars); }, iters);
            println!("  MSM_G1({k:5}) = {:7.1} µs", ns / 1e3);
            (k, ns)
        }).collect();

        let fft = FFT_SIZES.iter().map(|&d| {
            let domain = GeneralEvaluationDomain::<F>::new(d).unwrap();
            let v = rand_fr_vec(domain.size());
            let iters = if d <= 256 { 2000 } else if d <= 2048 { 400 } else { 80 };
            let ns = time_ns(|| { let _ = domain.fft(&v); }, iters);
            println!("  FFT({:5})    = {:7.1} µs", domain.size(), ns / 1e3);
            (domain.size(), ns)
        }).collect();

        let ifft = FFT_SIZES.iter().map(|&d| {
            let domain = GeneralEvaluationDomain::<F>::new(d).unwrap();
            let v = rand_fr_vec(domain.size());
            let iters = if d <= 256 { 2000 } else if d <= 2048 { 400 } else { 80 };
            let ns = time_ns(|| { let _ = domain.ifft(&v); }, iters);
            println!("  IFFT({:5})   = {:7.1} µs", domain.size(), ns / 1e3);
            (domain.size(), ns)
        }).collect();

        let x = F::rand(&mut ark_std::test_rng());
        let finv_ns = time_ns(|| { let _ = x.inverse().unwrap(); }, 10_000);
        println!("  Finv         = {:7.1} µs", finv_ns / 1e3);

        // Amortised cost per element for batch inversion at a representative size
        let k = 1024usize;
        let batch_inv_ns_per = {
            let v_orig = rand_fr_vec(k);
            let ns = time_ns(|| { let mut v = v_orig.clone(); ark_ff::batch_inversion(&mut v); }, 200);
            ns / k as f64
        };
        println!("  batch_inv/el = {:7.1} ns  (measured at k={k})", batch_inv_ns_per);

        let p = G1Affine::prime_subgroup_generator();
        let q = G2Affine::prime_subgroup_generator();
        let pairing_ns = time_ns(|| { let _ = E::pairing(p, q); }, 100);
        println!("  Pairing      = {:7.1} µs", pairing_ns / 1e3);

        let q2 = rand_g2();
        let s  = F::rand(&mut ark_std::test_rng());
        let g2_scalarmul_ns = time_ns(|| { let _ = q2.mul(s.into_repr()); }, 500);
        println!("  G2 scalarmul = {:7.1} µs", g2_scalarmul_ns / 1e3);

        Calibration { msm_g1, fft, ifft, finv_ns, batch_inv_ns_per, pairing_ns, g2_scalarmul_ns }
    }

    /// Interpolate cost for `size` from the measured table (log-log linear between neighbours).
    fn lookup(table: &[(usize, f64)], size: usize) -> f64 {
        match (
            table.iter().rev().find(|&&(s, _)| s <= size),
            table.iter().find(|&&(s, _)| s >= size),
        ) {
            (_, Some(&(s, t))) if s == size => t,
            (Some(&(s0, t0)), Some(&(s1, t1))) => {
                // linear interpolation in linear space
                t0 + (t1 - t0) * (size - s0) as f64 / (s1 - s0) as f64
            }
            (Some(&(s0, t0)), None) => {
                // Pippenger extrapolation: cost ∝ k / log₂(k)
                let ratio = (size as f64 / (size as f64).log2())
                    / (s0 as f64 / (s0 as f64).log2());
                t0 * ratio
            }
            (None, Some(&(_, t))) => t,
            (None, None) => 0.0,
        }
    }

    fn estimate_ns(&self, counts: &pfr::OpCounts) -> f64 {
        let mut total = 0.0f64;

        for &(size, count) in &counts.msm_g1 {
            total += count as f64 * Self::lookup(&self.msm_g1, size);
        }
        for &(size, count) in &counts.ffts {
            total += count as f64 * Self::lookup(&self.fft, size);
        }
        for &(size, count) in &counts.iffts {
            total += count as f64 * Self::lookup(&self.ifft, size);
        }
        // Each batch inversion: amortised cost per element × batch size
        for &(size, count) in &counts.batch_inv {
            total += count as f64 * size as f64 * self.batch_inv_ns_per;
        }
        // Individual field inversions (e.g. pointwise z_K⁻¹ in round 3)
        total += counts.field_inv as f64 * self.finv_ns;
        total += counts.pairings as f64 * self.pairing_ns;
        total += counts.g2_scalarmul as f64 * self.g2_scalarmul_ns;
        total
    }
}

// ---------------------------------------------------------------------------
// Part 3: Collect real counts + print estimation table
// ---------------------------------------------------------------------------

fn make_indices(n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let mut pairs = Vec::new();
    for r in 0..n {
        for c in (r + T)..n {
            pairs.push((r, c));
        }
    }
    let row: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].0).collect();
    let col: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].1).collect();
    (row, col)
}

fn collect_counts(n: usize, m: usize) -> (pfr::OpCounts, pfr::OpCounts, pfr::OpCounts) {
    let rng = &mut ark_std::test_rng();
    let pk = PfrPublicKey::<E>::setup(n, m, T, rng);
    let (row, col) = make_indices(n, m);

    // --- commit_statement ---
    counting::reset();
    let stmt = commit_statement(&pk, &row, &col, rng);
    let sc_counts = counting::take().unwrap();

    // --- prove ---
    counting::reset();
    let (proof, public_inputs) = prove(&pk, &row, &col, &stmt, rng);
    let pr_counts = counting::take().unwrap();

    // --- verify ---
    counting::reset();
    verify(&pk, &proof, &public_inputs.row_comm, &public_inputs.col_comm, &public_inputs.rowcol_comm);
    let vr_counts = counting::take().unwrap();

    (sc_counts, pr_counts, vr_counts)
}

fn bench_theoretical(c: &mut Criterion) {
    let cal = Calibration::run();

    // Collect counts + estimates for all sizes up front
    let rows: Vec<_> = PFR_SIZES.iter().map(|&(n, m)| {
        let (sc, pr, vr) = collect_counts(n, m);
        let sc_ms = cal.estimate_ns(&sc) / 1e6;
        let pr_ms = cal.estimate_ns(&pr) / 1e6;
        let vr_ms = cal.estimate_ns(&vr) / 1e6;
        (n, m, sc_ms, pr_ms, vr_ms)
    }).collect();

    println!();
    println!("╔═══════════════╦═══════════════════════════════╦═══════════════════════════════╦═══════════════════╗");
    println!("║               ║       stmt_commit             ║           prove               ║      verify       ║");
    println!("║    (n, m)     ║  theor(ms)  [measured(ms)]    ║  theor(ms)  [measured(ms)]    ║  theor   [meas]   ║");
    println!("╠═══════════════╬═══════════════════════════════╬═══════════════════════════════╬═══════════════════╣");
    for &(n, m, sc_ms, pr_ms, vr_ms) in &rows {
        println!(
            "║ n={n:4}, m={m:4} ║      {sc_ms:6.2}                      ║     {pr_ms:6.2}                      ║     {vr_ms:6.2}          ║",
        );
    }
    println!("╚═══════════════╩═══════════════════════════════╩═══════════════════════════════╩═══════════════════╝");

    // Run the criterion micro-benchmarks so they appear in the HTML report
    bench_msm_g1(c);
    bench_fft(c);
    bench_ifft(c);
    bench_finv(c);
    bench_batch_inv(c);
    bench_pairing(c);
    bench_g2_scalarmul(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1));
    targets = bench_theoretical
}
criterion_main!(benches);
