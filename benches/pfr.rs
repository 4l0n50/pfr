// Benchmarks for the ZK-PFR protocol (Appendix B, IMPR-FHFC paper).
//
// Run with:
//   cargo bench --bench pfr --features std
//
// Each benchmark covers one phase, parameterized by (n, m):
//   - round{1..5}_poly : polynomial arithmetic only
//   - round{1..3,5}_commit : KZG commitment only
//   - prove / verify : end-to-end


use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pfr::{prove, round_five, round_four, round_one, round_three, round_two, verify, PfrPublicKey};

type PC = ark_poly_commit::marlin_pc::MarlinKZG10<
    ark_bls12_381::Bls12_381,
    ark_poly::univariate::DensePolynomial<Fr>,
>;

// (n, m) pairs to benchmark: n = |H|, m = |K|, m must be a multiple of n.
const SIZES: &[(usize, usize)] = &[
    (256, 256),
    (256, 1024),
    (512, 512),
    (512, 2048),
    (1024, 1024),
    (1024, 4096),
];
const T: usize = 2;

fn make_indices(n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for r in 0..n {
        for c in (r + T)..n {
            pairs.push((r, c));
        }
    }
    let row: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].0).collect();
    let col: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].1).collect();
    (row, col)
}

fn bench_round1_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("round1_poly");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| round_one(&pk, &row, &col, &mut ark_std::test_rng()))
        });
    }
    group.finish();
}

fn bench_round1_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("round1_commit");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let state = round_one(&pk, &row, &col, rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| PC::commit(&pk.ck, state.polynomials.iter(), None).unwrap())
        });
    }
    group.finish();
}

fn bench_round2_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("round2_poly");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| round_two(&pk, &r1, beta, &mut ark_std::test_rng()))
        });
    }
    group.finish();
}

fn bench_round2_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("round2_commit");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| PC::commit(&pk.ck, r2.polynomials.iter(), None).unwrap())
        });
    }
    group.finish();
}

fn bench_round3_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("round3_poly");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);
        let eta = Fr::rand(rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| round_three(&pk, &r1, &r2, eta))
        });
    }
    group.finish();
}

fn bench_round3_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("round3_commit");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);
        let eta = Fr::rand(rng);
        let r3 = round_three(&pk, &r1, &r2, eta);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| PC::commit(&pk.ck, r3.polynomials.iter(), None).unwrap())
        });
    }
    group.finish();
}

fn bench_round4_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("round4_poly");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let alpha = Fr::rand(rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| round_four(&pk, &r1, alpha))
        });
    }
    group.finish();
}

fn bench_round5_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("round5_poly");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);
        let eta = Fr::rand(rng);
        let r3 = round_three(&pk, &r1, &r2, eta);
        let alpha = Fr::rand(rng);
        let r4 = round_four(&pk, &r1, alpha);
        let delta = Fr::rand(rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| round_five(&pk, &r1, &r2, &r3, &r4, delta))
        });
    }
    group.finish();
}

fn bench_round5_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("round5_commit");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let r1 = round_one(&pk, &row, &col, rng);
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);
        let eta = Fr::rand(rng);
        let r3 = round_three(&pk, &r1, &r2, eta);
        let alpha = Fr::rand(rng);
        let r4 = round_four(&pk, &r1, alpha);
        let delta = Fr::rand(rng);
        let q_poly = round_five(&pk, &r1, &r2, &r3, &r4, delta);
        let q_labeled = LabeledPolynomial::new("Q".into(), q_poly, None, None);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| PC::commit(&pk.ck, vec![&q_labeled], None).unwrap())
        });
    }
    group.finish();
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("prove");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| prove(&pk, &row, &col, &mut ark_std::test_rng()))
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let (proof, public_inputs) = prove(&pk, &row, &col, rng);
        group.bench_with_input(BenchmarkId::new("n,m", format!("{n},{m}")), &(n, m), |b, _| {
            b.iter(|| {
                verify(
                    &pk,
                    &proof,
                    &public_inputs.row_comm,
                    &public_inputs.col_comm,
                    &public_inputs.rowcol_comm,
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_round1_poly,
    bench_round1_commit,
    bench_round2_poly,
    bench_round2_commit,
    bench_round3_poly,
    bench_round3_commit,
    bench_round4_poly,
    bench_round5_poly,
    bench_round5_commit,
    bench_prove,
    bench_verify,
);
criterion_main!(benches);
