// Benchmarks for the ZK-PFR protocol (Appendix B, IMPR-FHFC paper).
//
// Run with:
//   cargo bench --bench pfr --features std
//
// For each (n, m) size, all rounds are set up once sequentially.
// Each round's poly and commit are benchmarked independently.


use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use criterion::{criterion_group, criterion_main, Criterion};
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

fn bench_all(c: &mut Criterion) {
    for &(n, m) in SIZES {
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, T, rng);
        let (row, col) = make_indices(n, m);
        let label = format!("{n},{m}");

        // ── Round 1 ──────────────────────────────────────────────────────────
        let r1 = round_one(&pk, &row, &col, rng);

        c.bench_function(&format!("round1_poly/{label}"), |b| {
            b.iter(|| round_one(&pk, &row, &col, &mut ark_std::test_rng()))
        });
        c.bench_function(&format!("round1_commit/{label}"), |b| {
            b.iter(|| PC::commit(&pk.ck, r1.polynomials.iter(), None).unwrap())
        });

        // ── Round 2 ──────────────────────────────────────────────────────────
        let beta = Fr::rand(rng);
        let r2 = round_two(&pk, &r1, beta, rng);

        c.bench_function(&format!("round2_poly/{label}"), |b| {
            b.iter(|| round_two(&pk, &r1, beta, &mut ark_std::test_rng()))
        });
        c.bench_function(&format!("round2_commit/{label}"), |b| {
            b.iter(|| PC::commit(&pk.ck, r2.polynomials.iter(), None).unwrap())
        });

        // ── Round 3 ──────────────────────────────────────────────────────────
        let eta = Fr::rand(rng);
        let r3 = round_three(&pk, &r1, &r2, eta);

        c.bench_function(&format!("round3_poly/{label}"), |b| {
            b.iter(|| round_three(&pk, &r1, &r2, eta))
        });
        c.bench_function(&format!("round3_commit/{label}"), |b| {
            b.iter(|| PC::commit(&pk.ck, r3.polynomials.iter(), None).unwrap())
        });

        // ── Round 4 (evaluations only, no commit) ────────────────────────────
        let alpha = Fr::rand(rng);
        let r4 = round_four(&pk, &r1, alpha);

        c.bench_function(&format!("round4_poly/{label}"), |b| {
            b.iter(|| round_four(&pk, &r1, alpha))
        });

        // ── Round 5 ──────────────────────────────────────────────────────────
        let delta = Fr::rand(rng);
        let q_poly = round_five(&pk, &r1, &r2, &r3, &r4, delta);
        let q_labeled = LabeledPolynomial::new("Q".into(), q_poly, None, None);

        c.bench_function(&format!("round5_poly/{label}"), |b| {
            b.iter(|| round_five(&pk, &r1, &r2, &r3, &r4, delta))
        });
        c.bench_function(&format!("round5_commit/{label}"), |b| {
            b.iter(|| PC::commit(&pk.ck, vec![&q_labeled], None).unwrap())
        });

        // ── End-to-end ───────────────────────────────────────────────────────
        let (proof, public_inputs) = prove(&pk, &row, &col, rng);

        c.bench_function(&format!("prove/{label}"), |b| {
            b.iter(|| prove(&pk, &row, &col, &mut ark_std::test_rng()))
        });
        c.bench_function(&format!("verify/{label}"), |b| {
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
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
