// Comparison benchmarks: PFR (this) vs TStrictlyLowerTriangular (theirs).
//
// Both use Bn254 and the same underlying matrix.  This convention is the
// transpose of theirs (their row ↔ our col), so the same valid pairs are
// encoded with row/col swapped.
//
// Phases benchmarked for each system:
//   init          – key generation (setup + trim + any one-time precomputation)
//   stmt_commit   – commit to the statement (row/col polynomials)
//   prove         – full proof generation given committed statement
//   verify        – verify the proof
//   proof_size    – serialized byte count, printed once per size
//
// Run with:
//   cargo bench --bench comparison --features std

use ark_bn254::Bn254;
use ark_ec::PairingEngine;
use ark_ff::to_bytes;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, UVPolynomial,
};
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use ark_serialize::CanonicalSerialize;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use fiat_shamir_rng::{FiatShamirRng as TheirFiatShamirRng, SimpleHashFiatShamirRng};
use homomorphic_poly_commit::marlin_kzg::KZG10;
use pfr::{commit_statement, prove, verify, PfrPublicKey};
use proof_of_function_relation::t_strictly_lower_triangular_test::TStrictlyLowerTriangular;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type E = Bn254;
type F = <E as PairingEngine>::Fr;

// Their PC is KZG10<Bn254> = homomorphic_poly_commit::marlin_kzg::KZG10<Bn254>,
// which is the same underlying type as our MarlinKZG10<Bn254, DensePolynomial<Fr>>.
type TheirPC = KZG10<E>;
type TheirFS = SimpleHashFiatShamirRng<blake2::Blake2s, rand_chacha::ChaChaRng>;

// ---------------------------------------------------------------------------
// Benchmark sizes: (n, m) pairs.  m must be a multiple of n.
// ---------------------------------------------------------------------------
const SIZES: &[(usize, usize)] = &[
    (64, 64),
    (64, 256),
    (128, 128),
    (128, 512),
    (256, 256),
    (256, 1024),
];
const T: usize = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate m valid (row, col) index pairs with row < col and col >= T.
fn make_indices(n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for r in 0..n {
        for c in (r + T)..n {
            pairs.push((r, c));
        }
    }
    assert!(!pairs.is_empty(), "no valid pairs for n={}, t={}", n, T);
    let row: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].0).collect();
    let col: Vec<usize> = (0..m).map(|i| pairs[i % pairs.len()].1).collect();
    (row, col)
}

/// Build their (row_evals, col_evals) from our (row, col) index arrays.
/// Their encoding is the transpose: their_row[i] = ω^{this_col[i]}, their_col[i] = ω^{this_row[i]}.
fn their_evals(
    domain_h: &GeneralEvaluationDomain<F>,
    this_row: &[usize],
    this_col: &[usize],
) -> (Vec<F>, Vec<F>) {
    let their_row: Vec<F> = this_col.iter().map(|&c| domain_h.element(c)).collect();
    let their_col: Vec<F> = this_row.iter().map(|&r| domain_h.element(r)).collect();
    (their_row, their_col)
}

/// Compute their degree parameters for a given m.
/// enforced_degree_bound = domain_k.size() + 1, where domain_k rounds m up to next power of 2.
fn their_degree_params(m: usize) -> (usize, usize, usize) {
    let domain_k_size = GeneralEvaluationDomain::<F>::new(m).unwrap().size();
    // FIX: Pad max_degree to allow the prover to commit to larger intermediate
    // polynomials (like quotient/blinding polys). domain_k_size + 3 gives us
    // 67 coefficients (degree 66) for m=64, which matches the error requirement.
    let max_degree = domain_k_size + 3;
    let enforced_degree_bound = domain_k_size + 1;
    let enforced_hiding_bound = 1;
    (max_degree, enforced_degree_bound, enforced_hiding_bound)
}

// ---------------------------------------------------------------------------
// Their init: PC::setup + PC::trim
// ---------------------------------------------------------------------------
struct TheirInit {
    ck: <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::CommitterKey,
    vk: <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::VerifierKey,
    enforced_degree_bound: usize,
    enforced_hiding_bound: usize,
    domain_k: GeneralEvaluationDomain<F>,
    domain_h: GeneralEvaluationDomain<F>,
}

fn their_init(n: usize, m: usize) -> TheirInit {
    let rng = &mut ark_std::test_rng();
    let (max_degree, enforced_degree_bound, enforced_hiding_bound) = their_degree_params(m);
    let pp = TheirPC::setup(max_degree, None, rng).unwrap();
    let (ck, vk) = TheirPC::trim(
        &pp,
        max_degree,
        enforced_hiding_bound,
        Some(&[2, enforced_degree_bound]),
    )
    .unwrap();
    TheirInit {
        ck,
        vk,
        enforced_degree_bound,
        enforced_hiding_bound,
        domain_k: GeneralEvaluationDomain::<F>::new(m).unwrap(),
        domain_h: GeneralEvaluationDomain::<F>::new(n).unwrap(),
    }
}

// ---------------------------------------------------------------------------
// Their statement: interpolate polys + commit
// ---------------------------------------------------------------------------
struct TheirStmt {
    row_poly: LabeledPolynomial<F, DensePolynomial<F>>,
    col_poly: LabeledPolynomial<F, DensePolynomial<F>>,
    row_rand: <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::Randomness,
    col_rand: <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::Randomness,
    row_labeled_comm: ark_poly_commit::LabeledCommitment<
        <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::Commitment,
    >,
    col_labeled_comm: ark_poly_commit::LabeledCommitment<
        <TheirPC as PolynomialCommitment<F, DensePolynomial<F>>>::Commitment,
    >,
}

fn their_commit_stmt(init: &TheirInit, their_row_evals: &[F], their_col_evals: &[F]) -> TheirStmt {
    let rng = &mut ark_std::test_rng();
    let row_poly =
        DensePolynomial::<F>::from_coefficients_slice(&init.domain_k.ifft(their_row_evals));
    let col_poly =
        DensePolynomial::<F>::from_coefficients_slice(&init.domain_k.ifft(their_col_evals));
    let row_labeled = LabeledPolynomial::new(
        "row_poly".into(),
        row_poly,
        Some(init.enforced_degree_bound),
        Some(init.enforced_hiding_bound),
    );
    let col_labeled = LabeledPolynomial::new(
        "col_poly".into(),
        col_poly,
        Some(init.enforced_degree_bound),
        Some(init.enforced_hiding_bound),
    );
    let (mut comms, mut rands) = TheirPC::commit(
        &init.ck,
        &[row_labeled.clone(), col_labeled.clone()],
        Some(rng),
    )
    .unwrap();
    let col_rand = rands.remove(1);
    let row_rand = rands.remove(0);
    let col_labeled_comm = comms.remove(1);
    let row_labeled_comm = comms.remove(0);
    TheirStmt {
        row_poly: row_labeled,
        col_poly: col_labeled,
        row_rand,
        col_rand,
        row_labeled_comm,
        col_labeled_comm,
    }
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

// Per-size precomputed state shared across bench functions.
struct SizeState {
    n: usize,
    m: usize,
    this_pk: PfrPublicKey<E>,
    this_row: Vec<usize>,
    this_col: Vec<usize>,
    this_stmt: pfr::PfrStatement<E>,
    their: TheirInit,
    their_stmt: TheirStmt,
    enforced_degree_bound: usize,
}

fn build_states() -> Vec<SizeState> {
    SIZES.iter().map(|&(n, m)| {
        let (this_row, this_col) = make_indices(n, m);
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let (their_row_evals, their_col_evals) = their_evals(&domain_h, &this_row, &this_col);
        let this_pk = PfrPublicKey::<E>::setup(n, m, T, &mut ark_std::test_rng());
        let this_stmt = commit_statement(&this_pk, &this_row, &this_col, &mut ark_std::test_rng());
        let their = their_init(n, m);
        let their_stmt = their_commit_stmt(&their, &their_row_evals, &their_col_evals);
        let (_, enforced_degree_bound, _) = their_degree_params(m);
        SizeState { n, m, this_pk, this_row, this_col, this_stmt, their, their_stmt, enforced_degree_bound }
    }).collect()
}

fn bench_this_prove(c: &mut Criterion) {
    let states = build_states();
    let mut group = c.benchmark_group("this/prove");
    group.sample_size(20);
    for s in &states {
        let label = format!("n={},m={}", s.n, s.m);
        group.bench_function(&label, |b| {
            b.iter(|| prove(&s.this_pk, &s.this_row, &s.this_col, &s.this_stmt, &mut ark_std::test_rng()))
        });
    }
    group.finish();
}

fn bench_theirs_prove(c: &mut Criterion) {
    let states = build_states();
    let mut group = c.benchmark_group("theirs/prove");
    group.sample_size(20);
    for s in &states {
        let label = format!("n={},m={}", s.n, s.m);
        group.bench_function(&label, |b| {
            b.iter(|| {
                let mut fs_rng = TheirFS::initialize(&to_bytes!(b"bench").unwrap());
                TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::prove(
                    &s.their.ck, T, &s.their.domain_k, &s.their.domain_h,
                    &s.their_stmt.row_poly, &s.their_stmt.row_labeled_comm, &s.their_stmt.row_rand,
                    &s.their_stmt.col_poly, &s.their_stmt.col_labeled_comm, &s.their_stmt.col_rand,
                    Some(s.enforced_degree_bound), &mut fs_rng, &mut ark_std::test_rng(),
                ).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_comparison(c: &mut Criterion) {
    for &(n, m) in SIZES {
        let label = format!("n={n},m={m}");
        let (this_row, this_col) = make_indices(n, m);
        let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let (their_row_evals, their_col_evals) = their_evals(&domain_h, &this_row, &this_col);

        let this_pk = PfrPublicKey::<E>::setup(n, m, T, &mut ark_std::test_rng());
        let their = their_init(n, m);
        let (max_degree, enforced_degree_bound, enforced_hiding_bound) = their_degree_params(m);

        // ── This: init ───────────────────────────────────────────────────────
        c.bench_function(&format!("this/init/{label}"), |b| {
            b.iter(|| PfrPublicKey::<E>::setup(n, m, T, &mut ark_std::test_rng()))
        });

        // ── Theirs: init ─────────────────────────────────────────────────────
        c.bench_function(&format!("theirs/init/{label}"), |b| {
            b.iter(|| {
                let rng = &mut ark_std::test_rng();
                let pp = TheirPC::setup(max_degree, None, rng).unwrap();
                TheirPC::trim(&pp, max_degree, enforced_hiding_bound, Some(&[2, enforced_degree_bound])).unwrap()
            })
        });

        // ── This: stmt_commit ────────────────────────────────────────────────
        let this_stmt = commit_statement(&this_pk, &this_row, &this_col, &mut ark_std::test_rng());
        c.bench_function(&format!("this/stmt_commit/{label}"), |b| {
            b.iter(|| commit_statement(&this_pk, &this_row, &this_col, &mut ark_std::test_rng()))
        });

        // ── Theirs: stmt_commit ──────────────────────────────────────────────
        let their_stmt = their_commit_stmt(&their, &their_row_evals, &their_col_evals);
        c.bench_function(&format!("theirs/stmt_commit/{label}"), |b| {
            b.iter(|| their_commit_stmt(&their, &their_row_evals, &their_col_evals))
        });

        // ── This: verify ─────────────────────────────────────────────────────
        let (this_proof, this_public_inputs) = prove(&this_pk, &this_row, &this_col, &this_stmt, &mut ark_std::test_rng());
        c.bench_function(&format!("this/verify/{label}"), |b| {
            b.iter(|| verify(&this_pk, &this_proof, &this_public_inputs.row_comm, &this_public_inputs.col_comm, &this_public_inputs.rowcol_comm))
        });

        // ── Theirs: verify ───────────────────────────────────────────────────
        let their_proof = {
            let mut fs_rng = TheirFS::initialize(&to_bytes!(b"bench").unwrap());
            TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::prove(
                &their.ck, T, &their.domain_k, &their.domain_h,
                &their_stmt.row_poly, &their_stmt.row_labeled_comm, &their_stmt.row_rand,
                &their_stmt.col_poly, &their_stmt.col_labeled_comm, &their_stmt.col_rand,
                Some(enforced_degree_bound), &mut fs_rng, &mut ark_std::test_rng(),
            ).unwrap()
        };
        c.bench_function(&format!("theirs/verify/{label}"), |b| {
            b.iter_batched(
                || {
                    let mut fs_rng = TheirFS::initialize(&to_bytes!(b"bench").unwrap());
                    TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::prove(
                        &their.ck, T, &their.domain_k, &their.domain_h,
                        &their_stmt.row_poly, &their_stmt.row_labeled_comm, &their_stmt.row_rand,
                        &their_stmt.col_poly, &their_stmt.col_labeled_comm, &their_stmt.col_rand,
                        Some(enforced_degree_bound), &mut fs_rng, &mut ark_std::test_rng(),
                    ).unwrap()
                },
                |proof| {
                    let mut fs_rng = TheirFS::initialize(&to_bytes!(b"bench").unwrap());
                    TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::verify(
                        &their.vk, &their.ck, T, &their.domain_k, &their.domain_h,
                        &their_stmt.row_labeled_comm, &their_stmt.col_labeled_comm,
                        Some(enforced_degree_bound), proof, &mut fs_rng,
                    ).is_ok()
                },
                BatchSize::SmallInput,
            )
        });

        // ── Proof sizes ───────────────────────────────────────────────────────
        let their_proof_size = their_proof.serialized_size();
        let this_proof_size = serialize_this_proof(this_proof);
        println!("proof_size  {label:20}  this={this_proof_size:6} B  theirs={their_proof_size:6} B");
    }
}

/// This proof size: Labeled commitments does not implement ark serialize, so
/// sum up serialized bytes of all G1 commitments + Fr field elements.
/// Each Comm<E> = LabeledCommitment<MarlinKZG10Commitment>; the inner commitment
/// contains a G1Affine point (and possibly a shifted G1Affine). We serialize
//// the inner .commitment() of each comm and the four Fr evaluations.
fn serialize_this_proof(this_proof: pfr::PfrProof<ark_ec::bn::Bn<ark_bn254::Parameters>>) -> usize {
    let this_proof_size = {
        let mut buf = Vec::new();
        this_proof.r_comm.commitment().serialize(&mut buf).unwrap();
        this_proof.c_comm.commitment().serialize(&mut buf).unwrap();
        this_proof.m_comm.commitment().serialize(&mut buf).unwrap();
        this_proof.s_comm.commitment().serialize(&mut buf).unwrap();
        this_proof
            .rowtilde_comm
            .commitment()
            .serialize(&mut buf)
            .unwrap();
        for fc in &this_proof.f_comms {
            fc.commitment().serialize(&mut buf).unwrap();
        }
        this_proof
            .r_star_comm
            .commitment()
            .serialize(&mut buf)
            .unwrap();
        this_proof.q_comm.commitment().serialize(&mut buf).unwrap();
        this_proof.h_alpha.serialize(&mut buf).unwrap();
        this_proof.r_alpha.serialize(&mut buf).unwrap();
        this_proof.c_alpha.serialize(&mut buf).unwrap();
        this_proof.row_alpha.serialize(&mut buf).unwrap();
        this_proof
            .q_poly_comm
            .commitment()
            .serialize(&mut buf)
            .unwrap();
        buf.len()
    };
    this_proof_size
}

criterion_group!(benches, bench_this_prove, bench_theirs_prove, bench_comparison);
criterion_main!(benches);
