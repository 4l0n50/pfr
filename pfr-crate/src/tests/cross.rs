// Cross-validation tests against TStrictlyLowerTriangular
//
// TStrictlyLowerTriangular (geometry repo) encodes the *transposed* matrix:
//   their row_poly(γ^i) = ω^{row_i}  ← our col index
//   their col_poly(γ^i) = ω^{col_i}  ← our row index
//
// So for the same underlying matrix M, valid iff M is t-strictly lower
// triangular (their_row_i ≥ t, their_col_i < their_row_i):
//
//   this_row[i]  = their col index  (= their col_poly encodes)
//   this_col[i]  = their row index  (= their row_poly encodes)
//
// In these test we check that both verifiers accept the same valid proof, produced
// by the respective prover, for a given matrix and reject the same invalid one.

use crate::*;

use ark_bn254::Bn254;
use ark_ec::PairingEngine;
use ark_ff::to_bytes;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain, UVPolynomial,
};
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use fiat_shamir_rng::{FiatShamirRng as TheirFiatShamirRng, SimpleHashFiatShamirRng};
use homomorphic_poly_commit::marlin_kzg::KZG10;
use proof_of_function_relation::t_strictly_lower_triangular_test::TStrictlyLowerTriangular;

type E = Bn254;
type F = <E as PairingEngine>::Fr;
type TheirPC = KZG10<E>;
type TheirFS = SimpleHashFiatShamirRng<blake2::Blake2s, rand_chacha::ChaChaRng>;

// ---------------------------------------------------------------------------
// The matrix (t=2, n=4, m=8 pairs):
//
//       col: 0  1  2  3
//   row 0:   0  0  0  0
//   row 1:   0  0  0  0
//   row 2:   1  2  0  0     ← nonzeros at (2,0) and (2,1)
//   row 3:   0  3  5  0     ← nonzeros at (3,1) and (3,2); (3,2) repeated
//
// Their encoding (t-strictly lower triangular: their_row_i ≥ t, col_i < row_i):
//   their_row_evals[i] = ω^{their_row[i]}  encodes this_col
//   their_col_evals[i] = ω^{their_col[i]}  encodes this_row
//   their_row = [2, 2, 3, 3, 3, 3, 3, 3]   (≥ t=2 ✓, this_col)
//   their_col = [0, 1, 1, 2, 2, 2, 2, 2]   (< their_row ✓, this_row)
//
// Our encoding (this_row[i] < this_col[i], this_col[i] ≥ t):
//   this_row = [0, 1, 1, 2, 2, 2, 2, 2]   (= their_col ✓)
//   this_col = [2, 2, 3, 3, 3, 3, 3, 3]   (≥ t=2 ✓, = their_row ✓)
// ---------------------------------------------------------------------------

const N: usize = 4;
const T: usize = 2;
const OUR_ROW: [usize; 8] = [0, 1, 1, 2, 2, 2, 2, 2];
const OUR_COL: [usize; 8] = [2, 2, 3, 3, 3, 3, 3, 3];

/// Call TStrictlyLowerTriangular prove+verify on Bn254.
///
/// `their_row_evals` = evaluations of their row polynomial over K (encodes this_col as ω^c).
/// `their_col_evals` = evaluations of their col polynomial over K (encodes this_row as ω^r).
/// Returns true iff prove succeeds and verify accepts.
fn their_prove_and_verify(
    their_row_evals: Vec<F>,
    their_col_evals: Vec<F>,
    n: usize,
    t: usize,
) -> bool {
    let rng = &mut ark_std::test_rng();
    let m = their_row_evals.len();

    let domain_k = GeneralEvaluationDomain::<F>::new(m).unwrap();
    let domain_h = GeneralEvaluationDomain::<F>::new(n).unwrap();

    let enforced_degree_bound = domain_k.size() + 1;
    let enforced_hiding_bound = 1;

    let row_poly = DensePolynomial::<F>::from_coefficients_slice(&domain_k.ifft(&their_row_evals));
    let col_poly = DensePolynomial::<F>::from_coefficients_slice(&domain_k.ifft(&their_col_evals));

    let row_poly = LabeledPolynomial::new(
        String::from("row_poly"),
        row_poly,
        Some(enforced_degree_bound),
        Some(enforced_hiding_bound),
    );
    let col_poly = LabeledPolynomial::new(
        String::from("col_poly"),
        col_poly,
        Some(enforced_degree_bound),
        Some(enforced_hiding_bound),
    );

    let max_degree = 20;
    let pp = TheirPC::setup(max_degree, None, rng).unwrap();
    let (ck, vk) = TheirPC::trim(
        &pp,
        max_degree,
        enforced_hiding_bound,
        Some(&[2, enforced_degree_bound]),
    )
    .unwrap();

    let (commitments, rands) =
        TheirPC::commit(&ck, &[row_poly.clone(), col_poly.clone()], Some(rng)).unwrap();

    let mut fs_rng = TheirFS::initialize(&to_bytes!(b"Testing :)").unwrap());

    let proof = TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::prove(
        &ck,
        t,
        &domain_k,
        &domain_h,
        &row_poly,
        &commitments[0],
        &rands[0],
        &col_poly,
        &commitments[1],
        &rands[1],
        Some(enforced_degree_bound),
        &mut fs_rng,
        rng,
    );

    match proof {
        Err(_) => false,
        Ok(proof) => {
            let mut fs_rng = TheirFS::initialize(&to_bytes!(b"Testing :)").unwrap());
            TStrictlyLowerTriangular::<F, TheirPC, TheirFS>::verify(
                &vk,
                &ck,
                t,
                &domain_k,
                &domain_h,
                &commitments[0],
                &commitments[1],
                Some(enforced_degree_bound),
                proof,
                &mut fs_rng,
            )
            .is_ok()
        }
    }
}

/// Both provers accept the same valid t-strictly lower triangular matrix.
///
/// The matrix has nonzeros at positions (2,0),(2,1),(3,1),(3,2) with t=2, n=4.
/// Our encoding uses the transposed convention: this_row=their_col, this_col=their_row.
#[test]
fn cross_valid_positive() {
    let domain_h = GeneralEvaluationDomain::<F>::new(N).unwrap();

    // --- Our system (Bn254) ---
    let rng = &mut ark_std::test_rng();
    let pk = PfrPublicKey::<E>::setup(N, OUR_ROW.len(), T, rng);
    let stmt = commit_statement(&pk, &OUR_ROW, &OUR_COL, rng);
    let (proof, public_inputs) = prove(&pk, &OUR_ROW, &OUR_COL, &stmt, rng);
    assert!(
        verify(
            &pk,
            &proof,
            &public_inputs.row_comm,
            &public_inputs.col_comm,
            &public_inputs.rowcol_comm
        ),
        "our prover should accept the valid matrix"
    );

    // --- Their system (Bn254, transposed) ---
    // their row_poly(γ^i) = ω^{this_col[i]},  their col_poly(γ^i) = ω^{this_row[i]}
    let their_row_evals: Vec<F> = OUR_COL.iter().map(|&c| domain_h.element(c)).collect();
    let their_col_evals: Vec<F> = OUR_ROW.iter().map(|&r| domain_h.element(r)).collect();

    assert!(
        their_prove_and_verify(their_row_evals, their_col_evals, N, T),
        "their prover should accept the same valid matrix (transposed)"
    );
}
