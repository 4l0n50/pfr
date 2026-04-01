use crate::*;
use ark_bls12_381::Bls12_381;

type E = Bls12_381;

/// Run prove + verify for an arbitrary configuration and assert the proof verifies.
fn prove_and_verify(n: usize, t: usize, row: &[usize], col: &[usize]) {
    let m = row.len();
    let rng = &mut ark_std::test_rng();
    let pk = PfrPublicKey::<E>::setup(n, m, t, rng);
    let stmt = commit_statement(&pk, row, col, rng);
    let (proof, public_inputs) = prove(&pk, row, col, &stmt, rng);
    assert!(
        verify(
            &pk,
            &proof,
            &public_inputs.row_comm,
            &public_inputs.col_comm,
            &public_inputs.rowcol_comm
        ),
        "verification failed for n={}, m={}, t={}, row={:?}, col={:?}",
        n,
        m,
        t,
        row,
        col,
    );
}

/// Baseline: the same 4-index example used throughout the module.
#[test]
fn e2e_baseline() {
    prove_and_verify(4, 1, &[0, 1, 2, 0], &[1, 2, 3, 3]);
}

/// n=8, m=8: K = H, larger table than the baseline.
#[test]
fn e2e_larger_table() {
    prove_and_verify(8, 1, &[0, 0, 1, 1, 2, 3, 4, 5], &[1, 3, 2, 5, 4, 6, 7, 7]);
}

/// n=4, m=8: K is twice the size of H (m/n = 2), so z_{K\H} is non-trivial.
#[test]
fn e2e_m_double_n() {
    // m = 8 pairs; column indices must be < n = 4 (into the table H).
    // All pairs (r_i, c_i) with 0 ≤ r_i < c_i < 4 and c_i ≥ t = 1.
    prove_and_verify(4, 1, &[0, 1, 0, 2, 0, 1, 2, 0], &[1, 2, 2, 3, 3, 3, 3, 1]);
}

/// n=8, m=8: index pairs that hit every row index at least once.
#[test]
fn e2e_dense_rows() {
    // Pairs cover rows 0–7 at least once (each r_i distinct).
    prove_and_verify(8, 1, &[0, 1, 2, 3, 4, 5, 6, 0], &[1, 2, 3, 4, 5, 6, 7, 7]);
}

/// t=2: every c_i ≥ 2, n = m = 4.
#[test]
fn e2e_t2() {
    // Pairs with r < c and c ≥ 2, all indices in [0, 4).
    prove_and_verify(4, 2, &[0, 0, 1, 0], &[2, 3, 3, 2]);
}

/// t=2 with n = m = 8.
#[test]
fn e2e_t2_n8() {
    prove_and_verify(8, 2, &[0, 1, 0, 2, 3, 4, 1, 2], &[2, 3, 4, 5, 6, 7, 6, 7]);
}

/// t=3 with n = m = 8.
#[test]
fn e2e_t3() {
    prove_and_verify(8, 3, &[0, 1, 0, 2, 0, 3, 1, 2], &[3, 4, 5, 6, 7, 7, 7, 7]);
}
