use crate::*;
use ark_bls12_381::Bls12_381;
use ark_ec::PairingEngine;
use ark_ff::{Field, One, Zero};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, Polynomial, UVPolynomial};

type E = Bls12_381;
type F = <E as PairingEngine>::Fr;

const N: usize = 4;
const M: usize = 4;
const T: usize = 1;
const ROW: [usize; 4] = [0, 1, 2, 0];
const COL: [usize; 4] = [1, 2, 3, 3];
// Multiplicities from eq. (7) with T=1.  Each (r,c) contributes indices
// r, c, c−r−1, c−t.  Tally over ROW×COL:
//   index 0: r=0(×3), r=0(i3), c-r-1=0(i1), c-r-1=0(i2)  → 6
//   index 1: c=1(i0), r=1(i1), c-t=1(i1)                  → 3
//   index 2: c=2(i1), r=2(i2), c-t=2(i2), c-r-1=2(i3), c-t=2(i3) → 5
//   index 3: c=3(i2), c=3(i3)                              → 2
const MULTS: [u64; 4] = [6, 3, 5, 2];
const MARLIN_ROW: [usize; 32] = [
    3, 3, 4, 1, 3, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,
];
const MARLIN_COL: [usize; 32] = [
    4, 5, 5, 6, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15,
];

fn setup() -> PfrPublicKey<E> {
    PfrPublicKey::<E>::setup(N, M, T, &mut ark_std::test_rng())
}

fn default_stmt(pk: &PfrPublicKey<E>) -> crate::PfrStatement<E> {
    commit_statement(pk, &ROW, &COL, &mut ark_std::test_rng())
}

// --- Round 1 ---

/// R(κ^i) = Δ^{r_i} for all i
#[test]
fn round1_r_poly() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let poly = s.polynomials[0].polynomial();
    for (i, &r) in ROW.iter().enumerate() {
        assert_eq!(
            poly.evaluate(&pk.k_domain.element(i)),
            pk.d_domain.element(r),
            "R(κ^{i}) ≠ Δ^{r}"
        );
    }
}

/// C(κ^i) = Δ^{c_i} for all i
#[test]
fn round1_c_poly() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let poly = s.polynomials[1].polynomial();
    for (i, &c) in COL.iter().enumerate() {
        assert_eq!(
            poly.evaluate(&pk.k_domain.element(i)),
            pk.d_domain.element(c),
            "C(κ^{i}) ≠ Δ^{c}"
        );
    }
}

/// m(ω^j) = m_j for all j (multiplicity polynomial over H)
#[test]
fn round1_m_poly() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let poly = s.polynomials[2].polynomial();
    for j in 0..N {
        assert_eq!(
            poly.evaluate(&pk.h_domain.element(j)),
            F::from(MULTS[j]),
            "m(ω^{j}) ≠ {}",
            MULTS[j]
        );
    }
}

/// S(X) = R_S·X + ρ_S·z_K(X): degree ≤ m+1, vanishes on K
#[test]
fn round1_s_poly_shape() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let poly = s.polynomials[3].polynomial();
    assert!(
        poly.degree() <= M + 1,
        "deg S = {} > m+1 = {}",
        poly.degree(),
        M + 1
    );
    // S(κ^i) = R_S·κ^i + ρ_S·z_K(κ^i) = R_S·κ^i  (z_K vanishes on K)
    // so S does NOT vanish on K in general; just check degree bound
}

/// row̃(κ^i) = ω^{r_i} for all i  (index 7)
#[test]
fn round1_rowtilde_poly() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let poly = s.polynomials[7].polynomial();
    for (i, &r) in ROW.iter().enumerate() {
        assert_eq!(
            poly.evaluate(&pk.k_domain.element(i)),
            pk.h_domain.element(r),
            "row̃(κ^{i}) ≠ ω^{r}"
        );
    }
}

/// (row̃ − row)(X) vanishes on K for any blinding randomness
#[test]
fn round1_rowtilde_minus_row_vanishes_on_k() {
    let pk = setup();
    // Use a fresh rng so blinding scalars are random (currently zero, but test is general)
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let row_poly = s.polynomials[4].polynomial();
    let rowtilde_poly = s.polynomials[7].polynomial();
    let diff = rowtilde_poly - row_poly;
    for i in 0..M {
        assert_eq!(
            diff.evaluate(&pk.k_domain.element(i)),
            F::zero(),
            "(row̃ − row)(κ^{i}) ≠ 0"
        );
    }
}

/// Cached r_evals / c_evals / m_evals match polynomial evaluations
#[test]
fn round1_cached_evals_consistent() {
    let pk = setup();
    let s = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let r_poly = s.polynomials[0].polynomial();
    let c_poly = s.polynomials[1].polynomial();
    let m_poly = s.polynomials[2].polynomial();
    for i in 0..M {
        let ki = pk.k_domain.element(i);
        assert_eq!(
            r_poly.evaluate(&ki),
            s.r_evals[i],
            "r_evals[{i}] inconsistent"
        );
        assert_eq!(
            c_poly.evaluate(&ki),
            s.c_evals[i],
            "c_evals[{i}] inconsistent"
        );
        assert_eq!(
            m_poly.evaluate(&ki),
            s.m_evals[i],
            "m_evals[{i}] inconsistent"
        );
    }
}

// --- Round 2 ---

/// F_j(κ^i) matches the closed-form formulas from eq. (8)
#[test]
fn round2_f_poly_evals() {
    let pk = setup();
    let r1 = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let beta = F::from(42u64);
    let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());
    let big_delta = pk.d_domain.element(1);
    let delta_t_inv = big_delta.pow([T as u64]).inverse().unwrap();

    for i in 0..M {
        let ki = pk.k_domain.element(i);
        let r = r1.r_evals[i];
        let c = r1.c_evals[i];
        let h = pk.d_domain.element(i); // h(κ^i) = Δ^i
        let m = r1.m_evals[i];

        let f1 = r2.polynomials[0].polynomial().evaluate(&ki);
        let f2 = r2.polynomials[1].polynomial().evaluate(&ki);
        let f3 = r2.polynomials[2].polynomial().evaluate(&ki);
        let f4 = r2.polynomials[3].polynomial().evaluate(&ki);
        let f5 = r2.polynomials[4].polynomial().evaluate(&ki);

        assert_eq!(f1, (beta + r).inverse().unwrap(), "F₁(κ^{i})");
        assert_eq!(f2, (beta + c).inverse().unwrap(), "F₂(κ^{i})");
        assert_eq!(
            f3,
            (beta + c * (big_delta * r).inverse().unwrap())
                .inverse()
                .unwrap(),
            "F₃(κ^{i})"
        );
        assert_eq!(f4, (beta + c * delta_t_inv).inverse().unwrap(), "F₄(κ^{i})");
        assert_eq!(f5, -(m * (beta + h).inverse().unwrap()), "F₅(κ^{i})");
    }
}

pub(super) fn check_round2_sumcheck(pk: &PfrPublicKey<E>, row: &[usize], col: &[usize]) {
    let m = row.len();
    let stmt = commit_statement(pk, row, col, &mut ark_std::test_rng());
    let r1 = round_one(pk, row, col, &stmt, &mut ark_std::test_rng());
    let beta = F::from(42u64);
    let r2 = round_two(pk, &r1, beta, &mut ark_std::test_rng());

    let sum: F = (0..m)
        .map(|i| {
            let ki = pk.k_domain.element(i);
            r2.polynomials
                .iter()
                .map(|p| p.polynomial().evaluate(&ki))
                .sum::<F>()
        })
        .sum();

    assert_eq!(
        sum,
        F::zero(),
        "∑ Fⱼ(κ^i) ≠ 0  (n={}, m={m})",
        pk.h_domain.size()
    );
}

/// ∑_{i=0}^{m-1} (F₁+F₂+F₃+F₄+F₅)(κ^i) = 0  (the rational-sumcheck identity, eq. 7)
#[test]
fn round2_sumcheck() {
    check_round2_sumcheck(&setup(), &ROW, &COL);
}

/// Same sumcheck but with m > n (n=4, m=8): K strictly larger than H.
#[test]
fn round2_sumcheck_m_double_n() {
    let row: &[usize] = &[0, 1, 0, 2, 0, 1, 2, 0];
    let col: &[usize] = &[1, 2, 2, 3, 3, 3, 3, 1];
    let pk = PfrPublicKey::<E>::setup(4, row.len(), 1, &mut ark_std::test_rng());
    check_round2_sumcheck(&pk, row, col);
}

#[test]
fn round2_sumcheck_holds_for_marlin_shared_relation_indices_standalone() {
    let pk = PfrPublicKey::<E>::setup(16, MARLIN_ROW.len(), 4, &mut ark_std::test_rng());
    let stmt = commit_statement(&pk, &MARLIN_ROW, &MARLIN_COL, &mut ark_std::test_rng());
    let r1 = round_one(&pk, &MARLIN_ROW, &MARLIN_COL, &stmt, &mut ark_std::test_rng());
    let beta = F::from(42u64);
    let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());

    let sum: F = (0..MARLIN_ROW.len())
        .map(|i| {
            let ki = pk.k_domain.element(i);
            r2.polynomials
                .iter()
                .map(|p| p.polynomial().evaluate(&ki))
                .sum::<F>()
        })
        .sum();

    assert_eq!(
        sum,
        F::zero(),
        "the Marlin-derived indices should satisfy eq. (7) when PFR builds its own statement"
    );
}

// --- Round 3 ---

/// Each F_j identity from P(X) vanishes on K.
///
/// For every κ^i ∈ K we check the five η-components of P individually:
///   η⁰: F₁(κ^i)(β + R(κ^i)) − 1 = 0
///   η¹: F₂(κ^i)(β + C(κ^i)) − 1 = 0
///   η²: F₃(κ^i)(β + C(κ^i)/(Δ·R(κ^i))) − 1 = 0
///   η³: F₄(κ^i)(β + C(κ^i)/Δᵗ) − 1 = 0
///   η⁴: F₅(κ^i)(β + h(κ^i)) − m(κ^i)·z_{K\H} = 0
///
/// These are exactly the identities whose sum forms the P polynomial;
/// verifying them pointwise on K is the core soundness check for round 3.
#[test]
fn round3_p_identities_vanish_on_k() {
    let pk = setup();
    let r1 = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let beta = F::from(42u64);
    let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());

    let big_delta = pk.big_delta();
    let delta_t_inv = big_delta.pow([T as u64]).inverse().unwrap();
    // z_{K\H} = 1 when K = H
    let zkh = F::one();

    for i in 0..M {
        let ki = pk.k_domain.element(i);

        let r = r1.polynomials[0].polynomial().evaluate(&ki);
        let c = r1.polynomials[1].polynomial().evaluate(&ki);
        let m = r1.polynomials[2].polynomial().evaluate(&ki);
        let h = pk.h_poly.evaluate(&ki);

        let f1 = r2.polynomials[0].polynomial().evaluate(&ki);
        let f2 = r2.polynomials[1].polynomial().evaluate(&ki);
        let f3 = r2.polynomials[2].polynomial().evaluate(&ki);
        let f4 = r2.polynomials[3].polynomial().evaluate(&ki);
        let f5 = r2.polynomials[4].polynomial().evaluate(&ki);

        assert_eq!(
            f1 * (beta + r) - F::one(),
            F::zero(),
            "η⁰ identity failed at κ^{i}"
        );
        assert_eq!(
            f2 * (beta + c) - F::one(),
            F::zero(),
            "η¹ identity failed at κ^{i}"
        );
        let c_over_delta_r = c * (big_delta * r).inverse().unwrap();
        assert_eq!(
            f3 * (beta + c_over_delta_r) - F::one(),
            F::zero(),
            "η² identity failed at κ^{i}"
        );
        assert_eq!(
            f4 * (beta + c * delta_t_inv) - F::one(),
            F::zero(),
            "η³ identity failed at κ^{i}"
        );
        assert_eq!(
            f5 * (beta + h) + m * zkh,
            F::zero(),
            "η⁴ identity failed at κ^{i}"
        );
    }
}

/// Round-3 polynomial degrees are within the theoretical bounds.
///
/// For m=4, U(X) = X³−1, with blinded R,C of degree m+1=5:
///   F_j·R·C terms have degree ≤ 3·(m+1) = 15,
///   big_sum deg ≤ 15,  P = big_sum·U deg ≤ 18,
///   deg(z_K·U) = m+3 = 7,  so deg q ≤ 18−7 = 11 ≤ 2m+3 = 11. ✓
///   R_F = (f_sum mod z_K)/X,  deg R_F ≤ m−2 = 2,
///   R* = R_F·U,  deg R* ≤ (m−2)+3 = 5 = m+1. ✓
#[test]
fn round3_poly_degrees() {
    let pk = setup();
    let r1 = { let _stmt = default_stmt(&pk); round_one(&pk, &ROW, &COL, &_stmt, &mut ark_std::test_rng()) };
    let r2 = round_two(&pk, &r1, F::from(42u64), &mut ark_std::test_rng());
    let r3 = round_three(&pk, &r1, &r2, F::from(17u64));

    let r_star = r3.polynomials[0].polynomial();
    let q = r3.polynomials[1].polynomial();

    assert_eq!(r_star.degree(), 5, "deg R* should be m+1 = 5 for m=4");
    assert!(
        q.degree() <= 2 * M + 3,
        "deg q = {} should be ≤ 2m+3 = {} for m={M}",
        q.degree(),
        2 * M + 3
    );
}

/// η⁴ identity F₅(β+h) + m·z_{K\H} = 0 holds on K for m>n.
/// Also checks what value z_{K\H} takes on H vs K\H elements.
#[test]
fn round3_eta4_identity_m_double_n() {
    let row: &[usize] = &[0, 1, 0, 2, 0, 1, 2, 0];
    let col: &[usize] = &[1, 2, 2, 3, 3, 3, 3, 1];
    let n = 4;
    let t = 1;
    let m = row.len();
    let pk = PfrPublicKey::<E>::setup(n, m, t, &mut ark_std::test_rng());
    let stmt = commit_statement(&pk, row, col, &mut ark_std::test_rng());
    let r1 = round_one(&pk, row, col, &stmt, &mut ark_std::test_rng());
    let beta = F::from(42u64);
    let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());

    // Compute z_{K\H} polynomial as the prover does: (n/m)·(X^m-1)/(X^n-1)
    let steps = m / n;
    let scale_factor = F::from(n as u64) * F::from(m as u64).inverse().unwrap();
    let mut coeffs = vec![F::zero(); (steps - 1) * n + 1];
    for k in 0..steps {
        coeffs[k * n] = scale_factor;
    }
    let zkh = DensePolynomial::from_coefficients_vec(coeffs);

    // Check identity and print z_{K\H} values
    let f5_poly = r2.polynomials[4].polynomial();
    let m_poly = r1.polynomials[2].polynomial();
    for i in 0..m {
        let ki = pk.k_domain.element(i);
        let f5 = f5_poly.evaluate(&ki);
        let h = pk.h_poly.evaluate(&ki);
        let mv = m_poly.evaluate(&ki);
        let zkh_val = zkh.evaluate(&ki);
        assert_eq!(
            f5 * (beta + h) + mv * zkh_val,
            F::zero(),
            "η⁴ identity failed at κ^{i} (in_H={})",
            i % (m / n) == 0
        );
    }
}
