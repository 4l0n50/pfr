use crate::counting;
use crate::types::*;
use ark_ec::PairingEngine;
use ark_ff::{to_bytes, Field, One, UniformRand, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, Polynomial, UVPolynomial,
};
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use ark_std::rand::RngCore;
use ark_std::{end_timer, start_timer};

use ark_marlin::{rng::FiatShamirRng, SimpleHashFiatShamirRng};

// ---------------------------------------------------------------------------
// Internal prover state (not sent to verifier)
// ---------------------------------------------------------------------------

/// Prover state after Round 1.
#[allow(dead_code)]
pub struct Round1State<E: PairingEngine> {
    /// Labeled polynomials [R, C, m, S, row, col, rowcol, rowtilde].
    ///
    /// | Index | Label       | Definition                               |
    /// |-------|-------------|------------------------------------------|
    /// | 0     | R           | R(κ^i) = Δ^{r_i}                         |
    /// | 1     | C           | C(κ^i) = Δ^{c_i}                         |
    /// | 2     | m           | m(ω^j) = m_j                             |
    /// | 3     | S           | S(X) = R_S·X + ρ_S·z_K                   |
    /// | 4     | row         | row(κ^i) = ω^{r_i}   (statement poly)    |
    /// | 5     | col         | col(κ^i) = ω^{c_i}   (statement poly)    |
    /// | 6     | rowcol      | rowcol(κ^i) = ω^{r_i·c_i} (statement)    |
    /// | 7     | rowtilde    | row̃(κ^i) = ω^{r_i} + ρ_row·z_K (blinded) |
    pub polynomials: [LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>; 8],
    /// Evaluation vector: r_evals[i] = R(κ^i) = Δ^{r_i}
    pub r_evals: Vec<E::Fr>,
    /// Evaluation vector: c_evals[i] = C(κ^i) = Δ^{c_i}
    pub c_evals: Vec<E::Fr>,
    /// Evaluation vector: m_evals[j] = m_j  (same as m(κ^j) when K = H)
    pub m_evals: Vec<E::Fr>,
    /// R_S: the leading coefficient of S(X) = R_S·X + ρ_S·z_K(X)
    r_s: E::Fr,
    /// ρ_S: the blinding scalar of S(X)
    rho_s: E::Fr,
    /// Evaluations of all 8 polynomials over pk.coset_domain (canonical coset of size 2m).
    /// Index order matches `polynomials`: [R, C, m, S, row, col, rowcol, rowtilde].
    pub coset_evals: [Vec<E::Fr>; 8],
    /// Commitment randomness (filled in by prove() after PC::commit)
    pub rands: Vec<Rand<E>>,
}

/// Prover state after Round 2.
#[allow(dead_code)]
pub struct Round2State<E: PairingEngine> {
    /// Labeled polynomials [F₁, …, F₅]; polynomials accessible via `.polynomial()`.
    pub polynomials: [LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>; 5],
    pub f_evals: [Vec<E::Fr>; 5],
    /// Evaluations of F1..F5 over pk.coset_domain.
    pub f_coset_evals: [Vec<E::Fr>; 5],
    /// β: verifier challenge that triggered Round 2.
    pub beta: E::Fr,
    /// Commitment randomness (filled in by prove() after PC::commit)
    pub rands: Vec<Rand<E>>,
}

/// Prover state after Round 3.
#[allow(dead_code)]
pub struct Round3State<E: PairingEngine> {
    /// Labeled polynomials [R*, q]; accessible via `.polynomial()`.
    ///
    /// | Index | Label  | Definition                      |
    /// |-------|--------|---------------------------------|
    /// | 0     | r_star | R*(X) = R_F(X) · U(X)           |
    /// | 1     | q      | q(X) = P(X) / (z_K(X) · U(X))  |
    pub polynomials: Vec<LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>>,
    /// U(X) = X³ − 1, stored to avoid recomputing in round_five.
    pub u_poly: DensePolynomial<E::Fr>,
    /// η: verifier challenge that triggered Round 3.
    pub eta: E::Fr,
    /// η⁹ (the last power of η used in P), stored for use in Lin(X).
    pub eta9: E::Fr,
    /// Commitment randomness (filled in by prove() after PC::commit)
    pub rands: Vec<Rand<E>>,
}

/// Prover state after Round 4.
#[allow(dead_code)]
pub struct Round4State<E: PairingEngine> {
    /// α: verifier challenge that triggered Round 4.
    pub alpha: E::Fr,
    /// h(α)
    pub h_alpha: E::Fr,
    /// R(α)
    pub r_alpha: E::Fr,
    /// C(α)
    pub c_alpha: E::Fr,
    /// row̃(α)  (= row(α) in no-ZK mode)
    pub row_alpha: E::Fr,
}

// ---------------------------------------------------------------------------
// Blinding helpers
// ---------------------------------------------------------------------------

/// Interpolate `evals` over `domain` and blind with ρ(X)·z(X),
/// where ρ is a random polynomial of degree `rho_degree`.
///
/// - `rho_degree = 1`: used for R, C, F₁–F₅, row̃  (blinding polynomial ∈ F≤1[X])
/// - `rho_degree = 0`: used for m  (scalar blinding)
fn blind_over_domain<F: ark_ff::PrimeField + UniformRand, R: RngCore>(
    evals: Vec<F>,
    domain: GeneralEvaluationDomain<F>,
    vanishing: &DensePolynomial<F>,
    rho_degree: usize,
    rng: &mut R,
) -> DensePolynomial<F> {
    let interp = EvaluationsOnDomain::from_vec_and_domain(evals, domain).interpolate();
    let rho =
        DensePolynomial::from_coefficients_vec((0..=rho_degree).map(|_| F::rand(rng)).collect());
    &interp + &(&rho * vanishing)
}

// ---------------------------------------------------------------------------
// Round 1
// ---------------------------------------------------------------------------

/// **Round 1**: build R(X), C(X), m(X), S(X), row(X), col(X), rowcol(X), row̃(X).
///
/// | Paper variable | Code variable   | Definition                                           |
/// |----------------|-----------------|------------------------------------------------------|
/// | R(X)           | `r_poly`        | R(κ^i) = Δ^{r_i}; blinded by ρ_R(X)·z_K(X)           |
/// | C(X)           | `c_poly`        | C(κ^i) = Δ^{c_i}; blinded by ρ_C(X)·z_K(X)           |
/// | m(X)           | `m_poly`        | m(ω^j) = m_j; blinded by ρ_m·z_H(X)                  |
/// | S(X)           | `s_poly`        | S(X) = R_S·X + ρ_S·z_K(X); R_S, ρ_S ← F              |
/// | row(X)         | `row_poly`      | row(κ^i) = ω^{r_i}; statement polynomial (unblinded) |
/// | col(X)         | `col_poly`      | col(κ^i) = ω^{c_i}; statement polynomial (unblinded) |
/// | rowcol(X)      | `rowcol_poly`   | rowcol(κ^i) = ω^{r_i·c_i}; statement (unblinded)     |
/// | row̃(X)         | `rowtilde_poly` | row̃ = row + ρ_row(X)·z_K(X); ρ_row ← F≤1[X]          |
pub fn round_one<E: PairingEngine, R: RngCore>(
    pk: &PfrPublicKey<E>,
    row_indices: &[usize],
    col_indices: &[usize],
    stmt: &PfrStatement<E>,
    rng: &mut R,
) -> Round1State<E> {
    let z_k: DensePolynomial<E::Fr> = pk.k_domain.vanishing_polynomial().into();
    let z_h: DensePolynomial<E::Fr> = pk.h_domain.vanishing_polynomial().into();

    // R(X): R(κ^i) = Δ^{r_i}, blinded by ρ_R(X)·z_K(X) with ρ_R ← F≤1[X]
    let r_evals: Vec<E::Fr> = row_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    counting::record_ifft(pk.k_domain.size()); // interpolate R over K
    let r_poly = blind_over_domain(r_evals.clone(), pk.k_domain, &z_k, 1, rng);

    // C(X): C(κ^i) = Δ^{c_i}, blinded by ρ_C(X)·z_K(X) with ρ_C ← F≤1[X]
    let c_evals: Vec<E::Fr> = col_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    counting::record_ifft(pk.k_domain.size()); // interpolate C over K
    let c_poly = blind_over_domain(c_evals.clone(), pk.k_domain, &z_k, 1, rng);

    // m(X): m(ω^j) = m_j, blinded by ρ_m·z_H(X) with ρ_m ← F
    let mults = pk.compute_multiplicities(row_indices, col_indices);
    let m_evals: Vec<E::Fr> = mults.iter().map(|&v| E::Fr::from(v)).collect();
    counting::record_ifft(pk.h_domain.size()); // interpolate m over H
    let m_poly = blind_over_domain(m_evals.clone(), pk.h_domain, &z_h, 0, rng);

    // S(X) = R_S·X + ρ_S·z_K(X),  R_S, ρ_S ← F
    let r_s = E::Fr::rand(rng);
    let rho_s = E::Fr::rand(rng);
    let s_poly =
        &DensePolynomial::from_coefficients_vec(vec![E::Fr::zero(), r_s]) + &(&z_k * rho_s);

    // row̃(X) = row(X) + ρ_row(X)·z_K(X),  ρ_row ← F≤1[X]
    // Built by blinding the already-interpolated row polynomial from stmt.
    let row_evals: Vec<E::Fr> = pk.k_domain
        .elements()
        .map(|x| stmt.row_poly.polynomial().evaluate(&x))
        .collect();
    counting::record_ifft(pk.k_domain.size()); // interpolate rowtilde over K
    let rowtilde_poly = blind_over_domain(row_evals.clone(), pk.k_domain, &z_k, 1, rng);

    // Clone statement polynomials (without degree bound / hiding) into the array.
    let row_poly = stmt.row_poly.polynomial().clone();
    let col_poly = stmt.col_poly.polynomial().clone();
    let rowcol_poly = stmt.rowcol_poly.polynomial().clone();

    let polys = [
        LabeledPolynomial::new("R".into(), r_poly, None, None),
        LabeledPolynomial::new("C".into(), c_poly, None, None),
        LabeledPolynomial::new("m".into(), m_poly, None, None),
        LabeledPolynomial::new("S".into(), s_poly, None, None),
        LabeledPolynomial::new("row".into(), row_poly, None, None),
        LabeledPolynomial::new("col".into(), col_poly, None, None),
        LabeledPolynomial::new("rowcol".into(), rowcol_poly, None, None),
        LabeledPolynomial::new("rowtilde".into(), rowtilde_poly, None, None),
    ];

    // Evaluate all 8 polynomials over the coset domain for use in round 3.
    // 8 × coset_FFT of size 2m
    for _ in 0..8 { counting::record_fft(pk.coset_domain.size()); }
    let coset_evals = [
        pk.coset_domain.coset_fft(&polys[0].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[1].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[2].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[3].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[4].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[5].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[6].polynomial().coeffs),
        pk.coset_domain.coset_fft(&polys[7].polynomial().coeffs),
    ];

    Round1State {
        polynomials: polys,
        r_evals,
        c_evals,
        m_evals,
        r_s,
        rho_s,
        coset_evals,
        rands: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Round 2
// ---------------------------------------------------------------------------

/// **Round 2**: given verifier challenge β, compute and commit to F₁, …, F₅.
///
/// The five sequences (eq. 8, evaluated at κ^i for i = 1, …, m):
///
/// | Paper variable | Code variable | Formula                             |
/// |--------------- |---------------|-------------------------------------|
/// | β              | `beta`        | verifier challenge                  |
/// | F₁(κ^i)        | `f1_evals`    | 1 / (β + R(κ^i))                    |
/// | F₂(κ^i)        | `f2_evals`    | 1 / (β + C(κ^i))                    |
/// | F₃(κ^i)        | `f3_evals`    | 1 / (β + C(κ^i)/(Δ·R(κ^i)))         |
/// | F₄(κ^i)        | `f4_evals`    | 1 / (β + C(κ^i)/Δ^t)                |
/// | F₅(κ^i)        | `f5_evals`    | −m(κ^i)·z_{K∖H}(κ^i) / (β + h(κ^i)) |
/// | Δ              | `big_delta`   | `d_domain.element(1)`               |
/// | z_{K∖H}{}      | `zkh_at_ki`   | z_{K∖H}(κ^i)                        |
pub fn round_two<E: PairingEngine, R: RngCore>(
    pk: &PfrPublicKey<E>,
    round1: &Round1State<E>,
    beta: E::Fr,
    rng: &mut R,
) -> Round2State<E> {
    // Δ = generator of D
    let big_delta: E::Fr = pk.big_delta();
    let big_delta_t: E::Fr = big_delta.pow([pk.t as u64]);

    // Reuse evaluation vectors stored in Round1State.
    // r_at_ki[i] = R(κ^i) = Δ^{r_i},  c_at_ki[i] = C(κ^i) = Δ^{c_i}
    let r_at_ki = &round1.r_evals;
    let c_at_ki = &round1.c_evals;

    // m(κ^i): when K = H, m_evals stores m(ω^i) = m_i directly.
    // When K ≠ H, use FFT to evaluate m over K in O(m log m).
    let m_at_ki_owned: Vec<E::Fr>;
    let m_at_ki: &Vec<E::Fr> = if pk.n == pk.m {
        &round1.m_evals
    } else {
        let m_poly = round1.polynomials[2].polynomial();
        m_at_ki_owned = m_poly.evaluate_over_domain_by_ref(pk.k_domain).evals;
        &m_at_ki_owned
    };

    // h(κ^i): when K = H, h(ω^i) = Δ^i (read from d_domain directly).
    // When K ≠ H, use FFT to evaluate h over K in O(m log m).
    let h_at_ki: Vec<E::Fr> = if pk.n == pk.m {
        (0..pk.m).map(|i| pk.d_domain.element(i)).collect()
    } else {
        pk.h_poly.evaluate_over_domain_by_ref(pk.k_domain).evals
    };

    // z_{K\H}(κ^i) = (n/m) · (X^m−1)/(X^n−1) evaluated at κ^i.
    // When n | m: X^m−1 = (X^n−1)·Q(X) with Q = X^{m−n}+…+1 (m/n terms).
    // At κ^i ∈ K: (κ^i)^m = 1 so numerator = 0 and denom = 0; use Q:
    //   z_{K\H}(κ^i) = (n/m)·Q(κ^i).  Each term (κ^{kn·i}) with k=0..m/n−1
    //   is a power of (κ^n)^i where κ^n is a primitive (m/n)-th root of unity.
    debug_assert_eq!(pk.m % pk.n, 0, "m must be a multiple of n for K ⊇ H");
    let steps = pk.m / pk.n;
    let scale = E::Fr::from(pk.n as u64) * E::Fr::from(pk.m as u64).inverse().unwrap();
    // κ^n has order m/n; let ζ = κ^n.
    let zeta = pk.k_domain.element(pk.n); // κ^n
    let zkh_at_ki: Vec<E::Fr> = (0..pk.m)
        .map(|i| {
            // Q(κ^i) = sum_{k=0}^{steps−1} (κ^{kn})^i = sum_{k=0}^{steps−1} ζ^{ki}
            let base = zeta.pow([i as u64]);
            let mut q_val = E::Fr::zero();
            let mut power = E::Fr::one();
            for _ in 0..steps {
                q_val += power;
                power *= base;
            }
            scale * q_val
        })
        .collect();

    // Batch-invert all denominators using Montgomery's trick.
    // Two passes:
    //   Pass 1: invert Δ·R(κ^i) to get (Δ·R)⁻¹ — needed to form F3 denoms.
    //   Pass 2: invert the 5 F-denom blocks simultaneously.
    //
    // Block layout for pass 2 (each block has m entries):
    //   block 0: β + C(κ^i)/(Δ·R(κ^i))  — F3 denom
    //   block 1: β + R(κ^i)              — F1 denom
    //   block 2: β + C(κ^i)/Δᵗ          — F4 denom
    //   block 3: β + C(κ^i)              — F2 denom
    //   block 4: β + h(κ^i)              — F5 denom

    // Δᵗ is a single scalar — one inversion outside the batch.
    let delta_t_inv = big_delta_t.inverse().unwrap();

    // Pass 1: batch-invert Δ·R(κ^i)
    counting::record_batch_inv(pk.m);
    let mut delta_r: Vec<E::Fr> = r_at_ki.iter().map(|&r| big_delta * r).collect();
    ark_ff::batch_inversion(&mut delta_r);

    // Pass 2: fill 5m denominators and batch-invert
    counting::record_batch_inv(5 * pk.m);
    let mut denoms: Vec<E::Fr> = Vec::with_capacity(5 * pk.m);
    for i in 0..pk.m {
        denoms.push(beta + c_at_ki[i] * delta_r[i]);
    } // block 0: F3
    for i in 0..pk.m {
        denoms.push(beta + r_at_ki[i]);
    } // block 1: F1
    for i in 0..pk.m {
        denoms.push(beta + c_at_ki[i] * delta_t_inv);
    } // block 2: F4
    for i in 0..pk.m {
        denoms.push(beta + c_at_ki[i]);
    } // block 3: F2
    for i in 0..pk.m {
        denoms.push(beta + h_at_ki[i]);
    } // block 4: F5
    ark_ff::batch_inversion(&mut denoms);

    let f3_evals: Vec<E::Fr> = denoms[0..pk.m].to_vec();
    let f1_evals: Vec<E::Fr> = denoms[pk.m..2 * pk.m].to_vec();
    let f4_evals: Vec<E::Fr> = denoms[2 * pk.m..3 * pk.m].to_vec();
    let f2_evals: Vec<E::Fr> = denoms[3 * pk.m..4 * pk.m].to_vec();
    // F₅(κ^i) = −m(κ^i) · z_{K∖H}(κ^i) · (β + h(κ^i))⁻¹
    let f5_evals: Vec<E::Fr> = m_at_ki
        .iter()
        .zip(zkh_at_ki.iter())
        .zip(denoms[4 * pk.m..5 * pk.m].iter())
        .map(|((&m_val, &zkh), &h_inv)| -m_val * zkh * h_inv)
        .collect();

    // Interpolate each F_j over K, then blind with ρ_j(X)·z_K(X), ρ_j ← F≤1[X]
    // 5 × IFFT(m)
    let z_k: DensePolynomial<E::Fr> = pk.k_domain.vanishing_polynomial().into();
    counting::record_ifft(pk.k_domain.size()); // F1
    let f1_poly = blind_over_domain(f1_evals.clone(), pk.k_domain, &z_k, 1, rng);
    counting::record_ifft(pk.k_domain.size()); // F2
    let f2_poly = blind_over_domain(f2_evals.clone(), pk.k_domain, &z_k, 1, rng);
    counting::record_ifft(pk.k_domain.size()); // F3
    let f3_poly = blind_over_domain(f3_evals.clone(), pk.k_domain, &z_k, 1, rng);
    counting::record_ifft(pk.k_domain.size()); // F4
    let f4_poly = blind_over_domain(f4_evals.clone(), pk.k_domain, &z_k, 1, rng);
    counting::record_ifft(pk.k_domain.size()); // F5
    let f5_poly = blind_over_domain(f5_evals.clone(), pk.k_domain, &z_k, 1, rng);

    let f_polys = [
        LabeledPolynomial::new("F1".into(), f1_poly, None, None),
        LabeledPolynomial::new("F2".into(), f2_poly, None, None),
        LabeledPolynomial::new("F3".into(), f3_poly, None, None),
        LabeledPolynomial::new("F4".into(), f4_poly, None, None),
        LabeledPolynomial::new("F5".into(), f5_poly, None, None),
    ];

    // Evaluate blinded F polynomials over coset domain for use in round 3.
    // 5 × coset_FFT(2m)
    for _ in 0..5 { counting::record_fft(pk.coset_domain.size()); }
    let f_coset_evals = [
        pk.coset_domain.coset_fft(&f_polys[0].polynomial().coeffs),
        pk.coset_domain.coset_fft(&f_polys[1].polynomial().coeffs),
        pk.coset_domain.coset_fft(&f_polys[2].polynomial().coeffs),
        pk.coset_domain.coset_fft(&f_polys[3].polynomial().coeffs),
        pk.coset_domain.coset_fft(&f_polys[4].polynomial().coeffs),
    ];

    Round2State {
        polynomials: f_polys,
        f_evals: [f1_evals, f2_evals, f3_evals, f4_evals, f5_evals],
        f_coset_evals,
        beta,
        rands: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Round 3
// ---------------------------------------------------------------------------

/// Compute polys (eq. 10 in the paper, Appendix B).
///
/// U(X) = X^{m-1} − 1  (i.e. X³ − 1 for m = 4).
/// R*(X) = R_F(X) · U(X), where R_F is the remainder of (∑Fⱼ)/z_K.
///
/// P(X) =
///   F₁(X)(β + R(X)) − 1
/// + η  (F₂(X)(β + C(X)) − 1)
/// + η² (F₃(X)(β + C(X)) − Δ·R(X))
/// + η³ (F₄(X)(β + C(X)) − Δᵗ)
/// − η⁴ (F₅(X)(β + h(X)) − m(X)·z_{K\H}(X))
/// + η⁵ (R²(X) − row(X))
/// + η⁶ (C²(X) − col(X))
/// + η⁷ (rowcol(X) − row̃(X)·col(X))
/// + η⁸ (row̃(X) − row(X))
/// + η⁹ (∑_{j=1}^{5} Fⱼ(X)) · U(X)
/// − η⁹ · X · R*(X)
///
/// and q(X) = P(X) / (z_K(X) · U(X)).
pub fn round_three<E: PairingEngine>(
    pk: &PfrPublicKey<E>,
    round1_state: &Round1State<E>,
    round2_state: &Round2State<E>,
    eta: E::Fr,
) -> Round3State<E> {
    let beta = round2_state.beta;
    let big_delta = pk.big_delta();
    let big_delta_t = big_delta.pow([pk.t as u64]);
    let z_k: DensePolynomial<E::Fr> = pk.k_domain.vanishing_polynomial().into();

    // U(X) = X³ − 1  (fixed; see eq. 9 and surrounding text in Appendix B)
    let u_poly = DensePolynomial::from_coefficients_vec(vec![
        -E::Fr::one(),
        E::Fr::zero(),
        E::Fr::zero(),
        E::Fr::one(),
    ]);

    // -----------------------------------------------------------------------
    // Compute R*(X) = (R_F(X) + η·R_S) · U(X) from eq. (9).
    // Use F polynomial coefficients directly (cheap polynomial additions).
    // -----------------------------------------------------------------------
    let s_poly = round1_state.polynomials[3].polynomial();
    let f1_poly = round2_state.polynomials[0].polynomial();
    let f2_poly = round2_state.polynomials[1].polynomial();
    let f3_poly = round2_state.polynomials[2].polynomial();
    let f4_poly = round2_state.polynomials[3].polynomial();
    let f5_poly = round2_state.polynomials[4].polynomial();

    let f_sum_poly = &(&(f1_poly + f2_poly) + &(f3_poly + f4_poly)) + f5_poly;
    let fs_sum_poly = {
        let mut t = DensePolynomial::zero();
        t += (eta, s_poly);
        &f_sum_poly + &t
    };
    let (_q_f, r_f) = fs_sum_poly.divide_by_vanishing_poly(pk.k_domain).unwrap();
    debug_assert!(
        r_f.coeffs
            .get(0)
            .map(|c| *c == E::Fr::zero())
            .unwrap_or(true),
        "r_f constant term is nonzero — sumcheck failed"
    );
    let r_f_over_x = if r_f.is_zero() {
        DensePolynomial::zero()
    } else {
        DensePolynomial::from_coefficients_slice(&r_f.coeffs[1..])
    };
    let r_star = &r_f_over_x * &u_poly;

    // -----------------------------------------------------------------------
    // Compute q(X) = big_sum(X) / z_K(X) using coset evaluations.
    //
    // big_sum(X) = ∑ ηʲ·termⱼ vanishes on K by construction, so big_sum/z_K
    // is a polynomial. We evaluate each term pointwise on the coset domain
    // (where z_K ≠ 0), divide pointwise, then IFFT to get the coefficients.
    //
    // Coset evals layout:
    //   round1_state.coset_evals: [R, C, m, S, row, col, rowcol, rowtilde]
    //   round2_state.f_coset_evals: [F1, F2, F3, F4, F5]
    //   pk.h_coset_evals, pk.zkh_coset_evals: precomputed at setup
    // -----------------------------------------------------------------------
    let cd = pk.coset_domain; // size 2m
    let nc = cd.size();

    let r_c = &round1_state.coset_evals[0];
    let c_c = &round1_state.coset_evals[1];
    let m_c = &round1_state.coset_evals[2];
    let _s_c = &round1_state.coset_evals[3]; // S included via fss_c
    let row_c = &round1_state.coset_evals[4];
    let col_c = &round1_state.coset_evals[5];
    let rc_c = &round1_state.coset_evals[6]; // rowcol
    let rt_c = &round1_state.coset_evals[7]; // rowtilde
    let f1_c = &round2_state.f_coset_evals[0];
    let f2_c = &round2_state.f_coset_evals[1];
    let f3_c = &round2_state.f_coset_evals[2];
    let f4_c = &round2_state.f_coset_evals[3];
    let f5_c = &round2_state.f_coset_evals[4];
    let h_c = &pk.h_coset_evals;
    let zkh_c = &pk.zkh_coset_evals;

    // Evaluate z_K and fs_sum over the coset. 2 × coset_FFT(2m)
    counting::record_fft(cd.size()); // z_K over coset
    counting::record_fft(cd.size()); // fs_sum over coset
    let zk_c = cd.coset_fft(&z_k.coeffs);
    let fss_c = cd.coset_fft(&fs_sum_poly.coeffs);

    // Precompute η powers.
    let mut eta_pows = vec![E::Fr::one(); 10];
    for i in 1..10 {
        eta_pows[i] = eta_pows[i - 1] * eta;
    }

    // Build big_sum pointwise over the coset.
    let big_sum_over_zk: Vec<E::Fr> = (0..nc)
        .map(|i| {
            let r = r_c[i];
            let c = c_c[i];
            let m = m_c[i];
            let row = row_c[i];
            let col = col_c[i];
            let rc = rc_c[i];
            let rt = rt_c[i];
            let f1 = f1_c[i];
            let f2 = f2_c[i];
            let f3 = f3_c[i];
            let f4 = f4_c[i];
            let f5 = f5_c[i];
            let h = h_c[i];
            let zkh = zkh_c[i];
            let fss = fss_c[i];

            // η⁰: F₁(β + R) − 1
            let mut s = f1 * (beta + r) - E::Fr::one();
            // η¹: F₂(β + C) − 1
            s += eta_pows[1] * (f2 * (beta + c) - E::Fr::one());
            // η²: F₃(β·Δ·R + C) − Δ·R
            let delta_r = big_delta * r;
            s += eta_pows[2] * (f3 * (beta * delta_r + c) - delta_r);
            // η³: F₄(β·Δᵗ + C) − Δᵗ
            s += eta_pows[3] * (f4 * (beta * big_delta_t + c) - big_delta_t);
            // η⁴: F₅(β + h) + m·z_{K\H}
            s += eta_pows[4] * (f5 * (beta + h) + m * zkh);
            // η⁵: R² − row
            s += eta_pows[5] * (r * r - row);
            // η⁶: C² − col
            s += eta_pows[6] * (c * c - col);
            // η⁷: rowcol − row̃·col
            s += eta_pows[7] * (rc - rt * col);
            // η⁸: row̃ − row
            s += eta_pows[8] * (rt - row);
            // η⁹: fs_sum
            s += eta_pows[9] * fss;

            s
        })
        .collect();

    // P(X) = (big_sum − η⁹·X·r_f_over_x) · U
    // since R* = r_f_over_x · U, so η⁹·X·R* = η⁹·X·r_f_over_x·U.
    // Therefore q = P / (z_K·U) = (big_sum − η⁹·X·r_f_over_x) / z_K.
    //
    // Both big_sum and X·r_f_over_x = r_f vanish on K, so the difference
    // is divisible by z_K. Compute it pointwise on the coset, then IFFT.

    let eta9 = eta_pows[9];

    // Evaluate η⁹·r_f over the coset. 1 × coset_FFT(2m)
    counting::record_fft(cd.size()); // r_f over coset
    let rf_c = cd.coset_fft(&r_f.coeffs);

    // Combine: (big_sum − η⁹·r_f) / z_K pointwise.
    // 2m individual field inversions (z_K values on the coset)
    for _ in 0..cd.size() { counting::record_field_inv(); }
    let mut q_evals: Vec<E::Fr> = big_sum_over_zk
        .into_iter()
        .zip(rf_c.iter())
        .zip(zk_c.iter())
        .map(|((bs, &rf), &zk)| (bs - eta9 * rf) * zk.inverse().unwrap())
        .collect();

    // IFFT to get q as polynomial coefficients. 1 × coset_IFFT(2m)
    counting::record_ifft(cd.size());
    cd.coset_ifft_in_place(&mut q_evals);
    let q_poly = DensePolynomial::from_coefficients_vec(q_evals);

    Round3State {
        polynomials: vec![
            LabeledPolynomial::new("r_star".into(), r_star, None, None),
            LabeledPolynomial::new("q".into(), q_poly, None, None),
        ],
        u_poly,
        eta,
        eta9,
        rands: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Round 4
// ---------------------------------------------------------------------------

/// **Round 4**: evaluate h, R, C, row at the verifier challenge α.
///
/// Evaluates row̃(X) (index 7) at α, which equals row(α) when ρ_row = 0.
pub fn round_four<E: PairingEngine>(
    pk: &PfrPublicKey<E>,
    round1_state: &Round1State<E>,
    alpha: E::Fr,
) -> Round4State<E> {
    Round4State {
        alpha,
        h_alpha: pk.h_poly.evaluate(&alpha),
        r_alpha: round1_state.polynomials[0].polynomial().evaluate(&alpha),
        c_alpha: round1_state.polynomials[1].polynomial().evaluate(&alpha),
        row_alpha: round1_state.polynomials[7].polynomial().evaluate(&alpha),
    }
}

// ---------------------------------------------------------------------------
// Round 5
// ---------------------------------------------------------------------------

/// **Round 5**: compute Lin(X) and the batched opening proof Q(X).
///
/// Lin(X) is obtained from P(X) by substituting the Round-4 scalars
/// (h_α, R_α, C_α, row̃_α) in place of h(X), R(X), C(X), row̃(X), while
/// keeping the Fⱼ(X) and q(X) as polynomials.  It satisfies Lin(α) = 0.
///
/// Corrected forms relative to the paper's typos (confirmed):
///   η²: F₃(X)(β·Δ·R_α + C_α) − Δ·R_α   (paper wrote β + C_α)
///   η³: F₄(X)(β·Δᵗ + C_α) − Δᵗ          (paper wrote β + C_α)
///
/// Full Lin(X) formula (page 39, with corrections):
///   Lin(X) = big_lin(X) · U(α) − η⁹ · α · R*(X) − q(X) · U(α) · z_K(α)
///
/// where big_lin is the same sum as big_sum in round_three but with scalar
/// substitutions for R, C, row, h.
///
/// Q(X) = [ (h(X)−h_α) + δ(R(X)−R_α) + δ²(C(X)−C_α)
///          + δ³(row̃(X)−row̃_α) + δ⁴·Lin(X) ] / (X − α)
pub fn round_five<E: PairingEngine>(
    pk: &PfrPublicKey<E>,
    round1_state: &Round1State<E>,
    round2_state: &Round2State<E>,
    round3_state: &Round3State<E>,
    round4_state: &Round4State<E>,
    delta: E::Fr,
) -> DensePolynomial<E::Fr> {
    let beta = round2_state.beta;
    let eta = round3_state.eta;
    let eta9 = round3_state.eta9;
    let alpha = round4_state.alpha;
    let h_alpha = round4_state.h_alpha;
    let r_alpha = round4_state.r_alpha;
    let c_alpha = round4_state.c_alpha;
    let row_alpha = round4_state.row_alpha;

    let big_delta = pk.big_delta();
    let big_delta_t = big_delta.pow([pk.t as u64]);

    // Round-1 polynomials
    let r_poly = round1_state.polynomials[0].polynomial();
    let c_poly = round1_state.polynomials[1].polynomial();
    let m_poly = round1_state.polynomials[2].polynomial();
    let row_poly = round1_state.polynomials[4].polynomial(); // public row(X)
    let col_poly = round1_state.polynomials[5].polynomial();
    let rowcol_poly = round1_state.polynomials[6].polynomial();
    let rowtilde_poly = round1_state.polynomials[7].polynomial(); // blinded row̃(X)
                                                                  // Round-2 polynomials
    let f1_poly = round2_state.polynomials[0].polynomial();
    let f2_poly = round2_state.polynomials[1].polynomial();
    let f3_poly = round2_state.polynomials[2].polynomial();
    let f4_poly = round2_state.polynomials[3].polynomial();
    let f5_poly = round2_state.polynomials[4].polynomial();
    // Round-3 precomputed values
    let u_poly = &round3_state.u_poly;
    let r_star = round3_state.polynomials[0].polynomial();
    let q_poly = round3_state.polynomials[1].polynomial();

    let scale = |s: E::Fr, p: &DensePolynomial<E::Fr>| -> DensePolynomial<E::Fr> {
        let mut out = DensePolynomial::zero();
        out += (s, p);
        out
    };
    let const_poly = |v: E::Fr| DensePolynomial::from_coefficients_vec(vec![v]);

    // z_{K\H}(X) = (n/m)·(X^m−1)/(X^n−1) as a polynomial (requires n | m).
    // Factor: X^m−1 = (X^n−1)·(X^{m−n}+X^{m−2n}+…+1), so z_{K\H}(X) = (n/m)·Q(X)
    // where Q has coeff (n/m) at degrees 0, n, 2n, …, m−n.
    let zkh: DensePolynomial<E::Fr> = {
        debug_assert_eq!(pk.m % pk.n, 0, "m must be a multiple of n");
        let steps = pk.m / pk.n;
        let scale_factor = E::Fr::from(pk.n as u64) * E::Fr::from(pk.m as u64).inverse().unwrap();
        let mut coeffs = vec![E::Fr::zero(); (steps - 1) * pk.n + 1];
        for k in 0..steps {
            coeffs[k * pk.n] = scale_factor;
        }
        DensePolynomial::from_coefficients_vec(coeffs)
    };

    let u_at_alpha = u_poly.evaluate(&alpha);
    let zk_at_alpha = pk.k_domain.vanishing_polynomial().evaluate(&alpha);

    // f_sum_poly = F₁ + … + F₅
    let f_sum_poly = &(&(f1_poly + f2_poly) + &(f3_poly + f4_poly)) + f5_poly;

    // Build big_lin: same structure as big_sum in round_three with scalar substitutions.

    // η⁰: F₁(X)(β + R_α) − 1
    let mut big_lin = &scale(beta + r_alpha, f1_poly) - &const_poly(E::Fr::one());
    let mut eta_pow = eta;

    // η¹: F₂(X)(β + C_α) − 1
    big_lin += &scale(
        eta_pow,
        &(&scale(beta + c_alpha, f2_poly) - &const_poly(E::Fr::one())),
    );

    // η²: F₃(X)(β·Δ·R_α + C_α) − Δ·R_α
    eta_pow *= eta;
    big_lin += &scale(
        eta_pow,
        &(&scale(beta * big_delta * r_alpha + c_alpha, f3_poly) - &const_poly(big_delta * r_alpha)),
    );

    // η³: F₄(X)(β·Δᵗ + C_α) − Δᵗ
    eta_pow *= eta;
    big_lin += &scale(
        eta_pow,
        &(&scale(beta * big_delta_t + c_alpha, f4_poly) - &const_poly(big_delta_t)),
    );

    // η⁴: F₅(X)(β + h_α) + m(X)·z_{K\H}(α)
    eta_pow *= eta;
    let zkh_at_alpha = zkh.evaluate(&alpha);
    big_lin += &scale(
        eta_pow,
        &(&scale(beta + h_alpha, f5_poly) + &scale(zkh_at_alpha, m_poly)),
    );

    // η⁵: R_α² − row(X)
    eta_pow *= eta;
    big_lin += &scale(eta_pow, &(&const_poly(r_alpha * r_alpha) - row_poly));

    // η⁶: C_α² − col(X)
    eta_pow *= eta;
    big_lin += &scale(eta_pow, &(&const_poly(c_alpha * c_alpha) - col_poly));

    // η⁷: rowcol(X) − row̃_α · col(X)
    eta_pow *= eta;
    big_lin += &scale(eta_pow, &(rowcol_poly - &scale(row_alpha, col_poly)));

    // η⁸: row̃(X) − row(X)
    eta_pow *= eta;
    big_lin += &scale(eta_pow, &(rowtilde_poly - row_poly));

    // η⁹: ∑Fⱼ(X) + η·S(X)  — matches round_three exactly
    let s_poly = round1_state.polynomials[3].polynomial();
    let fs_sum_poly = {
        let mut t = DensePolynomial::zero();
        t += (eta, s_poly);
        &f_sum_poly + &t
    };
    big_lin += &scale(eta9, &fs_sum_poly);

    // Lin(X) = big_lin · U(α) − η⁹ · α · R*(X) − q(X) · U(α) · z_K(α)
    let lin_poly = &(&scale(u_at_alpha, &big_lin) - &scale(eta9 * alpha, r_star))
        - &scale(u_at_alpha * zk_at_alpha, q_poly);

    // Q(X) = numerator / (X − α)
    let mut delta_pow = E::Fr::one();
    let mut numerator = &pk.h_poly - &const_poly(h_alpha);

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(r_poly - &const_poly(r_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(c_poly - &const_poly(c_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(rowtilde_poly - &const_poly(row_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &lin_poly);

    let x_minus_alpha = DensePolynomial::from_coefficients_vec(vec![-alpha, E::Fr::one()]);
    use ark_poly::univariate::DenseOrSparsePolynomial;
    let (q_open_poly, rem) = DenseOrSparsePolynomial::from(numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(x_minus_alpha))
        .unwrap();
    debug_assert!(
        rem.coeffs.iter().all(|c| *c == E::Fr::zero()),
        "Round-5 numerator is not divisible by (X − α)"
    );

    q_open_poly
}

// ---------------------------------------------------------------------------
// Statement commitment
// ---------------------------------------------------------------------------

/// Commit to the three public index polynomials row(X), col(X), rowcol(X).
///
/// This is the *statement* phase: it must be run before [`prove`] and the
/// resulting [`PfrStatement`] is passed into it.  Separating this step allows
/// the statement-commitment cost to be measured independently.
///
/// | Polynomial | Encoding at κ^i              |
/// |------------|-------------------------------|
/// | row(X)     | ω^{r_i}                       |
/// | col(X)     | ω^{c_i}                       |
/// | rowcol(X)  | ω^{r_i} · ω^{c_i} = ω^{r_i·c_i} (only when r_i+c_i < n) |
pub fn commit_statement<E: PairingEngine, R: RngCore>(
    pk: &PfrPublicKey<E>,
    row_indices: &[usize],
    col_indices: &[usize],
    rng: &mut R,
) -> PfrStatement<E> {
    let row_evals: Vec<E::Fr> = row_indices
        .iter()
        .map(|&j| pk.h_domain.element(j))
        .collect();
    counting::record_ifft(pk.k_domain.size());
    let row_poly =
        EvaluationsOnDomain::from_vec_and_domain(row_evals, pk.k_domain).interpolate();

    let col_evals: Vec<E::Fr> = col_indices
        .iter()
        .map(|&j| pk.h_domain.element(j))
        .collect();
    counting::record_ifft(pk.k_domain.size());
    let col_poly =
        EvaluationsOnDomain::from_vec_and_domain(col_evals, pk.k_domain).interpolate();

    let rowcol_evals: Vec<E::Fr> = row_indices
        .iter()
        .zip(col_indices.iter())
        .map(|(&r, &c)| pk.h_domain.element(r) * pk.h_domain.element(c))
        .collect();
    counting::record_ifft(pk.k_domain.size());
    let rowcol_poly =
        EvaluationsOnDomain::from_vec_and_domain(rowcol_evals, pk.k_domain).interpolate();

    let row_labeled = LabeledPolynomial::new("row".into(), row_poly, None, None);
    let col_labeled = LabeledPolynomial::new("col".into(), col_poly, None, None);
    let rowcol_labeled = LabeledPolynomial::new("rowcol".into(), rowcol_poly, None, None);

    // KZG commit: 3 polys of degree m-1, no hiding, no degree bound → 3 × MSM_G1(m)
    counting::record_msm_g1(pk.k_domain.size()); // row
    counting::record_msm_g1(pk.k_domain.size()); // col
    counting::record_msm_g1(pk.k_domain.size()); // rowcol
    let (mut comms, mut rands) = PC::<E>::commit(
        &pk.ck,
        [&row_labeled, &col_labeled, &rowcol_labeled].iter().copied(),
        Some(rng),
    )
    .unwrap();

    let rowcol_rand = rands.remove(2);
    let col_rand = rands.remove(1);
    let row_rand = rands.remove(0);
    let rowcol_comm = comms.remove(2);
    let col_comm = comms.remove(1);
    let row_comm = comms.remove(0);

    PfrStatement {
        row_poly: row_labeled,
        col_poly: col_labeled,
        rowcol_poly: rowcol_labeled,
        row_comm,
        col_comm,
        rowcol_comm,
        row_rand,
        col_rand,
        rowcol_rand,
    }
}

/// Commit to the statement polynomials using pre-computed field-element evaluations
/// (ω^{r_i}, ω^{c_i} already computed). Avoids re-deriving from integer indices.
pub fn commit_statement_from_evals<E: PairingEngine, R: RngCore>(
    pk: &PfrPublicKey<E>,
    row_evals: Vec<E::Fr>,
    col_evals: Vec<E::Fr>,
    rng: &mut R,
) -> PfrStatement<E> {
    assert_eq!(row_evals.len(), pk.k_domain.size());
    assert_eq!(col_evals.len(), pk.k_domain.size());

    let rowcol_evals: Vec<E::Fr> = row_evals
        .iter()
        .zip(col_evals.iter())
        .map(|(r, c)| *r * c)
        .collect();

    counting::record_ifft(pk.k_domain.size());
    let row_poly =
        EvaluationsOnDomain::from_vec_and_domain(row_evals, pk.k_domain).interpolate();
    counting::record_ifft(pk.k_domain.size());
    let col_poly =
        EvaluationsOnDomain::from_vec_and_domain(col_evals, pk.k_domain).interpolate();
    counting::record_ifft(pk.k_domain.size());
    let rowcol_poly =
        EvaluationsOnDomain::from_vec_and_domain(rowcol_evals, pk.k_domain).interpolate();

    let row_labeled = LabeledPolynomial::new("row".into(), row_poly, None, None);
    let col_labeled = LabeledPolynomial::new("col".into(), col_poly, None, None);
    let rowcol_labeled = LabeledPolynomial::new("rowcol".into(), rowcol_poly, None, None);

    counting::record_msm_g1(pk.k_domain.size());
    counting::record_msm_g1(pk.k_domain.size());
    counting::record_msm_g1(pk.k_domain.size());
    let (mut comms, mut rands) = PC::<E>::commit(
        &pk.ck,
        [&row_labeled, &col_labeled, &rowcol_labeled].iter().copied(),
        Some(rng),
    )
    .unwrap();

    let rowcol_rand = rands.remove(2);
    let col_rand = rands.remove(1);
    let row_rand = rands.remove(0);
    let rowcol_comm = comms.remove(2);
    let col_comm = comms.remove(1);
    let row_comm = comms.remove(0);

    PfrStatement {
        row_poly: row_labeled,
        col_poly: col_labeled,
        rowcol_poly: rowcol_labeled,
        row_comm,
        col_comm,
        rowcol_comm,
        row_rand,
        col_rand,
        rowcol_rand,
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Run the 5-round PFR interactive protocol (Appendix B) and return the proof.
///
/// ## Protocol outline
///
/// **Round 1** — Prover commits to witness polynomials:
///   R(X), C(X): R(κ^i) = Δ^{r_i}, C(κ^i) = Δ^{c_i}  (Δ-power encodings; since
///     Δ² = ω, these satisfy R(κ^i)² = ω^{r_i} = row(κ^i), i.e. R is the pointwise
///     square root of the standard Marlin row polynomial over K, and likewise for C)
///   m(X): multiplicity polynomial over H (satisfies eq. 7)
///   S(X) = 0: sumcheck blinding (zero in no-ZK mode)
///   row̃(X): auxiliary; row̃(κ^i) = ω^{r_i}
///   → Sends [R(τ), C(τ), m(τ), S(τ), row̃(τ)]₁
///
/// **Round 2** — After challenge β, prover commits to F₁, …, F₅:
///   F_j interpolates the j-th summand sequence from eq. (8) over K
///   → Sends [F₁(τ), …, F₅(τ)]₁
///
/// **Round 3** — After challenge η, prover batches polynomial identities:
///   Batched identity P(X) from eq. (10); quotient q(X); degree-check R*(X)
///   → Sends [R*(τ), q(τ)]₁
///
/// **Round 4** — After challenge α, prover opens four polynomials at α:
///   Sends h_α = h(α), R_α = R(α), C_α = C(α), row̃_α = row̃(α)
///
/// **Round 5** — After challenge δ, prover sends a batched KZG opening:
///   → Sends [Q(τ)]₁
///
pub fn prove<E: PairingEngine, R: RngCore>(
    pk: &PfrPublicKey<E>,
    row_indices: &[usize],
    col_indices: &[usize],
    stmt: &PfrStatement<E>,
    rng: &mut R,
) -> (PfrProof<E>, PfrPublicInputs<E>) {
    // Initialise the Fiat-Shamir transcript with the public-key commitment.
    let mut fs_rng = SimpleHashFiatShamirRng::<blake2::Blake2s, rand_chacha::ChaChaRng>::initialize(
        &to_bytes![pk.h_commitment.commitment()].unwrap(),
    );

    // --- Round 1 ---
    let mut round1_state = round_one(pk, row_indices, col_indices, stmt, rng);

    let first_round_comm_time = start_timer!(|| "Committing to Round 1 polynomials");
    // Commit to the 5 witness polynomials [R, C, m, S, rowtilde]; the statement
    // polynomials [row, col, rowcol] were already committed in commit_statement.
    let witness_polys = [
        &round1_state.polynomials[0], // R
        &round1_state.polynomials[1], // C
        &round1_state.polynomials[2], // m
        &round1_state.polynomials[3], // S
        &round1_state.polynomials[7], // rowtilde
    ];
    // 5 × MSM_G1(m+2): commit R, C, m, S, rowtilde (degree m+1, no hiding, no shifted comm)
    for _ in 0..5 { counting::record_msm_g1(pk.m + 2); }
    let (round1_comms, round1_rands) =
        PC::<E>::commit(&pk.ck, witness_polys.iter().copied(), None).unwrap();
    end_timer!(first_round_comm_time);
    // Reconstitute a full 8-element rands vec aligned with polynomials[]:
    // [R, C, m, S, row, col, rowcol, rowtilde].
    // Statement rands come from stmt; witness rands come from the commit above.
    round1_state.rands = vec![
        round1_rands[0].clone(), // R
        round1_rands[1].clone(), // C
        round1_rands[2].clone(), // m
        round1_rands[3].clone(), // S
        stmt.row_rand.clone(),   // row    (from stmt)
        stmt.col_rand.clone(),   // col    (from stmt)
        stmt.rowcol_rand.clone(),// rowcol (from stmt)
        round1_rands[4].clone(), // rowtilde
    ];

    let mut round1_comms = round1_comms;
    let r_comm = round1_comms.remove(0);       // [R(τ)]₁
    let c_comm = round1_comms.remove(0);       // [C(τ)]₁
    let m_comm = round1_comms.remove(0);       // [m(τ)]₁
    let s_comm = round1_comms.remove(0);       // [S(τ)]₁
    let rowtilde_comm = round1_comms.remove(0); // [row̃(τ)]₁

    // Statement commitments come from the pre-committed stmt.
    let row_comm = stmt.row_comm.clone();
    let col_comm = stmt.col_comm.clone();
    let rowcol_comm = stmt.rowcol_comm.clone();

    // Derive β by absorbing Round 1 commitments into the transcript.
    fs_rng.absorb(
        &to_bytes![
            r_comm.commitment(),
            c_comm.commitment(),
            m_comm.commitment(),
            s_comm.commitment(),
            rowtilde_comm.commitment()
        ]
        .unwrap(),
    );
    let beta = E::Fr::rand(&mut fs_rng);

    // --- Round 2 ---
    let mut round2_state = round_two(pk, &round1_state, beta, rng);

    let second_round_comm_time = start_timer!(|| "Committing to Round 2 polynomials");
    // 5 × MSM_G1(m+2): commit F1..F5 (degree m+1, no hiding, no shifted comm)
    for _ in 0..5 { counting::record_msm_g1(pk.m + 2); }
    let (f_comms, round2_rands) =
        PC::<E>::commit(&pk.ck, round2_state.polynomials.iter(), None).unwrap();
    end_timer!(second_round_comm_time);
    round2_state.rands = round2_rands;

    // Derive η by absorbing Round 2 commitments.
    fs_rng.absorb(
        &to_bytes![
            f_comms[0].commitment(),
            f_comms[1].commitment(),
            f_comms[2].commitment(),
            f_comms[3].commitment(),
            f_comms[4].commitment()
        ]
        .unwrap(),
    );
    let eta = E::Fr::rand(&mut fs_rng);

    // --- Round 3 ---
    let mut round3_state = round_three(pk, &round1_state, &round2_state, eta);

    let third_round_comm_time = start_timer!(|| "Committing to Round 3 polynomials");
    // MSM_G1(m+2) for r_star (deg ≤ m+1), MSM_G1(2m+4) for q (deg ≤ 2m+3)
    counting::record_msm_g1(pk.m + 2);
    counting::record_msm_g1(2 * pk.m + 4);
    let (round3_comms, round3_rands) =
        PC::<E>::commit(&pk.ck, round3_state.polynomials.iter(), None).unwrap();
    end_timer!(third_round_comm_time);
    round3_state.rands = round3_rands;

    let mut round3_comms = round3_comms;
    let r_star_comm = round3_comms.remove(0);
    let q_comm = round3_comms.remove(0);

    // --- Round 4 ---
    // Derive α by absorbing Round 3 commitments.
    fs_rng.absorb(&to_bytes![r_star_comm.commitment(), q_comm.commitment()].unwrap());
    let alpha = E::Fr::rand(&mut fs_rng);

    let round4_state = round_four(pk, &round1_state, alpha);

    // --- Round 5 ---
    // Derive δ by absorbing Round 4 evaluation values.
    fs_rng.absorb(
        &to_bytes![
            round4_state.h_alpha,
            round4_state.r_alpha,
            round4_state.c_alpha,
            round4_state.row_alpha
        ]
        .unwrap(),
    );
    let delta = E::Fr::rand(&mut fs_rng);

    let fifth_round_time = start_timer!(|| "Computing Round 5 Q polynomial");
    let q_open_poly = round_five(
        pk,
        &round1_state,
        &round2_state,
        &round3_state,
        &round4_state,
        delta,
    );
    end_timer!(fifth_round_time);

    let q_open_labeled = LabeledPolynomial::new("Q".into(), q_open_poly, None, None);
    // MSM_G1(2m+3) for Q (deg ≤ 2m+2)
    counting::record_msm_g1(2 * pk.m + 3);
    let (mut q_open_comms, _) = PC::<E>::commit(&pk.ck, vec![&q_open_labeled], None).unwrap();
    let q_poly_comm = q_open_comms.remove(0);

    let proof = PfrProof {
        r_comm,
        c_comm,
        m_comm,
        s_comm,
        rowtilde_comm,
        f_comms,
        r_star_comm,
        q_comm,
        h_alpha: round4_state.h_alpha,
        r_alpha: round4_state.r_alpha,
        c_alpha: round4_state.c_alpha,
        row_alpha: round4_state.row_alpha,
        q_poly_comm,
    };
    let public_inputs = PfrPublicInputs {
        row_comm,
        col_comm,
        rowcol_comm,
    };
    (proof, public_inputs)
}
