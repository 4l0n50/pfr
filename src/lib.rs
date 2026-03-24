//! Proof of Function Relation (PFR) — Appendix B, IMPR-FHFC paper.
//!
//! ## Notation (matches Appendix B)
//!
//! | Paper | Code                  | Meaning                                           |
//! |-------|-----------------------|---------------------------------------------------|
//! | n     | `pk.n`                | \|H\|: table domain size                          |
//! | m     | `pk.m`                | \|K\|: number of index pairs (non-zero entries)   |
//! | t     | `pk.t`                | strictly-lower-triangular offset                  |
//! | ω     | `h_domain.element(1)` | generator of H                                    |
//! | κ     | `k_domain.element(1)` | generator of K                                    |
//! | Δ     | `d_domain.element(1)` | generator of D, with Δ² = ω                       |
//! | r_i   | `row_indices[i]`      | row index of the i-th pair                        |
//! | c_i   | `col_indices[i]`      | column index of the i-th pair                     |
//! | m_j   | `mults[j]`            | multiplicity of h(ω^j) in the 4m-element multiset |
//!
//! ## Equation (7) — the lookup identity
//!
//! ```text
//!   m                                                           n-1
//!   ∑  [ 1/(R(κ^i)+X) + 1/(C(κ^i)+X)                     =       ∑   m_j / (h(ω^j)+X)
//!  i=1    + 1/(C(κ^i)/(Δ·R(κ^i))+X) + 1/(C(κ^i)/Δ^t+X) ]        j=0
//! ```
//!
//! ## 5-round protocol (Appendix B)
//!
//! | Round | Prover sends                        | Challenge |
//! |-------|-------------------------------------|-----------|
//! | 1     | \[R(τ), C(τ), m(τ), S(τ), row̃(τ)\]₁ | β         |
//! | 2     | \[F₁(τ), …, F₅(τ)\]₁                | η         |
//! | 3     | \[R\*(τ), q(τ)\]₁                   | α         |
//! | 4     | field elements h_α, R_α, C_α, row̃_α | δ         |
//! | 5     | \[Q(τ)\]₁                           | —         |

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{to_bytes, Field, One, UniformRand, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, Polynomial, UVPolynomial,
};
use ark_poly_commit::{
    marlin_pc::MarlinKZG10, LabeledCommitment, LabeledPolynomial, PolynomialCommitment,
};
use ark_std::rand::RngCore;
use ark_std::{end_timer, start_timer};
use blake2::Blake2s;
use rand_chacha::ChaChaRng;

use ark_marlin::{rng::FiatShamirRng, SimpleHashFiatShamirRng};
type FS = SimpleHashFiatShamirRng<Blake2s, ChaChaRng>;

// ---------------------------------------------------------------------------
// Convenience type aliases
// ---------------------------------------------------------------------------
type PC = MarlinKZG10<Bls12_381, DensePolynomial<Fr>>;
type Comm = LabeledCommitment<<PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::Commitment>;
type Rand = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::Randomness;
type CK = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::CommitterKey;
type VK = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::VerifierKey;

// ---------------------------------------------------------------------------
// Public key
// ---------------------------------------------------------------------------

/// Public parameters for the PFR (output of the key-generation phase).
///
/// Contains the table polynomial h(X) — committed once and reused across
/// many proofs — together with the KZG commitment keys and the three domains.
#[allow(dead_code)]
pub struct PfrPublicKey {
    /// n = |H|: size of the table domain H = <ω>.
    pub n: usize,
    /// m = |K|: number of index pairs (non-zero entries of the relation).
    pub m: usize,
    /// t: strictly-lower-triangular offset; every column index satisfies c_i ≥ t.
    pub t: usize,
    pub h_domain: GeneralEvaluationDomain<Fr>, // H = <ω>,  |H| = n
    pub d_domain: GeneralEvaluationDomain<Fr>, // D = <Δ>,  |D| = 2n,  Δ² = ω
    pub k_domain: GeneralEvaluationDomain<Fr>, // K = <κ>,  |K| = m
    /// Table polynomial: h(ω^j) = Δ^j for j = 0, …, n−1.
    pub h_poly: DensePolynomial<Fr>,
    pub ck: CK,
    pub vk: VK,
    pub h_commitment: Comm,
    pub h_randomness: Rand,
}

impl PfrPublicKey {
    /// Generate public parameters.
    ///
    /// - `n`: table size |H|
    /// - `m`: number of index pairs |K|
    /// - `t`: strictly-lower-triangular offset
    ///
    /// No zero-knowledge: `hiding_bound = None`, `supported_hiding_bound = 0`.
    pub fn setup<R: RngCore>(n: usize, m: usize, t: usize, rng: &mut R) -> Self {
        let h_domain = GeneralEvaluationDomain::<Fr>::new(n).expect("H domain must exist");
        let d_domain = GeneralEvaluationDomain::<Fr>::new(2 * n).expect("D domain must exist");
        let k_domain = GeneralEvaluationDomain::<Fr>::new(m).expect("K domain must exist");

        // h(X): unique polynomial of degree < n with h(ω^j) = Δ^j for j = 0, …, n−1.
        let h_evals: Vec<Fr> = d_domain.elements().take(n).collect();
        let h_poly = EvaluationsOnDomain::from_vec_and_domain(h_evals, h_domain).interpolate();

        // Round-3 P(X) has degree ≤ 2(m−1)+3 = 2m+1; q = P/(z_K·U) has degree ≤ 2m+1−m−3 = m−2.
        // R*(X) = R_F·U has degree ≤ (m−2)+3 = m+1.
        // The SRS must support committing to all of these.
        let max_degree = 2 * m + 1;
        let pp = PC::setup(max_degree, None, rng).unwrap();
        let (ck, vk) = PC::trim(&pp, max_degree, 0, None).unwrap();

        let h_labeled = LabeledPolynomial::new("h".into(), h_poly.clone(), None, None);
        let (mut comms, mut rands) = PC::commit(&ck, vec![&h_labeled], None).unwrap();

        Self {
            n,
            m,
            t,
            h_domain,
            d_domain,
            k_domain,
            h_poly,
            ck,
            vk,
            h_commitment: comms.remove(0),
            h_randomness: rands.remove(0),
        }
    }

    // ---------------------------------------------------------------------------
    // Multiplicity computation — equation (7)
    // ---------------------------------------------------------------------------

    /// Compute the multiplicity vector (m_0, …, m_{n-1}) satisfying equation (7).
    ///
    /// Each pair (r_i, c_i) contributes four table indices:
    ///
    /// | Term in eq. (7)| Value         |Table index|
    /// |----------------|---------------|-----------|
    /// | R(κ^i)         | Δ^{r_i}       | r_i       |
    /// | C(κ^i)         | Δ^{c_i}       | c_i       |
    /// | C/(Δ·R)(κ^i)   | Δ^{c_i−r_i−1} | c_i−r_i−1 |
    /// | C/Δ^t(κ^i)     | Δ^{c_i−t}     | c_i−t     |
    ///
    /// m_j counts how many times index j appears across all 4·m contributions.
    ///
    /// **Preconditions**: r_i < c_i, c_i ≥ t, all resulting indices ∈ [0, n−1].
    pub fn compute_multiplicities(&self, row_indices: &[usize], col_indices: &[usize]) -> Vec<u64> {
        let mut mults = vec![0u64; self.n];
        for (&r, &c) in row_indices.iter().zip(col_indices.iter()) {
            mults[r] += 1; // R(κ^i)    = Δ^r       = h(ω^r)
            mults[c] += 1; // C(κ^i)    = Δ^c       = h(ω^c)
            mults[c - r - 1] += 1; // C/(Δ·R)   = Δ^{c−r−1} = h(ω^{c−r−1})
            mults[c - self.t] += 1; // C/Δ^t     = Δ^{c−t}   = h(ω^{c−t})
        }
        mults
    }

    /// Return Δ
    pub fn delta(&self) -> Fr {
        self.d_domain.element(1)
    }
}

// ---------------------------------------------------------------------------
// Internal prover state (not sent to verifier)
// ---------------------------------------------------------------------------

/// Prover state after Round 1.
#[allow(dead_code)]
struct Round1State {
    /// Labeled polynomials [R, C, m, S, row̃].
    ///
    /// | Index | Label       | Definition              |
    /// |-------|-------------|-------------------------|
    /// | 0     | R           | R(κ^i) = Δ^{r_i}        |
    /// | 1     | C           | C(κ^i) = Δ^{c_i}        |
    /// | 2     | m           | m(ω^j) = m_j            |
    /// | 3     | S           | S(X) = 0                |
    /// | 4     | row         | row̃(κ^i) = ω^{r_i}      |
    /// | 5     | col         | row̃(κ^i) = ω^{c_i}      |
    /// | 6     | rowcol      | row̃(κ^i) = ω^{r_i*c_i}  |
    polynomials: [LabeledPolynomial<Fr, DensePolynomial<Fr>>; 7],
    /// Evaluation vector: r_evals[i] = R(κ^i) = Δ^{r_i}
    r_evals: Vec<Fr>,
    /// Evaluation vector: c_evals[i] = C(κ^i) = Δ^{c_i}
    c_evals: Vec<Fr>,
    /// Evaluation vector: m_evals[j] = m_j  (same as m(κ^j) when K = H)
    m_evals: Vec<Fr>,
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

/// Prover state after Round 2.
#[allow(dead_code)]
struct Round2State {
    /// Labeled polynomials [F₁, …, F₅]; polynomials accessible via `.polynomial()`.
    polynomials: [LabeledPolynomial<Fr, DensePolynomial<Fr>>; 5],
    f_evals: [Vec<Fr>; 5],
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

/// Prover state after Round 3.
#[allow(dead_code)]
struct Round3State {
    /// Labeled polynomials [F₁, …, F₅]; polynomials accessible via `.polynomial()`.
    polynomials: Vec<LabeledPolynomial<Fr, DensePolynomial<Fr>>>,
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

// ---------------------------------------------------------------------------
// Proof type (commitments only — opening proofs are TODO)
// ---------------------------------------------------------------------------

/// PFR proof produced by the prover.
///
/// Contains only polynomial commitments.  Opening proofs (rounds 4–5) are
/// TODO stubs to be completed in later rounds.
///
/// ### Round 1 — `[R(τ), C(τ), m(τ), S(τ), row̃(τ)]₁`
///   - `r_comm`:        R(X), square-root encoding of row indices
///   - `c_comm`:        C(X), square-root encoding of col indices
///   - `m_comm`:        m(X), multiplicity polynomial (satisfies eq. 7)
///   - `s_comm`:        S(X) = 0, sumcheck blinding (zero in no-ZK mode)
///   - `rowtilde_comm`: row̃(X), auxiliary polynomial; row̃(κ^i) = ω^{r_i}
///
/// ### Round 2 — `[F₁(τ), …, F₅(τ)]₁`
///   - `f_comms[j]`: F_{j+1}(X), the j-th rational-sum polynomial (eq. 8)
///
/// ### Round 3 — `[R*(τ), q(τ)]₁`
///   - `r_star_comm`: R*(X), degree-check polynomial
///   - `q_comm`:      q(X), quotient of the batched identity P(X)
#[allow(dead_code)]
pub struct PfrProof {
    // Round 1
    pub r_comm: Comm,
    pub c_comm: Comm,
    pub m_comm: Comm,
    pub s_comm: Comm,
    pub rowtilde_comm: Comm,
    // Round 2
    pub f_comms: Vec<Comm>, // [F₁(τ), F₂(τ), F₃(τ), F₄(τ), F₅(τ)]
    // Round 3
    pub r_star_comm: Comm,
    pub q_comm: Comm,
    // TODO: Round 4 — field elements h_α, R_α, C_α, row̃_α
    // TODO: Round 5 — [Q(τ)]₁
}

// ---------------------------------------------------------------------------
// Round 1
// ---------------------------------------------------------------------------

/// **Round 1**: build R(X), C(X), m(X), S(X), row̃(X).
///
/// | Paper variable | Code variable   | Definition                  |
/// |----------------|-----------------|-----------------------------|
/// | R(X)           | `r_poly`        | R(κ^i) = Δ^{r_i}            |
/// | C(X)           | `c_poly`        | C(κ^i) = Δ^{c_i}            |
/// | m(X)           | `m_poly`        | m(ω^j) = m_j (multiplicity) |
/// | S(X)=0         | `s_poly`        | blinding; zero in no-ZK     |
/// | row̃(X)         | `rowtilde_poly` | row̃(κ^i) = ω^{r_i}          |
fn round_one(pk: &PfrPublicKey, row_indices: &[usize], col_indices: &[usize]) -> Round1State {
    // R(X): R(κ^i) = Δ^{r_i}
    let r_evals: Vec<Fr> = row_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let r_poly =
        EvaluationsOnDomain::from_vec_and_domain(r_evals.clone(), pk.k_domain).interpolate();

    // C(X): C(κ^i) = Δ^{c_i}
    let c_evals: Vec<Fr> = col_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let c_poly =
        EvaluationsOnDomain::from_vec_and_domain(c_evals.clone(), pk.k_domain).interpolate();

    // m(X): m(ω^j) = m_j  (multiplicity polynomial over H)
    let mults = pk.compute_multiplicities(row_indices, col_indices);
    let m_evals: Vec<Fr> = mults.iter().map(|&v| Fr::from(v)).collect();
    let m_poly =
        EvaluationsOnDomain::from_vec_and_domain(m_evals.clone(), pk.h_domain).interpolate();

    // S(X) = 0  (no ZK: zero blinding polynomial)
    let s_poly = DensePolynomial::from_coefficients_vec(vec![]);

    // row(X): row(κ^i) = = ω^{r_i}  (note: ω-powers, not Δ-powers)
    let row_evals: Vec<Fr> = row_indices
        .iter()
        .map(|&j| pk.h_domain.element(j))
        .collect();
    let row_poly =
        EvaluationsOnDomain::from_vec_and_domain(row_evals.clone(), pk.k_domain).interpolate();

    // col(X): col(κ^i) = = ω^{c_i}  (note: ω-powers, not Δ-powers)
    let col_evals: Vec<Fr> = col_indices
        .iter()
        .map(|&j| pk.h_domain.element(j))
        .collect();
    let col_poly =
        EvaluationsOnDomain::from_vec_and_domain(col_evals.clone(), pk.k_domain).interpolate();

    // rowcol(X): rowcol(κ^i) = = ω^{r_i*c_i}  (note: ω-powers, not Δ-powers)
    let rowcol_evals: Vec<Fr> = row_indices
        .iter()
        .zip(col_indices.iter())
        .map(|(&j, &k)| pk.h_domain.element(j) * pk.h_domain.element(k))
        .collect();
    let rowcol_poly =
        EvaluationsOnDomain::from_vec_and_domain(rowcol_evals, pk.k_domain).interpolate();

    Round1State {
        polynomials: [
            LabeledPolynomial::new("R".into(), r_poly, None, None),
            LabeledPolynomial::new("C".into(), c_poly, None, None),
            LabeledPolynomial::new("m".into(), m_poly, None, None),
            LabeledPolynomial::new("S".into(), s_poly, None, None),
            LabeledPolynomial::new("row".into(), row_poly, None, None),
            LabeledPolynomial::new("col".into(), col_poly, None, None),
            LabeledPolynomial::new("rowcol".into(), rowcol_poly, None, None),
        ],
        r_evals,
        c_evals,
        m_evals,
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
/// | Δ              | `delta`       | `d_domain.element(1)`               |
/// | z_{K∖H}        | `zkh_at_ki`   | = 1 when K = H (our toy example)    |
fn round_two(pk: &PfrPublicKey, round1: &Round1State, beta: Fr) -> Round2State {
    // Δ = generator of D
    let delta: Fr = pk.delta();
    let delta_t: Fr = delta.pow([pk.t as u64]);

    // Reuse evaluation vectors stored in Round1State — no re-evaluation needed.
    // r_at_ki[i] = R(κ^i) = Δ^{r_i},  c_at_ki[i] = C(κ^i) = Δ^{c_i}
    let r_at_ki = &round1.r_evals;
    let c_at_ki = &round1.c_evals;
    // m_at_ki[j] = m_j  (equals m(κ^j) when K = H)
    let m_at_ki = &round1.m_evals;

    // h(κ^i) = h(ω^i) = Δ^i  (since K = H and h(ω^j) = Δ^j by definition).
    // Read directly from d_domain — no polynomial evaluation needed.
    let h_at_ki: Vec<Fr> = (0..pk.m).map(|i| pk.d_domain.element(i)).collect();

    // z_{K\H}(κ^i): vanishing polynomial ratio (n/m)·(X^m−1)/(X^n−1) at κ^i.
    // When K = H (n = m), simplifies to the constant 1.
    let zkh_at_ki: Vec<Fr> = if pk.n == pk.m {
        vec![Fr::from(1u64); pk.m]
    } else {
        // TODO: implement for n ≠ m
        todo!("z_{{K\\H}} for n ≠ m is not yet implemented")
    };

    // F₁(κ^i) = 1 / (β + R(κ^i))
    let f1_evals: Vec<Fr> = r_at_ki
        .iter()
        .map(|&r| (beta + r).inverse().unwrap())
        .collect();

    // F₂(κ^i) = 1 / (β + C(κ^i))
    let f2_evals: Vec<Fr> = c_at_ki
        .iter()
        .map(|&c| (beta + c).inverse().unwrap())
        .collect();

    // F₃(κ^i) = 1 / (β + C(κ^i) / (Δ · R(κ^i)))
    let f3_evals: Vec<Fr> = r_at_ki
        .iter()
        .zip(c_at_ki.iter())
        .map(|(&r, &c)| {
            let c_over_delta_r = c * (delta * r).inverse().unwrap();
            (beta + c_over_delta_r).inverse().unwrap()
        })
        .collect();

    // F₄(κ^i) = 1 / (β + C(κ^i) / Δ^t)
    let delta_t_inv = delta_t.inverse().unwrap();
    let f4_evals: Vec<Fr> = c_at_ki
        .iter()
        .map(|&c| (beta + c * delta_t_inv).inverse().unwrap())
        .collect();

    // F₅(κ^i) = −m(κ^i) · z_{K∖H}(κ^i) / (β + h(κ^i))
    let f5_evals: Vec<Fr> = m_at_ki
        .iter()
        .zip(h_at_ki.iter())
        .zip(zkh_at_ki.iter())
        .map(|((&m_val, &h_val), &zkh)| {
            let denom_inv = (beta + h_val).inverse().unwrap();
            -m_val * zkh * denom_inv
        })
        .collect();

    // Interpolate each F_j over K
    let f1_poly =
        EvaluationsOnDomain::from_vec_and_domain(f1_evals.clone(), pk.k_domain).interpolate();
    let f2_poly =
        EvaluationsOnDomain::from_vec_and_domain(f2_evals.clone(), pk.k_domain).interpolate();
    let f3_poly =
        EvaluationsOnDomain::from_vec_and_domain(f3_evals.clone(), pk.k_domain).interpolate();
    let f4_poly =
        EvaluationsOnDomain::from_vec_and_domain(f4_evals.clone(), pk.k_domain).interpolate();
    let f5_poly =
        EvaluationsOnDomain::from_vec_and_domain(f5_evals.clone(), pk.k_domain).interpolate();

    Round2State {
        polynomials: [
            LabeledPolynomial::new("F1".into(), f1_poly, None, None),
            LabeledPolynomial::new("F2".into(), f2_poly, None, None),
            LabeledPolynomial::new("F3".into(), f3_poly, None, None),
            LabeledPolynomial::new("F4".into(), f4_poly, None, None),
            LabeledPolynomial::new("F5".into(), f5_poly, None, None),
        ],
        f_evals: [f1_evals, f2_evals, f3_evals, f4_evals, f5_evals],
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
fn round_three(
    pk: &PfrPublicKey,
    round1_state: &Round1State,
    round2_state: &Round2State,
    beta: Fr,
    eta: Fr,
) -> Round3State {
    let delta = pk.delta();
    let delta_t = delta.pow([pk.t as u64]);

    // z_{K\H}(X) = 1 when K = H (n = m).
    let zkh: DensePolynomial<Fr> = if pk.n == pk.m {
        DensePolynomial::from_coefficients_vec(vec![Fr::one()])
    } else {
        todo!("z_{{K\\H}} for n ≠ m is not yet implemented")
    };

    // Round-1 polynomials: [R, C, m, S, row, col, rowcol].
    let r_poly      = round1_state.polynomials[0].polynomial();
    let c_poly      = round1_state.polynomials[1].polynomial();
    let m_poly      = round1_state.polynomials[2].polynomial();
    let row_poly    = round1_state.polynomials[4].polynomial(); // row̃ = row (no ZK)
    let col_poly    = round1_state.polynomials[5].polynomial();
    let rowcol_poly = round1_state.polynomials[6].polynomial();
    // Round-2 polynomials: [F1, F2, F3, F4, F5].
    let f1_poly = round2_state.polynomials[0].polynomial();
    let f2_poly = round2_state.polynomials[1].polynomial();
    let f3_poly = round2_state.polynomials[2].polynomial();
    let f4_poly = round2_state.polynomials[3].polynomial();
    let f5_poly = round2_state.polynomials[4].polynomial();

    // Constant polynomials used in the terms below.
    let beta_poly    = DensePolynomial::from_coefficients_vec(vec![beta]);
    let one_poly     = DensePolynomial::from_coefficients_vec(vec![Fr::one()]);
    let delta_poly   = DensePolynomial::from_coefficients_vec(vec![delta]);
    let delta_t_poly = DensePolynomial::from_coefficients_vec(vec![delta_t]);
    let z_k: DensePolynomial<Fr> = pk.k_domain.vanishing_polynomial().into();
    // U(X) = X³ − 1  (fixed; see eq. 9 and surrounding text in Appendix B)
    let u_poly = DensePolynomial::from_coefficients_vec(
        vec![-Fr::one(), Fr::zero(), Fr::zero(), Fr::one()]);

    // -----------------------------------------------------------------------
    // Compute R*(X) = R_F(X) · U(X) from eq. (9):
    //   ∑Fⱼ(X) = q_F(X)·z_K(X) + R_F(X)·X,  deg R_F ≤ m−2.
    // So: R_F = (∑Fⱼ − q_F·z_K) / X = (remainder of ∑Fⱼ / z_K) / X.
    // The sumcheck identity ∑Fⱼ(κ^i) = 0 guarantees r_f[0] = 0.
    // -----------------------------------------------------------------------
    let f_sum_poly = &(&(f1_poly + f2_poly) + &(f3_poly + f4_poly)) + f5_poly;
    let (_q_f, r_f) = f_sum_poly.divide_by_vanishing_poly(pk.k_domain).unwrap();
    debug_assert!(
        r_f.coeffs.get(0).map(|c| *c == Fr::zero()).unwrap_or(true),
        "r_f constant term is nonzero — sumcheck failed"
    );
    // R_F = r_f / X (drop the zero constant coefficient)
    let r_f_over_x = if r_f.is_zero() {
        DensePolynomial::zero()
    } else {
        DensePolynomial::from_coefficients_slice(&r_f.coeffs[1..])
    };
    let r_star = &r_f_over_x * &u_poly;

    // -----------------------------------------------------------------------
    // Build P(X) term by term (eq. 10, Appendix B).
    // Helper: scale a polynomial by a field scalar.
    // -----------------------------------------------------------------------
    let scale = |s: Fr, q: &DensePolynomial<Fr>| -> DensePolynomial<Fr> {
        let mut out = DensePolynomial::zero();
        out += (s, q);
        out
    };

    // TODO: build big_sum using pointwise evaluation arithmetic on a domain of size ≥ 2*(2m+1)
    // (to avoid aliasing), then a single IFFT, instead of repeated polynomial multiplications.
    // Divide the result by z_K first (divide_by_vanishing_poly) then by U (degree 3).
    // Build the "big sum" S(X) = ∑ ηʲ·termⱼ, then P(X) = S(X)·U(X) − η⁹·X·R*(X).
    // This follows eq. (10): P = (∑Fⱼ-identities + η⁹·∑Fⱼ + η¹⁰·S)·U − η⁹·X·R*.
    // Each inner term vanishes on K; multiplying by U gives divisibility by z_K·U.

    // η⁰: F₁(β + R) − 1
    let mut big_sum = &(f1_poly * &(&beta_poly + r_poly)) - &one_poly;

    let mut eta_pow = eta; // η¹

    // η¹: F₂(β + C) − 1
    let term = &(f2_poly * &(&beta_poly + c_poly)) - &one_poly;
    big_sum += &scale(eta_pow, &term);

    // η²: F₃(β·Δ·R + C) − Δ·R  (cleared-denominator form; vanishes on K)
    eta_pow *= eta;
    let delta_r_poly = &delta_poly * r_poly;
    let term = &(f3_poly * &(&(&beta_poly * &delta_r_poly) + c_poly)) - &delta_r_poly;
    big_sum += &scale(eta_pow, &term);

    // η³: F₄(β·Δᵗ + C) − Δᵗ  (cleared-denominator form; vanishes on K)
    eta_pow *= eta;
    let term = &(f4_poly * &(&(&beta_poly * &delta_t_poly) + c_poly)) - &delta_t_poly;
    big_sum += &scale(eta_pow, &term);

    // η⁴: F₅(β + h) + m·z_{K\H}  (vanishes on K: F₅ = −m·z_{K\H}/(β+h))
    eta_pow *= eta;
    let term = &(f5_poly * &(&beta_poly + &pk.h_poly)) + &(m_poly * &zkh);
    big_sum += &scale(eta_pow, &term);

    // η⁵: R² − row
    eta_pow *= eta;
    let term = &(r_poly * r_poly) - row_poly;
    big_sum += &scale(eta_pow, &term);

    // η⁶: C² − col
    eta_pow *= eta;
    let term = &(c_poly * c_poly) - col_poly;
    big_sum += &scale(eta_pow, &term);

    // η⁷: rowcol − row̃·col  (row̃ = row in no-ZK mode)
    eta_pow *= eta;
    let term = rowcol_poly - &(row_poly * col_poly);
    big_sum += &scale(eta_pow, &term);

    // η⁸: row̃ − row = 0  (no ZK)
    eta_pow *= eta;
    // big_sum += 0

    // η⁹: ∑Fⱼ  (S(X)=0 in this toy, so no η¹⁰ term)
    eta_pow *= eta;
    big_sum += &scale(eta_pow, &f_sum_poly);

    // P(X) = big_sum · U(X) − η⁹ · X · R*(X)
    let x_poly = DensePolynomial::from_coefficients_vec(vec![Fr::zero(), Fr::one()]);
    let p = &(&big_sum * &u_poly) - &scale(eta_pow, &(&x_poly * &r_star));

    // -----------------------------------------------------------------------
    // q(X) = P(X) / (z_K(X) · U(X))
    // -----------------------------------------------------------------------
    let divisor = &z_k * &u_poly;

    use ark_poly::univariate::DenseOrSparsePolynomial;
    let (q_poly, rem) =
        DenseOrSparsePolynomial::from(p)
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(divisor))
            .unwrap();
    debug_assert!(
        rem.coeffs.iter().all(|c| *c == Fr::zero()),
        "P is not divisible by z_K · U: remainder has {} nonzero coeffs (deg {})",
        rem.coeffs.iter().filter(|c| **c != Fr::zero()).count(),
        rem.degree()
    );

    Round3State {
        polynomials: vec![
            LabeledPolynomial::new("r_star".into(), r_star, None, None),
            LabeledPolynomial::new("q".into(), q_poly, None, None),
        ],
        rands: Vec::new(),
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
///   # TODO: Round 3
///
/// **Round 4** — After challenge α, prover opens four polynomials at α:
///   Sends h_α = h(α), R_α = R(α), C_α = C(α), row̃_α = row̃(α)
///   # TODO: Round 4
///
/// **Round 5** — After challenge δ, prover sends a batched KZG opening:
///   → Sends [Q(τ)]₁
///   # TODO: Round 5
///
pub fn prove(pk: &PfrPublicKey, row_indices: &[usize], col_indices: &[usize]) -> PfrProof {
    // Initialise the Fiat-Shamir transcript with the public-key commitment.
    let mut fs_rng = FS::initialize(&to_bytes![pk.h_commitment.commitment()].unwrap());

    // --- Round 1 ---
    let mut round1_state = round_one(pk, row_indices, col_indices);

    let first_round_comm_time = start_timer!(|| "Committing to Round 1 polynomials");
    let (round1_comms, round1_rands) =
        PC::commit(&pk.ck, round1_state.polynomials.iter(), None).unwrap();
    end_timer!(first_round_comm_time);
    round1_state.rands = round1_rands;

    let mut round1_comms = round1_comms;
    let r_comm = round1_comms.remove(0);
    let c_comm = round1_comms.remove(0);
    let m_comm = round1_comms.remove(0);
    let s_comm = round1_comms.remove(0);
    let rowtilde_comm = round1_comms.remove(0);

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
    let beta = Fr::rand(&mut fs_rng);

    // --- Round 2 ---
    let mut round2_state = round_two(pk, &round1_state, beta);

    let second_round_comm_time = start_timer!(|| "Committing to Round 2 polynomials");
    let (f_comms, round2_rands) =
        PC::commit(&pk.ck, round2_state.polynomials.iter(), None).unwrap();
    end_timer!(second_round_comm_time);
    round2_state.rands = round2_rands;

    // Derive η by absorbing Round 2 commitments (used in Round 3, TODO).
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
    let eta = Fr::rand(&mut fs_rng);

    // --- Round 3 ---
    let mut round3_state = round_three(pk, &round1_state, &round2_state, beta, eta);

    let third_round_comm_time = start_timer!(|| "Committing to Round 3 polynomials");
    let (round3_comms, round3_rands) =
        PC::commit(&pk.ck, round3_state.polynomials.iter(), None).unwrap();
    end_timer!(third_round_comm_time);
    round3_state.rands = round3_rands;

    let mut round3_comms = round3_comms;
    let r_star_comm = round3_comms.remove(0);
    let q_comm = round3_comms.remove(0);

    // TODO: Round 4 — receive α; evaluate h(α), R(α), C(α), row̃(α); send values.
    // TODO: Round 5 — receive δ; compute Lin(X) and Q(X) from eq. (12); send [Q(τ)]₁.

    PfrProof {
        r_comm,
        c_comm,
        m_comm,
        s_comm,
        rowtilde_comm,
        f_comms,
        r_star_comm,
        q_comm,
    }
}

// ---------------------------------------------------------------------------
// Verifier (stub)
// ---------------------------------------------------------------------------

/// Verify the PFR proof.
///
/// Currently a stub — full verification will be implemented alongside rounds 3–5.
pub fn verify(_pk: &PfrPublicKey, _proof: &PfrProof) -> bool {
    // TODO: implement full verification after all 5 rounds are complete.
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;
    use ark_poly::Polynomial;

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

    fn setup() -> PfrPublicKey {
        PfrPublicKey::setup(N, M, T, &mut ark_std::test_rng())
    }

    // --- Round 1 ---

    /// R(κ^i) = Δ^{r_i} for all i
    #[test]
    fn round1_r_poly() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL);
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
        let s = round_one(&pk, &ROW, &COL);
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
        let s = round_one(&pk, &ROW, &COL);
        let poly = s.polynomials[2].polynomial();
        for j in 0..N {
            assert_eq!(
                poly.evaluate(&pk.h_domain.element(j)),
                Fr::from(MULTS[j]),
                "m(ω^{j}) ≠ {}",
                MULTS[j]
            );
        }
    }

    /// S(X) = 0 (no-ZK blinding)
    #[test]
    fn round1_s_poly_is_zero() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL);
        let poly = s.polynomials[3].polynomial();
        for i in 0..M {
            assert_eq!(
                poly.evaluate(&pk.k_domain.element(i)),
                Fr::zero(),
                "S(κ^{i}) ≠ 0"
            );
        }
    }

    /// row̃(κ^i) = ω^{r_i} for all i
    #[test]
    fn round1_rowtilde_poly() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL);
        let poly = s.polynomials[4].polynomial();
        for (i, &r) in ROW.iter().enumerate() {
            assert_eq!(
                poly.evaluate(&pk.k_domain.element(i)),
                pk.h_domain.element(r),
                "row̃(κ^{i}) ≠ ω^{r}"
            );
        }
    }

    /// Cached r_evals / c_evals / m_evals match polynomial evaluations
    #[test]
    fn round1_cached_evals_consistent() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL);
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
        let r1 = round_one(&pk, &ROW, &COL);
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta);
        let delta = pk.d_domain.element(1);
        let delta_t_inv = delta.pow([T as u64]).inverse().unwrap();

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
                (beta + c * (delta * r).inverse().unwrap())
                    .inverse()
                    .unwrap(),
                "F₃(κ^{i})"
            );
            assert_eq!(f4, (beta + c * delta_t_inv).inverse().unwrap(), "F₄(κ^{i})");
            assert_eq!(f5, -(m * (beta + h).inverse().unwrap()), "F₅(κ^{i})");
        }
    }

    /// ∑_{i=0}^{m-1} (F₁+F₂+F₃+F₄+F₅)(κ^i) = 0  (the rational-sumcheck identity, eq. 7)
    #[test]
    fn round2_sumcheck() {
        let pk = setup();
        let r1 = round_one(&pk, &ROW, &COL);
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta);

        let sum: Fr = (0..M)
            .map(|i| {
                let ki = pk.k_domain.element(i);
                r2.polynomials
                    .iter()
                    .map(|p| p.polynomial().evaluate(&ki))
                    .sum::<Fr>()
            })
            .sum();

        assert_eq!(sum, Fr::zero(), "∑ Fⱼ(κ^i) ≠ 0");
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
        let r1 = round_one(&pk, &ROW, &COL);
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta);

        let delta = pk.delta();
        let delta_t_inv = delta.pow([T as u64]).inverse().unwrap();
        // z_{K\H} = 1 when K = H
        let zkh = Fr::one();

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
                f1 * (beta + r) - Fr::one(),
                Fr::zero(),
                "η⁰ identity failed at κ^{i}"
            );
            assert_eq!(
                f2 * (beta + c) - Fr::one(),
                Fr::zero(),
                "η¹ identity failed at κ^{i}"
            );
            let c_over_delta_r = c * (delta * r).inverse().unwrap();
            assert_eq!(
                f3 * (beta + c_over_delta_r) - Fr::one(),
                Fr::zero(),
                "η² identity failed at κ^{i}"
            );
            assert_eq!(
                f4 * (beta + c * delta_t_inv) - Fr::one(),
                Fr::zero(),
                "η³ identity failed at κ^{i}"
            );
            assert_eq!(
                f5 * (beta + h) + m * zkh,
                Fr::zero(),
                "η⁴ identity failed at κ^{i}"
            );
        }
    }

    /// q(X) is a polynomial: P(X) is exactly divisible by z_K(X) · U(X).
    ///
    /// Equivalently, P(κ^i) = 0 for all κ^i ∈ K when the identities hold.
    /// We verify this by evaluating the full batched P at each κ^i and
    /// checking the result is zero for an arbitrary challenge η.
    #[test]
    fn round3_p_divisible_by_zk_u() {
        let pk = setup();
        let r1 = round_one(&pk, &ROW, &COL);
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta);
        let eta = Fr::from(17u64);
        let r3 = round_three(&pk, &r1, &r2, beta, eta);

        let q_poly = r3.polynomials[1].polynomial();

        // Recover P(X) = q(X) · z_K(X) · U(X) and check P(κ^i) = 0 for all κ^i ∈ K.
        // z_K(X) = X^m − 1,  U(X) = X^{m−1}(1 − X).
        // Since z_K(κ^i) = 0 by definition, the product is always 0 on K — but
        // we compute q(κ^i) · z_K(κ^i) · U(κ^i) explicitly so that any error in
        // q (e.g. a non-zero remainder from the division) would surface as a
        // non-zero intermediate before z_K cancels.  We therefore evaluate the
        // full product as a polynomial identity, not pointwise.
        let z_k: DensePolynomial<Fr> = pk.k_domain.vanishing_polynomial().into();
        // U(X) = X^{m−1} − X^m  (i.e. X^{m−1}(1 − X))
        let mut u_coeffs = vec![Fr::zero(); M + 1];
        u_coeffs[M - 1] = Fr::one();
        u_coeffs[M] = -Fr::one();
        let u_poly = DensePolynomial::from_coefficients_vec(u_coeffs);

        let p_recovered = q_poly * &(&z_k * &u_poly);

        for i in 0..M {
            let ki = pk.k_domain.element(i);
            assert_eq!(
                p_recovered.evaluate(&ki),
                Fr::zero(),
                "(q · z_K · U)(κ^{i}) ≠ 0"
            );
        }
    }
}

