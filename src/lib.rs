//! Proof of Function Relation (PFR) — Appendix B, IMPR-FHFC paper.
//!
//! ## Notation (Appendix B)
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
    /// Coset domain for round 3: a coset of K of size 2m, disjoint from K.
    /// Used to evaluate polynomials without hitting zeros of z_K.
    pub coset_domain: GeneralEvaluationDomain<Fr>,
    /// Table polynomial: h(ω^j) = Δ^j for j = 0, …, n−1.
    pub h_poly: DensePolynomial<Fr>,
    /// h(X) evaluated over coset_domain, precomputed at setup time.
    pub h_coset_evals: Vec<Fr>,
    /// z_{K\H}(X) evaluated over coset_domain, precomputed at setup time.
    pub zkh_coset_evals: Vec<Fr>,
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

        // Polynomials committed across the 5 rounds and their degrees (with ZK blinding):
        //   Round 1: R, C  deg m+1  (interp deg m−1, + ρ·z_K deg m+1)
        //            m     deg n−1,  S deg m+1,  row/col/rowcol deg m−1
        //   Round 2: F₁–F₅  deg m+1  (+ ρ·z_K blinding, added in later step)
        //   Round 3: R*(X) = (R_F + η·R_S)·U,  deg R* ≤ m+1
        //            q(X) = P/(z_K·U):
        //              P involves F_j(X)·R(X)·C(X) products; with blinded R,C of deg m+1
        //              and blinded F_j of deg m+1, worst case deg P = 3(m+1)+3 = 3m+6,
        //              so deg q ≤ (3m+6) − (m+3) = 2m+3.
        //   Round 5: Q(X), deg Q ≤ max committed deg − 1 = 2m+2
        // P(X) itself is never committed (only q = P/(z_K·U) is).
        // Maximum degree to support: max(n−1, 2m+3).
        let max_degree = (n - 1).max(2 * m + 3);
        let pp = PC::setup(max_degree, None, rng).unwrap();
        let (ck, vk) = PC::trim(&pp, max_degree, 1, None).unwrap();

        let h_labeled = LabeledPolynomial::new("h".into(), h_poly.clone(), None, None);
        let (mut comms, mut rands) = PC::commit(&ck, vec![&h_labeled], None).unwrap();

        // Coset domain of size 2m for round 3: uses ark-poly's canonical coset
        // (multiplied internally by a fixed generator g), which is disjoint from K.
        let coset_domain =
            GeneralEvaluationDomain::<Fr>::new(2 * m).expect("coset domain must exist");

        // Precompute h over the coset using coset_fft (evaluates coeffs over g·K).
        let h_coset_evals = coset_domain.coset_fft(&h_poly.coeffs);

        // z_{K\H}(X) = (n/m)·(X^{m−n} + X^{m−2n} + … + 1), coeffs at multiples of n.
        let steps = m / n;
        let scale = Fr::from(n as u64) * Fr::from(m as u64).inverse().unwrap();
        let mut zkh_coeffs = vec![Fr::zero(); (steps - 1) * n + 1];
        for k in 0..steps {
            zkh_coeffs[k * n] = scale;
        }
        let zkh_poly = DensePolynomial::from_coefficients_vec(zkh_coeffs);
        let zkh_coset_evals = coset_domain.coset_fft(&zkh_poly.coeffs);

        Self {
            n,
            m,
            t,
            h_domain,
            d_domain,
            k_domain,
            coset_domain,
            h_poly,
            h_coset_evals,
            zkh_coset_evals,
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
    pub fn big_delta(&self) -> Fr {
        self.d_domain.element(1)
    }
}

// ---------------------------------------------------------------------------
// Internal prover state (not sent to verifier)
// ---------------------------------------------------------------------------

/// Prover state after Round 1.
#[allow(dead_code)]
pub(crate) struct Round1State {
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
    /// | 6     | rowcol      | rowcol(κ^i) = ω^{r_i·c_i} (statement)   |
    /// | 7     | rowtilde    | row̃(κ^i) = ω^{r_i} + ρ_row·z_K (blinded)|
    pub(crate) polynomials: [LabeledPolynomial<Fr, DensePolynomial<Fr>>; 8],
    /// Evaluation vector: r_evals[i] = R(κ^i) = Δ^{r_i}
    r_evals: Vec<Fr>,
    /// Evaluation vector: c_evals[i] = C(κ^i) = Δ^{c_i}
    c_evals: Vec<Fr>,
    /// Evaluation vector: m_evals[j] = m_j  (same as m(κ^j) when K = H)
    m_evals: Vec<Fr>,
    /// R_S: the leading coefficient of S(X) = R_S·X + ρ_S·z_K(X)
    r_s: Fr,
    /// ρ_S: the blinding scalar of S(X)
    rho_s: Fr,
    /// Evaluations of all 8 polynomials over pk.coset_domain (canonical coset of size 2m).
    /// Index order matches `polynomials`: [R, C, m, S, row, col, rowcol, rowtilde].
    coset_evals: [Vec<Fr>; 8],
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

/// Prover state after Round 2.
#[allow(dead_code)]
pub(crate) struct Round2State {
    /// Labeled polynomials [F₁, …, F₅]; polynomials accessible via `.polynomial()`.
    pub(crate) polynomials: [LabeledPolynomial<Fr, DensePolynomial<Fr>>; 5],
    f_evals: [Vec<Fr>; 5],
    /// Evaluations of F1..F5 over pk.coset_domain.
    f_coset_evals: [Vec<Fr>; 5],
    /// β: verifier challenge that triggered Round 2.
    beta: Fr,
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

/// Prover state after Round 3.
#[allow(dead_code)]
pub(crate) struct Round3State {
    /// Labeled polynomials [R*, q]; accessible via `.polynomial()`.
    ///
    /// | Index | Label  | Definition                      |
    /// |-------|--------|---------------------------------|
    /// | 0     | r_star | R*(X) = R_F(X) · U(X)           |
    /// | 1     | q      | q(X) = P(X) / (z_K(X) · U(X))  |
    pub(crate) polynomials: Vec<LabeledPolynomial<Fr, DensePolynomial<Fr>>>,
    /// U(X) = X³ − 1, stored to avoid recomputing in round_five.
    u_poly: DensePolynomial<Fr>,
    /// η: verifier challenge that triggered Round 3.
    eta: Fr,
    /// η⁹ (the last power of η used in P), stored for use in Lin(X).
    eta9: Fr,
    /// Commitment randomness (filled in by prove() after PC::commit)
    rands: Vec<Rand>,
}

/// Prover state after Round 4.
#[allow(dead_code)]
pub(crate) struct Round4State {
    /// α: verifier challenge that triggered Round 4.
    alpha: Fr,
    /// h(α)
    h_alpha: Fr,
    /// R(α)
    r_alpha: Fr,
    /// C(α)
    c_alpha: Fr,
    /// row̃(α)  (= row(α) in no-ZK mode)
    row_alpha: Fr,
}

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

/// PFR proof produced by the prover.
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
///
/// ### Round 4 — field elements h_α, R_α, C_α, row̃_α
///   - `h_alpha`:   h(α)
///   - `r_alpha`:   R(α)
///   - `c_alpha`:   C(α)
///   - `row_alpha`: row̃(α) — in no-ZK mode row̃ = row, so this is row(α);
///                  in the ZK version row̃(α) ≠ row(α).
///
/// ### Round 5 — `[Q(τ)]₁`
///   - `q_poly_comm`: Q(X), the batched KZG opening proof polynomial
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
    // Round 4
    pub h_alpha: Fr,
    pub r_alpha: Fr,
    pub c_alpha: Fr,
    pub row_alpha: Fr,
    // Round 5
    pub q_poly_comm: Comm,
}

/// Public inputs to the PFR verifier — commitments known to both parties
/// before the proof is generated.
///
/// In the paper these are committed as part of the relation description,
/// not as part of `π_PFR`.
#[allow(dead_code)]
pub struct PfrPublicInputs {
    /// [row(τ)]₁: commitment to the row-index polynomial row(X)
    pub row_comm: Comm,
    /// [col(τ)]₁: commitment to the column-index polynomial col(X)
    pub col_comm: Comm,
    /// [rowcol(τ)]₁: commitment to the rowcol polynomial rowcol(X)
    pub rowcol_comm: Comm,
}

// ---------------------------------------------------------------------------
// Blinding helpers
// ---------------------------------------------------------------------------

/// Interpolate `evals` over `domain` and blind with ρ(X)·z(X),
/// where ρ is a random polynomial of degree `rho_degree`.
///
/// - `rho_degree = 1`: used for R, C, F₁–F₅, row̃  (blinding polynomial ∈ F≤1[X])
/// - `rho_degree = 0`: used for m  (scalar blinding)
fn blind_over_domain<R: RngCore>(
    evals: Vec<Fr>,
    domain: GeneralEvaluationDomain<Fr>,
    vanishing: &DensePolynomial<Fr>,
    rho_degree: usize,
    rng: &mut R,
) -> DensePolynomial<Fr> {
    let interp = EvaluationsOnDomain::from_vec_and_domain(evals, domain).interpolate();
    let rho =
        DensePolynomial::from_coefficients_vec((0..=rho_degree).map(|_| Fr::rand(rng)).collect());
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
pub(crate) fn round_one<R: RngCore>(
    pk: &PfrPublicKey,
    row_indices: &[usize],
    col_indices: &[usize],
    rng: &mut R,
) -> Round1State {
    let z_k: DensePolynomial<Fr> = pk.k_domain.vanishing_polynomial().into();
    let z_h: DensePolynomial<Fr> = pk.h_domain.vanishing_polynomial().into();

    // R(X): R(κ^i) = Δ^{r_i}, blinded by ρ_R(X)·z_K(X) with ρ_R ← F≤1[X]
    let r_evals: Vec<Fr> = row_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let r_poly = blind_over_domain(r_evals.clone(), pk.k_domain, &z_k, 1, rng);

    // C(X): C(κ^i) = Δ^{c_i}, blinded by ρ_C(X)·z_K(X) with ρ_C ← F≤1[X]
    let c_evals: Vec<Fr> = col_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let c_poly = blind_over_domain(c_evals.clone(), pk.k_domain, &z_k, 1, rng);

    // m(X): m(ω^j) = m_j, blinded by ρ_m·z_H(X) with ρ_m ← F
    let mults = pk.compute_multiplicities(row_indices, col_indices);
    let m_evals: Vec<Fr> = mults.iter().map(|&v| Fr::from(v)).collect();
    let m_poly = blind_over_domain(m_evals.clone(), pk.h_domain, &z_h, 0, rng);

    // S(X) = R_S·X + ρ_S·z_K(X),  R_S, ρ_S ← F
    let r_s = Fr::rand(rng);
    let rho_s = Fr::rand(rng);
    let s_poly = &DensePolynomial::from_coefficients_vec(vec![Fr::zero(), r_s]) + &(&z_k * rho_s);

    // row(X): row(κ^i) = ω^{r_i}  (note: ω-powers, not Δ-powers) — statement polynomial
    let row_evals: Vec<Fr> = row_indices
        .iter()
        .map(|&j| pk.h_domain.element(j))
        .collect();
    let row_poly =
        EvaluationsOnDomain::from_vec_and_domain(row_evals.clone(), pk.k_domain).interpolate();

    // row̃(X) = row(X) + ρ_row(X)·z_K(X),  ρ_row ← F≤1[X]
    let rowtilde_poly = blind_over_domain(row_evals.clone(), pk.k_domain, &z_k, 1, rng);

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
/// | z_{K∖H}        | `zkh_at_ki`   | z_{K∖H}(κ^i)                        |
pub(crate) fn round_two<R: RngCore>(
    pk: &PfrPublicKey,
    round1: &Round1State,
    beta: Fr,
    rng: &mut R,
) -> Round2State {
    // Δ = generator of D
    let big_delta: Fr = pk.big_delta();
    let big_delta_t: Fr = big_delta.pow([pk.t as u64]);

    // Reuse evaluation vectors stored in Round1State.
    // r_at_ki[i] = R(κ^i) = Δ^{r_i},  c_at_ki[i] = C(κ^i) = Δ^{c_i}
    let r_at_ki = &round1.r_evals;
    let c_at_ki = &round1.c_evals;

    // m(κ^i): when K = H, m_evals stores m(ω^i) = m_i directly.
    // When K ≠ H, use FFT to evaluate m over K in O(m log m).
    let m_at_ki_owned: Vec<Fr>;
    let m_at_ki: &Vec<Fr> = if pk.n == pk.m {
        &round1.m_evals
    } else {
        let m_poly = round1.polynomials[2].polynomial();
        m_at_ki_owned = m_poly.evaluate_over_domain_by_ref(pk.k_domain).evals;
        &m_at_ki_owned
    };

    // h(κ^i): when K = H, h(ω^i) = Δ^i (read from d_domain directly).
    // When K ≠ H, use FFT to evaluate h over K in O(m log m).
    let h_at_ki: Vec<Fr> = if pk.n == pk.m {
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
    let scale = Fr::from(pk.n as u64) * Fr::from(pk.m as u64).inverse().unwrap();
    // κ^n has order m/n; let ζ = κ^n.
    let zeta = pk.k_domain.element(pk.n); // κ^n
    let zkh_at_ki: Vec<Fr> = (0..pk.m)
        .map(|i| {
            // Q(κ^i) = sum_{k=0}^{steps−1} (κ^{kn})^i = sum_{k=0}^{steps−1} ζ^{ki}
            let base = zeta.pow([i as u64]);
            let mut q_val = Fr::zero();
            let mut power = Fr::one();
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
    let mut delta_r: Vec<Fr> = r_at_ki.iter().map(|&r| big_delta * r).collect();
    ark_ff::batch_inversion(&mut delta_r);

    // Pass 2: fill 5m denominators and batch-invert
    let mut denoms: Vec<Fr> = Vec::with_capacity(5 * pk.m);
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

    let f3_evals: Vec<Fr> = denoms[0..pk.m].to_vec();
    let f1_evals: Vec<Fr> = denoms[pk.m..2 * pk.m].to_vec();
    let f4_evals: Vec<Fr> = denoms[2 * pk.m..3 * pk.m].to_vec();
    let f2_evals: Vec<Fr> = denoms[3 * pk.m..4 * pk.m].to_vec();
    // F₅(κ^i) = −m(κ^i) · z_{K∖H}(κ^i) · (β + h(κ^i))⁻¹
    let f5_evals: Vec<Fr> = m_at_ki
        .iter()
        .zip(zkh_at_ki.iter())
        .zip(denoms[4 * pk.m..5 * pk.m].iter())
        .map(|((&m_val, &zkh), &h_inv)| -m_val * zkh * h_inv)
        .collect();

    // Interpolate each F_j over K, then blind with ρ_j(X)·z_K(X), ρ_j ← F≤1[X]
    let z_k: DensePolynomial<Fr> = pk.k_domain.vanishing_polynomial().into();
    let f1_poly = blind_over_domain(f1_evals.clone(), pk.k_domain, &z_k, 1, rng);
    let f2_poly = blind_over_domain(f2_evals.clone(), pk.k_domain, &z_k, 1, rng);
    let f3_poly = blind_over_domain(f3_evals.clone(), pk.k_domain, &z_k, 1, rng);
    let f4_poly = blind_over_domain(f4_evals.clone(), pk.k_domain, &z_k, 1, rng);
    let f5_poly = blind_over_domain(f5_evals.clone(), pk.k_domain, &z_k, 1, rng);

    let f_polys = [
        LabeledPolynomial::new("F1".into(), f1_poly, None, None),
        LabeledPolynomial::new("F2".into(), f2_poly, None, None),
        LabeledPolynomial::new("F3".into(), f3_poly, None, None),
        LabeledPolynomial::new("F4".into(), f4_poly, None, None),
        LabeledPolynomial::new("F5".into(), f5_poly, None, None),
    ];

    // Evaluate blinded F polynomials over coset domain for use in round 3.
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
pub(crate) fn round_three(
    pk: &PfrPublicKey,
    round1_state: &Round1State,
    round2_state: &Round2State,
    eta: Fr,
) -> Round3State {
    let beta = round2_state.beta;
    let big_delta = pk.big_delta();
    let big_delta_t = big_delta.pow([pk.t as u64]);
    let z_k: DensePolynomial<Fr> = pk.k_domain.vanishing_polynomial().into();

    // U(X) = X³ − 1  (fixed; see eq. 9 and surrounding text in Appendix B)
    let u_poly =
        DensePolynomial::from_coefficients_vec(vec![-Fr::one(), Fr::zero(), Fr::zero(), Fr::one()]);

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
        r_f.coeffs.get(0).map(|c| *c == Fr::zero()).unwrap_or(true),
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

    // Evaluate z_K and fs_sum over the coset.
    let zk_c = cd.coset_fft(&z_k.coeffs);
    let fss_c = cd.coset_fft(&fs_sum_poly.coeffs);

    // Precompute η powers.
    let mut eta_pows = vec![Fr::one(); 10];
    for i in 1..10 {
        eta_pows[i] = eta_pows[i - 1] * eta;
    }

    // Build big_sum pointwise over the coset.
    let big_sum_over_zk: Vec<Fr> = (0..nc)
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
            let mut s = f1 * (beta + r) - Fr::one();
            // η¹: F₂(β + C) − 1
            s += eta_pows[1] * (f2 * (beta + c) - Fr::one());
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

    // Evaluate η⁹·r_f over the coset.
    let rf_c = cd.coset_fft(&r_f.coeffs);

    // Combine: (big_sum − η⁹·r_f) / z_K pointwise.
    let mut q_evals: Vec<Fr> = big_sum_over_zk
        .into_iter()
        .zip(rf_c.iter())
        .zip(zk_c.iter())
        .map(|((bs, &rf), &zk)| (bs - eta9 * rf) * zk.inverse().unwrap())
        .collect();

    // IFFT to get q as polynomial coefficients.
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
pub(crate) fn round_four(pk: &PfrPublicKey, round1_state: &Round1State, alpha: Fr) -> Round4State {
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
pub(crate) fn round_five(
    pk: &PfrPublicKey,
    round1_state: &Round1State,
    round2_state: &Round2State,
    round3_state: &Round3State,
    round4_state: &Round4State,
    delta: Fr,
) -> DensePolynomial<Fr> {
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

    let scale = |s: Fr, p: &DensePolynomial<Fr>| -> DensePolynomial<Fr> {
        let mut out = DensePolynomial::zero();
        out += (s, p);
        out
    };
    let const_poly = |v: Fr| DensePolynomial::from_coefficients_vec(vec![v]);

    // z_{K\H}(X) = (n/m)·(X^m−1)/(X^n−1) as a polynomial (requires n | m).
    // Factor: X^m−1 = (X^n−1)·(X^{m−n}+X^{m−2n}+…+1), so z_{K\H}(X) = (n/m)·Q(X)
    // where Q has coeff (n/m) at degrees 0, n, 2n, …, m−n.
    let zkh: DensePolynomial<Fr> = {
        debug_assert_eq!(pk.m % pk.n, 0, "m must be a multiple of n");
        let steps = pk.m / pk.n;
        let scale_factor = Fr::from(pk.n as u64) * Fr::from(pk.m as u64).inverse().unwrap();
        let mut coeffs = vec![Fr::zero(); (steps - 1) * pk.n + 1];
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
    let mut big_lin = &scale(beta + r_alpha, f1_poly) - &const_poly(Fr::one());
    let mut eta_pow = eta;

    // η¹: F₂(X)(β + C_α) − 1
    big_lin += &scale(
        eta_pow,
        &(&scale(beta + c_alpha, f2_poly) - &const_poly(Fr::one())),
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
    let mut delta_pow = Fr::one();
    let mut numerator = &pk.h_poly - &const_poly(h_alpha);

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(r_poly - &const_poly(r_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(c_poly - &const_poly(c_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &(rowtilde_poly - &const_poly(row_alpha)));

    delta_pow *= delta;
    numerator += &scale(delta_pow, &lin_poly);

    let x_minus_alpha = DensePolynomial::from_coefficients_vec(vec![-alpha, Fr::one()]);
    use ark_poly::univariate::DenseOrSparsePolynomial;
    let (q_open_poly, rem) = DenseOrSparsePolynomial::from(numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(x_minus_alpha))
        .unwrap();
    debug_assert!(
        rem.coeffs.iter().all(|c| *c == Fr::zero()),
        "Round-5 numerator is not divisible by (X − α)"
    );

    q_open_poly
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
pub fn prove<R: RngCore>(
    pk: &PfrPublicKey,
    row_indices: &[usize],
    col_indices: &[usize],
    rng: &mut R,
) -> (PfrProof, PfrPublicInputs) {
    // Initialise the Fiat-Shamir transcript with the public-key commitment.
    let mut fs_rng = FS::initialize(&to_bytes![pk.h_commitment.commitment()].unwrap());

    // --- Round 1 ---
    let mut round1_state = round_one(pk, row_indices, col_indices, rng);

    let first_round_comm_time = start_timer!(|| "Committing to Round 1 polynomials");
    let (round1_comms, round1_rands) =
        PC::commit(&pk.ck, round1_state.polynomials.iter(), None).unwrap();
    end_timer!(first_round_comm_time);
    round1_state.rands = round1_rands;

    let mut round1_comms = round1_comms;
    let r_comm = round1_comms.remove(0); // [R(τ)]₁
    let c_comm = round1_comms.remove(0); // [C(τ)]₁
    let m_comm = round1_comms.remove(0); // [m(τ)]₁
    let s_comm = round1_comms.remove(0); // [S(τ)]₁
    let row_comm = round1_comms.remove(0); // [row(τ)]₁  — statement polynomial
    let col_comm = round1_comms.remove(0); // [col(τ)]₁  — statement polynomial
    let rowcol_comm = round1_comms.remove(0); // [rowcol(τ)]₁ — statement polynomial
    let rowtilde_comm = round1_comms.remove(0); // [row̃(τ)]₁  — blinded, in proof

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
    let mut round2_state = round_two(pk, &round1_state, beta, rng);

    let second_round_comm_time = start_timer!(|| "Committing to Round 2 polynomials");
    let (f_comms, round2_rands) =
        PC::commit(&pk.ck, round2_state.polynomials.iter(), None).unwrap();
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
    let eta = Fr::rand(&mut fs_rng);

    // --- Round 3 ---
    let mut round3_state = round_three(pk, &round1_state, &round2_state, eta);

    let third_round_comm_time = start_timer!(|| "Committing to Round 3 polynomials");
    let (round3_comms, round3_rands) =
        PC::commit(&pk.ck, round3_state.polynomials.iter(), None).unwrap();
    end_timer!(third_round_comm_time);
    round3_state.rands = round3_rands;

    let mut round3_comms = round3_comms;
    let r_star_comm = round3_comms.remove(0);
    let q_comm = round3_comms.remove(0);

    // --- Round 4 ---
    // Derive α by absorbing Round 3 commitments.
    fs_rng.absorb(&to_bytes![r_star_comm.commitment(), q_comm.commitment()].unwrap());
    let alpha = Fr::rand(&mut fs_rng);

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
    let delta = Fr::rand(&mut fs_rng);

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
    let (mut q_open_comms, _) = PC::commit(&pk.ck, vec![&q_open_labeled], None).unwrap();
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

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify the PFR proof.
///
/// Implements the verification equation from page 39 of the paper:
///
///   Compute [y]₁ = [h(τ)]₁ − h_α·[1]₁
///                + δ([R(τ)]₁ − R_α·[1]₁)
///                + δ²([C(τ)]₁ − C_α·[1]₁)
///                + δ³([row̃(τ)]₁ − row̃_α·[1]₁)
///                + δ⁴[Lin(τ)]₁
///
///   where [Lin(τ)]₁ = U(α)·[big_lin(τ)]₁ − η⁹·α·[R*(τ)]₁ − U(α)·z_K(α)·[q(τ)]₁
///
///   and [big_lin(τ)]₁ is computed as an MSM over proof and public-input commitments:
///     η⁰·(β+R_α)·[F₁(τ)]₁ − η⁰·[1]₁
///   + η¹·(β+C_α)·[F₂(τ)]₁ − η¹·[1]₁
///   + η²·(β·Δ·R_α+C_α)·[F₃(τ)]₁ − η²·Δ·R_α·[1]₁
///   + η³·(β·Δᵗ+C_α)·[F₄(τ)]₁ − η³·Δᵗ·[1]₁
///   + η⁴·(β+h_α)·[F₅(τ)]₁ + η⁴·z_{K∖H}(α)·[m(τ)]₁
///   + η⁵·R_α²·[1]₁ − η⁵·[row(τ)]₁
///   + η⁶·C_α²·[1]₁ − η⁶·[col(τ)]₁
///   + η⁷·[rowcol(τ)]₁ − η⁷·row̃_α·[col(τ)]₁
///   + η⁸·[row̃(τ)]₁ − η⁸·[row(τ)]₁
///   + η⁹·(∑ⱼ[Fⱼ(τ)]₁ + η·[S(τ)]₁)
///
///   Check: e([y]₁, [1]₂) = e([Q(τ)]₁, [τ − α]₂)
///
/// `col_comm` and `rowcol_comm` are public inputs.
pub fn verify(
    pk: &PfrPublicKey,
    proof: &PfrProof,
    row_comm: &Comm,
    col_comm: &Comm,
    rowcol_comm: &Comm,
) -> bool {
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::G1Affine;
    use ark_ec::{msm::VariableBaseMSM, AffineCurve, PairingEngine, ProjectiveCurve};
    use ark_ff::PrimeField;

    // Derive Fiat-Shamir challenges
    let mut fs_rng = FS::initialize(&to_bytes![pk.h_commitment.commitment()].unwrap());

    fs_rng.absorb(
        &to_bytes![
            proof.r_comm.commitment(),
            proof.c_comm.commitment(),
            proof.m_comm.commitment(),
            proof.s_comm.commitment(),
            proof.rowtilde_comm.commitment()
        ]
        .unwrap(),
    );
    let beta = Fr::rand(&mut fs_rng);

    fs_rng.absorb(
        &to_bytes![
            proof.f_comms[0].commitment(),
            proof.f_comms[1].commitment(),
            proof.f_comms[2].commitment(),
            proof.f_comms[3].commitment(),
            proof.f_comms[4].commitment()
        ]
        .unwrap(),
    );
    let eta = Fr::rand(&mut fs_rng);

    fs_rng.absorb(&to_bytes![proof.r_star_comm.commitment(), proof.q_comm.commitment()].unwrap());
    let alpha = Fr::rand(&mut fs_rng);

    fs_rng
        .absorb(&to_bytes![proof.h_alpha, proof.r_alpha, proof.c_alpha, proof.row_alpha].unwrap());
    let delta = Fr::rand(&mut fs_rng);

    let big_delta = pk.big_delta();
    let big_delta_t = big_delta.pow([pk.t as u64]);

    let h_alpha = proof.h_alpha;
    let r_alpha = proof.r_alpha;
    let c_alpha = proof.c_alpha;
    let row_alpha = proof.row_alpha;

    // U(α) = α³ − 1,  z_K(α)
    // z_{K\H}(α) = (n/m)·(α^m−1)/(α^n−1) — same normalized polynomial as in round_three
    let u_at_alpha: Fr = alpha * alpha * alpha - Fr::one();
    let zk_at_alpha: Fr = pk.k_domain.vanishing_polynomial().evaluate(&alpha);
    let zh_at_alpha: Fr = pk.h_domain.vanishing_polynomial().evaluate(&alpha);
    let nm_ratio: Fr = Fr::from(pk.n as u64) * Fr::from(pk.m as u64).inverse().unwrap();
    let zkh_at_alpha: Fr = nm_ratio * zk_at_alpha * zh_at_alpha.inverse().unwrap();

    // η⁹: the last η power used in big_lin (η⁰ through η⁹, skipping η⁸)
    let eta9 = eta.pow([9u64]);

    // -----------------------------------------------------------------------
    // Build [Lin(τ)]₁ as a MSM over proof commitments.
    // Mirrors big_lin in round_five, but operating on group elements.
    // Constant-polynomial terms  −c  become  −c·[1]₁.
    // -----------------------------------------------------------------------
    let g1 = pk.vk.vk.g; // [1]₁

    let mut bases: Vec<G1Affine> = Vec::new();
    let mut scalars: Vec<Fr> = Vec::new();

    macro_rules! add_comm {
        ($s:expr, $c:expr) => {
            bases.push(($c).commitment().comm.0);
            scalars.push($s);
        };
    }
    macro_rules! add_g1 {
        ($s:expr) => {
            bases.push(g1);
            scalars.push($s);
        };
    }

    // big_lin terms (before multiplying by U(α))
    let mut eta_pow = Fr::one(); // η⁰

    // η⁰: F₁(τ)(β + R_α) − 1
    add_comm!(eta_pow * (beta + r_alpha), &proof.f_comms[0]);
    add_g1!(-eta_pow);

    eta_pow *= eta; // η¹
                    // η¹: F₂(τ)(β + C_α) − 1
    add_comm!(eta_pow * (beta + c_alpha), &proof.f_comms[1]);
    add_g1!(-eta_pow);

    eta_pow *= eta; // η²
                    // η²: F₃(τ)(β·Δ·R_α + C_α) − Δ·R_α
    add_comm!(
        eta_pow * (beta * big_delta * r_alpha + c_alpha),
        &proof.f_comms[2]
    );
    add_g1!(-(eta_pow * big_delta * r_alpha));

    eta_pow *= eta; // η³
                    // η³: F₄(τ)(β·Δᵗ + C_α) − Δᵗ
    add_comm!(eta_pow * (beta * big_delta_t + c_alpha), &proof.f_comms[3]);
    add_g1!(-(eta_pow * big_delta_t));

    eta_pow *= eta; // η⁴
                    // η⁴: F₅(τ)(β + h_α) + m(τ)·z_{K\H}(α)
    add_comm!(eta_pow * (beta + h_alpha), &proof.f_comms[4]);
    add_comm!(eta_pow * zkh_at_alpha, &proof.m_comm);

    eta_pow *= eta; // η⁵
                    // η⁵: R_α² − row(τ)  (public statement polynomial)
    add_g1!(eta_pow * r_alpha * r_alpha);
    add_comm!(-eta_pow, row_comm);

    eta_pow *= eta; // η⁶
                    // η⁶: C_α² − col(τ)
    add_g1!(eta_pow * c_alpha * c_alpha);
    add_comm!(-eta_pow, col_comm);

    eta_pow *= eta; // η⁷
                    // η⁷: rowcol(τ) − row̃_α · col(τ)
    add_comm!(eta_pow, rowcol_comm);
    add_comm!(-(eta_pow * row_alpha), col_comm);

    eta_pow *= eta; // η⁸: row̃(τ) − row(τ)
    add_comm!(eta_pow, &proof.rowtilde_comm);
    add_comm!(-eta_pow, row_comm);

    eta_pow *= eta; // η⁹ = eta9
                    // η⁹: ∑Fⱼ(τ) + η·S(τ)
    for fj in &proof.f_comms {
        add_comm!(eta_pow, fj);
    }
    add_comm!(eta_pow * eta, &proof.s_comm); // η¹⁰·S(τ)

    // Multiply big_lin by U(α), then add −η⁹·α·[R*(τ)]₁ and −U(α)·z_K(α)·[q(τ)]₁
    for s in &mut scalars {
        *s *= u_at_alpha;
    }
    add_comm!(-(eta9 * alpha), &proof.r_star_comm);
    add_comm!(-(u_at_alpha * zk_at_alpha), &proof.q_comm);

    // -----------------------------------------------------------------------
    // [y]₁ = [h(τ)]₁ − h_α·[1]₁
    //       + δ  ([R(τ)]₁  − R_α·[1]₁)
    //       + δ² ([C(τ)]₁  − C_α·[1]₁)
    //       + δ³ ([row̃(τ)]₁ − row̃_α·[1]₁)
    //       + δ⁴ · [Lin(τ)]₁
    // -----------------------------------------------------------------------
    // Scale all existing Lin terms by δ⁴
    let delta4 = delta.pow([4u64]);
    for s in &mut scalars {
        *s *= delta4;
    }

    // h term
    add_comm!(Fr::one(), &pk.h_commitment);
    add_g1!(-h_alpha);

    // δ · R term
    add_comm!(delta, &proof.r_comm);
    add_g1!(-(delta * r_alpha));

    // δ² · C term
    let delta2 = delta * delta;
    add_comm!(delta2, &proof.c_comm);
    add_g1!(-(delta2 * c_alpha));

    // δ³ · row̃ term
    let delta3 = delta2 * delta;
    add_comm!(delta3, &proof.rowtilde_comm);
    add_g1!(-(delta3 * row_alpha));

    // Compute [y]₁ via MSM
    let scalars_repr: Vec<_> = scalars.iter().map(|s| s.into_repr()).collect();
    let y_proj = VariableBaseMSM::multi_scalar_mul(&bases, &scalars_repr);
    let y = y_proj.into_affine();

    // -----------------------------------------------------------------------
    // Pairing check: e([y]₁, [1]₂) = e([Q(τ)]₁, [τ−α]₂)
    // [τ−α]₂ = [τ]₂ − α·[1]₂ = beta_h − α·h
    // -----------------------------------------------------------------------
    let h = pk.vk.vk.h; // [1]₂
    let tau_h = pk.vk.vk.beta_h; // [τ]₂
    let tau_minus_alpha_h = (tau_h.into_projective() - h.mul(alpha.into_repr())).into_affine();

    let q_aff = proof.q_poly_comm.commitment().comm.0;

    let lhs = Bls12_381::pairing(y, h);
    let rhs = Bls12_381::pairing(q_aff, tau_minus_alpha_h);

    lhs == rhs
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
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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

    /// S(X) = R_S·X + ρ_S·z_K(X): degree ≤ m+1, vanishes on K
    #[test]
    fn round1_s_poly_shape() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
        let row_poly = s.polynomials[4].polynomial();
        let rowtilde_poly = s.polynomials[7].polynomial();
        let diff = rowtilde_poly - row_poly;
        for i in 0..M {
            assert_eq!(
                diff.evaluate(&pk.k_domain.element(i)),
                Fr::zero(),
                "(row̃ − row)(κ^{i}) ≠ 0"
            );
        }
    }

    /// Cached r_evals / c_evals / m_evals match polynomial evaluations
    #[test]
    fn round1_cached_evals_consistent() {
        let pk = setup();
        let s = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
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
        let r1 = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
        let beta = Fr::from(42u64);
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

    fn check_round2_sumcheck(pk: &PfrPublicKey, row: &[usize], col: &[usize]) {
        let m = row.len();
        let r1 = round_one(pk, row, col, &mut ark_std::test_rng());
        let beta = Fr::from(42u64);
        let r2 = round_two(pk, &r1, beta, &mut ark_std::test_rng());

        let sum: Fr = (0..m)
            .map(|i| {
                let ki = pk.k_domain.element(i);
                r2.polynomials
                    .iter()
                    .map(|p| p.polynomial().evaluate(&ki))
                    .sum::<Fr>()
            })
            .sum();

        assert_eq!(
            sum,
            Fr::zero(),
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
        let pk = PfrPublicKey::setup(4, row.len(), 1, &mut ark_std::test_rng());
        check_round2_sumcheck(&pk, row, col);
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
        let r1 = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());

        let big_delta = pk.big_delta();
        let delta_t_inv = big_delta.pow([T as u64]).inverse().unwrap();
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
            let c_over_delta_r = c * (big_delta * r).inverse().unwrap();
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
        let r1 = round_one(&pk, &ROW, &COL, &mut ark_std::test_rng());
        let r2 = round_two(&pk, &r1, Fr::from(42u64), &mut ark_std::test_rng());
        let r3 = round_three(&pk, &r1, &r2, Fr::from(17u64));

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
        let pk = PfrPublicKey::setup(n, m, t, &mut ark_std::test_rng());
        let r1 = round_one(&pk, row, col, &mut ark_std::test_rng());
        let beta = Fr::from(42u64);
        let r2 = round_two(&pk, &r1, beta, &mut ark_std::test_rng());

        // Compute z_{K\H} polynomial as the prover does: (n/m)·(X^m-1)/(X^n-1)
        let steps = m / n;
        let scale_factor = Fr::from(n as u64) * Fr::from(m as u64).inverse().unwrap();
        let mut coeffs = vec![Fr::zero(); (steps - 1) * n + 1];
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
                Fr::zero(),
                "η⁴ identity failed at κ^{i} (in_H={})",
                i % (m / n) == 0
            );
        }
    }

    // --- End-to-end ---

    /// Run prove + verify for an arbitrary configuration and assert the proof verifies.
    fn prove_and_verify(n: usize, t: usize, row: &[usize], col: &[usize]) {
        let m = row.len();
        let rng = &mut ark_std::test_rng();
        let pk = PfrPublicKey::setup(n, m, t, rng);
        let (proof, public_inputs) = prove(&pk, row, col, rng);
        assert!(
            verify(
                &pk,
                &proof,
                &public_inputs.row_comm,
                &public_inputs.col_comm,
                &public_inputs.rowcol_comm
            ),
            "verification failed for n={}, m={}, t={}, row={:?}, col={:?}", n, m, t, row, col,
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
}
