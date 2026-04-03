use ark_ec::PairingEngine;
use ark_ff::{Field, Zero};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain, UVPolynomial,
};
use ark_poly_commit::{
    marlin_pc::MarlinKZG10, LabeledCommitment, LabeledPolynomial, PolynomialCommitment,
};
use ark_std::rand::RngCore;

// ---------------------------------------------------------------------------
// Convenience type aliases
// ---------------------------------------------------------------------------
pub type PC<E> = MarlinKZG10<E, DensePolynomial<<E as PairingEngine>::Fr>>;
pub type Comm<E> = LabeledCommitment<
    <PC<E> as PolynomialCommitment<
        <E as PairingEngine>::Fr,
        DensePolynomial<<E as PairingEngine>::Fr>,
    >>::Commitment,
>;
pub type Rand<E> = <PC<E> as PolynomialCommitment<
    <E as PairingEngine>::Fr,
    DensePolynomial<<E as PairingEngine>::Fr>,
>>::Randomness;
pub type CK<E> = <PC<E> as PolynomialCommitment<
    <E as PairingEngine>::Fr,
    DensePolynomial<<E as PairingEngine>::Fr>,
>>::CommitterKey;
pub type VK<E> = <PC<E> as PolynomialCommitment<
    <E as PairingEngine>::Fr,
    DensePolynomial<<E as PairingEngine>::Fr>,
>>::VerifierKey;

// ---------------------------------------------------------------------------
// Public key
// ---------------------------------------------------------------------------

/// Public parameters for the PFR (output of the key-generation phase).
///
/// Contains the table polynomial h(X) — committed once and reused across
/// many proofs — together with the KZG commitment keys and the three domains.
#[allow(dead_code)]
pub struct PfrPublicKey<E: PairingEngine> {
    /// n = |H|: size of the table domain H = <ω>.
    pub n: usize,
    /// m = |K|: number of index pairs (non-zero entries of the relation).
    pub m: usize,
    /// t: strictly-lower-triangular offset; every column index satisfies c_i ≥ t.
    pub t: usize,
    pub h_domain: GeneralEvaluationDomain<E::Fr>, // H = <ω>,  |H| = n
    pub d_domain: GeneralEvaluationDomain<E::Fr>, // D = <Δ>,  |D| = 2n,  Δ² = ω
    pub k_domain: GeneralEvaluationDomain<E::Fr>, // K = <κ>,  |K| = m
    /// Coset domain for round 3: a coset of K of size 2m, disjoint from K.
    /// Used to evaluate polynomials without hitting zeros of z_K.
    pub coset_domain: GeneralEvaluationDomain<E::Fr>,
    /// Table polynomial: h(ω^j) = Δ^j for j = 0, …, n−1.
    pub h_poly: DensePolynomial<E::Fr>,
    /// h(X) evaluated over coset_domain, precomputed at setup time.
    pub h_coset_evals: Vec<E::Fr>,
    /// z_{K\H}(X) evaluated over coset_domain, precomputed at setup time.
    pub zkh_coset_evals: Vec<E::Fr>,
    pub ck: CK<E>,
    pub vk: VK<E>,
    pub h_commitment: Comm<E>,
    pub h_randomness: Rand<E>,
}

impl<E: PairingEngine> PfrPublicKey<E> {
    /// Generate public parameters.
    ///
    /// - `n`: table size |H|
    /// - `m`: number of index pairs |K|
    /// - `t`: strictly-lower-triangular offset
    ///
    /// No zero-knowledge: `hiding_bound = None`, `supported_hiding_bound = 0`.
    pub fn setup<R: RngCore>(n: usize, m: usize, t: usize, rng: &mut R) -> Self {
        let h_domain = GeneralEvaluationDomain::<E::Fr>::new(n).expect("H domain must exist");
        let d_domain = GeneralEvaluationDomain::<E::Fr>::new(2 * n).expect("D domain must exist");
        let k_domain = GeneralEvaluationDomain::<E::Fr>::new(m).expect("K domain must exist");

        // h(X): unique polynomial of degree < n with h(ω^j) = Δ^j for j = 0, …, n−1.
        let h_evals: Vec<E::Fr> = d_domain.elements().take(n).collect();
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
        // Maximum degree to support: max(n−1, 2m+3, 2m).
        let max_degree = (n - 1).max(2 * m + 3).max(2 * m);
        let pp = PC::<E>::setup(max_degree, None, rng).unwrap();
        let (ck, vk) = PC::<E>::trim(&pp, max_degree, 1, None).unwrap();

        let h_labeled = LabeledPolynomial::new("h".into(), h_poly.clone(), None, None);
        let (mut comms, mut rands) = PC::<E>::commit(&ck, vec![&h_labeled], None).unwrap();

        // Coset domain of size 2m for round 3: uses ark-poly's canonical coset
        // (multiplied internally by a fixed generator g), which is disjoint from K.
        let coset_domain =
            GeneralEvaluationDomain::<E::Fr>::new(2 * m).expect("coset domain must exist");

        // Precompute h over the coset using coset_fft (evaluates coeffs over g·K).
        let h_coset_evals = coset_domain.coset_fft(&h_poly.coeffs);

        // z_{K\H}(X) = (n/m)·(X^{m−n} + X^{m−2n} + … + 1), coeffs at multiples of n.
        let steps = m / n;
        let scale = E::Fr::from(n as u64) * E::Fr::from(m as u64).inverse().unwrap();
        let mut zkh_coeffs = vec![E::Fr::zero(); (steps - 1) * n + 1];
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
    pub fn big_delta(&self) -> E::Fr {
        self.d_domain.element(1)
    }

    /// Build a `PfrPublicKey` reusing an externally-provided `(ck, vk)`.
    ///
    /// Use this when the KZG SRS was already generated by another component
    /// (e.g. Marlin's `universal_setup`), so that both proofs share the same
    /// trusted setup. The caller is responsible for ensuring `ck` supports
    /// the required maximum degree `max(n-1, 2m+3)`.
    pub fn with_keys<R: RngCore>(ck: CK<E>, vk: VK<E>, n: usize, m: usize, t: usize, rng: &mut R) -> Self {
        let h_domain = GeneralEvaluationDomain::<E::Fr>::new(n).expect("H domain must exist");
        let d_domain = GeneralEvaluationDomain::<E::Fr>::new(2 * n).expect("D domain must exist");
        let k_domain = GeneralEvaluationDomain::<E::Fr>::new(m).expect("K domain must exist");
        let coset_domain =
            GeneralEvaluationDomain::<E::Fr>::new(2 * m).expect("coset domain must exist");

        let h_evals: Vec<E::Fr> = d_domain.elements().take(n).collect();
        let h_poly = EvaluationsOnDomain::from_vec_and_domain(h_evals, h_domain).interpolate();
        let h_coset_evals = coset_domain.coset_fft(&h_poly.coeffs);

        let steps = m / n;
        let scale = E::Fr::from(n as u64) * E::Fr::from(m as u64).inverse().unwrap();
        let mut zkh_coeffs = vec![E::Fr::zero(); (steps - 1) * n + 1];
        for k in 0..steps {
            zkh_coeffs[k * n] = scale;
        }
        let zkh_poly = DensePolynomial::from_coefficients_vec(zkh_coeffs);
        let zkh_coset_evals = coset_domain.coset_fft(&zkh_poly.coeffs);

        let h_labeled = LabeledPolynomial::new("h".into(), h_poly.clone(), None, None);
        let (mut comms, mut rands) = PC::<E>::commit(&ck, vec![&h_labeled], None).unwrap();

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

pub struct PfrProof<E: PairingEngine> {
    // Round 1
    pub r_comm: Comm<E>,
    pub c_comm: Comm<E>,
    pub m_comm: Comm<E>,
    pub s_comm: Comm<E>,
    pub rowtilde_comm: Comm<E>,
    // Round 2
    pub f_comms: Vec<Comm<E>>, // [F₁(τ), F₂(τ), F₃(τ), F₄(τ), F₅(τ)]
    // Round 3
    pub r_star_comm: Comm<E>,
    pub q_comm: Comm<E>,
    // Round 4
    pub h_alpha: E::Fr,
    pub r_alpha: E::Fr,
    pub c_alpha: E::Fr,
    pub row_alpha: E::Fr,
    // Round 5
    pub q_poly_comm: Comm<E>,
}

/// Public inputs to the PFR verifier — commitments known to both parties
/// before the proof is generated.
///
/// In the paper these are committed as part of the relation description,
/// not as part of `π_PFR`.
#[allow(dead_code)]
pub struct PfrPublicInputs<E: PairingEngine> {
    /// [row(τ)]₁: commitment to the row-index polynomial row(X)
    pub row_comm: Comm<E>,
    /// [col(τ)]₁: commitment to the column-index polynomial col(X)
    pub col_comm: Comm<E>,
    /// [rowcol(τ)]₁: commitment to the rowcol polynomial rowcol(X)
    pub rowcol_comm: Comm<E>,
}

/// Precomputed statement: polynomials, commitments, and randomness for
/// the three public index polynomials row(X), col(X), rowcol(X).
///
/// Produced by [`crate::prover::commit_statement`] and consumed by [`crate::prover::prove`].
/// Separating this step allows the statement commitment cost to be benchmarked
/// and amortised independently of the proof generation.
#[allow(dead_code)]
pub struct PfrStatement<E: PairingEngine> {
    /// row(κ^i) = ω^{r_i}: statement polynomial for row indices.
    pub row_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
    /// col(κ^i) = ω^{c_i}: statement polynomial for column indices.
    pub col_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
    /// rowcol(κ^i) = ω^{r_i·c_i}: statement polynomial for row×col.
    pub rowcol_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
    /// [row(τ)]₁
    pub row_comm: Comm<E>,
    /// [col(τ)]₁
    pub col_comm: Comm<E>,
    /// [rowcol(τ)]₁
    pub rowcol_comm: Comm<E>,
    /// KZG randomness for row(X) (trivial in no-ZK mode).
    pub row_rand: Rand<E>,
    /// KZG randomness for col(X) (trivial in no-ZK mode).
    pub col_rand: Rand<E>,
    /// KZG randomness for rowcol(X) (trivial in no-ZK mode).
    pub rowcol_rand: Rand<E>,
}

impl<E: PairingEngine> PfrStatement<E> {
    /// Build a PFR statement from externally committed statement polynomials.
    ///
    /// This is the entry point for reusing a statement committed by another
    /// protocol, such as Marlin's indexed `row/col/rowcol` commitments.
    /// The caller is responsible for ensuring that:
    /// - the three polynomials share the same K-domain evaluations expected by PFR,
    /// - the commitments and randomness correspond to these exact polynomials,
    /// - any blinding on the polynomials is accounted for by the PFR equations.
    pub fn from_existing(
        row_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
        col_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
        rowcol_poly: LabeledPolynomial<E::Fr, DensePolynomial<E::Fr>>,
        row_comm: Comm<E>,
        col_comm: Comm<E>,
        rowcol_comm: Comm<E>,
        row_rand: Rand<E>,
        col_rand: Rand<E>,
        rowcol_rand: Rand<E>,
    ) -> Self {
        Self {
            row_poly,
            col_poly,
            rowcol_poly,
            row_comm,
            col_comm,
            rowcol_comm,
            row_rand,
            col_rand,
            rowcol_rand,
        }
    }
}
