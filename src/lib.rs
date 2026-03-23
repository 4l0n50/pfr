//! Proof of Function Relation (PFR) — Appendix B, IMPR-FHFC paper.
//!
//! Equation (7) with t = 1:
//!
//!   ∑_{i=1}^m [ 1/(R(κ^i)+X) + 1/(C(κ^i)+X) + 1/(C(κ^i)/(ΔR(κ^i))+X) + 1/(C(κ^i)/Δ+X) ]
//!   = ∑_{j=0}^{n-1} m_j / (h(ω^j) + X)
//!
//! The LHS has four terms per index pair (r_i, c_i).  Since R(κ^i) = Δ^{r_i} and
//! C(κ^i) = Δ^{c_i}, those four Δ-powers are:
//!
//!   Δ^{r_i},  Δ^{c_i},  Δ^{c_i - r_i - 1},  Δ^{c_i - 1}
//!
//! and by Lemma 5 they must all lie in {h(ω^j) = Δ^j : j ∈ [0, n-1]}.
//! The m_j are therefore the multiplicities of h(ω^j) in that 4m-element multiset.
//!
//! Module structure
//! ----------------
//! Public key (indexer output):  [`PfrPublicKey`] — h(X), domains, KZG keys.
//! Prover output:                [`PfrProof`]     — commitments to R, C, m + openings.
//! API:  [`PfrPublicKey::setup`], [`compute_multiplicities`], [`prove`], [`verify`].

use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    GeneralEvaluationDomain,
};
use ark_poly_commit::{
    marlin_pc::MarlinKZG10, LabeledCommitment, LabeledPolynomial, PolynomialCommitment,
};
use ark_std::rand::RngCore;

// ---------------------------------------------------------------------------
// Convenience type aliases for the concrete PC instantiation
// ---------------------------------------------------------------------------
type PC = MarlinKZG10<Bls12_381, DensePolynomial<Fr>>;
type Comm = LabeledCommitment<<PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::Commitment>;
type Rand = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::Randomness;
type PcProof = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::Proof;
type CK = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::CommitterKey;
type VK = <PC as PolynomialCommitment<Fr, DensePolynomial<Fr>>>::VerifierKey;

// ---------------------------------------------------------------------------
// Public key
// ---------------------------------------------------------------------------

/// Public parameters for the PFR (output of the indexer / key-generation phase).
///
/// Contains the table polynomial h(X) — the only polynomial whose commitment is
/// fixed at setup time and reused across many proofs — together with the KZG
/// commitment keys and the three evaluation domains H, D, K.
pub struct PfrPublicKey {
    /// Size of the table H (number of rows/columns).
    pub n: usize,
    /// Number of non-zero entries (index pairs).
    pub m: usize,
    /// Number of public input variables in the R1CS (Definition 22).
    /// The matrices A, B are t-strictly lower triangular (strictly lower triangular
    /// with the first t rows zero), so every nonzero entry has column index ≥ t.
    /// The fourth term of equation (7), C(κ^i)/Δ^t, encodes the lookup
    /// col(K) ⊆ {ω^t, …, ω^{n-1}} that enforces this column-side condition.
    pub t: usize,
    pub h_domain: GeneralEvaluationDomain<Fr>, // H = <ω>, |H| = n
    pub d_domain: GeneralEvaluationDomain<Fr>, // D = <Δ>, |D| = 2n
    pub k_domain: GeneralEvaluationDomain<Fr>, // K = <κ>, |K| = m
    /// Table polynomial: h(ω^j) = Δ^j for j = 0, …, n-1.
    pub h_poly: DensePolynomial<Fr>,
    pub ck: CK,
    pub vk: VK,
    pub h_commitment: Comm,
    pub h_randomness: Rand,
}

impl PfrPublicKey {
    /// Generate public parameters for a PFR with an n-element table, m index pairs,
    /// and exponent `t` for the fourth term of equation (7).
    ///
    /// No zero-knowledge: `hiding_bound = None`, `supported_hiding_bound = 0`,
    /// no rng passed to commit/open.
    pub fn setup<R: RngCore>(n: usize, m: usize, t: usize, rng: &mut R) -> Self {
        let h_domain = GeneralEvaluationDomain::<Fr>::new(n).expect("H domain must exist");
        let d_domain = GeneralEvaluationDomain::<Fr>::new(2 * n).expect("D domain must exist");
        let k_domain = GeneralEvaluationDomain::<Fr>::new(m).expect("K domain must exist");

        // h(X): unique deg < n poly with h(ω^j) = Δ^j
        let h_evals: Vec<Fr> = d_domain.elements().take(n).collect();
        let h_poly = EvaluationsOnDomain::from_vec_and_domain(h_evals, h_domain).interpolate();

        let max_degree = (n - 1).max(m - 1);
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
    // Multiplicity computation
    // ---------------------------------------------------------------------------

    /// Compute the multiplicity vector (m_0, …, m_{n-1}) for equation (7):
    ///
    /// ```text
    ///   m                                                         n-1
    ///   ∑  [ 1/(R(κ^i)+X) + 1/(C(κ^i)+X)                    =      ∑   m_j / (h(ω^j)+X)
    ///  i=1    + 1/(C(κ^i)/(ΔR(κ^i))+X) + 1/(C(κ^i)/Δ^t+X) ]       j=0
    /// ```
    ///
    /// Since R(κ^i) = Δ^{r_i}, C(κ^i) = Δ^{c_i}, and h(ω^j) = Δ^j, the four LHS
    /// terms per pair (r_i, c_i) correspond to table indices r_i, c_i, c_i−r_i−1, c_i−t.
    /// m_j counts how many times index j appears across all pairs and all four terms.
    ///
    /// **Preconditions** (from the t-strictly lower triangular structure: r_i < c_i
    /// and c_i ≥ t for all i, with all resulting indices ∈ [0, n-1]):
    ///   - c_i − r_i − 1 ≥ 0  (because r_i < c_i)
    ///   - c_i − t       ≥ 0  (because column indices satisfy c_i ≥ t)
    ///   - both values   < n
    pub fn compute_multiplicities(&self, row_indices: &[usize], col_indices: &[usize]) -> Vec<u64> {
        let mut mults = vec![0u64; self.n];
        for (&r, &c) in row_indices.iter().zip(col_indices.iter()) {
            mults[r] += 1; //         R(κ^i)  = Δ^r       = h(ω^r)
            mults[c] += 1; //         C(κ^i)  = Δ^c       = h(ω^c)
            mults[c - r - 1] += 1; // C/(ΔR)  = Δ^{c-r-1} = h(ω^{c-r-1})
            mults[c - self.t] += 1; // C/Δ^t  = Δ^{c-t}   = h(ω^{c-t})
        }
        mults
    }
}

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

/// PFR proof output by the prover.
///
/// Contains:
///   - KZG commitments to R(X), C(X), m(X).
///   - The raw multiplicity values m_j (for display / sanity checks).
///   - Opening proofs demonstrating consistency for index i = 0:
///       R(κ^0) = Δ^{r_0},  C(κ^0) = Δ^{c_0},
///       h(ω^{r_0}) = Δ^{r_0},  h(ω^{c_0}) = Δ^{c_0}.
pub struct PfrProof {
    pub r_commitment: Comm,
    pub c_commitment: Comm,
    pub m_commitment: Comm,
    /// m_j: number of times h(ω^j) appears in the 4m-element multiset of eq (7).
    pub multiplicities: Vec<u64>,
    /// Batch opening proof for R and C at κ^0.
    pub proof_rc: PcProof,
    /// Opening proof for h at ω^{r_0} (shows h(ω^{r_0}) = R(κ^0)).
    pub proof_h_r: PcProof,
    /// Opening proof for h at ω^{c_0} (shows h(ω^{c_0}) = C(κ^0)).
    pub proof_h_c: PcProof,
    /// Opening proof for m at ω^0 (shows m(ω^0) = m_0).
    pub proof_m: PcProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Build R(X), C(X), m(X), commit to them, and produce opening proofs for i = 0.
pub fn prove(
    pk: &PfrPublicKey,
    row_indices: &[usize],
    col_indices: &[usize],
    opening_challenge: Fr,
) -> PfrProof {
    // R(X): R(κ^i) = Δ^{r_i}
    let r_evals: Vec<Fr> = row_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let r_poly = EvaluationsOnDomain::from_vec_and_domain(r_evals, pk.k_domain).interpolate();

    // C(X): C(κ^i) = Δ^{c_i}
    let c_evals: Vec<Fr> = col_indices
        .iter()
        .map(|&j| pk.d_domain.element(j))
        .collect();
    let c_poly = EvaluationsOnDomain::from_vec_and_domain(c_evals, pk.k_domain).interpolate();

    // m(X): m(ω^j) = m_j — the multiplicity polynomial over H
    let multiplicities = pk.compute_multiplicities(row_indices, col_indices);
    let m_evals: Vec<Fr> = multiplicities.iter().map(|&v| Fr::from(v)).collect();
    let m_poly = EvaluationsOnDomain::from_vec_and_domain(m_evals, pk.h_domain).interpolate();

    // Commit to R, C, m (no hiding)
    let r_labeled = LabeledPolynomial::new("R".into(), r_poly.clone(), None, None);
    let c_labeled = LabeledPolynomial::new("C".into(), c_poly.clone(), None, None);
    let m_labeled = LabeledPolynomial::new("m".into(), m_poly.clone(), None, None);

    let (mut comms, mut rands) =
        PC::commit(&pk.ck, vec![&r_labeled, &c_labeled, &m_labeled], None).unwrap();
    let m_rand = rands.remove(2);
    let c_rand = rands.remove(1);
    let r_rand = rands.remove(0);
    let m_comm = comms.remove(2);
    let c_comm = comms.remove(1);
    let r_comm = comms.remove(0);

    // --- Open R and C at κ^0 ---
    let kappa_0 = pk.k_domain.element(0);
    let rc_comms = vec![r_comm.clone(), c_comm.clone()];
    let proof_rc = PC::open(
        &pk.ck,
        vec![&r_labeled, &c_labeled],
        &rc_comms,
        &kappa_0,
        opening_challenge,
        vec![&r_rand, &c_rand],
        None,
    )
    .unwrap();

    // --- Open m at ω^0 ---
    let omega_0 = pk.h_domain.element(0);
    let proof_m = PC::open(
        &pk.ck,
        vec![&m_labeled],
        &[m_comm.clone()],
        &omega_0,
        opening_challenge,
        vec![&m_rand],
        None,
    )
    .unwrap();

    // --- Open h at ω^{r_0} ---
    let h_labeled = LabeledPolynomial::new("h".into(), pk.h_poly.clone(), None, None);
    let omega_r0 = pk.h_domain.element(row_indices[0]);
    let proof_h_r = PC::open(
        &pk.ck,
        vec![&h_labeled],
        &[pk.h_commitment.clone()],
        &omega_r0,
        opening_challenge,
        vec![&pk.h_randomness],
        None,
    )
    .unwrap();

    // --- Open h at ω^{c_0} ---
    let omega_c0 = pk.h_domain.element(col_indices[0]);
    let proof_h_c = PC::open(
        &pk.ck,
        vec![&h_labeled],
        &[pk.h_commitment.clone()],
        &omega_c0,
        opening_challenge,
        vec![&pk.h_randomness],
        None,
    )
    .unwrap();

    PfrProof {
        r_commitment: r_comm,
        c_commitment: c_comm,
        m_commitment: m_comm,
        multiplicities,
        proof_rc,
        proof_h_r,
        proof_h_c,
        proof_m,
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify the PFR proof: check all three opening proofs for index i = 0.
pub fn verify(
    pk: &PfrPublicKey,
    row_indices: &[usize],
    col_indices: &[usize],
    proof: &PfrProof,
    opening_challenge: Fr,
    rng: &mut impl RngCore,
) -> bool {
    let kappa_0 = pk.k_domain.element(0);
    let r_at_0 = pk.d_domain.element(row_indices[0]); // expected Δ^{r_0}
    let c_at_0 = pk.d_domain.element(col_indices[0]); // expected Δ^{c_0}

    // R(κ^0) = Δ^{r_0}  and  C(κ^0) = Δ^{c_0}
    let valid_rc = PC::check(
        &pk.vk,
        &[proof.r_commitment.clone(), proof.c_commitment.clone()],
        &kappa_0,
        [r_at_0, c_at_0],
        &proof.proof_rc,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    // h(ω^{r_0}) = Δ^{r_0}  →  h(ω^{r_0}) = R(κ^0)
    let omega_r0 = pk.h_domain.element(row_indices[0]);
    let valid_h_r = PC::check(
        &pk.vk,
        &[pk.h_commitment.clone()],
        &omega_r0,
        [r_at_0],
        &proof.proof_h_r,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    // h(ω^{c_0}) = Δ^{c_0}  →  h(ω^{c_0}) = C(κ^0)
    let omega_c0 = pk.h_domain.element(col_indices[0]);
    let valid_h_c = PC::check(
        &pk.vk,
        &[pk.h_commitment.clone()],
        &omega_c0,
        [c_at_0],
        &proof.proof_h_c,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    // m(ω^0) = m_0
    let omega_0 = pk.h_domain.element(0);
    let valid_m = PC::check(
        &pk.vk,
        &[proof.m_commitment.clone()],
        &omega_0,
        [Fr::from(proof.multiplicities[0])],
        &proof.proof_m,
        opening_challenge,
        Some(rng),
    )
    .unwrap();

    valid_rc && valid_h_r && valid_h_c && valid_m
}
