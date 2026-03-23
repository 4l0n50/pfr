// Toy PFR (Polynomial Functional Relation) example — Appendix B, IMPR-FHFC paper.
//
// Demonstrates a Proof of Function Relation using the `pfr` module:
//
//   - Public key (indexer):  h(X) with h(ω^j) = Δ^j, KZG keys, domains.
//   - Prover:                R(X), C(X) — row/col indices as Δ-powers;
//                            m(X) — multiplicity polynomial from equation (7).
//
// Domains
//   H (size n=4, generator ω):  table domain
//   D (size 2n=8, generator Δ): Δ^2 = ω, so Δ-powers cover both H and extra
//   K (size m=4, generator κ):  index domain
//
// Concrete index pairs (r_i, c_i) with r_i < c_i:
//   i=0: (0,1)   i=1: (1,2)   i=2: (2,3)   i=3: (0,3)
//
// No zero-knowledge: hiding_bound = None, no rng passed to commit/open.
//
// Run with:
//   cargo run --example commit_and_open


use ark_bls12_381::Fr;
use ark_poly::{EvaluationDomain, Polynomial}; // for .element() and .degree()
use pfr::{prove, verify, PfrPublicKey};

fn main() {
    let rng = &mut ark_std::test_rng();

    // -----------------------------------------------------------------------
    // Public key (indexer output)
    // h(X): unique deg < n poly with h(ω^j) = Δ^j  (the table polynomial)
    // -----------------------------------------------------------------------
    let n = 4usize; // |H|
    let m = 4usize; // |K| = number of index pairs
    let pk = PfrPublicKey::setup(n, m, 1, rng);
    println!("=== Public Key ===");
    println!("h(X) degree: {}", pk.h_poly.degree());
    // Sanity-check: h(ω^j) = Δ^j
    for j in 0..n {
        assert_eq!(
            pk.h_poly.evaluate(&pk.h_domain.element(j)),
            pk.d_domain.element(j),
            "h(ω^{j}) ≠ Δ^{j}"
        );
    }
    println!("h(ω^j) = Δ^j for all j ∈ [0,{n}) ✓");

    // -----------------------------------------------------------------------
    // Prover input: index pairs (r_i, c_i) with r_i < c_i
    // -----------------------------------------------------------------------
    let row_indices: [usize; 4] = [0, 1, 2, 0];
    let col_indices: [usize; 4] = [1, 2, 3, 3];

    // -----------------------------------------------------------------------
    // Multiplicities from equation (7) with t=1.
    // Four terms per i: Δ^{r_i}, Δ^{c_i}, Δ^{c_i-r_i-1}, Δ^{c_i-1}.
    // m_j = how many times h(ω^j) appears in that 4m-element multiset.
    // -----------------------------------------------------------------------
    println!("\n=== Multiplicities (eq. 7, t=1) ===");
    let mults = pk.compute_multiplicities(&row_indices, &col_indices);
    for (j, &mj) in mults.iter().enumerate() {
        println!("  m_{j} = {mj}  (h(ω^{j}) = Δ^{j} appears {mj} times)");
    }
    let total: u64 = mults.iter().sum();
    println!("  Total = {total}  (should be 4·m = {})", 4 * m);
    assert_eq!(total, (4 * m) as u64);

    // -----------------------------------------------------------------------
    // Prover: commit to R, C, m and produce opening proofs for i=0
    // -----------------------------------------------------------------------
    println!("\n=== Prover ===");
    let opening_challenge = Fr::from(42u64);
    let proof = prove(&pk, &row_indices, &col_indices, opening_challenge);

    // Verify expected values at i=0
    let r_at_0 = pk.d_domain.element(row_indices[0]); // Δ^{r_0} = Δ^0 = 1
    let c_at_0 = pk.d_domain.element(col_indices[0]); // Δ^{c_0} = Δ^1
    println!("R(κ^0) = Δ^{} = {:?}", row_indices[0], r_at_0);
    println!("C(κ^0) = Δ^{} = {:?}", col_indices[0], c_at_0);

    // -----------------------------------------------------------------------
    // Verifier
    // -----------------------------------------------------------------------
    println!("\n=== Verifier ===");
    let valid = verify(
        &pk,
        &row_indices,
        &col_indices,
        &proof,
        opening_challenge,
        rng,
    );
    println!("R, C opening at κ^0 valid: {valid}");
    println!(
        "h(ω^{}) = R(κ^0) and h(ω^{}) = C(κ^0): {valid}",
        row_indices[0], col_indices[0]
    );
    assert!(valid, "PFR proof verification failed");
    println!("\nPFR proof valid ✓");
}
