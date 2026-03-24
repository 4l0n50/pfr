// Toy PFR (Polynomial Functional Relation) example — Appendix B, IMPR-FHFC paper.
//
// Demonstrates the first two rounds of the PFR protocol using the `pfr` module:
//
//   Public key (indexer):  h(X) with h(ω^j) = Δ^j, KZG keys, domains.
//   Round 1 (prover):      R(X), C(X), m(X), S(X)=0, row̃(X) — commitments.
//   Round 2 (prover):      F₁(X), …, F₅(X) — rational-sum commitments (eq. 8).
//   Rounds 3–5:            TODO stubs.
//
// Domains
//   H (size n=4, generator ω):  table domain
//   D (size 2n=8, generator Δ): Δ² = ω
//   K (size m=4, generator κ):  index domain  (K = H in this toy example)
//
// Concrete index pairs (r_i, c_i) with r_i < c_i and c_i ≥ t = 1:
//   i=0: (0,1)   i=1: (1,2)   i=2: (2,3)   i=3: (0,3)


use ark_poly::{EvaluationDomain, Polynomial};
use pfr::{prove, verify, PfrPublicKey};

fn main() {
    let rng = &mut ark_std::test_rng();

    // -----------------------------------------------------------------------
    // Public key (indexer output)
    // h(X): h(ω^j) = Δ^j  (the table polynomial)
    // -----------------------------------------------------------------------
    let n = 4usize; // |H|: table size
    let m = 4usize; // |K|: number of index pairs
    let t = 1usize; // strictly-lower-triangular offset
    let pk = PfrPublicKey::setup(n, m, t, rng);

    println!("=== Public Key ===");
    println!("h(X) degree: {}", pk.h_poly.degree());
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
    println!("\n=== Multiplicities (eq. 7, t={t}) ===");
    let mults = pk.compute_multiplicities(&row_indices, &col_indices);
    for (j, &mj) in mults.iter().enumerate() {
        println!("  m_{j} = {mj}  (h(ω^{j}) = Δ^{j} appears {mj} times)");
    }
    let total: u64 = mults.iter().sum();
    println!("  Total = {total}  (should be 4·m = {})", 4 * m);
    assert_eq!(total, (4 * m) as u64);

    // -----------------------------------------------------------------------
    // Prover: Round 1 + Round 2
    // beta and eta are the Fiat-Shamir challenges (hardcoded for this toy example).
    // -----------------------------------------------------------------------
    println!("\n=== Prover ===");
    let proof = prove(&pk, &row_indices, &col_indices);

    println!("Round 1 commitments: R, C, m, S, row̃ ✓");
    println!("Round 2 commitments: F₁, F₂, F₃, F₄, F₅ ({} polys) ✓", proof.f_comms.len());

    // -----------------------------------------------------------------------
    // Verifier (stub — full verification is TODO)
    // -----------------------------------------------------------------------
    println!("\n=== Verifier (stub) ===");
    let valid = verify(&pk, &proof);
    println!("Verify: {valid}  (stub — rounds 3–5 not yet implemented)");
    assert!(valid);

    println!("\nPFR rounds 1–2 complete ✓");
}
