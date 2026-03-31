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

pub mod counting;
mod prover;
mod types;
mod verifier;

#[cfg(test)]
mod tests;

pub use prover::{commit_statement, prove, round_five, round_four, round_one, round_three, round_two};
pub use prover::{Round1State, Round2State, Round3State, Round4State};
pub use counting::OpCounts;
pub use types::{PfrProof, PfrPublicInputs, PfrPublicKey, PfrStatement};
pub use verifier::verify;
