use crate::types::*;
use ark_ec::{msm::VariableBaseMSM, AffineCurve, PairingEngine, ProjectiveCurve};
use ark_ff::{to_bytes, Field, One, PrimeField, UniformRand};
use ark_poly::{EvaluationDomain, Polynomial};
use ark_marlin::{rng::FiatShamirRng, SimpleHashFiatShamirRng};
use blake2::Blake2s;
use rand_chacha::ChaChaRng;

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
pub fn verify<E: PairingEngine>(
    pk: &PfrPublicKey<E>,
    proof: &PfrProof<E>,
    row_comm: &Comm<E>,
    col_comm: &Comm<E>,
    rowcol_comm: &Comm<E>,
) -> bool {
    // Derive Fiat-Shamir challenges
    let mut fs_rng = SimpleHashFiatShamirRng::<Blake2s, ChaChaRng>::initialize(
        &to_bytes![pk.h_commitment.commitment()].unwrap(),
    );

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
    let beta = E::Fr::rand(&mut fs_rng);

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
    let eta = E::Fr::rand(&mut fs_rng);

    fs_rng.absorb(&to_bytes![proof.r_star_comm.commitment(), proof.q_comm.commitment()].unwrap());
    let alpha = E::Fr::rand(&mut fs_rng);

    fs_rng
        .absorb(&to_bytes![proof.h_alpha, proof.r_alpha, proof.c_alpha, proof.row_alpha].unwrap());
    let delta = E::Fr::rand(&mut fs_rng);

    let big_delta = pk.big_delta();
    let big_delta_t = big_delta.pow([pk.t as u64]);

    let h_alpha = proof.h_alpha;
    let r_alpha = proof.r_alpha;
    let c_alpha = proof.c_alpha;
    let row_alpha = proof.row_alpha;

    // U(α) = α³ − 1,  z_K(α)
    // z_{K\H}(α) = (n/m)·(α^m−1)/(α^n−1) — same normalized polynomial as in round_three
    let u_at_alpha: E::Fr = alpha * alpha * alpha - E::Fr::one();
    let zk_at_alpha: E::Fr = pk.k_domain.vanishing_polynomial().evaluate(&alpha);
    let zh_at_alpha: E::Fr = pk.h_domain.vanishing_polynomial().evaluate(&alpha);
    let nm_ratio: E::Fr = E::Fr::from(pk.n as u64) * E::Fr::from(pk.m as u64).inverse().unwrap();
    let zkh_at_alpha: E::Fr = nm_ratio * zk_at_alpha * zh_at_alpha.inverse().unwrap();

    // η⁹: the last η power used in big_lin (η⁰ through η⁹, skipping η⁸)
    let eta9 = eta.pow([9u64]);

    // -----------------------------------------------------------------------
    // Build [Lin(τ)]₁ as a MSM over proof commitments.
    // Mirrors big_lin in round_five, but operating on group elements.
    // Constant-polynomial terms  −c  become  −c·[1]₁.
    // -----------------------------------------------------------------------
    let g1 = pk.vk.vk.g; // [1]₁

    let mut bases: Vec<E::G1Affine> = Vec::new();
    let mut scalars: Vec<E::Fr> = Vec::new();

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
    let mut eta_pow = E::Fr::one(); // η⁰

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
    add_comm!(E::Fr::one(), &pk.h_commitment);
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

    let lhs = E::pairing(y, h);
    let rhs = E::pairing(q_aff, tau_minus_alpha_h);

    lhs == rhs
}
