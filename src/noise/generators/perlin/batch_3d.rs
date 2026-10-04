use simply_simd::{Arch, Simd, enable_targets};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Perlin;
use crate::noise::util::constants::{BYTE_SHUFFLE, GRAD_TABLE, HASH_PRIME, X_GRAD_ENCODING, Y_GRAD_ENCODING, Z_GRAD_ENCODING};

#[enable_targets(A)]
impl BatchGenerator<3> for Perlin {
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 3],
        freq: [Simd<f32, A>; 3],
    ) -> Simd<f32, A> {
        // Constants.
        let six: Simd<f32, A> = Simd::splat(6.0);
        let ten: Simd<f32, A> = Simd::splat(10.0);
        let fifteen: Simd<f32, A> = Simd::splat(15.0);
        let one: Simd<f32, A> = Simd::splat(1.0);
        let three_int: Simd<u32, A> = Simd::splat(3);

        let x_encoding: Simd<u32, A> = Simd::splat(X_GRAD_ENCODING);
        let y_encoding: Simd<u32, A> = Simd::splat(Y_GRAD_ENCODING);
        let z_encoding: Simd<u32, A> = Simd::splat(Z_GRAD_ENCODING);

        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(HASH_PRIME);

        // Scale: 3
        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];
        let z_scaled = input[2] * freq[2];

        // Gridpoints and distances: 12
        let x_scaled_floored = x_scaled.floor();
        let y_scaled_floored = y_scaled.floor();
        let z_scaled_floored = z_scaled.floor();

        let x_grid_lo = x_scaled_floored.cast_int_trunc();
        let y_grid_lo = y_scaled_floored.cast_int_trunc();
        let z_grid_lo = z_scaled_floored.cast_int_trunc();

        let x_dist_lo = x_scaled - x_scaled_floored;
        let y_dist_lo = y_scaled - y_scaled_floored;
        let z_dist_lo = z_scaled - z_scaled_floored;
        let x_dist_hi = x_dist_lo - one;
        let y_dist_hi = y_dist_lo - one;
        let z_dist_hi = z_dist_lo - one;

        // Lerp fade calculation: 15
        let t = x_dist_lo;
        let s = y_dist_lo;
        let u = z_dist_lo;
        let x_lerp = t * t * t * t.mul_add(t.mul_sub(six, fifteen), ten);
        let y_lerp = s * s * s * s.mul_add(s.mul_sub(six, fifteen), ten);
        let z_lerp = u * u * u * u.mul_add(u.mul_sub(six, fifteen), ten);

        // Hash: 26
        let x1: Simd<u32, A> = x_grid_lo.raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid_lo.raw_cast() * channel_seed;
        let z1: Simd<u32, A> = z_grid_lo.raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;
        let z2 = z1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let z1_shuf = z1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;
        let z2_shuf = z2.permute_8(shuffle_indices) ^ prime;

        let mix_tlf = x1_shuf * y1_shuf * z1_shuf;
        let mix_trf = x1_shuf * y1_shuf * z2_shuf;
        let mix_blf = x1_shuf * y2_shuf * z1_shuf;
        let mix_brf = x1_shuf * y2_shuf * z2_shuf;
        let mix_tlb = x2_shuf * y1_shuf * z1_shuf;
        let mix_trb = x2_shuf * y1_shuf * z2_shuf;
        let mix_blb = x2_shuf * y2_shuf * z1_shuf;
        let mix_brb = x2_shuf * y2_shuf * z2_shuf;

        // Products: 88
        let indices_tlf = (mix_tlf >> 28) << 1;
        let indices_trf = (mix_trf >> 28) << 1;
        let indices_blf = (mix_blf >> 28) << 1;
        let indices_brf = (mix_brf >> 28) << 1;
        let indices_tlb = (mix_tlb >> 28) << 1;
        let indices_trb = (mix_trb >> 28) << 1;
        let indices_blb = (mix_blb >> 28) << 1;
        let indices_brb = (mix_brb >> 28) << 1;

        let x_grads_tlf = ((x_encoding >> indices_tlf) & three_int).gather(&GRAD_TABLE);
        let x_grads_trf = ((x_encoding >> indices_trf) & three_int).gather(&GRAD_TABLE);
        let x_grads_blf = ((x_encoding >> indices_blf) & three_int).gather(&GRAD_TABLE);
        let x_grads_brf = ((x_encoding >> indices_brf) & three_int).gather(&GRAD_TABLE);
        let x_grads_tlb = ((x_encoding >> indices_tlb) & three_int).gather(&GRAD_TABLE);
        let x_grads_trb = ((x_encoding >> indices_trb) & three_int).gather(&GRAD_TABLE);
        let x_grads_blb = ((x_encoding >> indices_blb) & three_int).gather(&GRAD_TABLE);
        let x_grads_brb = ((x_encoding >> indices_brb) & three_int).gather(&GRAD_TABLE);
        let y_grads_tlf = ((y_encoding >> indices_tlf) & three_int).gather(&GRAD_TABLE);
        let y_grads_trf = ((y_encoding >> indices_trf) & three_int).gather(&GRAD_TABLE);
        let y_grads_blf = ((y_encoding >> indices_blf) & three_int).gather(&GRAD_TABLE);
        let y_grads_brf = ((y_encoding >> indices_brf) & three_int).gather(&GRAD_TABLE);
        let y_grads_tlb = ((y_encoding >> indices_tlb) & three_int).gather(&GRAD_TABLE);
        let y_grads_trb = ((y_encoding >> indices_trb) & three_int).gather(&GRAD_TABLE);
        let y_grads_blb = ((y_encoding >> indices_blb) & three_int).gather(&GRAD_TABLE);
        let y_grads_brb = ((y_encoding >> indices_brb) & three_int).gather(&GRAD_TABLE);
        let z_grads_tlf = ((z_encoding >> indices_tlf) & three_int).gather(&GRAD_TABLE);
        let z_grads_trf = ((z_encoding >> indices_trf) & three_int).gather(&GRAD_TABLE);
        let z_grads_blf = ((z_encoding >> indices_blf) & three_int).gather(&GRAD_TABLE);
        let z_grads_brf = ((z_encoding >> indices_brf) & three_int).gather(&GRAD_TABLE);
        let z_grads_tlb = ((z_encoding >> indices_tlb) & three_int).gather(&GRAD_TABLE);
        let z_grads_trb = ((z_encoding >> indices_trb) & three_int).gather(&GRAD_TABLE);
        let z_grads_blb = ((z_encoding >> indices_blb) & three_int).gather(&GRAD_TABLE);
        let z_grads_brb = ((z_encoding >> indices_brb) & three_int).gather(&GRAD_TABLE);

        // Interpolation: 38
        let prod_tlf = x_grads_tlf.mul_add(
            x_dist_lo,
            y_grads_tlf.mul_add(y_dist_lo, z_grads_tlf * z_dist_lo),
        );
        let prod_trf = x_grads_trf.mul_add(
            x_dist_lo,
            y_grads_trf.mul_add(y_dist_lo, z_grads_trf * z_dist_hi),
        );
        let prod_blf = x_grads_blf.mul_add(
            x_dist_lo,
            y_grads_blf.mul_add(y_dist_hi, z_grads_blf * z_dist_lo),
        );
        let prod_brf = x_grads_brf.mul_add(
            x_dist_lo,
            y_grads_brf.mul_add(y_dist_hi, z_grads_brf * z_dist_hi),
        );
        let prod_tlb = x_grads_tlb.mul_add(
            x_dist_hi,
            y_grads_tlb.mul_add(y_dist_lo, z_grads_tlb * z_dist_lo),
        );
        let prod_trb = x_grads_trb.mul_add(
            x_dist_hi,
            y_grads_trb.mul_add(y_dist_lo, z_grads_trb * z_dist_hi),
        );
        let prod_blb = x_grads_blb.mul_add(
            x_dist_hi,
            y_grads_blb.mul_add(y_dist_hi, z_grads_blb * z_dist_lo),
        );
        let prod_brb = x_grads_brb.mul_add(
            x_dist_hi,
            y_grads_brb.mul_add(y_dist_hi, z_grads_brb * z_dist_hi),
        );

        let lerp_tf = z_lerp.mul_add(prod_trf - prod_tlf, prod_tlf);
        let lerp_bf = z_lerp.mul_add(prod_brf - prod_blf, prod_blf);
        let lerp_tb = z_lerp.mul_add(prod_trb - prod_tlb, prod_tlb);
        let lerp_bb = z_lerp.mul_add(prod_brb - prod_blb, prod_blb);

        let lerp_front = y_lerp.mul_add(lerp_bf - lerp_tf, lerp_tf);
        let lerp_back = y_lerp.mul_add(lerp_bb - lerp_tb, lerp_tb);

        x_lerp.mul_add(lerp_back - lerp_front, lerp_front)
    }
}
