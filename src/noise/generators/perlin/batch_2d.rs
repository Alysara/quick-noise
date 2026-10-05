use std::f32::consts::SQRT_2;

use simply_simd::{Arch, Simd, enable_targets};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Perlin;
use crate::noise::util::constants::{BYTE_SHUFFLE, HASH_PRIME};

pub const X_GRADIENTS_2D: [f32; 8] = [
    SQRT_2,
    1.0000000000000000,
    0.0000000000000000,
    -1.0000000000000000,
    -SQRT_2,
    -1.0000000000000000,
    0.0000000000000000,
    1.0000000000000000,
];

pub const Y_GRADIENTS_2D: [f32; 8] = [
    0.0000000000000000,
    1.0000000000000000,
    SQRT_2,
    1.0000000000000000,
    0.0000000000000000,
    -1.0000000000000000,
    -SQRT_2,
    -1.0000000000000000,
];

// #[enable_targets(A)]
impl BatchGenerator<2> for Perlin {
    #[inline(always)]
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 2],
        freq: [Simd<f32, A>; 2],
    ) -> Simd<f32, A> {
        // Constants.
        let six: Simd<f32, A> = Simd::splat(6.0);
        let ten: Simd<f32, A> = Simd::splat(10.0);
        let fifteen: Simd<f32, A> = Simd::splat(15.0);
        let one: Simd<f32, A> = Simd::splat(1.0);

        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(HASH_PRIME);

        // Scale: 2
        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];

        // Gridpoints and distances: 8
        let x_scaled_floored = x_scaled.floor();
        let y_scaled_floored = y_scaled.floor();

        let x_grid_lo = x_scaled_floored.cast_int_trunc();
        let y_grid_lo = y_scaled_floored.cast_int_trunc();

        let x_dist_lo = x_scaled - x_scaled_floored;
        let y_dist_lo = y_scaled - y_scaled_floored;
        let x_dist_hi = x_dist_lo - one;
        let y_dist_hi = y_dist_lo - one;

        // Lerp fade calculation: 10
        let t = x_dist_lo;
        let s = y_dist_lo;
        let x_lerp = t * t * t * t.mul_add(t.mul_sub(six, fifteen), ten);
        let y_lerp = s * s * s * s.mul_add(s.mul_sub(six, fifteen), ten);

        // Hash: 16
        let x1: Simd<u32, A> = x_grid_lo.raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid_lo.raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

        let mix_tl = x1_shuf * y1_shuf;
        let mix_tr = x1_shuf * y2_shuf;
        let mix_bl = x2_shuf * y1_shuf;
        let mix_br = x2_shuf * y2_shuf;

        // Permute Gather: 12
        let indices_tl = mix_tl >> 29;
        let indices_tr = mix_tr >> 29;
        let indices_bl = mix_bl >> 29;
        let indices_br = mix_br >> 29;

        let x_grads_tl = indices_tl.gather(&X_GRADIENTS_2D);
        let y_grads_tl = indices_tl.gather(&Y_GRADIENTS_2D);
        let x_grads_tr = indices_tr.gather(&X_GRADIENTS_2D);
        let y_grads_tr = indices_tr.gather(&Y_GRADIENTS_2D);
        let x_grads_bl = indices_bl.gather(&X_GRADIENTS_2D);
        let y_grads_bl = indices_bl.gather(&Y_GRADIENTS_2D);
        let x_grads_br = indices_br.gather(&X_GRADIENTS_2D);
        let y_grads_br = indices_br.gather(&Y_GRADIENTS_2D);

        // Interpolation: 14
        let prod_tl = x_grads_tl.mul_add(x_dist_lo, y_grads_tl * y_dist_lo);
        let prod_tr = x_grads_tr.mul_add(x_dist_lo, y_grads_tr * y_dist_hi);
        let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);

        let prod_bl = x_grads_bl.mul_add(x_dist_hi, y_grads_bl * y_dist_lo);
        let prod_br = x_grads_br.mul_add(x_dist_hi, y_grads_br * y_dist_hi);
        let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

        x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp)
    }
}
