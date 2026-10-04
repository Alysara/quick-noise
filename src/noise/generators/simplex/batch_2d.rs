use std::f32::consts::SQRT_2;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Simplex;

const SQRT_3: f32 = 1.732_050_8;
const SKEW_2D: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

const SCALE: f32 = 80.0;
const SCALED_SQRT: f32 = (SQRT_2 / 2.0) * SCALE;

const A: f32 = SCALE;
const B: f32 = SCALED_SQRT;
const C: f32 = 0.0;
pub const X_GRADIENTS_2D: [f32; 8] = [A, B, C, -B, -A, -B, C, B];
pub const Y_GRADIENTS_2D: [f32; 8] = [C, B, A, B, C, -B, -A, -B];

#[enable_targets(A)]
impl BatchGenerator<2> for Simplex {
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 2],
        freq: [Simd<f32, A>; 2],
    ) -> Simd<f32, A> {
        // Constants.
        let skew: Simd<f32, A> = Simd::splat(SKEW_2D);
        let unskew: Simd<f32, A> = Simd::splat(UNSKEW_2D);
        let subbed_unskew: Simd<f32, A> = Simd::splat(UNSKEW_2D - 1.0);
        let hi_skew_offset: Simd<f32, A> = Simd::splat(2.0 * UNSKEW_2D - 1.0);
        let half: Simd<f32, A> = Simd::splat(0.5);
        let zero: Simd<f32, A> = Simd::splat(0.0);
        let t_hi_coef = Simd::splat(2.0 * SQRT_3 / 3.0);
        let neg_two_thirds = Simd::splat(-2.0 / 3.0);

        // Hash constants.
        const BYTE_SHUFFLE: [u8; 64] = [
            3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8,
            10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2,
            1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13,
        ];

        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(0x85ebca6b_u32);

        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];

        // Gridpoints and distances: 19
        let s = (x_scaled + y_scaled) * skew;
        let x_grid = (x_scaled + s).floor();
        let y_grid = (y_scaled + s).floor();

        let unskew_sub = (x_grid + y_grid) * unskew;
        let x_dist_lo = x_scaled - x_grid + unskew_sub;
        let y_dist_lo = y_scaled - y_grid + unskew_sub;
        let triangle_mask = x_dist_lo.simd_gt(y_dist_lo);

        let x_dist_mi_offset = triangle_mask.select(subbed_unskew, unskew);
        let y_dist_mi_offset = triangle_mask.select(unskew, subbed_unskew);
        let x_dist_mi = x_dist_lo + x_dist_mi_offset;
        let y_dist_mi = y_dist_lo + y_dist_mi_offset;

        let x_dist_hi = x_dist_lo + hi_skew_offset;
        let y_dist_hi = y_dist_lo + hi_skew_offset;

        // Hash: 22
        let x1: Simd<u32, A> = x_grid.cast_int_trunc().raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid.cast_int_trunc().raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

        let mix_lo = (x1_shuf * y1_shuf) ^ x1_shuf;
        let mix_hi = (x2_shuf * y2_shuf) ^ x2_shuf;

        let x_shuf_mi = triangle_mask.raw_cast().select(x2_shuf, x1_shuf);
        let y_shuf_mi = triangle_mask.raw_cast().select(y1_shuf, y2_shuf);
        let mix_mi = (x_shuf_mi * y_shuf_mi) ^ x_shuf_mi;

        // Gradient lookup: 9
        let indices_lo = mix_lo >> 29;
        let indices_mi = mix_mi >> 29;
        let indices_hi = mix_hi >> 29;

        let x_grads_lo = indices_lo.gather(&X_GRADIENTS_2D);
        let y_grads_lo = indices_lo.gather(&Y_GRADIENTS_2D);
        let x_grads_mi = indices_mi.gather(&X_GRADIENTS_2D);
        let y_grads_mi = indices_mi.gather(&Y_GRADIENTS_2D);
        let x_grads_hi = indices_hi.gather(&X_GRADIENTS_2D);
        let y_grads_hi = indices_hi.gather(&Y_GRADIENTS_2D);

        // Sum of products: 27
        let t_lo_pre = half - x_dist_lo.mul_add(x_dist_lo, y_dist_lo * y_dist_lo);
        let t_mi_pre = half - x_dist_mi.mul_add(x_dist_mi, y_dist_mi * y_dist_mi);
        let t_hi_pre = t_lo_pre + t_hi_coef.mul_add(x_dist_lo + y_dist_lo, neg_two_thirds);

        let t_lo = t_lo_pre.max(zero);
        let t_mi = t_mi_pre.max(zero);
        let t_hi = t_hi_pre.max(zero);

        let t2_lo = t_lo * t_lo;
        let t2_mi = t_mi * t_mi;
        let t2_hi = t_hi * t_hi;

        let t4_lo = t2_lo * t2_lo;
        let t4_mi = t2_mi * t2_mi;
        let t4_hi = t2_hi * t2_hi;

        let dot_lo = x_grads_lo.mul_add(x_dist_lo, y_grads_lo * y_dist_lo);
        let dot_mi = x_grads_mi.mul_add(x_dist_mi, y_grads_mi * y_dist_mi);
        let dot_hi = x_grads_hi.mul_add(x_dist_hi, y_grads_hi * y_dist_hi);

        t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "image")]
    use crate::emit::NoiseImageExt;
    #[cfg(feature = "image")]
    use simply_simd::Simd;
    #[cfg(feature = "image")]
    use crate::api::batch::interface::BatchGenerator;
    #[cfg(feature = "image")]
    use crate::api::seed::gen_octave_seed;
    #[cfg(feature = "image")]
    use crate::simd::StaticArch;
    #[cfg(feature = "image")]
    use crate::{ Fbm, Grid, Simplex };

    #[test]
    #[cfg(feature = "image")]
    fn batch_image() {
        let grid = Grid::<2>::new(256, 256).seed(42).sample_position(-128, -128);

        grid.builder::<Fbm, Simplex>()
            .frequency(1.0 / 32.0)
            .into_iter()
            .to_grayscale_image(256, 256, "test_images/batch_2d_simplex_seeded.png");
    }
}
