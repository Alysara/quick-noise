use simply_simd::{Arch, Simd, enable_targets};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Simplex;
use crate::noise::util::constants::{
    BYTE_SHUFFLE, X_GRAD_ENCODING, Y_GRAD_ENCODING, Z_GRAD_ENCODING,
};

const SKEW_3D: f32 = 1.0 / 3.0;
const UNSKEW_3D: f32 = 1.0 / 6.0;

impl BatchGenerator<3> for Simplex {
    #[inline(always)]
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 3],
        freq: [Simd<f32, A>; 3],
    ) -> Simd<f32, A> {
        // Constants.
        let skew: Simd<f32, A> = Simd::splat(SKEW_3D);
        let unskew: Simd<f32, A> = Simd::splat(UNSKEW_3D);
        let subbed_unskew: Simd<f32, A> = Simd::splat(UNSKEW_3D - 1.0);
        let hi_skew_offset: Simd<f32, A> = Simd::splat(3.0 * UNSKEW_3D - 1.0);
        let two_unskew: Simd<f32, A> = Simd::splat(2.0 * UNSKEW_3D);
        let mi2_skew_offset: Simd<f32, A> = Simd::splat(2.0 * UNSKEW_3D - 1.0);
        let half: Simd<f32, A> = Simd::splat(0.5);
        let zero: Simd<f32, A> = Simd::splat(0.0);
        let three_int: Simd<u32, A> = Simd::splat(3);

        let x_encoding: Simd<u32, A> = Simd::splat(X_GRAD_ENCODING);
        let y_encoding: Simd<u32, A> = Simd::splat(Y_GRAD_ENCODING);
        let z_encoding: Simd<u32, A> = Simd::splat(Z_GRAD_ENCODING);

        // Hash constants.
        // TODO: Figure out what this needs to be.
        const S: f32 = 100.0;
        const GRAD_TABLE: [f32; 4] = [0.0, S, -S, 0.0];

        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(0x85ebca6b_u32);

        // Scale: 3
        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];
        let z_scaled = input[2] * freq[2];

        // Gridpoints and distances: 39
        let s = (x_scaled + y_scaled + z_scaled) * skew;
        let x_grid = (x_scaled + s).floor();
        let y_grid = (y_scaled + s).floor();
        let z_grid = (z_scaled + s).floor();

        let unskew_sub = (x_grid + y_grid + z_grid) * unskew;
        let x_dist_lo = x_scaled - x_grid + unskew_sub;
        let y_dist_lo = y_scaled - y_grid + unskew_sub;
        let z_dist_lo = z_scaled - z_grid + unskew_sub;

        let x_gt_y = x_dist_lo.simd_gt(y_dist_lo);
        let x_gt_z = x_dist_lo.simd_gt(z_dist_lo);
        let ny_gt_z = y_dist_lo.simd_le(z_dist_lo);

        let nx_gt_y = x_dist_lo.simd_le(y_dist_lo);
        let nx_gt_z = x_dist_lo.simd_le(z_dist_lo);
        let y_gt_z = y_dist_lo.simd_gt(z_dist_lo);

        let i1 = x_gt_y & x_gt_z;
        let j1 = nx_gt_y & y_gt_z;
        let k1 = nx_gt_z & ny_gt_z;

        let i2 = x_gt_y | x_gt_z;
        let j2 = nx_gt_y | y_gt_z;
        let k2 = nx_gt_z | ny_gt_z;

        let x_dist_mi1 = x_dist_lo + i1.select(subbed_unskew, unskew);
        let y_dist_mi1 = y_dist_lo + j1.select(subbed_unskew, unskew);
        let z_dist_mi1 = z_dist_lo + k1.select(subbed_unskew, unskew);

        let x_dist_mi2 = x_dist_lo + i2.select(mi2_skew_offset, two_unskew);
        let y_dist_mi2 = y_dist_lo + j2.select(mi2_skew_offset, two_unskew);
        let z_dist_mi2 = z_dist_lo + k2.select(mi2_skew_offset, two_unskew);

        let x_dist_hi = x_dist_lo + hi_skew_offset;
        let y_dist_hi = y_dist_lo + hi_skew_offset;
        let z_dist_hi = z_dist_lo + hi_skew_offset;

        // Hash: 35
        let x1: Simd<u32, A> = x_grid.cast_int_trunc().raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid.cast_int_trunc().raw_cast() * channel_seed;
        let z1: Simd<u32, A> = z_grid.cast_int_trunc().raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;
        let z2 = z1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let z1_shuf = z1.permute_8(shuffle_indices) ^ prime;

        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;
        let z2_shuf = z2.permute_8(shuffle_indices) ^ prime;

        let x_mi1_shuf = i1.raw_cast().select(x2_shuf, x1_shuf);
        let y_mi1_shuf = j1.raw_cast().select(y2_shuf, y1_shuf);
        let z_mi1_shuf = k1.raw_cast().select(z2_shuf, z1_shuf);

        let x_mi2_shuf = i2.raw_cast().select(x2_shuf, x1_shuf);
        let y_mi2_shuf = j2.raw_cast().select(y2_shuf, y1_shuf);
        let z_mi2_shuf = k2.raw_cast().select(z2_shuf, z1_shuf);

        let mix_lo = x1_shuf * y1_shuf * z1_shuf;
        let mix_hi = x2_shuf * y2_shuf * z2_shuf;
        let mix_mi1 = x_mi1_shuf * y_mi1_shuf * z_mi1_shuf;
        let mix_mi2 = x_mi2_shuf * y_mi2_shuf * z_mi2_shuf;

        // Gradient lookup: 44
        let indices_lo = (mix_lo >> 28) << 1;
        let indices_mi1 = (mix_mi1 >> 28) << 1;
        let indices_mi2 = (mix_mi2 >> 28) << 1;
        let indices_hi = (mix_hi >> 28) << 1;

        let x_grads_lo = ((x_encoding >> indices_lo) & three_int).gather(&GRAD_TABLE);
        let y_grads_lo = ((y_encoding >> indices_lo) & three_int).gather(&GRAD_TABLE);
        let z_grads_lo = ((z_encoding >> indices_lo) & three_int).gather(&GRAD_TABLE);
        let x_grads_mi1 = ((x_encoding >> indices_mi1) & three_int).gather(&GRAD_TABLE);
        let y_grads_mi1 = ((y_encoding >> indices_mi1) & three_int).gather(&GRAD_TABLE);
        let z_grads_mi1 = ((z_encoding >> indices_mi1) & three_int).gather(&GRAD_TABLE);
        let x_grads_mi2 = ((x_encoding >> indices_mi2) & three_int).gather(&GRAD_TABLE);
        let y_grads_mi2 = ((y_encoding >> indices_mi2) & three_int).gather(&GRAD_TABLE);
        let z_grads_mi2 = ((z_encoding >> indices_mi2) & three_int).gather(&GRAD_TABLE);
        let x_grads_hi = ((x_encoding >> indices_hi) & three_int).gather(&GRAD_TABLE);
        let y_grads_hi = ((y_encoding >> indices_hi) & three_int).gather(&GRAD_TABLE);
        let z_grads_hi = ((z_encoding >> indices_hi) & three_int).gather(&GRAD_TABLE);

        // Sum of products: 44
        let t_lo = (half
            - x_dist_lo.mul_add(
                x_dist_lo,
                y_dist_lo.mul_add(y_dist_lo, z_dist_lo * z_dist_lo),
            ))
        .max(zero);
        let t_mi1 = (half
            - x_dist_mi1.mul_add(
                x_dist_mi1,
                y_dist_mi1.mul_add(y_dist_mi1, z_dist_mi1 * z_dist_mi1),
            ))
        .max(zero);
        let t_mi2 = (half
            - x_dist_mi2.mul_add(
                x_dist_mi2,
                y_dist_mi2.mul_add(y_dist_mi2, z_dist_mi2 * z_dist_mi2),
            ))
        .max(zero);
        let t_hi = (half
            - x_dist_hi.mul_add(
                x_dist_hi,
                y_dist_hi.mul_add(y_dist_hi, z_dist_hi * z_dist_hi),
            ))
        .max(zero);

        let t2_lo = t_lo * t_lo;
        let t2_mi1 = t_mi1 * t_mi1;
        let t2_mi2 = t_mi2 * t_mi2;
        let t2_hi = t_hi * t_hi;

        let t4_lo = t2_lo * t2_lo;
        let t4_mi1 = t2_mi1 * t2_mi1;
        let t4_mi2 = t2_mi2 * t2_mi2;
        let t4_hi = t2_hi * t2_hi;

        let dot_lo = x_grads_lo.mul_add(
            x_dist_lo,
            y_grads_lo.mul_add(y_dist_lo, z_dist_lo * z_grads_lo),
        );
        let dot_mi1 = x_grads_mi1.mul_add(
            x_dist_mi1,
            y_grads_mi1.mul_add(y_dist_mi1, z_dist_mi1 * z_grads_mi1),
        );
        let dot_mi2 = x_grads_mi2.mul_add(
            x_dist_mi2,
            y_grads_mi2.mul_add(y_dist_mi2, z_dist_mi2 * z_grads_mi2),
        );
        let dot_hi = x_grads_hi.mul_add(
            x_dist_hi,
            y_grads_hi.mul_add(y_dist_hi, z_dist_hi * z_grads_hi),
        );

        t4_lo.mul_add(
            dot_lo,
            t4_mi1.mul_add(dot_mi1, t4_mi2.mul_add(dot_mi2, t4_hi * dot_hi)),
        )
    }
}
