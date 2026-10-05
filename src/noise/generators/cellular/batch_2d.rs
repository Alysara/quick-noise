use simply_simd::{Arch, Simd};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Cellular;
use crate::noise::util::constants::{BYTE_SHUFFLE, CELLULAR_EXP_MASK, HASH_MASK, HASH_PRIME};

impl BatchGenerator<2> for Cellular {
    #[inline(always)]
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 2],
        freq: [Simd<f32, A>; 2],
    ) -> Simd<f32, A> {
        // Constants.
        let three_halves: Simd<f32, A> = Simd::splat(1.5);
        let one: Simd<f32, A> = Simd::splat(1.0);

        let hash_mask: Simd<u32, A> = Simd::splat(HASH_MASK);
        let exp_bits: Simd<u32, A> = Simd::splat(CELLULAR_EXP_MASK);

        // Hash constants.
        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(HASH_PRIME);

        // Scale: 2
        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];

        // Gridpoints and distances: 8
        let x_grid_lo = x_scaled.floor();
        let y_grid_lo = y_scaled.floor();

        let x_dist_lo = x_scaled - x_grid_lo - three_halves;
        let y_dist_lo = y_scaled - y_grid_lo - three_halves;
        let x_dist_hi = one - x_dist_lo;
        let y_dist_hi = one - y_dist_lo;

        // Threshold: 6
        let close_edge_lo = x_dist_lo.min(y_dist_lo) + Simd::splat(2.0);
        let close_edge_hi = x_dist_hi.min(y_dist_hi) - one;
        let closest_edge_dist = close_edge_lo.min(close_edge_hi);
        let threshold = closest_edge_dist * closest_edge_dist;

        // Hash: 22
        let x1: Simd<u32, A> = x_grid_lo.cast_int_trunc().raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid_lo.cast_int_trunc().raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

        let hash_tl = (x1_shuf * y1_shuf) ^ x1_shuf;
        let hash_tr = (x1_shuf * y2_shuf) ^ x1_shuf;
        let hash_bl = (x2_shuf * y1_shuf) ^ x2_shuf;
        let hash_br = (x2_shuf * y2_shuf) ^ x2_shuf;

        // Distance Calc: 35
        let x_dist_tl = ((hash_tl & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_tr = ((hash_tr & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_bl = ((hash_bl & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
        let x_dist_br = ((hash_br & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;

        let y_dist_tl = ((hash_tl >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_tr = ((hash_tr >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
        let y_dist_bl = ((hash_bl >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_br = ((hash_br >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;

        let dist_tl = x_dist_tl.mul_add(x_dist_tl, y_dist_tl * y_dist_tl);
        let dist_tr = x_dist_tr.mul_add(x_dist_tr, y_dist_tr * y_dist_tr);
        let dist_bl = x_dist_bl.mul_add(x_dist_bl, y_dist_bl * y_dist_bl);
        let dist_br = x_dist_br.mul_add(x_dist_br, y_dist_br * y_dist_br);

        let mut min_dist = dist_tl.min(dist_tr).min(dist_bl).min(dist_br);

        // Branch: 2 + Branch prediction.
        let is_far = min_dist.simd_gt(threshold);

        // Only triggers 2-3% of the time. -25% Throughput despite this.
        if !is_far.all_false() {
            let x0 = x1 - channel_seed;
            let y0 = y1 - channel_seed;
            let x3 = x2 + channel_seed;
            let y3 = y2 + channel_seed;

            let x0_shuf = x0.permute_8(shuffle_indices) ^ prime;
            let y0_shuf = y0.permute_8(shuffle_indices) ^ prime;
            let x3_shuf = x3.permute_8(shuffle_indices) ^ prime;
            let y3_shuf = y3.permute_8(shuffle_indices) ^ prime;

            let hash_ttl = (x0_shuf * y1_shuf) ^ x0_shuf;
            let hash_tll = (x1_shuf * y0_shuf) ^ x1_shuf;
            let hash_ttr = (x0_shuf * y2_shuf) ^ x0_shuf;
            let hash_trr = (x1_shuf * y3_shuf) ^ x1_shuf;

            let hash_bbl = (x3_shuf * y1_shuf) ^ x3_shuf;
            let hash_bll = (x2_shuf * y0_shuf) ^ x2_shuf;
            let hash_bbr = (x3_shuf * y2_shuf) ^ x3_shuf;
            let hash_brr = (x2_shuf * y3_shuf) ^ x2_shuf;

            let x_dist_ttl =
                ((hash_ttl & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_tll = ((hash_tll & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_ttr =
                ((hash_ttr & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_trr = ((hash_trr & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_bbl =
                ((hash_bbl & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_bll = ((hash_bll & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_bbr =
                ((hash_bbr & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_brr = ((hash_brr & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;

            let y_dist_ttl = ((hash_ttl >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_tll = ((hash_tll >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_ttr = ((hash_ttr >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_trr = ((hash_trr >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;
            let y_dist_bbl = ((hash_bbl >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_bll = ((hash_bll >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_bbr = ((hash_bbr >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_brr = ((hash_brr >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;

            let dist_ttl = x_dist_ttl.mul_add(x_dist_ttl, y_dist_ttl * y_dist_ttl);
            let dist_tll = x_dist_tll.mul_add(x_dist_tll, y_dist_tll * y_dist_tll);
            let dist_ttr = x_dist_ttr.mul_add(x_dist_ttr, y_dist_ttr * y_dist_ttr);
            let dist_trr = x_dist_trr.mul_add(x_dist_trr, y_dist_trr * y_dist_trr);
            let dist_bbl = x_dist_bbl.mul_add(x_dist_bbl, y_dist_bbl * y_dist_bbl);
            let dist_bll = x_dist_bll.mul_add(x_dist_bll, y_dist_bll * y_dist_bll);
            let dist_bbr = x_dist_bbr.mul_add(x_dist_bbr, y_dist_bbr * y_dist_bbr);
            let dist_brr = x_dist_brr.mul_add(x_dist_brr, y_dist_brr * y_dist_brr);

            let outer_min = dist_ttl
                .min(dist_tll)
                .min(dist_ttr)
                .min(dist_trr)
                .min(dist_bbl)
                .min(dist_bll)
                .min(dist_bbr)
                .min(dist_brr);

            min_dist = min_dist.min(outer_min);
        }

        // Sqrt: 1
        min_dist.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::seed::gen_octave_seed;
    use crate::math::random::Random;
    use crate::noise::generators::cellular::grid_2d::{hash_cell, split_hash};
    use crate::simd::StaticArch;

    #[test]
    fn cellular_batch_2d_reference() {
        const W: usize = 64;
        const H: usize = 64;
        const FREQ: f32 = 1.0 / 32.0;
        let seed = 123456789i64;

        let grid_seed = Random::mix_u64(seed as u64);
        let base_seed = Random::mix_u64_pair(grid_seed, 0xD5E7B3C94F8A1E6B);
        let octave_seed = gen_octave_seed([FREQ, FREQ], base_seed);

        let max_diff = compare_batch(octave_seed, FREQ, W, H, (-5, 3));
        assert!(
            max_diff < 1e-4,
            "Batch cellular diverges from the brute-force Cellular by {max_diff}"
        );

        let max_diff = compare_batch(octave_seed, FREQ, 32, 96, (-5, 3));
        assert!(
            max_diff < 1e-4,
            "Tall batch cellular diverges from the brute-force Cellular by {max_diff}"
        );
    }

    fn compare_batch(octave_seed: u32, freq: f32, w: usize, h: usize, pos: (i32, i32)) -> f32 {
        let lanes = Simd::<f32, StaticArch>::LANES;

        // Same sample coordinates as the grid: `position + index` scaled by freq.
        let xs: Vec<f32> = (0..w * h).map(|i| pos.0 as f32 + (i % w) as f32).collect();
        let ys: Vec<f32> = (0..w * h).map(|i| pos.1 as f32 + (i / w) as f32).collect();

        let mut max_diff = 0.0f32;
        for (block, (x_block, y_block)) in xs
            .chunks_exact(lanes)
            .zip(ys.chunks_exact(lanes))
            .enumerate()
        {
            let input = [
                Simd::<f32, StaticArch>::from_slice(x_block),
                Simd::<f32, StaticArch>::from_slice(y_block),
            ];
            let freq_simd = [Simd::<f32, StaticArch>::splat(freq); 2];

            let actual =
                Cellular::sample_batch::<StaticArch>(octave_seed, input, freq_simd).to_array();

            for (lane, actual) in actual.iter().enumerate().take(lanes) {
                let i = block * lanes + lane;
                let px = xs[i] * freq;
                let py = ys[i] * freq;
                let reference = reference_cellular(octave_seed, px, py);
                max_diff = max_diff.max((actual - reference).abs());
            }
        }
        max_diff
    }

    fn reference_cellular(seed: u32, px: f32, py: f32) -> f32 {
        let cell_x = px.floor() as i32;
        let cell_y = py.floor() as i32;
        let sx = px - px.floor();
        let sy = py - py.floor();

        let mut min_dist = f32::MAX;
        for ox in -3..=3 {
            for oy in -3..=3 {
                let (jx, jy) = split_hash(hash_cell::<StaticArch>(
                    (cell_x + ox) as u32,
                    (cell_y + oy) as u32,
                    seed,
                ));
                let dx = sx - (ox as f32 + jx);
                let dy = sy - (oy as f32 + jy);
                min_dist = min_dist.min(dx * dx + dy * dy);
            }
        }
        min_dist.sqrt()
    }
}
