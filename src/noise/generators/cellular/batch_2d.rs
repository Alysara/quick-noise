use simply_simd::{Arch, Simd, enable_targets};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::Cellular;

#[enable_targets(A)]
impl BatchGenerator<2> for Cellular {
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 2],
        freq: [Simd<f32, A>; 2],
    ) -> Simd<f32, A> {
        // Constants.
        let three_halves: Simd<f32, A> = Simd::splat(1.5);
        let one: Simd<f32, A> = Simd::splat(1.0);

        let hash_mask: Simd<u32, A> = Simd::splat(0x007FFFFF);
        let exp_bits: Simd<u32, A> = Simd::splat(0x3F800000);

        // Hash constants.
        const BYTE_SHUFFLE: [u8; 64] = [
            3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8,
            10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2,
            1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13,
        ];

        let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);
        let channel_seed = Simd::splat(seed);
        let prime = Simd::splat(0x85ebca6b_u32);

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