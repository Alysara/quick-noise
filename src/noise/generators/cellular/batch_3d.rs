use simply_simd::{Arch, Simd};

use crate::api::batch::interface::BatchGenerator;
use crate::noise::generators::{Cellular, Euclidean};
use crate::noise::util::constants::{BYTE_SHUFFLE, CELLULAR_EXP_MASK, HASH_MASK, HASH_PRIME};

impl BatchGenerator<3> for Cellular<Euclidean> {
    #[inline(always)]
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; 3],
        freq: [Simd<f32, A>; 3],
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

        // Scale: 3
        let x_scaled = input[0] * freq[0];
        let y_scaled = input[1] * freq[1];
        let z_scaled = input[2] * freq[2];

        // Gridpoints and distances: 12
        let x_grid_lo = x_scaled.floor();
        let y_grid_lo = y_scaled.floor();
        let z_grid_lo = z_scaled.floor();

        let x_dist_lo = x_scaled - x_grid_lo - three_halves;
        let y_dist_lo = y_scaled - y_grid_lo - three_halves;
        let z_dist_lo = z_scaled - z_grid_lo - three_halves;
        let x_dist_hi = one - x_dist_lo;
        let y_dist_hi = one - y_dist_lo;
        let z_dist_hi = one - z_dist_lo;

        // Threshold: 8
        let close_edge_lo = x_dist_lo.min(y_dist_lo).min(z_dist_lo) + Simd::splat(2.0);
        let close_edge_hi = x_dist_hi.min(y_dist_hi).min(z_dist_hi) - one;
        let closest_edge_dist = close_edge_lo.min(close_edge_hi);
        let threshold = closest_edge_dist * closest_edge_dist;

        // Hash: 37
        let x1: Simd<u32, A> = x_grid_lo.cast_int_trunc().raw_cast() * channel_seed;
        let y1: Simd<u32, A> = y_grid_lo.cast_int_trunc().raw_cast() * channel_seed;
        let z1: Simd<u32, A> = z_grid_lo.cast_int_trunc().raw_cast() * channel_seed;
        let x2 = x1 + channel_seed;
        let y2 = y1 + channel_seed;
        let z2 = z1 + channel_seed;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let z1_shuf = z1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;
        let z2_shuf = z2.permute_8(shuffle_indices) ^ prime;

        let hash_tlf = x1_shuf * y1_shuf * z1_shuf;
        let hash_trf = x1_shuf * y1_shuf * z2_shuf;
        let hash_blf = x1_shuf * y2_shuf * z1_shuf;
        let hash_brf = x1_shuf * y2_shuf * z2_shuf;
        let hash_tlb = x2_shuf * y1_shuf * z1_shuf;
        let hash_trb = x2_shuf * y1_shuf * z2_shuf;
        let hash_blb = x2_shuf * y2_shuf * z1_shuf;
        let hash_brb = x2_shuf * y2_shuf * z2_shuf;

        // Distance Calc: 111
        let x_dist_tlf = ((hash_tlf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_trf = ((hash_trf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_blf = ((hash_blf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_brf = ((hash_brf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
        let x_dist_tlb = ((hash_tlb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
        let x_dist_trb = ((hash_trb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
        let x_dist_blb = ((hash_blb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
        let x_dist_brb = ((hash_brb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;

        let y_dist_tlf = ((hash_tlf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_trf = ((hash_trf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_blf = ((hash_blf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
        let y_dist_brf = ((hash_brf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
        let y_dist_tlb = ((hash_tlb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_trb = ((hash_trb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
        let y_dist_blb = ((hash_blb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
        let y_dist_brb = ((hash_brb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;

        let z_dist_tlf = (((hash_tlf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
        let z_dist_trf = (((hash_trf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
        let z_dist_blf = (((hash_blf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
        let z_dist_brf = (((hash_brf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
        let z_dist_tlb = (((hash_tlb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
        let z_dist_trb = (((hash_trb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
        let z_dist_blb = (((hash_blb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
        let z_dist_brb = (((hash_brb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;

        let dist_tlf = x_dist_tlf.mul_add(
            x_dist_tlf,
            y_dist_tlf.mul_add(y_dist_tlf, z_dist_tlf * z_dist_tlf),
        );
        let dist_trf = x_dist_trf.mul_add(
            x_dist_trf,
            y_dist_trf.mul_add(y_dist_trf, z_dist_trf * z_dist_trf),
        );
        let dist_blf = x_dist_blf.mul_add(
            x_dist_blf,
            y_dist_blf.mul_add(y_dist_blf, z_dist_blf * z_dist_blf),
        );
        let dist_brf = x_dist_brf.mul_add(
            x_dist_brf,
            y_dist_brf.mul_add(y_dist_brf, z_dist_brf * z_dist_brf),
        );
        let dist_tlb = x_dist_tlb.mul_add(
            x_dist_tlb,
            y_dist_tlb.mul_add(y_dist_tlb, z_dist_tlb * z_dist_tlb),
        );
        let dist_trb = x_dist_trb.mul_add(
            x_dist_trb,
            y_dist_trb.mul_add(y_dist_trb, z_dist_trb * z_dist_trb),
        );
        let dist_blb = x_dist_blb.mul_add(
            x_dist_blb,
            y_dist_blb.mul_add(y_dist_blb, z_dist_blb * z_dist_blb),
        );
        let dist_brb = x_dist_brb.mul_add(
            x_dist_brb,
            y_dist_brb.mul_add(y_dist_brb, z_dist_brb * z_dist_brb),
        );

        let mut min_dist = dist_tlf
            .min(dist_trf)
            .min(dist_blf)
            .min(dist_brf)
            .min(dist_tlb)
            .min(dist_trb)
            .min(dist_blb)
            .min(dist_brb);

        // Branch: 2 + Branch prediction.
        let is_far = min_dist.simd_gt(threshold);

        if !is_far.all_false() {
            let x0 = x1 - channel_seed;
            let y0 = y1 - channel_seed;
            let z0 = z1 - channel_seed;
            let x3 = x2 + channel_seed;
            let y3 = y2 + channel_seed;
            let z3 = z2 + channel_seed;

            let x0_shuf = x0.permute_8(shuffle_indices) ^ prime;
            let y0_shuf = y0.permute_8(shuffle_indices) ^ prime;
            let z0_shuf = z0.permute_8(shuffle_indices) ^ prime;
            let x3_shuf = x3.permute_8(shuffle_indices) ^ prime;
            let y3_shuf = y3.permute_8(shuffle_indices) ^ prime;
            let z3_shuf = z3.permute_8(shuffle_indices) ^ prime;

            let hash_tlff = x0_shuf * y1_shuf * z1_shuf;
            let hash_ttlf = x1_shuf * y0_shuf * z1_shuf;
            let hash_tllf = x1_shuf * y1_shuf * z0_shuf;
            let hash_trff = x0_shuf * y1_shuf * z2_shuf;
            let hash_ttrf = x1_shuf * y0_shuf * z2_shuf;
            let hash_trrf = x1_shuf * y1_shuf * z3_shuf;
            let hash_blff = x0_shuf * y2_shuf * z1_shuf;
            let hash_bblf = x1_shuf * y3_shuf * z1_shuf;
            let hash_bllf = x1_shuf * y2_shuf * z0_shuf;
            let hash_brff = x0_shuf * y2_shuf * z2_shuf;
            let hash_bbrf = x1_shuf * y3_shuf * z2_shuf;
            let hash_brrf = x1_shuf * y2_shuf * z3_shuf;
            let hash_tlbb = x3_shuf * y1_shuf * z1_shuf;
            let hash_ttlb = x2_shuf * y0_shuf * z1_shuf;
            let hash_tllb = x2_shuf * y1_shuf * z0_shuf;
            let hash_trbb = x3_shuf * y1_shuf * z2_shuf;
            let hash_ttrb = x2_shuf * y0_shuf * z2_shuf;
            let hash_trrb = x2_shuf * y1_shuf * z3_shuf;
            let hash_blbb = x3_shuf * y2_shuf * z1_shuf;
            let hash_bblb = x2_shuf * y3_shuf * z1_shuf;
            let hash_bllb = x2_shuf * y2_shuf * z0_shuf;
            let hash_brbb = x3_shuf * y2_shuf * z2_shuf;
            let hash_bbrb = x2_shuf * y3_shuf * z2_shuf;
            let hash_brrb = x2_shuf * y2_shuf * z3_shuf;

            let x_dist_tlff =
                ((hash_tlff & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_ttlf = ((hash_ttlf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_tllf = ((hash_tllf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_trff =
                ((hash_trff & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_ttrf = ((hash_ttrf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_trrf = ((hash_trrf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_blff =
                ((hash_blff & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_bblf = ((hash_bblf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_bllf = ((hash_bllf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_brff =
                ((hash_brff & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo + one;
            let x_dist_bbrf = ((hash_bbrf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_brrf = ((hash_brrf & hash_mask) | exp_bits).raw_cast::<f32>() + x_dist_lo;
            let x_dist_tlbb =
                ((hash_tlbb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_ttlb = ((hash_ttlb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_tllb = ((hash_tllb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_trbb =
                ((hash_trbb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_ttrb = ((hash_ttrb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_trrb = ((hash_trrb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_blbb =
                ((hash_blbb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_bblb = ((hash_bblb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_bllb = ((hash_bllb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_brbb =
                ((hash_brbb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi - one;
            let x_dist_bbrb = ((hash_bbrb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;
            let x_dist_brrb = ((hash_brrb & hash_mask) | exp_bits).raw_cast::<f32>() - x_dist_hi;

            let y_dist_tlff = ((hash_tlff >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_ttlf = ((hash_ttlf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_tllf = ((hash_tllf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_trff = ((hash_trff >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_ttrf = ((hash_ttrf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_trrf = ((hash_trrf >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_blff = ((hash_blff >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_bblf = ((hash_bblf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;
            let y_dist_bllf = ((hash_bllf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_brff = ((hash_brff >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_bbrf = ((hash_bbrf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;
            let y_dist_brrf = ((hash_brrf >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_tlbb = ((hash_tlbb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_ttlb = ((hash_ttlb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_tllb = ((hash_tllb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_trbb = ((hash_trbb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_ttrb = ((hash_ttrb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo + one;
            let y_dist_trrb = ((hash_trrb >> 9) | exp_bits).raw_cast::<f32>() + y_dist_lo;
            let y_dist_blbb = ((hash_blbb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_bblb = ((hash_bblb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;
            let y_dist_bllb = ((hash_bllb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_brbb = ((hash_brbb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;
            let y_dist_bbrb = ((hash_bbrb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi - one;
            let y_dist_brrb = ((hash_brrb >> 9) | exp_bits).raw_cast::<f32>() - y_dist_hi;

            let z_dist_tlff =
                (((hash_tlff << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_ttlf =
                (((hash_ttlf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_tllf =
                (((hash_tllf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo + one;
            let z_dist_trff =
                (((hash_trff << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_ttrf =
                (((hash_ttrf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_trrf =
                (((hash_trrf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi - one;
            let z_dist_blff =
                (((hash_blff << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_bblf =
                (((hash_bblf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_bllf =
                (((hash_bllf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo + one;
            let z_dist_brff =
                (((hash_brff << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_bbrf =
                (((hash_bbrf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_brrf =
                (((hash_brrf << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi - one;
            let z_dist_tlbb =
                (((hash_tlbb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_ttlb =
                (((hash_ttlb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_tllb =
                (((hash_tllb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo + one;
            let z_dist_trbb =
                (((hash_trbb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_ttrb =
                (((hash_ttrb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_trrb =
                (((hash_trrb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi - one;
            let z_dist_blbb =
                (((hash_blbb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_bblb =
                (((hash_bblb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo;
            let z_dist_bllb =
                (((hash_bllb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() + z_dist_lo + one;
            let z_dist_brbb =
                (((hash_brbb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_bbrb =
                (((hash_bbrb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi;
            let z_dist_brrb =
                (((hash_brrb << 12) & hash_mask) | exp_bits).raw_cast::<f32>() - z_dist_hi - one;

            let dist_tlff = x_dist_tlff.mul_add(
                x_dist_tlff,
                y_dist_tlff.mul_add(y_dist_tlff, z_dist_tlff * z_dist_tlff),
            );
            let dist_ttlf = x_dist_ttlf.mul_add(
                x_dist_ttlf,
                y_dist_ttlf.mul_add(y_dist_ttlf, z_dist_ttlf * z_dist_ttlf),
            );
            let dist_tllf = x_dist_tllf.mul_add(
                x_dist_tllf,
                y_dist_tllf.mul_add(y_dist_tllf, z_dist_tllf * z_dist_tllf),
            );
            let dist_trff = x_dist_trff.mul_add(
                x_dist_trff,
                y_dist_trff.mul_add(y_dist_trff, z_dist_trff * z_dist_trff),
            );
            let dist_ttrf = x_dist_ttrf.mul_add(
                x_dist_ttrf,
                y_dist_ttrf.mul_add(y_dist_ttrf, z_dist_ttrf * z_dist_ttrf),
            );
            let dist_trrf = x_dist_trrf.mul_add(
                x_dist_trrf,
                y_dist_trrf.mul_add(y_dist_trrf, z_dist_trrf * z_dist_trrf),
            );
            let dist_blff = x_dist_blff.mul_add(
                x_dist_blff,
                y_dist_blff.mul_add(y_dist_blff, z_dist_blff * z_dist_blff),
            );
            let dist_bblf = x_dist_bblf.mul_add(
                x_dist_bblf,
                y_dist_bblf.mul_add(y_dist_bblf, z_dist_bblf * z_dist_bblf),
            );
            let dist_bllf = x_dist_bllf.mul_add(
                x_dist_bllf,
                y_dist_bllf.mul_add(y_dist_bllf, z_dist_bllf * z_dist_bllf),
            );
            let dist_brff = x_dist_brff.mul_add(
                x_dist_brff,
                y_dist_brff.mul_add(y_dist_brff, z_dist_brff * z_dist_brff),
            );
            let dist_bbrf = x_dist_bbrf.mul_add(
                x_dist_bbrf,
                y_dist_bbrf.mul_add(y_dist_bbrf, z_dist_bbrf * z_dist_bbrf),
            );
            let dist_brrf = x_dist_brrf.mul_add(
                x_dist_brrf,
                y_dist_brrf.mul_add(y_dist_brrf, z_dist_brrf * z_dist_brrf),
            );
            let dist_tlbb = x_dist_tlbb.mul_add(
                x_dist_tlbb,
                y_dist_tlbb.mul_add(y_dist_tlbb, z_dist_tlbb * z_dist_tlbb),
            );
            let dist_ttlb = x_dist_ttlb.mul_add(
                x_dist_ttlb,
                y_dist_ttlb.mul_add(y_dist_ttlb, z_dist_ttlb * z_dist_ttlb),
            );
            let dist_tllb = x_dist_tllb.mul_add(
                x_dist_tllb,
                y_dist_tllb.mul_add(y_dist_tllb, z_dist_tllb * z_dist_tllb),
            );
            let dist_trbb = x_dist_trbb.mul_add(
                x_dist_trbb,
                y_dist_trbb.mul_add(y_dist_trbb, z_dist_trbb * z_dist_trbb),
            );
            let dist_ttrb = x_dist_ttrb.mul_add(
                x_dist_ttrb,
                y_dist_ttrb.mul_add(y_dist_ttrb, z_dist_ttrb * z_dist_ttrb),
            );
            let dist_trrb = x_dist_trrb.mul_add(
                x_dist_trrb,
                y_dist_trrb.mul_add(y_dist_trrb, z_dist_trrb * z_dist_trrb),
            );
            let dist_blbb = x_dist_blbb.mul_add(
                x_dist_blbb,
                y_dist_blbb.mul_add(y_dist_blbb, z_dist_blbb * z_dist_blbb),
            );
            let dist_bblb = x_dist_bblb.mul_add(
                x_dist_bblb,
                y_dist_bblb.mul_add(y_dist_bblb, z_dist_bblb * z_dist_bblb),
            );
            let dist_bllb = x_dist_bllb.mul_add(
                x_dist_bllb,
                y_dist_bllb.mul_add(y_dist_bllb, z_dist_bllb * z_dist_bllb),
            );
            let dist_brbb = x_dist_brbb.mul_add(
                x_dist_brbb,
                y_dist_brbb.mul_add(y_dist_brbb, z_dist_brbb * z_dist_brbb),
            );
            let dist_bbrb = x_dist_bbrb.mul_add(
                x_dist_bbrb,
                y_dist_bbrb.mul_add(y_dist_bbrb, z_dist_bbrb * z_dist_bbrb),
            );
            let dist_brrb = x_dist_brrb.mul_add(
                x_dist_brrb,
                y_dist_brrb.mul_add(y_dist_brrb, z_dist_brrb * z_dist_brrb),
            );

            let outer_min = dist_tlff
                .min(dist_ttlf)
                .min(dist_tllf)
                .min(dist_trff)
                .min(dist_ttrf)
                .min(dist_trrf)
                .min(dist_blff)
                .min(dist_bblf)
                .min(dist_bllf)
                .min(dist_brff)
                .min(dist_bbrf)
                .min(dist_brrf)
                .min(dist_tlbb)
                .min(dist_ttlb)
                .min(dist_tllb)
                .min(dist_trbb)
                .min(dist_ttrb)
                .min(dist_trrb)
                .min(dist_blbb)
                .min(dist_bblb)
                .min(dist_bllb)
                .min(dist_brbb)
                .min(dist_bbrb)
                .min(dist_brrb);

            min_dist = min_dist.min(outer_min);
        }

        // Sqrt: 1
        min_dist.sqrt()
    }
}
