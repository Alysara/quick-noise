use simply_simd::{ Arch, Mask, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::SimplexGridData;
use crate::noise::util::grid_helpers::{
    maybe_tail_load,
    maybe_tail_store,
    validate_grid_size,
    validate_state_size,
};
use crate::{ GridGenerator, Simplex };
use crate::noise::util::constants::{
    BYTE_SHUFFLE,
    HASH_PRIME,
    X_GRADIENTS_2D,
    Y_GRADIENTS_2D,
    SKEW_2D,
    UNSKEW_2D,
    SQRT_3,
};

/// Resolves the gradient of a single lattice vertex `(i, j)`. Used only by the
/// scalar tail of a row
#[inline(always)]
fn gradient<A: Arch>(i: i32, j: i32, seed: u32) -> (f32, f32) {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(HASH_PRIME);
    let x_shuf = (
        Simd::<u32, A>::splat((i as u32).wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    let y_shuf = (
        Simd::<u32, A>::splat((j as u32).wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    let mix = x_shuf.wrapping_mul(y_shuf) ^ x_shuf;
    let idx = (mix >> 29) as usize;
    (X_GRADIENTS_2D[idx], Y_GRADIENTS_2D[idx])
}

/// Simplex calculation. `x0`/`y0` are the
/// low-corner distances for this cell.
#[inline(always)]
fn simplex_calc(x0: f32, y0: f32, grads: &[(f32, f32); 4], upper: bool) -> f32 {
    let subbed_unskew = UNSKEW_2D - 1.0;
    let hi_skew_offset = 2.0 * UNSKEW_2D - 1.0;

    let (gx_lo, gy_lo) = grads[0];
    let (gx_mi, gy_mi) = if upper { grads[1] } else { grads[2] };
    let (gx_hi, gy_hi) = grads[3];

    let (x_mi, y_mi) = if upper {
        (x0 + subbed_unskew, y0 + UNSKEW_2D)
    } else {
        (x0 + UNSKEW_2D, y0 + subbed_unskew)
    };
    let (x_hi, y_hi) = (x0 + hi_skew_offset, y0 + hi_skew_offset);

    let t_lo_pre = 0.5 - x0.mul_add(x0, y0 * y0);
    let t_mi = (0.5 - x_mi.mul_add(x_mi, y_mi * y_mi)).max(0.0);
    let t_hi_pre = t_lo_pre + ((2.0 * SQRT_3) / 3.0).mul_add(x0 + y0, -2.0 / 3.0);
    let t_lo = t_lo_pre.max(0.0);
    let t_hi = t_hi_pre.max(0.0);

    let t2_lo = t_lo * t_lo;
    let t2_mi = t_mi * t_mi;
    let t2_hi = t_hi * t_hi;

    let dot_lo = gx_lo.mul_add(x0, gy_lo * y0);
    let dot_mi = gx_mi.mul_add(x_mi, gy_mi * y_mi);
    let dot_hi = gx_hi.mul_add(x_hi, gy_hi * y_hi);

    let t4_lo = t2_lo * t2_lo;
    let t4_mi = t2_mi * t2_mi;
    let t4_hi = t2_hi * t2_hi;
    t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi))
}

/// Vectorised simplex calculation.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simplex_calc_simd<A: Arch>(
    x0: Simd<f32, A>,
    y0: Simd<f32, A>,
    t_lo_pre: Simd<f32, A>,
    gx_lo: Simd<f32, A>,
    gy_lo: Simd<f32, A>,
    gx_mi: Simd<f32, A>,
    gy_mi: Simd<f32, A>,
    gx_hi: Simd<f32, A>,
    gy_hi: Simd<f32, A>,
    upper: Mask<f32, A>
) -> Simd<f32, A> {
    let subbed_unskew = Simd::<f32, A>::splat(UNSKEW_2D - 1.0);
    let hi_skew_offset = Simd::<f32, A>::splat(2.0 * UNSKEW_2D - 1.0);
    let unskew = Simd::<f32, A>::splat(UNSKEW_2D);
    let half = Simd::<f32, A>::splat(0.5);
    let zero = Simd::<f32, A>::splat(0.0);
    let t_hi_coef = Simd::<f32, A>::splat((2.0 * SQRT_3) / 3.0);
    let neg_two_thirds = Simd::<f32, A>::splat(-2.0 / 3.0);

    let x_mi = x0 + upper.select(subbed_unskew, unskew);
    let y_mi = y0 + upper.select(unskew, subbed_unskew);
    let (x_hi, y_hi) = (x0 + hi_skew_offset, y0 + hi_skew_offset);

    let t_mi = (half - x_mi.mul_add(x_mi, y_mi * y_mi)).max(zero);
    let t_hi_pre = t_lo_pre + t_hi_coef.mul_add(x0 + y0, neg_two_thirds);
    let t_lo = t_lo_pre.max(zero);
    let t_hi = t_hi_pre.max(zero);

    let t2_lo = t_lo * t_lo;
    let t2_mi = t_mi * t_mi;
    let t2_hi = t_hi * t_hi;

    let dot_lo = gx_lo.mul_add(x0, gy_lo * y0);
    let dot_mi = gx_mi.mul_add(x_mi, gy_mi * y_mi);
    let dot_hi = gx_hi.mul_add(x_hi, gy_hi * y_hi);

    let t4_lo = t2_lo * t2_lo;
    let t4_mi = t2_mi * t2_mi;
    let t4_hi = t2_hi * t2_hi;
    t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi))
}

#[enable_targets(A)]
impl GridGenerator<2> for Simplex {
    type GenConfig = ();
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        combiner: C::Config,
        _generator_config: <Self as GridGenerator<2>>::GenConfig,
        state: &mut [f32],
        dst: &mut [f32]
    ) {
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, dst.len());

        let grid_data = SimplexGridData::new(&params);
        let row_width = grid_data.grid_size[0];

        for oy in 0..grid_data.grid_size[1] {
            simplex_fill_row::<A, C, INIT, FINAL>(
                &grid_data,
                &combiner,
                params.seed,
                oy,
                row_width,
                state,
                dst
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn simplex_fill_row<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    grid_data: &SimplexGridData<2>,
    combiner: &C::Config,
    seed: u32,
    oy: usize,
    row_width: usize,
    state: &mut [f32],
    dst: &mut [f32]
) {
    let lanes = Simd::<f32, A>::LANES;
    let lanes_f = lanes as f32;
    let weight = grid_data.weight;
    let row_start = oy * row_width;

    // Per-step skewed-space increments along the x-axis
    let dsx = grid_data.increment[0] * (1.0 + SKEW_2D);
    let dsy = grid_data.increment[0] * SKEW_2D;

    // Sample-space y is fixed across the whole row
    let y = grid_data.origin[1] + (oy as f32) * grid_data.increment[1];

    let s0 = (grid_data.origin[0] + y) * SKEW_2D;
    let sx = grid_data.origin[0] + s0;
    let sy = y + s0;

    // Lane-index vector and per-step skewed increments
    let iota_f = Simd::<i32, A>::iota(0).cast_float();
    let sx_stride = Simd::<f32, A>::splat(dsx * lanes_f);
    let sy_stride = Simd::<f32, A>::splat(dsy * lanes_f);

    // Running vectors of sample-space x and skewed coords for the current
    // chunk, advanced by `lanes` each iteration
    let x0_stride = Simd::<f32, A>::splat(grid_data.increment[0] * lanes_f);
    let mut lane_x0 =
        Simd::<f32, A>::splat(grid_data.origin[0]) +
        iota_f * Simd::<f32, A>::splat(grid_data.increment[0]);
    let mut sxv = Simd::<f32, A>::splat(sx) + iota_f * Simd::<f32, A>::splat(dsx);
    let mut syv = Simd::<f32, A>::splat(sy) + iota_f * Simd::<f32, A>::splat(dsy);

    let unskew_v = Simd::<f32, A>::splat(UNSKEW_2D);
    let y_v = Simd::<f32, A>::splat(y);
    let weight_v = Simd::<f32, A>::splat(weight);
    let half_v = Simd::<f32, A>::splat(0.5);

    // Hash constants.
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(HASH_PRIME);
    let seed_v = Simd::<u32, A>::splat(seed);

    let mut ox = 0;

    while ox + lanes <= row_width {
        let ic = sxv.floor();
        let jc = syv.floor();
        let ic_v = ic.cast_int_trunc();
        let jc_v = jc.cast_int_trunc();

        let sum = (ic + jc) * unskew_v;
        let cx = ic - sum;
        let cy = jc - sum;
        let x0 = lane_x0 - cx;
        let y0 = y_v - cy;
        let upper = x0.simd_gt(y0);

        let ic_u = ic_v.raw_cast::<u32>();
        let jc_u = jc_v.raw_cast::<u32>();
        let x1 = ic_u * seed_v;
        let y1 = jc_u * seed_v;
        let x2 = x1 + seed_v;
        let y2 = y1 + seed_v;

        let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
        let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
        let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
        let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

        let mix_lo = (x1_shuf * y1_shuf) ^ x1_shuf;
        let mix_hi = (x2_shuf * y2_shuf) ^ x2_shuf;

        let upper_u = upper.raw_cast::<u32>();
        let x_shuf_mi = upper_u.select(x2_shuf, x1_shuf);
        let y_shuf_mi = upper_u.select(y1_shuf, y2_shuf);
        let mix_mi = (x_shuf_mi * y_shuf_mi) ^ x_shuf_mi;

        let indices_lo = mix_lo >> 29;
        let indices_mi = mix_mi >> 29;
        let indices_hi = mix_hi >> 29;

        let t_lo_pre = half_v - x0.mul_add(x0, y0 * y0);

        let value =
            simplex_calc_simd::<A>(
                x0,
                y0,
                t_lo_pre,
                indices_lo.gather(&X_GRADIENTS_2D),
                indices_lo.gather(&Y_GRADIENTS_2D),
                indices_mi.gather(&X_GRADIENTS_2D),
                indices_mi.gather(&Y_GRADIENTS_2D),
                indices_hi.gather(&X_GRADIENTS_2D),
                indices_hi.gather(&Y_GRADIENTS_2D),
                upper
            ) * weight_v;

        fill_block::<A, C, INIT, FINAL, false>(
            grid_data,
            combiner,
            state,
            dst,
            row_start + ox,
            lanes,
            value
        );
        // Advance all tracking
        sxv += sx_stride;
        syv += sy_stride;
        lane_x0 += x0_stride;
        ox += lanes;
    }

    // Scalar tail: remaining samples after the last full SIMD chunk
    let mut sx = grid_data.origin[0] + s0 + dsx * (ox as f32);
    let mut sy = y + s0 + dsy * (ox as f32);
    while ox < row_width {
        let ic = sx.floor() as i32;
        let jc = sy.floor() as i32;
        let c = grid_data.unskew(&[ic, jc]);
        let x = grid_data.origin[0] + (ox as f32) * grid_data.increment[0];
        let x0 = x - c[0];
        let y0 = y - c[1];
        let upper = x0 > y0;

        let grads = [
            gradient::<A>(ic, jc, seed),
            gradient::<A>(ic + 1, jc, seed),
            gradient::<A>(ic, jc + 1, seed),
            gradient::<A>(ic + 1, jc + 1, seed),
        ];
        let value = Simd::<f32, A>::splat(simplex_calc(x0, y0, &grads, upper) * weight);

        fill_block::<A, C, INIT, FINAL, true>(
            grid_data,
            combiner,
            state,
            dst,
            row_start + ox,
            1,
            value
        );

        sx += dsx;
        sy += dsy;
        ox += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_block<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool, const IS_TAIL: bool>(
    grid_data: &SimplexGridData<2>,
    combiner: &C::Config,
    state: &mut [f32],
    dst: &mut [f32],
    sample_start: usize,
    lanes: usize,
    value: Simd<f32, A>
) {
    let sample_end = sample_start + lanes;

    let (cur_state, mut result) = if INIT {
        C::initialize_sample(combiner, value)
    } else {
        let mut cur_state = C::State::<A>::default();
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            cur_state[i] = unsafe {
                maybe_tail_load::<A, IS_TAIL>(sample_start + offset..sample_end + offset, state)
            };
        }
        let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(sample_start..sample_end, dst) };
        C::apply_sample(combiner, cur_state, cur_result, value)
    };

    if !FINAL {
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            unsafe {
                maybe_tail_store::<A, IS_TAIL>(
                    sample_start + offset..sample_end + offset,
                    cur_state[i],
                    state
                );
            }
        }
    }

    if FINAL {
        result = C::finalize_sample(combiner, cur_state, result);
    }

    unsafe { maybe_tail_store::<A, IS_TAIL>(sample_start..sample_end, result, dst) }
}

#[cfg(test)]
mod tests {
    use simply_simd::Simd;

    use crate::api::batch::interface::BatchGenerator;
    use crate::api::seed::gen_octave_seed;
    use crate::math::random::Random;
    use crate::simd::StaticArch;
    use crate::{ Fbm, Grid, Simplex };
    #[cfg(feature = "image")]
    use crate::emit::NoiseImageExt;

    fn reference(seed: u32, px: f32, py: f32, freq: f32) -> f32 {
        let gain = Simplex::sample_batch::<StaticArch>(
            seed,
            [Simd::splat(px), Simd::splat(py)],
            [Simd::splat(freq), Simd::splat(freq)]
        );
        gain.to_array()[0]
    }

    #[test]
    fn simplex_grid_2d_reference() {
        let seed = 123456789i64;

        for freq in [1.0 / 32.0, 1.0 / 8.0, 1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0] {
            check_reference(64, 64, seed, -5.0, 3.0, freq);
        }

        check_reference(32, 96, seed, -5.0, 3.0, 1.0 / 6.0);
    }

    fn check_reference(w: usize, h: usize, seed: i64, offset_x: f32, offset_y: f32, freq: f32) {
        let grid = Grid::<2>::new(w, h).seed(seed).sample_position(offset_x, offset_y);
        let grid_seed = Random::mix_u64(seed as u64);
        let base_seed = Random::mix_u64_pair(grid_seed, 0xd5e7b3c94f8a1e6b);
        let octave_seed = gen_octave_seed([freq, freq], base_seed);

        let mut result = vec![0.0; w * h];
        grid.builder::<Fbm, Simplex>().frequency(freq).fill(result.as_mut_slice());

        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let px = (offset_x + (x as f32)) as f32;
                let py = (offset_y + (y as f32)) as f32;
                let reference = reference(octave_seed, px, py, freq);
                let actual = result[y * w + x];
                max_diff = max_diff.max((actual - reference).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "Grid simplex at freq {freq} diverges from the brute-force Simplex by {max_diff}"
        );
    }
}
