use std::mem::MaybeUninit;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::generators::DistanceFn;
use crate::noise::util::grid_data::CellularGridData;
use crate::noise::util::constants::{ BYTE_SHUFFLE, CELLULAR_EXP_MASK, HASH_MASK, HASH_PRIME };
use crate::noise::util::grid_helpers::{
    Arena,
    ArenaBuffer,
    MaybeUninitSliceSimdExt,
    maybe_tail_load,
    maybe_tail_store,
    pad_grid_size,
    validate_grid_size,
    validate_state_size,
    rem_euclid_v,
};
use crate::{ Cellular, GridGenerator };

/// Candidate offsets for the 12 candidates (4 base + 8 ring). The ring is
/// grouped by bounding edge and stay aligned with the near/far gate
///        (0,-1) (1,-1)
/// (-1,0) (0,0) (1,0) (2,0)
/// (-1,1) (0,1) (1,1) (2,1)
///        (0,2) (1,2)
const NEIGHBORS_12: [(i32, i32); 12] = [
    ( 0,  0),
    ( 1,  0),
    ( 0,  1),
    ( 1,  1),

    (-1,  0),
    (-1,  1),
    ( 0, -1),
    ( 1, -1),
    
    ( 2,  0),
    ( 2,  1),
    ( 0,  2),
    ( 1,  2),
];

/// 2D Plane Window
struct Window {
    rows: [*mut f32; 4],
    width: usize,
}

impl Window {
    fn new(arena: &mut Arena, width: usize) -> Self {
        let len = width * 2;
        Self {
            rows: std::array::from_fn(|_| { arena.allocate::<f32>(len).as_mut_ptr().cast() }),
            width,
        }
    }

    #[inline(always)]
    fn fill_row<A: Arch>(
        params: &GridNoiseParams<2>,
        grid_data: &CellularGridData<2>,
        buff: &mut [(f32, f32)],
        lx_start: i32,
        ly: i32
    ) {
        let ly = grid_data.octave_tiling[1].map_or(ly, |t| ly.rem_euclid(t as i32));
        let y_shuf = hash_axis::<A>(ly as u32, params.seed); // Scalar
        let y_shuf_v = Simd::<u32, A>::splat(y_shuf); // Vector
        let lanes = Simd::<f32, A>::LANES;

        let mut x_it = 0;
        while x_it + lanes <= buff.len() {
            let lx = lx_start.wrapping_add(x_it as i32); // Scalar
            let lx_v = Simd::<i32, A>::splat(lx) + Simd::<i32, A>::iota(0); // Vector
            let lx_t = if let Some(t) = grid_data.octave_tiling[0] {
                // Tiling
                rem_euclid_v::<A>(lx_v, t as i32)
            } else {
                lx_v
            };
            // Tiled hashes (if octave tiling was set)
            let hashes = hash_row::<A>(lx_t.raw_cast(), y_shuf_v, params.seed);
            // Tiled jitter offsets
            let (tx, ty) = split_hash_v::<A>(hashes);
            let tx_arr = tx.to_array();
            let ty_arr = ty.to_array();
            for i in 0..lanes {
                buff[x_it + i] = (tx_arr[i], ty_arr[i]);
            }
            x_it += lanes;
        }
        for i in x_it..buff.len() {
            let lx = lx_start.wrapping_add(i as i32);
            let lx = grid_data.octave_tiling[0].map_or(lx, |t| lx.rem_euclid(t as i32));
            buff[i] = split_hash(hash_cell_with::<A>(lx as u32, y_shuf, params.seed));
        }
    }

    fn row(&self, index: usize) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.rows[index].cast(), self.width) }
    }

    fn row_mut(&mut self, index: usize) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.rows[index].cast(), self.width) }
    }

    fn rows(&self) -> [&[(f32, f32)]; 4] {
        [self.row(0), self.row(1), self.row(2), self.row(3)]
    }

    #[inline(always)]
    fn swap(&mut self) {
        self.rows.rotate_left(1);
    }
}

#[inline(always)]
fn hash_axis<A: Arch>(a: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(HASH_PRIME);
    (Simd::<u32, A>::splat(a.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime).to_array()[0]
}

/// Hashes `LANES` consecutive lattice columns from a pre-built `lx_v` vector
/// at fixed y in one shot
#[inline(always)]
fn hash_row<A: Arch>(lx_v: Simd<u32, A>, y_shuf: Simd<u32, A>, seed: u32) -> Simd<u32, A> {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(HASH_PRIME);
    let seed_v = Simd::<u32, A>::splat(seed);

    let x_shuf = (lx_v * seed_v).permute_8(shuffle_indices) ^ prime;
    (x_shuf * y_shuf) ^ x_shuf
}

#[inline(always)]
fn hash_cell_with<A: Arch>(x: u32, shuf: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(HASH_PRIME);
    let x_shuf = (
        Simd::<u32, A>::splat(x.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    x_shuf.wrapping_mul(shuf) ^ x_shuf
}

/// Split a hash into the per-axis jitter offsets. The lower 23 bits
/// become the x-mantissa and the next 23 bits become the y-mantissa, giving
/// unform values in the range `[1, 2)`
#[inline(always)]
fn split_hash(hash: u32) -> (f32, f32) {
    let exp_bits = CELLULAR_EXP_MASK;
    let hash_mask = HASH_MASK;
    let tx = 1.5 - f32::from_bits((hash & hash_mask) | exp_bits);
    let ty = 1.5 - f32::from_bits((hash >> 9) | exp_bits);
    (tx, ty)
}

#[inline(always)]
fn split_hash_v<A: Arch>(hash: Simd<u32, A>) -> (Simd<f32, A>, Simd<f32, A>) {
    let exp_bits = Simd::<u32, A>::splat(0x3f800000);
    let hash_mask = Simd::<u32, A>::splat(0x007fffff);
    let one_halves = Simd::<f32, A>::splat(1.5);

    let tx = one_halves - ((hash & hash_mask) | exp_bits).raw_cast::<f32>();
    let ty = one_halves - ((hash >> Simd::<u32, A>::splat(9)) | exp_bits).raw_cast::<f32>();
    (tx, ty)
}

/// Jitter offsets for the 12 candidates (4 base + 8 ring).
/// Stored as a flat array of 24 floats: [x0,y0, x1,y1, ..., x11,y11].
struct CellJitters {
    parts: *mut MaybeUninit<f32>,
}

impl CellJitters {
    fn new(arena: &mut Arena) -> Self {
        Self {
            parts: arena.allocate(24).as_mut_ptr(),
        }
    }

    #[inline(always)]
    fn write(&mut self, rows: [&[(f32, f32)]; 4], x_it: usize) {
        for (i, &(ox, oy)) in NEIGHBORS_12.iter().enumerate() {
            // The oy is + 1 away from the indicies. - 1 becomes 0 and everything that follows becomes a usable index
            let row = rows[(oy + 1) as usize];
            let (jx, jy) = row[x_it + ((ox + 1) as usize)];
            unsafe {
                self.parts.add(i * 2).write(MaybeUninit::new(jx + (ox as f32)));
                self.parts.add(i * 2 + 1).write(MaybeUninit::new(jy + (oy as f32)));
            }
        }
    }
}

#[enable_targets(A)]
impl<F: DistanceFn> GridGenerator<2> for Cellular<F> {
    type GenConfig = F::DistanceConfig;
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        combiner: C::Config,
        gen_cfg: <Self as GridGenerator<2>>::GenConfig,
        state: &mut [f32],
        dst: &mut [f32]
    ) {
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, _>(params.grid_size);

        let required_cache =
            padded_size[1] * 3 + padded_size[0] * 3 + (padded_size[0] + 1) * 8 + 24;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);
        let mut sub_arena = arena.allocate_arena(padded_size[0] * 3 + padded_size[1] * 3);

        let grid_data = CellularGridData::new::<A>(&params, &mut sub_arena, &padded_size);

        let row_len = grid_data.num_loops[0] + 3;
        let mut window = Window::new(&mut arena, row_len);
        let mut cell_jitters = CellJitters::new(&mut arena);

        // Pre-fill the 4 rows: fir (y-1), sec (y=0), thi (y=1), fou (y=2)
        let lx_offset = grid_data.grid_start[0] - 1;
        for (i, offset) in [-1, 0, 1, 2].iter().enumerate() {
            Window::fill_row::<A>(
                &params,
                &grid_data,
                window.row_mut(i),
                lx_offset,
                grid_data.grid_start[1] + offset,
            );
        }

        let mut y_idx = 0;
        for y_it in 0..grid_data.num_loops[1] {
            let y_next_idx = unsafe {
                grid_data.grid_indices[1].get_unchecked(y_it).assume_init() as usize
            };

            let mut x_idx = 0;
            for x_it in 0..grid_data.num_loops[0] {
                let x_next_idx = unsafe {
                    grid_data.grid_indices[0].get_unchecked(x_it).assume_init() as usize
                };

                cell_jitters.write(window.rows(), x_it);

                fill_cell::<A, C, F, INIT, FINAL>(
                    &grid_data,
                    &cell_jitters,
                    x_idx,
                    x_next_idx,
                    y_idx,
                    y_next_idx,
                    dst,
                    state,
                    &combiner,
                    gen_cfg
                );

                x_idx = x_next_idx;
            }

            // Swap rows: sec->fir, thi->sec, fou->thi, old fir->fou
            window.swap();

            // Fill the new last row
            Window::fill_row::<A>(
                &params,
                &grid_data,
                window.row_mut(3),
                lx_offset,
                grid_data.grid_start[1] + (y_it as i32) + 3,
            );

            y_idx = y_next_idx;
        }
    }
}

#[inline(always)]
fn fill_cell<A: Arch, C: Combiner, F: DistanceFn, const INIT: bool, const FINAL: bool>(
    grid_data: &CellularGridData<2>,
    jit: &CellJitters,
    x_idx: usize,
    x_next: usize,
    y_idx: usize,
    y_next: usize,
    dst: &mut [f32],
    state: &mut [f32],
    combiner_config: &C::Config,
    dst_cfg: F::DistanceConfig
) {
    let lanes = Simd::<f32, A>::LANES;
    let row_width = grid_data.grid_size[0];

    let weight_vec = Simd::<f32, A>::splat(grid_data.weight);

    for y in y_idx..y_next {
        let sy = unsafe { grid_data.distances[1].get_unchecked(y).assume_init() };
        let row_start = y * row_width;

        let dy_mod: [MaybeUninit<Simd<f32, A>>; 12] = std::array::from_fn(|c| {
            // Stride 2, y is component 1
            let jy = unsafe { (*jit.parts.add(c * 2 + 1)).assume_init() };
            let dy = Simd::<f32, A>::splat(sy) - Simd::<f32, A>::splat(jy);
            MaybeUninit::new(F::modify_delta::<A>(dy, &dst_cfg))
        });

        let mut index = x_idx;
        while index + lanes <= x_next {
            fill_row_block::<A, C, F, INIT, FINAL, false>(
                grid_data,
                jit,
                &dy_mod,
                weight_vec,
                sy,
                row_start,
                x_next,
                index,
                dst,
                state,
                combiner_config,
                dst_cfg
            );
            index += lanes;
        }
        if index < x_next {
            fill_row_block::<A, C, F, INIT, FINAL, true>(
                grid_data,
                jit,
                &dy_mod,
                weight_vec,
                sy,
                row_start,
                x_next,
                index,
                dst,
                state,
                combiner_config,
                dst_cfg
            );
        }
    }
}

#[inline(always)]
fn fill_row_block<
    A: Arch,
    C: Combiner,
    F: DistanceFn,
    const INIT: bool,
    const FINAL: bool,
    const IS_TAIL: bool
>(
    grid_data: &CellularGridData<2>,
    jit: &CellJitters,
    dy_mod: &[MaybeUninit<Simd<f32, A>>; 12],
    weight_vec: Simd<f32, A>,
    sy: f32,
    row_start: usize,
    x_next: usize,
    index: usize,
    dst: &mut [f32],
    state: &mut [f32],
    combiner_config: &C::Config,
    dst_cfg: F::DistanceConfig
) {
    let lanes = if IS_TAIL { x_next - index } else { Simd::<f32, A>::LANES };
    let sample_start = row_start + index;
    let sample_end = sample_start + lanes;

    let sx: Simd<f32, A> = unsafe { grid_data.distances[0].load_simd(index) };

    let mut min = Simd::<f32, A>::splat(f32::MAX);
    for c in 0..4 {
        // Stride 2, x is component 0
        let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
        let dy = unsafe { *dy_mod[c].assume_init_ref() };
        let dx = F::modify_delta(sx - jx, &dst_cfg);
        let deltas = [dx, dy];
        min = min.min(F::dist(deltas, &dst_cfg));
    }

    let x_edge_lo = sx + Simd::<f32, A>::splat(0.5);
    let x_edge_hi = Simd::<f32, A>::splat(1.5) - sx;
    let y_edge_lo = Simd::<f32, A>::splat(sy + 0.5);
    let y_edge_hi = Simd::<f32, A>::splat(1.5 - sy);

    let near_sq = {
        let edge = x_edge_lo.min(y_edge_lo);
        F::gate_val(edge, &dst_cfg)
    };
    let far_sq = {
        let edge = x_edge_hi.min(y_edge_hi);
        F::gate_val(edge, &dst_cfg)
    };

    let any_near = min.simd_gt(near_sq).to_bits() != 0;
    let any_far = min.simd_gt(far_sq).to_bits() != 0;

    if any_near {
        // all low-edge candidates:
        // x/y/z offsets containing -1
        for c in 4..8 {
            let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
            let dy = unsafe { *dy_mod[c].assume_init_ref() };
            let dx = F::modify_delta(sx - jx, &dst_cfg);
            let deltas = [dx, dy];
            min = min.min(F::dist(deltas, &dst_cfg));
        }
    }

    if any_far {
        // all high-edge candidates:
        // x/y/z offsets containing 2
        for c in 8..12 {
            let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
            let dy = unsafe { *dy_mod[c].assume_init_ref() };
            let dx = F::modify_delta(sx - jx, &dst_cfg);
            let deltas = [dx, dy];
            min = min.min(F::dist(deltas, &dst_cfg));
        }
    }

    let mut raw_val = F::finalize(min, &dst_cfg);
    if !C::MANUAL_WEIGHT_PROCESSING {
        raw_val *= weight_vec;
    }

    let (cur_state, mut result) = if INIT {
        C::initialize_sample(combiner_config, weight_vec, raw_val)
    } else {
        let mut cur_state = C::State::<A>::default();
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            cur_state[i] = unsafe {
                maybe_tail_load::<A, IS_TAIL>(sample_start + offset..sample_end + offset, state)
            };
        }
        let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(sample_start..sample_end, dst) };
        C::apply_sample(combiner_config, cur_state, weight_vec, cur_result, raw_val)
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
        result = C::finalize_sample(combiner_config, cur_state, result);
    }

    unsafe {
        maybe_tail_store::<A, IS_TAIL>(sample_start..sample_end, result, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::seed::gen_octave_seed;
    use crate::math::random::Random;
    use crate::simd::StaticArch;
    use crate::{ BatchGenerator, EuclideanSquared };
    use crate::{ Cellular, Fbm, Grid };

    #[test]
    fn cellular_grid_2d_sanity() {
        let grid = Grid::<2>::new(32, 32);
        let mut result = [0.0; 1024];
        grid.builder::<Fbm, Cellular<EuclideanSquared>>().fill(result.as_mut_slice());
        verify_slice(result.as_slice());
    }

    fn verify_slice(slice: &[f32]) {
        let mut min = f32::MAX;
        let mut max = f32::NEG_INFINITY;
        let mut dif_total = 0.0;
        let mut prev = slice[0];

        for val in slice.iter() {
            min = val.min(min);
            max = val.max(max);

            let dif = (*val - prev).abs();
            dif_total += dif;
            prev = *val;
        }

        assert!(min >= 0.0, "Cellular distance of {min} was negative!");
        assert!(max < 10.0, "Maximum value of {max} was above 10!");
        assert!(dif_total > 0.0, "Output is constant of {}!", slice[0]);
    }

    #[test]
    fn cellular_grid_2d_reference() {
        let seed = 123456789i64;

        for freq in [1.0 / 32.0, 1.0 / 8.0, 1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0] {
            check_reference(64, 64, seed, -5.0, 3.0, freq);
        }

        check_reference(32, 96, seed, -5.0, 3.0, 1.0 / 6.0);
    }

    fn check_reference(w: usize, h: usize, seed: i64, offset_x: f64, offset_y: f64, freq: f32) {
        let grid = Grid::<2>::new(w, h).seed(seed).sample_position(offset_x, offset_y);
        let grid_seed = Random::mix_u64(seed as u64);
        let base_seed = Random::mix_u64_pair(grid_seed, 0xd5e7b3c94f8a1e6b);
        let octave_seed = gen_octave_seed([freq, freq], base_seed);

        let mut result = vec![0.0; w * h];
        grid.builder::<Fbm, Cellular<EuclideanSquared>>()
            .frequency(freq)
            .fill(result.as_mut_slice());

        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let px = (offset_x as f32) + (x as f32);
                let py = (offset_y as f32) + (y as f32);
                let reference = reference(octave_seed, px, py, freq);
                let actual = result[y * w + x];
                max_diff = max_diff.max((actual - reference).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "Grid cellular at freq {freq} diverges from the batch Cellular by {max_diff}"
        );
    }

    fn reference(seed: u32, px: f32, py: f32, freq: f32) -> f32 {
        let gain = Cellular::<EuclideanSquared>::sample_batch::<StaticArch>(
            seed,
            [Simd::splat(px), Simd::splat(py)],
            [Simd::splat(freq), Simd::splat(freq)],
            ()
        );
        gain.to_array()[0]
    }
}
