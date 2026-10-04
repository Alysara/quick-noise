use std::mem::MaybeUninit;

use simply_simd::{ Arch, Simd, enable_targets };

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{ Combiner, CombinerState };
use crate::noise::util::grid_data::{ GridData, Lerp };
use crate::noise::util::grid_helpers::{
    Arena,
    ArenaBuffer,
    MaybeUninitSliceSimdExt,
    maybe_tail_load,
    maybe_tail_store,
    pad_grid_size,
    validate_grid_size,
    validate_state_size,
};
use crate::{ Cellular, GridGenerator };

const LERP: u8 = Lerp::Quintic as u8;

/// Candidate offsets for the 12 candidates (4 base + 8 ring). The ring is
/// grouped by bounding edge and stay aligned with the near/far gate
///        (0,-1) (1,-1)
/// (-1,0) (0,0) (1,0) (2,0)
/// (-1,1) (0,1) (1,1) (2,1)
///        (0,2) (1,2)
const NEIGHBORS_12: [(i32, i32); 12] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),

    (-1, 0),
    (-1, 1),

    (0, -1),
    (1, -1),

    (2, 0),
    (2, 1),

    (0, 2),
    (1, 2),
];

const BYTE_SHUFFLE: [u8; 64] = [
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13, 
    3, 0, 2, 1,  7, 4, 6, 5,  11, 8, 10, 9,  15, 12, 14, 13,
];

struct RowWindow {
    top: *mut f32,
    sec: *mut f32,
    thi: *mut f32,
    bot: *mut f32,
    width: usize,
}

impl RowWindow {
    fn new(arena: &mut Arena, width: usize) -> Self {
        Self {
            top: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            sec: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            thi: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            bot: arena
                .allocate::<f32>(width * 2)
                .as_mut_ptr()
                .cast(),
            width,
        }
    }

    #[inline(always)]
    fn fill_row<A: Arch>(
        params: &GridNoiseParams<2>,
        grid_data: &GridData<2>,
        buff: &mut [(f32, f32)],
        cy: i32,
        cx_start: i32
    ) {
        let cy = grid_data.octave_tiling[1].map_or(cy, |t| cy.rem_euclid(t as i32));
        let y_shuf = hash_cell_y::<A>(cy as u32, params.seed);
        let y_shuf_v = Simd::<u32, A>::splat(y_shuf);
        let lanes = Simd::<f32, A>::LANES;

        let mut x_it = 0;
        while x_it + lanes <= buff.len() {
            let cx_base = cx_start.wrapping_add(x_it as i32);
            let cx_v = Simd::<i32, A>::splat(cx_base) + Simd::<i32, A>::iota(0);
            let cx_v = if let Some(t) = grid_data.octave_tiling[0] {
                simd_rem_euclid_i32::<A>(cx_v, t as i32)
            } else {
                cx_v
            };
            let hashes = hash_cells_row::<A>(cx_v.raw_cast(), y_shuf_v, params.seed);
            let (tx, ty) = split_hash_batch::<A>(hashes);
            let tx_arr = tx.to_array();
            let ty_arr = ty.to_array();
            for i in 0..lanes {
                buff[x_it + i] = (tx_arr[i], ty_arr[i]);
            }
            x_it += lanes;
        }
        for i in x_it..buff.len() {
            let cx = cx_start.wrapping_add(i as i32);
            let cx = grid_data.octave_tiling[0].map_or(cx, |t| cx.rem_euclid(t as i32));
            buff[i] = split_hash(hash_cell_with_y::<A>(cx as u32, y_shuf, params.seed));
        }
    }

    fn top(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.top.cast::<(f32, f32)>(), self.width) }
    }

    fn sec(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.sec.cast::<(f32, f32)>(), self.width) }
    }

    fn thi(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.thi.cast::<(f32, f32)>(), self.width) }
    }

    fn bot(&self) -> &[(f32, f32)] {
        unsafe { std::slice::from_raw_parts(self.bot.cast::<(f32, f32)>(), self.width) }
    }

    fn top_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.top.cast::<(f32, f32)>(), self.width) }
    }

    fn sec_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.sec.cast::<(f32, f32)>(), self.width) }
    }

    fn thi_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.thi.cast::<(f32, f32)>(), self.width) }
    }

    fn bot_mut(&mut self) -> &mut [(f32, f32)] {
        unsafe { std::slice::from_raw_parts_mut(self.bot.cast::<(f32, f32)>(), self.width) }
    }

    #[inline(always)]
    fn swap(&mut self) {
        let tmp = self.top;
        self.top = self.sec;
        self.sec = self.thi;
        self.thi = self.bot;
        self.bot = tmp;
    }
}

#[inline(always)]
pub(super) fn hash_cell<A: Arch>(x: u32, y: u32, seed: u32) -> u32 {
    hash_cell_with_y::<A>(x, hash_cell_y::<A>(y, seed), seed)
}

/// Hashes `LANES` consecutive lattice columns from a pre-built `cx_v` vector
/// at fixed y in one shot
#[inline(always)]
fn hash_cells_row<A: Arch>(cx_v: Simd<u32, A>, y_shuf: Simd<u32, A>, seed: u32) -> Simd<u32, A> {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    let seed_v = Simd::<u32, A>::splat(seed);

    let x_shuf = (cx_v * seed_v).permute_8(shuffle_indices) ^ prime;
    (x_shuf * y_shuf) ^ x_shuf
}

#[inline(always)]
fn hash_cell_y<A: Arch>(y: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    (Simd::<u32, A>::splat(y.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime).to_array()[0]
}

#[inline(always)]
fn hash_cell_with_y<A: Arch>(x: u32, y_shuf: u32, seed: u32) -> u32 {
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let prime = Simd::<u32, A>::splat(0x85ebca6b_u32);
    let x_shuf = (
        Simd::<u32, A>::splat(x.wrapping_mul(seed)).permute_8(shuffle_indices) ^ prime
    ).to_array()[0];
    x_shuf.wrapping_mul(y_shuf) ^ x_shuf
}

/// Split a hash into the per-axis jitter offsets. The lower 23 bits
/// become the x-mantissa and the next 23 bits become the y-mantissa, giving
/// unform values in the range `[1, 2)`
#[inline(always)]
pub(super) fn split_hash(hash: u32) -> (f32, f32) {
    let exp_bits = 0x3f800000;
    let hash_mask = 0x007fffff;
    let tx = 1.5 - f32::from_bits((hash & hash_mask) | exp_bits);
    let ty = 1.5 - f32::from_bits((hash >> 9) | exp_bits);
    (tx, ty)
}

#[inline(always)]
fn split_hash_batch<A: Arch>(hash: Simd<u32, A>) -> (Simd<f32, A>, Simd<f32, A>) {
    let exp_bits = Simd::<u32, A>::splat(0x3f800000);
    let hash_mask = Simd::<u32, A>::splat(0x007fffff);
    let one_halves = Simd::<f32, A>::splat(1.5);

    let tx = one_halves - ((hash & hash_mask) | exp_bits).raw_cast::<f32>();
    let ty = one_halves - ((hash >> Simd::<u32, A>::splat(9)) | exp_bits).raw_cast::<f32>();
    (tx, ty)
}

#[inline(always)]
fn simd_rem_euclid_i32<A: Arch>(x: Simd<i32, A>, t: i32) -> Simd<i32, A> {
    let t_f = Simd::<f32, A>::splat(t as f32);
    let x_f = x.cast_float();
    (x_f - (x_f / t_f).floor() * t_f).cast_int_trunc()
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
    fn write<A: Arch>(
        &mut self,
        top: &[(f32, f32)],
        sec: &[(f32, f32)],
        thi: &[(f32, f32)],
        bot: &[(f32, f32)],
        x_it: usize
    ) {
        for (i, &(ox, oy)) in NEIGHBORS_12.iter().enumerate() {
            let row: &[(f32, f32)] = match oy {
                -1 => top,
                0 => sec,
                1 => thi,
                2 => bot,
                _ => unreachable!(),
            };
            let (jx, jy) = row[x_it + ((ox + 1) as usize)];
            unsafe {
                self.parts.add(i * 2).write(MaybeUninit::new(jx + (ox as f32)));
                self.parts.add(i * 2 + 1).write(MaybeUninit::new(jy + (oy as f32)));
            }
        }
    }
}

#[enable_targets(A)]
impl GridGenerator<2> for Cellular {
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        combiner: C::Config,
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

        let grid_data = GridData::new::<A, LERP>(&params, &mut sub_arena, &padded_size);

        let row_len = grid_data.num_loops[0] + 3;
        let mut window = RowWindow::new(&mut arena, row_len);
        let mut cell_jitters = CellJitters::new(&mut arena);

        // Pre-fill the 4 rows: top (y-1), sec (y=0), thi (y=1), bot (y=2)
        let cx_offset = grid_data.grid_start[0] - 1;
        RowWindow::fill_row::<A>(
            &params,
            &grid_data,
            window.top_mut(),
            grid_data.grid_start[1] - 1,
            cx_offset
        );
        RowWindow::fill_row::<A>(
            &params,
            &grid_data,
            window.sec_mut(),
            grid_data.grid_start[1],
            cx_offset
        );
        RowWindow::fill_row::<A>(
            &params,
            &grid_data,
            window.thi_mut(),
            grid_data.grid_start[1] + 1,
            cx_offset
        );
        RowWindow::fill_row::<A>(
            &params,
            &grid_data,
            window.bot_mut(),
            grid_data.grid_start[1] + 2,
            cx_offset
        );

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

                cell_jitters.write::<A>(
                    window.top(),
                    window.sec(),
                    window.thi(),
                    window.bot(),
                    x_it
                );

                grid_cellular_fill::<A, C, INIT, FINAL>(
                    &grid_data,
                    &cell_jitters,
                    x_idx,
                    x_next_idx,
                    y_idx,
                    y_next_idx,
                    dst,
                    state,
                    &combiner
                );

                x_idx = x_next_idx;
            }

            // Swap rows: sec->top, thi->sec, bot->thi, old top->bot
            window.swap();

            // Fill the new bot row
            RowWindow::fill_row::<A>(
                &params,
                &grid_data,
                window.bot_mut(),
                grid_data.grid_start[1] + (y_it as i32) + 3,
                cx_offset
            );

            y_idx = y_next_idx;
        }
    }
}

#[inline(always)]
fn grid_cellular_fill<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    grid_data: &GridData<2>,
    jit: &CellJitters,
    x_idx: usize,
    x_next: usize,
    y_idx: usize,
    y_next: usize,
    dst: &mut [f32],
    state: &mut [f32],
    combiner_config: &C::Config
) {
    let lanes = Simd::<f32, A>::LANES;
    let row_width = grid_data.grid_size[0];

    let weight_vec = Simd::<f32, A>::splat(grid_data.weight);

    for y in y_idx..y_next {
        let sy = unsafe { grid_data.distances[1].get_unchecked(y).assume_init() };
        let row_start = y * row_width;

        // (sy - jy)^2` is identical for every block in this cell segment, 
        // so they are computed once per row instead of once per block.
        let mut dysq: [MaybeUninit<Simd<f32, A>>; 12] =
            std::array::from_fn(|_| MaybeUninit::uninit());
        for c in 0..12 {
            let jy = unsafe { (*jit.parts.add(c * 2 + 1)).assume_init() };
            let dy_sq =
                (Simd::<f32, A>::splat(sy) - Simd::<f32, A>::splat(jy)) *
                (Simd::<f32, A>::splat(sy) - Simd::<f32, A>::splat(jy));
            dysq[c].write(dy_sq);
        }
        let dysq = &dysq;

        let mut index = x_idx;
        while index + lanes <= x_next {
            grid_cellular_fill_block::<A, C, INIT, FINAL, false>(
                grid_data,
                jit,
                &dysq,
                weight_vec,
                sy,
                row_start,
                x_next,
                index,
                dst,
                state,
                combiner_config
            );
            index += lanes;
        }
        if index < x_next {
            grid_cellular_fill_block::<A, C, INIT, FINAL, true>(
                grid_data,
                jit,
                &dysq,
                weight_vec,
                sy,
                row_start,
                x_next,
                index,
                dst,
                state,
                combiner_config
            );
        }
    }
}

#[inline(always)]
fn grid_cellular_fill_block<
    A: Arch,
    C: Combiner,
    const INIT: bool,
    const FINAL: bool,
    const IS_TAIL: bool
>(
    grid_data: &GridData<2>,
    jit: &CellJitters,
    dysq: &[MaybeUninit<Simd<f32, A>>; 12],
    weight_vec: Simd<f32, A>,
    sy: f32,
    row_start: usize,
    x_next: usize,
    index: usize,
    dst: &mut [f32],
    state: &mut [f32],
    combiner_config: &C::Config
) {
    let lanes = if IS_TAIL { x_next - index } else { Simd::<f32, A>::LANES };
    let sample_start = row_start + index;
    let sample_end = sample_start + lanes;

    let sx: Simd<f32, A> = unsafe { grid_data.distances[0].load_simd(index) };

    let mut min_sq = Simd::<f32, A>::splat(f32::MAX);
    for c in 0..4 {
        let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
        let dy_sq = unsafe { *dysq[c].assume_init_ref() };
        let dx = sx - jx;
        min_sq = min_sq.min(dx.mul_add(dx, dy_sq));
    }

    let x_edge_lo = sx + Simd::<f32, A>::splat(0.5);
    let x_edge_hi = Simd::<f32, A>::splat(1.5) - sx;
    let y_edge_lo = Simd::<f32, A>::splat(sy + 0.5);
    let y_edge_hi = Simd::<f32, A>::splat(1.5 - sy);

    let near_sq = {
        let edge = x_edge_lo.min(y_edge_lo);
        edge * edge
    };
    let far_sq = {
        let edge = x_edge_hi.min(y_edge_hi);
        edge * edge
    };
    let any_near = min_sq.simd_gt(near_sq).to_bits() != 0;
    let any_far = min_sq.simd_gt(far_sq).to_bits() != 0;

    if any_near {
        for c in 4..8 {
            let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
            let dy_sq = unsafe { *dysq[c].assume_init_ref() };
            let dx = sx - jx;
            min_sq = min_sq.min(dx.mul_add(dx, dy_sq));
        }
    }

    if any_far {
        for c in 8..12 {
            let jx = Simd::<f32, A>::splat(unsafe { (*jit.parts.add(c * 2)).assume_init() });
            let dy_sq = unsafe { *dysq[c].assume_init_ref() };
            let dx = sx - jx;
            min_sq = min_sq.min(dx.mul_add(dx, dy_sq));
        }
    }

    let raw_val = min_sq.sqrt() * weight_vec;

    let (cur_state, mut result) = if INIT {
        C::initialize_sample(combiner_config, raw_val)
    } else {
        let mut cur_state = C::State::<A>::default();
        for i in 0..C::State::<A>::STATE_SIZE {
            let offset = i * grid_data.total_size;
            cur_state[i] = unsafe {
                maybe_tail_load::<A, IS_TAIL>(sample_start + offset..sample_end + offset, state)
            };
        }
        let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(sample_start..sample_end, dst) };
        C::apply_sample(combiner_config, cur_state, cur_result, raw_val)
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
    use crate::{ Cellular, Fbm, Grid };

    #[test]
    fn cellular_grid_2d_sanity() {
        let grid = Grid::<2>::new(32, 32);
        let mut result = [0.0; 1024];
        grid.builder::<Fbm, Cellular>().fill(result.as_mut_slice());
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

    fn check_reference(
        w: usize,
        h: usize,
        seed: i64,
        offset_x: f32,
        offset_y: f32,
        freq: f32
    ) {
        let grid = Grid::<2>::new(w, h)
            .seed(seed)
            .sample_position(offset_x, offset_y);
        let grid_seed = Random::mix_u64(seed as u64);
        let base_seed = Random::mix_u64_pair(grid_seed, 0xd5e7b3c94f8a1e6b);
        let octave_seed = gen_octave_seed([freq, freq], base_seed);

        let mut result = vec![0.0; w * h];
        grid.builder::<Fbm, Cellular>()
            .frequency(freq)
            .fill(result.as_mut_slice());

        let mut max_diff = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let px = (offset_x + x as f32) * freq;
                let py = (offset_y + y as f32) * freq;
                let reference = reference_cellular(octave_seed, px, py);
                let actual = result[y * w + x];
                max_diff = max_diff.max((actual - reference).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "Grid cellular at freq {freq} diverges from the brute-force Cellular by {max_diff}"
        );
    }

    fn reference_cellular(seed: u32, px: f32, py: f32) -> f32 {
        let cell_x = px.floor() as i32;
        let cell_y = py.floor() as i32;
        let sx = px - px.floor();
        let sy = py - py.floor();

        let mut min_dist = f32::MAX;
        for ox in -3..=3 {
            for oy in -3..=3 {
                let (jx, jy) = split_hash(
                    hash_cell::<StaticArch>((cell_x + ox) as u32, (cell_y + oy) as u32, seed)
                );
                let dx = sx - ((ox as f32) + jx);
                let dy = sy - ((oy as f32) + jy);
                min_dist = min_dist.min(dx.mul_add(dx, dy * dy));
            }
        }
        min_dist.sqrt()
    }
}