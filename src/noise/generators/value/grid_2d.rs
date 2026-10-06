use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Simd, enable_targets};

use crate::GridGenerator;
use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};
use crate::noise::generators::Value;
use crate::noise::util::grid_data::{PVGridData, Lerp};
use crate::noise::util::grid_helpers::{
    Arena, ArenaBuffer, InterpolationConfig, MaybeUninitSliceSimdExt, 
    pad_grid_size, validate_grid_size, validate_state_size,
};
use crate::noise::util::constants::{BYTE_SHUFFLE, VALUE_EXP_MASK, HASH_MASK, HASH_PRIME};
use crate::noise::util::grid_helpers::*;

pub struct ValueGradients2D<'a> {
    pub tl: &'a mut [MaybeUninit<f32>],
    pub tr: &'a mut [MaybeUninit<f32>],
    pub bl: &'a mut [MaybeUninit<f32>],
    pub br: &'a mut [MaybeUninit<f32>],
}

impl<'a> ValueGradients2D<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, size: usize) -> Self {
        Self {
            tl: arena.allocate(size),
            tr: arena.allocate(size),
            bl: arena.allocate(size),
            br: arena.allocate(size),
        }
    }

    #[inline(always)]
    pub fn swap_top_bottom(&mut self) {
        std::mem::swap(&mut self.tl, &mut self.bl);
        std::mem::swap(&mut self.tr, &mut self.br);
    }
}

const LERP: u8 = Lerp::Cubic as u8;
#[enable_targets(A)]
impl GridGenerator<2> for Value {
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        fractal_config: C::Config,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, 2>(params.grid_size);

        let required_cache = padded_size[1] * 3 + padded_size[0] * 8;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);

        // SIMD Slice constants.
        let num_blocks = BilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS;
        let bilerp_config = InterpolationConfig::new(num_blocks, params.grid_size[0]);

        let mut sub_arena = arena.allocate_arena(padded_size[0] * 3 + padded_size[1] * 3);
        let mut grid_data = PVGridData::new::<A, LERP>(&params, &mut sub_arena, &padded_size);

        // Allocate scratch buffer for gradients.
        let grad_scratch = arena.allocate(padded_size[0]);

        // Initialize gradient vectors.
        let mut gradients = ValueGradients2D::new(&mut arena, padded_size[0]);

        // Set the top gradients.
        fill_gradients_2d::<A>(
            &params,
            &mut grid_data,
            grad_scratch,
            gradients.tl,
            gradients.tr,
            0,
        );

        // Iterate through single y chunks but full x chunks.
        let mut y_cur_index = 0;
        for y_it in 0..grid_data.num_loops[1] {
            let y_next_index =
                unsafe { grid_data.grid_indices[1].get_unchecked(y_it).assume_init() as usize };

            // Set bottom gradients.
            fill_gradients_2d::<A>(
                &params,
                &mut grid_data,
                grad_scratch,
                gradients.bl,
                gradients.br,
                y_it + 1,
            );

            let y_range = y_cur_index..y_next_index;
            bilerp::<A, C, INIT, FINAL>(
                &bilerp_config,
                &fractal_config,
                &grid_data,
                &gradients,
                y_range,
                (state, dst),
            );

            // Reuse the top and bottom gradients.
            gradients.swap_top_bottom();

            y_cur_index = y_next_index;
        }
    }
}

#[inline(always)]
pub(super) fn fill_gradients_2d<'a, A: Arch>(
    params: &GridNoiseParams<2>,
    grid_data: &mut PVGridData<2>,
    grad_buffer: &mut [MaybeUninit<f32>],
    left: &'a mut [MaybeUninit<f32>],
    right: &'a mut [MaybeUninit<f32>],
    y_it: usize,
) {
    let lanes = Simd::<f32, A>::LANES;
    let y_start = grid_data.grid_start[1] + y_it as i32;
    let y_rem = grid_data.octave_tiling[1].map_or(y_start, |t| {
        let t = t as i32;
        // Assert t > 0 to avoid checks after validation.
        unsafe { core::hint::assert_unchecked(t > 0) };
        y_start.rem_euclid(t)
    });

    let y_vec = Simd::splat((y_rem as u32).wrapping_mul(params.seed));

    let prime = Simd::splat(HASH_PRIME);
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let y_shuf = y_vec.permute_8(shuffle_indices) ^ prime;
    let y_shuf = y_shuf * y_shuf;

    let hash_mask: Simd<u32, A> = Simd::splat(HASH_MASK);
    let exp_bits: Simd<u32, A> = Simd::splat(VALUE_EXP_MASK);
    let three: Simd<f32, A> = Simd::splat(3.0);

    if let Some(x_tiling) = grid_data.octave_tiling[0] {
        let x_tiling = Simd::splat(x_tiling as f32);
        let mut x_vec = Simd::splat(grid_data.grid_start[0]) + Simd::iota(0);
        let x_vec_stride = Simd::splat(lanes as i32);
        let seed_vec = Simd::splat(params.seed);

        let end_index = grid_data.num_loops[0] + 1;
        for i in (0..end_index).step_by(lanes) {
            let x_floats = x_vec.cast_float();
            let x_rem = x_floats - (x_floats / x_tiling).floor() * x_tiling;
            let x_seeded = x_rem.cast_int_round().raw_cast() * seed_vec;

            let x_shuf = x_seeded.permute_8(shuffle_indices) ^ prime;
            let hash = y_shuf * x_shuf;
            let grad = ((hash & hash_mask) | exp_bits).raw_cast() - three;
            unsafe { grad_buffer.write_simd_aligned(i, grad) };
            x_vec += x_vec_stride;
        }
    } else {
        let iota_vec = Simd::iota(0) * Simd::splat(params.seed);
        let mut x_vec =
            Simd::splat((grid_data.grid_start[0] as u32).wrapping_mul(params.seed)) + iota_vec;
        let x_vec_stride = Simd::splat((lanes as u32).wrapping_mul(params.seed));

        // Main vectorized bit mixing loop.
        let end_index = grid_data.num_loops[0] + 1;
        for i in (0..end_index).step_by(lanes) {
            let x_shuf = x_vec.permute_8(shuffle_indices) ^ prime;
            let hash = y_shuf * x_shuf;
            let grad = ((hash & hash_mask) | exp_bits).raw_cast() - three;
            unsafe { grad_buffer.write_simd_aligned(i, grad) };
            x_vec += x_vec_stride;
        }
    }

    // Loop through the x chunks.
    let mut x_cur_index = 0;
    for x_it in 0..grid_data.num_loops[0] {
        // Find range of gradients to set.
        let x_next_index = unsafe { grid_data.grid_indices[0].get_unchecked(x_it).assume_init() };
        let mut amount = (x_next_index - x_cur_index) as isize;

        unsafe {
            let l = grad_buffer.get_unchecked(x_it).assume_init();
            let r = grad_buffer.get_unchecked(x_it + 1).assume_init();

            let mut index = x_cur_index as usize;
            while amount > 0 {
                left.write_simd(index, Simd::<f32, A>::splat(l));
                right.write_simd(index, Simd::<f32, A>::splat(r));
                amount -= lanes as isize;
                index += lanes;
            }
        }

        x_cur_index = x_next_index;
    }
}

/// Handles interpolation execution state and fills
/// the dst slice with interpolated values from gradient dot produtcts.
pub(crate) struct BilerpExecuter<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool> {
    config: &'a InterpolationConfig<A>,
    fractal_config: &'a C::Config,
    grid_data: &'a PVGridData<'a, 2>,
    gradients: &'a ValueGradients2D<'a>,
    y_range: Range<usize>,
    top: A::Block2<f32>,
    dif: A::Block2<f32>,
    weight: Simd<f32, A>,
}

/// Fills the dst slice with interpolated dot products from gradients.
#[inline(always)]
pub(super) fn bilerp<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    config: &InterpolationConfig<A>,
    fractal_config: &C::Config,
    grid_data: &PVGridData<2>,
    gradients: &ValueGradients2D,
    y_range: Range<usize>,
    output: (&mut [f32], &mut [f32]),
) {
    let mut executer = BilerpExecuter::<A, C, INIT, FINAL> {
        config,
        fractal_config,
        grid_data,
        gradients,
        y_range,
        top: Default::default(),
        dif: Default::default(),
        weight: Simd::splat(grid_data.weight),
    };

    let (state, dst) = output;

    if config.has_block_head {
        executer.interpolate::<false, 0, FULL>(state, dst);
    }

    if config.has_block_tail {
        std::hint::cold_path();
        let block_tail_size = config.block_tail_size;
        let has_partial = config.has_partial;
        let has_block_head = config.has_block_head;
        match BilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS {
            1 => match (block_tail_size, has_partial, has_block_head) {
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(state, dst),
                _ => {}
            },
            2 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(state, dst),
                _ => {}
            },
            4 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(state, dst),
                (2, false, _) => executer.interpolate::<true, 2, FULL>(state, dst),
                (3, false, _) => executer.interpolate::<true, 3, FULL>(state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(state, dst),
                (2, true, _) => executer.interpolate::<true, 2, PARTIAL>(state, dst),
                (3, true, _) => executer.interpolate::<true, 3, PARTIAL>(state, dst),
                _ => {}
            },
            8 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(state, dst),
                (2, false, _) => executer.interpolate::<true, 2, FULL>(state, dst),
                (3, false, _) => executer.interpolate::<true, 3, FULL>(state, dst),
                (4, false, _) => executer.interpolate::<true, 4, FULL>(state, dst),
                (5, false, _) => executer.interpolate::<true, 5, FULL>(state, dst),
                (6, false, _) => executer.interpolate::<true, 6, FULL>(state, dst),
                (7, false, _) => executer.interpolate::<true, 7, FULL>(state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(state, dst),
                (2, true, _) => executer.interpolate::<true, 2, PARTIAL>(state, dst),
                (3, true, _) => executer.interpolate::<true, 3, PARTIAL>(state, dst),
                (4, true, _) => executer.interpolate::<true, 4, PARTIAL>(state, dst),
                (5, true, _) => executer.interpolate::<true, 5, PARTIAL>(state, dst),
                (6, true, _) => executer.interpolate::<true, 6, PARTIAL>(state, dst),
                (7, true, _) => executer.interpolate::<true, 7, PARTIAL>(state, dst),
                _ => {}
            },
            _ => {}
        }
    }
}

impl<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>
    BilerpExecuter<'a, A, C, INIT, FINAL>
{
    const MAX_BLOCKS: usize = A::NUM_SIMD_REG / 4;

    #[inline(always)]
    pub fn interpolate<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        let range = if IS_TAIL {
            self.config.block_tail_start..self.grid_data.grid_size[0]
        } else {
            0..self.config.block_tail_start
        };

        let process = |this: &mut Self, state: &mut [f32], dst: &mut [f32], x, y, index| {
            this.process_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(x, y, index, state, dst)
        };

        let mut x = range.start;
        while x < range.end {
            self.initialize_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(x);

            let mut y = self.y_range.start;
            let y_step = self.grid_data.grid_size[0];
            let mut idx = y * y_step;
            while y < self.y_range.end {
                if y + 4 > self.y_range.end {
                    // Slow single y iteration path.
                    std::hint::cold_path();
                    process(self, state, dst, x, y, idx);
                    y += 1;
                    idx += y_step;
                } else {
                    // Fast unrolled 4x y iteration path.
                    process(self, state, dst, x, y, idx);
                    process(self, state, dst, x, y + 1, idx + y_step);
                    process(self, state, dst, x, y + 2, idx + 2 * y_step);
                    process(self, state, dst, x, y + 3, idx + 3 * y_step);
                    y += 4;
                    idx += y_step * 4;
                }
            }
            x += self.config.block_lanes;
        }
    }

    #[inline(always)]
    fn initialize_factors<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        x: usize,
    ) {
        let num_blocks = if IS_TAIL {
            NUM_BLOCKS
        } else {
            Self::MAX_BLOCKS
        };

        // These blocked loops will get entirely unrolled by the compiler.
        for block in 0..num_blocks {
            // Load gradients into registers.
            let index = x + Simd::<f32, A>::LANES * block;
            self.initialize_factors_block::<FULL>(index, block);
        }

        if ACCESS_MODE != FULL {
            self.initialize_factors_block::<ACCESS_MODE>(0, NUM_BLOCKS);
        }
    }

    #[inline(always)]
    fn initialize_factors_block<const ACCESS_MODE: u8>(&mut self, index: usize, block: usize) {
        let (x_lerp, tl, tr, bl, br) = unsafe {
            (
                self.grid_data.fade_factors[0].ld_buf::<ACCESS_MODE>(index, self.config),
                self.gradients.tl.ld_buf::<ACCESS_MODE>(index, self.config),
                self.gradients.tr.ld_buf::<ACCESS_MODE>(index, self.config),
                self.gradients.bl.ld_buf::<ACCESS_MODE>(index, self.config),
                self.gradients.br.ld_buf::<ACCESS_MODE>(index, self.config),
            )
        };

        // Base interpolation.
        unsafe {
            *self.top.get_unchecked_mut(block) = x_lerp.mul_add(tr - tl, tl) * self.weight;
            let bottom = x_lerp.mul_add(br - bl, bl) * self.weight;
            *self.dif.get_unchecked_mut(block) = bottom - *self.top.get_unchecked(block);
        }
    }

    #[inline(always)]
    fn process_factors<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        x: usize,
        y: usize,
        index: usize,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        let y_lerp = Simd::splat(unsafe {
            self.grid_data.fade_factors[1]
                .get_unchecked(y)
                .assume_init()
        });

        let num_blocks = if IS_TAIL {
            NUM_BLOCKS
        } else {
            Self::MAX_BLOCKS
        };

        for block in 0..num_blocks {
            let index = index + x + block * Simd::<f32, A>::LANES;

            self.process_factors_block::<FULL>(block, y_lerp, index, state, dst);
        }

        if ACCESS_MODE != FULL {
            self.process_factors_block::<ACCESS_MODE>(NUM_BLOCKS, y_lerp, index, state, dst);
        }
    }

    #[inline(always)]
    fn process_factors_block<const ACCESS_MODE: u8>(
        &mut self,
        block: usize,
        y_lerp: Simd<f32, A>,
        index: usize,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        let access_mode = SimdAccessMode::from_u8(ACCESS_MODE);
        let output = unsafe {
            y_lerp.mul_add(
                *self.dif.get_unchecked(block),
                *self.top.get_unchecked(block),
            )
        };

        // Initialize sample if it is the first and apply combiner algorithm
        // for subsequent samples.
        let (cur_state, mut result) = if INIT {
            C::initialize_sample(self.fractal_config, output)
        } else {
            let mut cur_state = C::State::<A>::default();
            for i in 0..C::State::<A>::STATE_SIZE {
                let offset = i * self.grid_data.total_size;
                let index = index + offset;
                cur_state[i] = unsafe { load_simd_rw(state, index, self.config, access_mode) };
            }

            let cur_result = unsafe { load_simd_rw(dst, index, self.config, access_mode) };
            C::apply_sample(self.fractal_config, cur_state, cur_result, output)
        };

        // Save changes to state.
        if !FINAL {
            for i in 0..C::State::<A>::STATE_SIZE {
                let offset = i * self.grid_data.total_size;
                let index = index + offset;
                unsafe { write_simd_rw(state, index, cur_state[i], self.config, access_mode) };
            }
        }

        // Finalize result if it is the last one.
        if FINAL {
            result = C::finalize_sample(self.fractal_config, cur_state, result);
        }

        // Save the result.
        unsafe { write_simd_rw(dst, index, result, self.config, access_mode) };
    }
}
