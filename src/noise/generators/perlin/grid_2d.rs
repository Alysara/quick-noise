use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Simd, enable_targets};

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};
use crate::noise::generators::perlin::batch_2d::{X_GRADIENTS_2D, Y_GRADIENTS_2D};
use crate::noise::util::constants::{BYTE_SHUFFLE, HASH_PRIME};
use crate::noise::util::grid_data::{Lerp, PVGridData};
use crate::noise::util::grid_helpers::{
    Arena, ArenaBuffer, InterpolationConfig, MaybeUninitSliceSimdExt, pad_grid_size,
    validate_grid_size, validate_state_size, *,
};
use crate::{GridGenerator, Perlin};

pub struct PerlinGradients2D<'a> {
    pub top: [&'a mut [MaybeUninit<f32>]; 2],
    pub bot: [&'a mut [MaybeUninit<f32>]; 2],
}

impl<'a> PerlinGradients2D<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, size: usize) -> Self {
        Self {
            top: [arena.allocate(size), arena.allocate(size)],
            bot: [arena.allocate(size), arena.allocate(size)],
        }
    }

    #[inline(always)]
    pub fn swap_top_bottom(&mut self) {
        std::mem::swap(&mut self.top, &mut self.bot);
    }
}

const LERP: u8 = Lerp::Quintic as u8;
#[enable_targets(A)]
impl GridGenerator<2> for Perlin {
    type GenConfig = ();
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<2>,
        fractal_config: C::Config,
        _generator_config: (),
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        // Validate the inputs.
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, _>(params.grid_size);

        // 3 * padded_size[1] + 3 * padded_size[0] for grid data,
        // 2 * padded_size[0] for the per-pixel left/right cell hashes,
        // 4 * padded_size[0] for the two x-lerped gradient rows (x and y each).
        let required_cache = padded_size[1] * 3 + padded_size[0] * 9;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);

        let mut sub_arena = arena.allocate_arena(padded_size[0] * 3 + padded_size[1] * 3);
        let grid_data = PVGridData::new::<A, LERP>(&params, &mut sub_arena, &padded_size);

        // Initialize gradient vectors.
        let mut gradients = PerlinGradients2D::new(&mut arena, padded_size[0]);

        // Initialize gradient filler and interpolator
        let mut gradient_filler = GradientFiller::<A>::new(&params, &grid_data);
        let mut interpolator = DottedBilerpExecuter::<A, C, INIT, FINAL>::new(
            &params,
            &fractal_config,
            &grid_data,
            state,
            dst,
        );

        // Set the top gradients.
        gradient_filler.fill_gradients(0, &mut gradients.top);

        // Get y_grid_indices for conveient accessing.
        let y_grid_indices = unsafe { grid_data.grid_indices[1].assume_init_ref() };

        // Iterate through single y chunks but full x chunks.
        let mut y_cur_index = 0;
        for y_it in 0..grid_data.num_loops[1] {
            // Get the next boundary along the y-axis.
            let y_next_index = unsafe { *y_grid_indices.get_unchecked(y_it) as usize };

            // Set bottom gradients.
            gradient_filler.fill_gradients(y_it as u32 + 1, &mut gradients.bot);

            // Interpolate full x-rows along the current y-cell (y_cur_index..y_next_index).
            interpolator.fill_interpolations(&gradients, y_cur_index, y_next_index);

            // Reuse the bottom gradients by setting them as the new top gradients.
            gradients.swap_top_bottom();

            y_cur_index = y_next_index;
        }
    }
}

pub(super) struct GradientFiller<'a, A: Arch> {
    params: &'a GridNoiseParams<2>,
    grid_data: &'a PVGridData<'a, 2>,
    prime: Simd<u32, A>,
    shuffle_indices: Simd<u8, A>,
    x_cur: Simd<i32, A>,
    y_shuf: Simd<u32, A>,
    x_stride: Simd<u32, A>,
    x_tiling: Simd<f32, A>,
    seed_vec: Simd<u32, A>,
    has_tiling: bool,
    x_prev_grad: Simd<f32, A>,
    y_prev_grad: Simd<f32, A>,
    x_cur_index: u32,
    end_index: u32,
    cur_loop: u32,
}

impl<'a, A: Arch> GradientFiller<'a, A> {
    #[inline(always)]
    pub(crate) fn new(params: &'a GridNoiseParams<2>, grid_data: &'a PVGridData<2>) -> Self {
        let x_tiling = Simd::splat(grid_data.octave_tiling[0].unwrap_or(0) as f32);
        let has_tiling = grid_data.octave_tiling[0].is_some();
        Self {
            params,
            grid_data,
            prime: Simd::splat(HASH_PRIME),
            shuffle_indices: unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) },
            x_cur: Simd::zero(),
            y_shuf: Simd::zero(),
            x_stride: Simd::zero(),
            seed_vec: Simd::splat(params.seed),
            x_tiling,
            has_tiling,
            x_prev_grad: Simd::zero(),
            y_prev_grad: Simd::zero(),
            x_cur_index: 0,
            end_index: 0,
            cur_loop: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn fill_gradients(&mut self, y_it: u32, out: &mut [&mut [MaybeUninit<f32>]; 2]) {
        let lanes = Simd::<f32, A>::LANES;

        self.end_index = self.grid_data.num_loops[0] as u32 + 1;
        self.x_cur_index = 0;
        self.cur_loop = 0;

        if !self.has_tiling {
            // Normal path, set up strides with seed baked in.
            let iota_seed = Simd::iota(0) * Simd::<i32, A>::splat(self.params.seed as i32);
            let start_seed = (self.grid_data.grid_start[0] as u32).wrapping_mul(self.params.seed);

            self.y_shuf = self.compute_y_shuf(y_it);
            self.x_cur = Simd::<i32, A>::splat(start_seed as i32) + iota_seed;
            self.x_stride = Simd::<u32, A>::splat((lanes as u32).wrapping_mul(self.params.seed));

            self.dispatch_grad_filler::<false>(out);
        } else {
            // Tiling path, unbaked strides.
            self.y_shuf = self.compute_y_shuf(y_it);
            self.x_cur = Simd::splat(self.grid_data.grid_start[0]) + Simd::iota(0);
            self.x_stride = Simd::splat(lanes as u32);

            self.dispatch_grad_filler::<true>(out);
        }
    }

    #[inline(always)]
    pub(crate) fn dispatch_grad_filler<const TILING: bool>(
        &mut self,
        out: &mut [&mut [MaybeUninit<f32>]; 2],
    ) {
        let num_full = self.end_index / Simd::<f32, A>::LANES as u32;
        let has_tail = !self.end_index.is_multiple_of(Simd::<f32, A>::LANES as u32);

        // Dispatch to optimal asm for each case (INIT, FINAL).
        if num_full == 0 {
            self.fill_grads_inner::<TILING, true, true>(out);
            return;
        }

        self.fill_grads_inner::<TILING, true, false>(out);

        for _ in 0..(num_full - 1) {
            self.fill_grads_inner::<TILING, false, false>(out);
        }

        if has_tail {
            self.fill_grads_inner::<TILING, false, true>(out);
        }
    }

    #[inline(always)]
    pub(crate) fn fill_grads_inner<const TILING: bool, const FIRST: bool, const LAST: bool>(
        &mut self,
        out: &mut [&mut [MaybeUninit<f32>]; 2],
    ) {
        let one = Simd::splat(1);

        let mut cur_permute = Simd::<u32, A>::zero();
        let (x_grads, y_grads) = self.compute_grads::<TILING>();

        let num_set_loops = match (FIRST, LAST) {
            (true, true) => self.end_index - 1,
            (true, false) => Simd::<f32, A>::LANES as u32 - 1,
            (false, true) => self.end_index - self.cur_loop - 1,
            (false, false) => Simd::<f32, A>::LANES as u32,
        };

        if FIRST {
            self.x_prev_grad = x_grads.permute_32(cur_permute);
            self.y_prev_grad = y_grads.permute_32(cur_permute);
            cur_permute += one;
        }

        for _ in 0..num_set_loops {
            let x_next_index = unsafe {
                self.grid_data.grid_indices[0]
                    .get_unchecked(self.cur_loop as usize)
                    .assume_init()
            };
            let x_next_grad = x_grads.permute_32(cur_permute);
            let y_next_grad = y_grads.permute_32(cur_permute);

            let y_dif = y_next_grad - self.y_prev_grad;
            let x_dif = x_next_grad - self.x_prev_grad;

            let end = x_next_index as usize;
            let mut index = self.x_cur_index as usize;
            while index < end {
                let dist = unsafe { self.grid_data.distances[0].load_simd(index) };
                let x_fade: Simd<f32, A> =
                    unsafe { self.grid_data.fade_factors[0].load_simd(index) };

                // Get the dot products.
                let left_dot = self.x_prev_grad * dist;
                let x_dotted_dif = x_dif.mul_sub(dist, x_next_grad); // Equal to x(d - 1).

                // Interpolate with the x-fade factory.
                let x_component = x_fade.mul_add(x_dotted_dif, left_dot);
                let y_component = x_fade.mul_add(y_dif, self.y_prev_grad);

                unsafe {
                    out[0].write_simd(index, x_component);
                    out[1].write_simd(index, y_component);
                }

                index += Simd::<f32, A>::LANES;
            }

            cur_permute += one;
            self.x_prev_grad = x_next_grad;
            self.y_prev_grad = y_next_grad;
            self.x_cur_index = x_next_index;
            self.cur_loop += 1;
        }

        self.x_cur += self.x_stride.raw_cast();
    }

    #[inline(always)]
    fn compute_y_shuf(&self, y_it: u32) -> Simd<u32, A> {
        let y_start = self.grid_data.grid_start[1] + y_it as i32;
        let y_rem = self.grid_data.octave_tiling[1].map_or(y_start, |t| {
            let t = t as i32;
            // Assert t > 0 to avoid checks after validation.
            unsafe { core::hint::assert_unchecked(t > 0) };
            y_start.rem_euclid(t)
        });
        let y_vec = Simd::<u32, A>::splat((y_rem as u32).wrapping_mul(self.params.seed));

        let prime = Simd::<u32, A>::splat(HASH_PRIME);
        let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
        y_vec.permute_8(shuffle_indices) ^ prime
    }

    #[inline(always)]
    fn compute_grads<const TILING: bool>(&self) -> (Simd<f32, A>, Simd<f32, A>) {
        // If tiling, wrap the coordinates before seeding.
        // Otherwise just take the current x_cur as it has strides
        // with the seed baked in.
        let x_cur = match TILING {
            false => self.x_cur.raw_cast(),
            true => {
                let x_floats = self.x_cur.cast_float();
                let x_rem = x_floats - (x_floats / self.x_tiling).floor() * self.x_tiling;
                x_rem.cast_int_round().raw_cast() * self.seed_vec
            }
        };

        let x_shuf = x_cur.permute_8(self.shuffle_indices).raw_cast() ^ self.prime;
        let indices: Simd<u32, A> = (self.y_shuf * x_shuf) >> 29;

        let x_grads = indices.gather(&X_GRADIENTS_2D);
        let y_grads = indices.gather(&Y_GRADIENTS_2D);

        (x_grads, y_grads)
    }
}

/// Handles interpolation execution state and fills
/// the dst slice with interpolated values from gradient dot produtcts.
pub(crate) struct DottedBilerpExecuter<
    'a,
    A: Arch,
    C: Combiner,
    const INIT: bool,
    const FINAL: bool,
> {
    config: InterpolationConfig<A>,
    fractal_config: &'a C::Config,
    grid_data: &'a PVGridData<'a, 2>,
    y_range: Range<usize>,
    top: A::Block2<f32>,
    dif: A::Block2<f32>,
    d_top: A::Block2<f32>,
    d_dif: A::Block2<f32>,
    weight_vec: Simd<f32, A>,
    y_weighted_increment: Simd<f32, A>,
    y_upper_increment: Simd<f32, A>,
    y_lower_increment: Simd<f32, A>,
    state: &'a mut [f32],
    dst: &'a mut [f32],
}

impl<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>
    DottedBilerpExecuter<'a, A, C, INIT, FINAL>
{
    const MAX_BLOCKS: usize = A::NUM_SIMD_REG / 8;

    #[inline(always)]
    pub fn new(
        params: &GridNoiseParams<2>,
        fractal_config: &'a C::Config,
        grid_data: &'a PVGridData<2>,
        state: &'a mut [f32],
        dst: &'a mut [f32],
    ) -> Self {
        let config = InterpolationConfig::<A>::new(Self::MAX_BLOCKS, params.grid_size[0]);
        Self {
            config,
            fractal_config,
            grid_data,
            y_range: 0..0,
            top: Default::default(),
            dif: Default::default(),
            d_top: Default::default(),
            d_dif: Default::default(),
            weight_vec: Simd::splat(grid_data.weight),
            y_weighted_increment: Simd::splat(grid_data.increment[1] * grid_data.weight),
            y_upper_increment: Simd::zero(),
            y_lower_increment: Simd::zero(),
            state,
            dst,
        }
    }

    #[inline(always)]
    pub fn fill_interpolations(&mut self, grads: &PerlinGradients2D, y_start: usize, y_end: usize) {
        let y_frac_start = unsafe {
            self.grid_data.distances[1]
                .get_unchecked(y_start)
                .assume_init()
        };

        self.y_upper_increment = Simd::splat(y_frac_start);
        self.y_lower_increment = Simd::splat(y_frac_start - 1.0);
        self.y_range = y_start..y_end;

        if self.config.has_block_head {
            self.bilerp::<false, 0, FULL>(grads);
        }

        if self.config.has_block_tail {
            std::hint::cold_path();
            let block_tail_size = self.config.block_tail_size;
            let has_partial = self.config.has_partial;
            let has_block_head = self.config.has_block_head;
            match DottedBilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS {
                1 => match (block_tail_size, has_partial, has_block_head) {
                    (0, true, false) => self.bilerp::<true, 0, SCALAR>(grads),
                    (0, true, true) => self.bilerp::<true, 0, PARTIAL>(grads),
                    _ => {}
                },
                2 => match (block_tail_size, has_partial, has_block_head) {
                    (1, false, _) => self.bilerp::<true, 1, FULL>(grads),
                    (0, true, false) => self.bilerp::<true, 0, SCALAR>(grads),
                    (0, true, true) => self.bilerp::<true, 0, PARTIAL>(grads),
                    (1, true, _) => self.bilerp::<true, 1, PARTIAL>(grads),
                    _ => {}
                },
                4 => match (block_tail_size, has_partial, has_block_head) {
                    (1, false, _) => self.bilerp::<true, 1, FULL>(grads),
                    (2, false, _) => self.bilerp::<true, 2, FULL>(grads),
                    (3, false, _) => self.bilerp::<true, 3, FULL>(grads),
                    (0, true, false) => self.bilerp::<true, 0, SCALAR>(grads),
                    (0, true, true) => self.bilerp::<true, 0, PARTIAL>(grads),
                    (1, true, _) => self.bilerp::<true, 1, PARTIAL>(grads),
                    (2, true, _) => self.bilerp::<true, 2, PARTIAL>(grads),
                    (3, true, _) => self.bilerp::<true, 3, PARTIAL>(grads),
                    _ => {}
                },
                8 => match (block_tail_size, has_partial, has_block_head) {
                    (1, false, _) => self.bilerp::<true, 1, FULL>(grads),
                    (2, false, _) => self.bilerp::<true, 2, FULL>(grads),
                    (3, false, _) => self.bilerp::<true, 3, FULL>(grads),
                    (4, false, _) => self.bilerp::<true, 4, FULL>(grads),
                    (5, false, _) => self.bilerp::<true, 5, FULL>(grads),
                    (6, false, _) => self.bilerp::<true, 6, FULL>(grads),
                    (7, false, _) => self.bilerp::<true, 7, FULL>(grads),
                    (0, true, false) => self.bilerp::<true, 0, SCALAR>(grads),
                    (0, true, true) => self.bilerp::<true, 0, PARTIAL>(grads),
                    (1, true, _) => self.bilerp::<true, 1, PARTIAL>(grads),
                    (2, true, _) => self.bilerp::<true, 2, PARTIAL>(grads),
                    (3, true, _) => self.bilerp::<true, 3, PARTIAL>(grads),
                    (4, true, _) => self.bilerp::<true, 4, PARTIAL>(grads),
                    (5, true, _) => self.bilerp::<true, 5, PARTIAL>(grads),
                    (6, true, _) => self.bilerp::<true, 6, PARTIAL>(grads),
                    (7, true, _) => self.bilerp::<true, 7, PARTIAL>(grads),
                    _ => {}
                },
                _ => {}
            }
        }
    }

    #[inline(always)]
    pub fn bilerp<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        gradients: &PerlinGradients2D,
    ) {
        let range = if IS_TAIL {
            self.config.block_tail_start..self.grid_data.grid_size[0]
        } else {
            0..self.config.block_tail_start
        };

        let process = |this: &mut Self, x, y, index| {
            this.process_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(x, y, index)
        };

        let mut x = range.start;
        while x < range.end {
            self.initialize_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(gradients, x);

            let mut y = self.y_range.start;
            let y_step = self.grid_data.grid_size[0];
            let mut idx = y * y_step;

            let fast_end = self.y_range.end.saturating_sub(3);

            // Fast unrolled 4x y iteration path.
            while y < fast_end {
                process(self, x, y, idx);
                process(self, x, y + 1, idx + y_step);
                process(self, x, y + 2, idx + 2 * y_step);
                process(self, x, y + 3, idx + 3 * y_step);
                y += 4;
                idx += y_step * 4;
            }

            // Slow single y iteration path.
            while y < self.y_range.end {
                process(self, x, y, idx);
                y += 1;
                idx += y_step;
            }
            x += self.config.block_lanes;
        }
    }

    #[inline(always)]
    fn initialize_factors<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        gradients: &PerlinGradients2D,
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
            self.initialize_factors_block::<FULL>(gradients, index, block);
        }

        if ACCESS_MODE != FULL {
            self.initialize_factors_block::<ACCESS_MODE>(gradients, 0, NUM_BLOCKS);
        }
    }

    #[inline(always)]
    fn initialize_factors_block<const ACCESS_MODE: u8>(
        &mut self,
        gradients: &PerlinGradients2D,
        index: usize,
        block: usize,
    ) {
        // The x-lerp was already applied while filling gradients, so only the
        // y direction remains.
        let (ax_t, ay_t, ax_b, ay_b) = unsafe {
            (
                gradients.top[0].ld_buf::<ACCESS_MODE>(index, &self.config),
                gradients.top[1].ld_buf::<ACCESS_MODE>(index, &self.config),
                gradients.bot[0].ld_buf::<ACCESS_MODE>(index, &self.config),
                gradients.bot[1].ld_buf::<ACCESS_MODE>(index, &self.config),
            )
        };

        // Dot product at the current y offset for the top and bottom grid rows.
        let top = ay_t.mul_add(self.y_upper_increment, ax_t) * self.weight_vec;
        let bot = ay_b.mul_add(self.y_lower_increment, ax_b) * self.weight_vec;

        unsafe {
            *self.top.get_unchecked_mut(block) = top;
            *self.dif.get_unchecked_mut(block) = bot - top;

            // Per-row increments.
            *self.d_top.get_unchecked_mut(block) = ay_t * self.y_weighted_increment;
            *self.d_dif.get_unchecked_mut(block) = (ay_b - ay_t) * self.y_weighted_increment;
        }
    }

    #[inline(always)]
    fn process_factors<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        x: usize,
        y: usize,
        index: usize,
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

            self.process_factors_block::<FULL>(block, y_lerp, index);
        }

        if ACCESS_MODE != FULL {
            self.process_factors_block::<ACCESS_MODE>(NUM_BLOCKS, y_lerp, index);
        }
    }

    #[inline(always)]
    fn process_factors_block<const ACCESS_MODE: u8>(
        &mut self,
        block: usize,
        y_lerp: Simd<f32, A>,
        index: usize,
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
                cur_state[i] =
                    unsafe { load_simd_rw(self.state, index, &self.config, access_mode) };
            }

            let cur_result = unsafe { load_simd_rw(self.dst, index, &self.config, access_mode) };
            C::apply_sample(self.fractal_config, cur_state, cur_result, output)
        };

        // Save changes to state.
        if !FINAL {
            for i in 0..C::State::<A>::STATE_SIZE {
                let offset = i * self.grid_data.total_size;
                let index = index + offset;
                unsafe {
                    write_simd_rw(self.state, index, cur_state[i], &self.config, access_mode)
                };
            }
        }

        // Finalize result if it is the last one.
        if FINAL {
            result = C::finalize_sample(self.fractal_config, cur_state, result);
        }

        // Save the result.
        unsafe { write_simd_rw(self.dst, index, result, &self.config, access_mode) };

        // Walk along the grid by accumulating interpolation components.
        unsafe {
            *self.dif.get_unchecked_mut(block) += *self.d_dif.get_unchecked(block);
            *self.top.get_unchecked_mut(block) += *self.d_top.get_unchecked(block);
        }
    }
}
