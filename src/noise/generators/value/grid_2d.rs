use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Simd, enable_targets};

use crate::GridGenerator;
use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};
use crate::noise::generators::Value;
use crate::noise::util::grid_data::{PVGridData, Lerp};
use crate::noise::util::grid_helpers::{
    Arena, ArenaBuffer, InterpolationConfig, MaybeUninitSliceSimdExt, assume_init_slice,
    maybe_tail_load, maybe_tail_store, pad_grid_size, validate_grid_size, validate_state_size,
};

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

impl<'a> fmt::Debug for ValueGradients2D<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            f.debug_struct("PVGridData")
                .field("tl", &assume_init_slice(self.tl))
                .field("tr", &assume_init_slice(self.tr))
                .field("bl", &assume_init_slice(self.bl))
                .field("br", &assume_init_slice(self.br))
                .finish()
        }
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
        let num_blocks = A::NUM_SIMD_REG / 4;
        let bilerp_config = InterpolationConfig::new(num_blocks, params.grid_size[0]);

        let mut sub_arena = arena.allocate_arena(padded_size[0] * 3 + padded_size[1] * 3);
        let mut grid_data = PVGridData::new::<A, LERP>(&params, &mut sub_arena, &padded_size);

        // Allocate scratch buffer for gradients.
        let grad_scratch = arena.allocate(padded_size[0]);

        // Initialize gradient vectors.
        let mut gradients = ValueGradients2D::new(&mut arena, padded_size[0]);

        // Set the top gradients.
        grid_gradients_2d::<A>(
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
            grid_gradients_2d::<A>(
                &params,
                &mut grid_data,
                grad_scratch,
                gradients.bl,
                gradients.br,
                y_it + 1,
            );

            let y_range = y_cur_index..y_next_index;
            grid_bilerp::<A, C, INIT, FINAL>(
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
pub(super) fn grid_gradients_2d<'a, A: Arch>(
    params: &GridNoiseParams<2>,
    grid_data: &mut PVGridData<2>,
    grad_buffer: &mut [MaybeUninit<f32>],
    left: &'a mut [MaybeUninit<f32>],
    right: &'a mut [MaybeUninit<f32>],
    y_it: usize,
) {
    let lanes = Simd::<f32, A>::LANES;
    let y_start = grid_data.grid_start[1] + y_it as i32;
    let y_rem = grid_data.octave_tiling[1].map_or(y_start, |t| y_start.rem_euclid(t as i32));
    let y_vec = Simd::splat((y_rem as u32).wrapping_mul(params.seed));

    let prime = Simd::splat(0x85ebca6b_u32);
    const BYTE_SHUFFLE: [u8; 64] = [
        3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9,
        15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6,
        5, 11, 8, 10, 9, 15, 12, 14, 13,
    ];
    let shuffle_indices = unsafe { Simd::<u8, A>::from_slice_unchecked(&BYTE_SHUFFLE[..]) };
    let y_shuf = y_vec.permute_8(shuffle_indices) ^ prime;
    let y_shuf = y_shuf * y_shuf;

    let hash_mask: Simd<u32, A> = Simd::splat(0x007FFFFF);
    let exp_bits: Simd<u32, A> = Simd::splat(0x40000000);
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
pub(super) fn grid_bilerp<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
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
        executer.interpolate::<false>(state, dst);
    }

    if config.has_block_tail {
        executer.interpolate::<true>(state, dst);
        std::hint::cold_path();
    }
}

impl<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>
    BilerpExecuter<'a, A, C, INIT, FINAL>
{
    #[inline(always)]
    pub fn interpolate<const IS_TAIL: bool>(&mut self, state: &mut [f32], dst: &mut [f32]) {
        let range = if IS_TAIL {
            self.config.block_tail_start..self.grid_data.grid_size[0]
        } else {
            0..self.config.block_tail_start
        };

        for x in range.step_by(self.config.block_lanes) {
            self.initialize_factors::<IS_TAIL>(x);

            let mut y = self.y_range.start;
            while y < self.y_range.end {
                if y + 4 > self.y_range.end {
                    self.process_factors::<IS_TAIL>(x, y, state, dst);
                    y += 1;
                } else {
                    self.process_factors::<IS_TAIL>(x, y, state, dst);
                    self.process_factors::<IS_TAIL>(x, y + 1, state, dst);
                    self.process_factors::<IS_TAIL>(x, y + 2, state, dst);
                    self.process_factors::<IS_TAIL>(x, y + 3, state, dst);
                    y += 4;
                }
            }
        }
    }

    #[inline(always)]
    fn initialize_factors<const IS_TAIL: bool>(&mut self, x: usize) {
        let num_blocks = if IS_TAIL {
            self.config.block_tail_size
        } else {
            self.config.num_blocks
        };

        // These blocked loops will get entirely unrolled by the compiler.
        for block in 0..num_blocks {
            // Load gradients into registers.
            let index = x + Simd::<f32, A>::LANES * block;

            let x_lerp = unsafe { self.grid_data.fade_factors[0].load_simd_aligned(index) };
            let tl = unsafe { self.gradients.tl.load_simd_aligned(index) };
            let tr = unsafe { self.gradients.tr.load_simd_aligned(index) };
            let bl = unsafe { self.gradients.bl.load_simd_aligned(index) };
            let br = unsafe { self.gradients.br.load_simd_aligned(index) };

            // Base interpolation.
            self.top[block] = x_lerp.mul_add(tr - tl, tl) * self.weight;
            let bottom = x_lerp.mul_add(br - bl, bl) * self.weight;
            self.dif[block] = bottom - self.top[block];
        }
    }

    #[inline(always)]
    fn process_factors<const IS_TAIL: bool>(
        &mut self,
        x: usize,
        y: usize,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        let y_lerp = Simd::splat(unsafe {
            self.grid_data.fade_factors[1]
                .get_unchecked(y)
                .assume_init()
        });

        let range = if IS_TAIL {
            0..self.config.block_tail_size
        } else {
            0..self.config.num_blocks
        };

        let index = y * self.grid_data.grid_size[0] + x;
        let tail_end = index + self.config.tail_size;
        for block in range {
            let index = index + block * Simd::<f32, A>::LANES;
            let output = y_lerp.mul_add(self.dif[block], self.top[block]);

            let (cur_state, mut result) = if INIT {
                C::initialize_sample(self.fractal_config, output)
            } else {
                let mut cur_state = C::State::<A>::default();
                for i in 0..C::State::<A>::STATE_SIZE {
                    let offset = i * self.grid_data.total_size;
                    let index = index + offset;
                    let tail_end = tail_end + offset;
                    cur_state[i] = unsafe { maybe_tail_load::<A, IS_TAIL>(index..tail_end, state) };
                }
                let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(index..tail_end, dst) };
                C::apply_sample(self.fractal_config, cur_state, cur_result, output)
            };

            // Save changes to state.
            if !FINAL {
                for i in 0..C::State::<A>::STATE_SIZE {
                    let offset = i * self.grid_data.total_size;
                    let index = index + offset;
                    let tail_end = tail_end + offset;
                    unsafe { maybe_tail_store::<A, IS_TAIL>(index..tail_end, cur_state[i], state) };
                }
            }

            if FINAL {
                result = C::finalize_sample(self.fractal_config, cur_state, result);
            }

            unsafe { maybe_tail_store::<A, IS_TAIL>(index..tail_end, result, dst) };
        }
    }
}
