use std::array::from_fn;
use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Simd, enable_targets};

use crate::GridGenerator;
use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};
use crate::noise::generators::Value;
use crate::noise::util::grid_data::{PVGridData, Lerp};
use crate::noise::util::grid_helpers::{
    Arena, ArenaBuffer, InterpolationConfig, MaybeUninitSliceSimdExt, maybe_tail_load,
    maybe_tail_store, pad_grid_size, validate_grid_size, validate_state_size,
};
use crate::noise::util::constants::{BYTE_SHUFFLE, VALUE_EXP_MASK, HASH_MASK, HASH_PRIME};
use crate::noise::util::grid_helpers::*;

pub struct ValueGradients3D<'a> {
    pub tlf: &'a mut [MaybeUninit<f32>],
    pub trf: &'a mut [MaybeUninit<f32>],
    pub blf: &'a mut [MaybeUninit<f32>],
    pub brf: &'a mut [MaybeUninit<f32>],
    pub tlb: &'a mut [MaybeUninit<f32>],
    pub trb: &'a mut [MaybeUninit<f32>],
    pub blb: &'a mut [MaybeUninit<f32>],
    pub brb: &'a mut [MaybeUninit<f32>],
    pub grad_buffers: [&'a mut [MaybeUninit<f32>]; 2],
}

impl<'a> ValueGradients3D<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, size: usize) -> Self {
        Self {
            tlf: arena.allocate(size),
            trf: arena.allocate(size),
            blf: arena.allocate(size),
            brf: arena.allocate(size),
            tlb: arena.allocate(size),
            trb: arena.allocate(size),
            blb: arena.allocate(size),
            brb: arena.allocate(size),
            grad_buffers: from_fn(|_| arena.allocate(size)),
        }
    }

    #[inline(always)]
    pub fn swap_top_bottom(&mut self) {
        std::mem::swap(&mut self.tlf, &mut self.blf);
        std::mem::swap(&mut self.trf, &mut self.brf);
        std::mem::swap(&mut self.tlb, &mut self.blb);
        std::mem::swap(&mut self.trb, &mut self.brb);
    }
}

const LERP: u8 = Lerp::Quintic as u8;
#[enable_targets(A)]
impl GridGenerator<3> for Value {
    type GenConfig = ();
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<3>,
        fractal_config: C::Config,
        _generator_config: <Self as GridGenerator<2>>::GenConfig,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        // Validate and pad grid size.
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, 3>(params.grid_size);

        // Arena setup.
        let required_cache = padded_size[0] * 17 + padded_size[1] * 3 + padded_size[2] * 3;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);
        let mut data_arena = arena.allocate_arena(padded_size.iter().fold(0, |n, x| n + 3 * x));
        let mut trilerp_arena = arena.allocate_arena(padded_size[0] * 4);

        // Allocation setup.
        let num_blocks = TrilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS;
        let bilerp_config = InterpolationConfig::new(num_blocks, params.grid_size[0]);
        let grid_data = PVGridData::new::<A, LERP>(&params, &mut data_arena, &padded_size);
        let mut trilerp_buffers = TrilerpBuffers::new(&mut trilerp_arena, padded_size[0]);
        let mut gradients = ValueGradients3D::new(&mut arena, padded_size[0]);

        // Iterate through single y chunks but full x chunks.
        let mut z_cur_index = 0;
        for z_it in 0..grid_data.num_loops[2] {
            let z_next_index =
                unsafe { grid_data.grid_indices[2].get_unchecked(z_it).assume_init() as usize };
            let z_range = z_cur_index..z_next_index;

            // Set the top gradients.
            fill_gradients_3d::<A>(&params, &grid_data, &mut gradients, 0, z_it);
            gradients.swap_top_bottom();

            let mut y_cur_index = 0;
            for y_it in 0..grid_data.num_loops[1] {
                let y_next_index =
                    unsafe { grid_data.grid_indices[1].get_unchecked(y_it).assume_init() as usize };
                let y_range = y_cur_index..y_next_index;

                // Set bottom gradients.
                fill_gradients_3d::<A>(&params, &grid_data, &mut gradients, y_it + 1, z_it);

                trilerp::<A, C, INIT, FINAL>(
                    &mut trilerp_buffers,
                    &bilerp_config,
                    &fractal_config,
                    &grid_data,
                    &gradients,
                    (y_range, z_range.clone()),
                    (state, dst),
                );

                // Reuse the top and bottom gradients.
                gradients.swap_top_bottom();

                y_cur_index = y_next_index;
            }
            z_cur_index = z_next_index;
        }
    }
}

#[inline(always)]
pub(super) fn fill_gradients_3d<'a, A: Arch>(
    params: &GridNoiseParams<3>,
    grid_data: &PVGridData<3>,
    gradients: &mut ValueGradients3D<'a>,
    y_it: usize,
    z_it: usize,
) {
    let lanes = Simd::<f32, A>::LANES;
    let y_start = y_it as i32 + grid_data.grid_start[1];
    let z_start = z_it as i32 + grid_data.grid_start[2];
    let (z1, z2) = match grid_data.octave_tiling[2] {
        None => (
            (z_start as u32).wrapping_mul(params.seed),
            (z_start as u32)
                .wrapping_mul(params.seed)
                .wrapping_add(params.seed),
        ),
        Some(t) => (
            (z_start.rem_euclid(t as i32)) as u32,
            ((z_start + 1).rem_euclid(t as i32)) as u32,
        ),
    };
    let z_vec = [Simd::splat(z1), Simd::splat(z2)];

    let y_rem = grid_data.octave_tiling[1].map_or(y_start, |t| y_start.rem_euclid(t as i32));
    let y_vec = Simd::splat((y_rem as u32).wrapping_mul(params.seed));

    let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);

    let prime = Simd::splat(HASH_PRIME);
    let z_shuf: [_; 2] = from_fn(|i| z_vec[i].permute_8(shuffle_indices) ^ prime);
    let y_shuf = y_vec.permute_8(shuffle_indices) ^ prime;
    let zy_mix: [_; 2] = from_fn(|i| z_shuf[i] * y_shuf);

    // Main vectorized bit mixing loop.
    let end_index = grid_data.num_loops[0] + 1;
    let hash_mask: Simd<u32, A> = Simd::splat(HASH_MASK);
    let exp_bits: Simd<u32, A> = Simd::splat(VALUE_EXP_MASK);
    let three: Simd<f32, A> = Simd::splat(3.0);

    if let Some(x_tiling) = grid_data.octave_tiling[0] {
        let x_tiling = Simd::splat(x_tiling as f32);
        let mut x_vec = Simd::splat(grid_data.grid_start[0]) + Simd::iota(0);
        let x_vec_stride = Simd::splat(lanes as i32);
        let seed_vec = Simd::splat(params.seed);

        for i in (0..end_index).step_by(lanes) {
            let x_floats = x_vec.cast_float();
            let x_rem = x_floats - (x_floats / x_tiling).floor() * x_tiling;
            let x_seeded = x_rem.cast_int_round().raw_cast() * seed_vec;
            let x_shuf = x_seeded.permute_8(shuffle_indices) ^ prime;
            let hashes: [_; 2] = from_fn(|i| zy_mix[i] + x_shuf * y_shuf);
            let grads: [_; 2] =
                from_fn(|i| ((hashes[i] & hash_mask) | exp_bits).raw_cast() - three);

            unsafe {
                gradients.grad_buffers[0].write_simd_aligned(i, grads[0]);
                gradients.grad_buffers[1].write_simd_aligned(i, grads[1]);
            };

            x_vec += x_vec_stride;
        }
    } else {
        let iota_vec = Simd::iota(0) * Simd::splat(params.seed);
        let x_start_seeded = (grid_data.grid_start[0] as u32).wrapping_mul(params.seed);
        let mut x_vec = Simd::splat(x_start_seeded) + iota_vec;
        let x_vec_stride = Simd::splat((lanes as u32).wrapping_mul(params.seed));

        for i in (0..end_index).step_by(lanes) {
            let x_shuf = x_vec.permute_8(shuffle_indices) ^ prime;
            let hashes: [_; 2] = from_fn(|i| zy_mix[i] + x_shuf * y_shuf);
            let grads: [_; 2] =
                from_fn(|i| ((hashes[i] & hash_mask) | exp_bits).raw_cast() - three);

            unsafe {
                gradients.grad_buffers[0].write_simd_aligned(i, grads[0]);
                gradients.grad_buffers[1].write_simd_aligned(i, grads[1]);
            };
            x_vec += x_vec_stride;
        }
    }

    fill_gradients_3d_set_loop::<A, true>(grid_data, gradients);
    fill_gradients_3d_set_loop::<A, false>(grid_data, gradients);
}

#[inline(always)]
pub(super) fn grid_gradients_3d_set_loop<'a, A: Arch, const IS_FRONT: bool>(
    grid_data: &PVGridData<3>,
    gradients: &mut ValueGradients3D<'a>,
) {
    let (grad_buffer, left, right) = if IS_FRONT {
        (
            &mut gradients.grad_buffers[0],
            &mut gradients.blf,
            &mut gradients.brf,
        )
    } else {
        (
            &mut gradients.grad_buffers[1],
            &mut gradients.blb,
            &mut gradients.brb,
        )
    };

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
                left.write_simd(index, Simd::<_, A>::splat(l));
                right.write_simd(index, Simd::<_, A>::splat(r));

                amount -= Simd::<f32, A>::LANES as isize;
                index += Simd::<f32, A>::LANES;
            }
        }

        x_cur_index = x_next_index;
    }
}

pub(crate) struct TrilerpBuffers<'a> {
    tf_base: &'a mut [MaybeUninit<f32>],
    bf_base: &'a mut [MaybeUninit<f32>],
    top_base_dif: &'a mut [MaybeUninit<f32>],
    bottom_base_dif: &'a mut [MaybeUninit<f32>],
}

impl<'a> TrilerpBuffers<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, x_size: usize) -> Self {
        Self {
            tf_base: arena.allocate(x_size),
            bf_base: arena.allocate(x_size),
            top_base_dif: arena.allocate(x_size),
            bottom_base_dif: arena.allocate(x_size),
        }
    }
}

#[inline(always)]
pub(super) fn fill_gradients_3d_set_loop<'a, A: Arch, const IS_FRONT: bool>(
    grid_data: &PVGridData<3>,
    gradients: &mut ValueGradients3D<'a>,
) {
    let (grad_buffer, left, right) = if IS_FRONT {
        (
            &mut gradients.grad_buffers[0],
            &mut gradients.blf,
            &mut gradients.brf,
        )
    } else {
        (
            &mut gradients.grad_buffers[1],
            &mut gradients.blb,
            &mut gradients.brb,
        )
    };

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
                left.write_simd(index, Simd::<_, A>::splat(l));
                right.write_simd(index, Simd::<_, A>::splat(r));

                amount -= Simd::<f32, A>::LANES as isize;
                index += Simd::<f32, A>::LANES;
            }
        }

        x_cur_index = x_next_index;
    }
}

/// Handles interpolation execution state and fills
/// the dst slice with interpolated values from gradient dot produtcts.
pub(crate) struct TrilerpExecuter<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool> {
    config: &'a InterpolationConfig<A>,
    fractal_config: &'a C::Config,
    grid_data: &'a PVGridData<'a, 3>,
    gradients: &'a ValueGradients3D<'a>,
    y_range: Range<usize>,
    z_range: Range<usize>,
    top: A::Block2<f32>,
    dif: A::Block2<f32>,
    weight: Simd<f32, A>,
}

/// Fills the dst slice with interpolated dot products from gradients.
#[inline(always)]
pub(super) fn trilerp<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    buffers: &mut TrilerpBuffers,
    config: &InterpolationConfig<A>,
    fractal_config: &C::Config,
    grid_data: &PVGridData<3>,
    gradients: &ValueGradients3D,
    ranges: (Range<usize>, Range<usize>),
    output: (&mut [f32], &mut [f32]),
) {
    let mut executer = TrilerpExecuter::<A, C, INIT, FINAL> {
        config,
        fractal_config,
        grid_data,
        gradients,
        y_range: ranges.0,
        z_range: ranges.1,
        top: Default::default(),
        dif: Default::default(),
        weight: Simd::splat(grid_data.weight),
    };

    executer.initialize_trilerp_buffers(buffers);

    let (state, dst) = output;

    if config.has_block_head {
        executer.interpolate::<false, 0, FULL>(buffers, state, dst);
    }

    if config.has_block_tail {
        std::hint::cold_path();
        let block_tail_size = config.block_tail_size;
        let has_partial = config.has_partial;
        let has_block_head = config.has_block_head;
        match TrilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS {
            1 => match (block_tail_size, has_partial, has_block_head) {
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(buffers, state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(buffers, state, dst),
                _ => {}
            },
            2 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(buffers, state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(buffers, state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(buffers, state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(buffers, state, dst),
                _ => {}
            },
            4 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(buffers, state, dst),
                (2, false, _) => executer.interpolate::<true, 2, FULL>(buffers, state, dst),
                (3, false, _) => executer.interpolate::<true, 3, FULL>(buffers, state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(buffers, state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(buffers, state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(buffers, state, dst),
                (2, true, _) => executer.interpolate::<true, 2, PARTIAL>(buffers, state, dst),
                (3, true, _) => executer.interpolate::<true, 3, PARTIAL>(buffers, state, dst),
                _ => {}
            },
            8 => match (block_tail_size, has_partial, has_block_head) {
                (1, false, _) => executer.interpolate::<true, 1, FULL>(buffers, state, dst),
                (2, false, _) => executer.interpolate::<true, 2, FULL>(buffers, state, dst),
                (3, false, _) => executer.interpolate::<true, 3, FULL>(buffers, state, dst),
                (4, false, _) => executer.interpolate::<true, 4, FULL>(buffers, state, dst),
                (5, false, _) => executer.interpolate::<true, 5, FULL>(buffers, state, dst),
                (6, false, _) => executer.interpolate::<true, 6, FULL>(buffers, state, dst),
                (7, false, _) => executer.interpolate::<true, 7, FULL>(buffers, state, dst),
                (0, true, false) => executer.interpolate::<true, 0, SCALAR>(buffers, state, dst),
                (0, true, true) => executer.interpolate::<true, 0, PARTIAL>(buffers, state, dst),
                (1, true, _) => executer.interpolate::<true, 1, PARTIAL>(buffers, state, dst),
                (2, true, _) => executer.interpolate::<true, 2, PARTIAL>(buffers, state, dst),
                (3, true, _) => executer.interpolate::<true, 3, PARTIAL>(buffers, state, dst),
                (4, true, _) => executer.interpolate::<true, 4, PARTIAL>(buffers, state, dst),
                (5, true, _) => executer.interpolate::<true, 5, PARTIAL>(buffers, state, dst),
                (6, true, _) => executer.interpolate::<true, 6, PARTIAL>(buffers, state, dst),
                (7, true, _) => executer.interpolate::<true, 7, PARTIAL>(buffers, state, dst),
                _ => {}
            },
            _ => {}
        }
    }
}

impl<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>
    TrilerpExecuter<'a, A, C, INIT, FINAL>
{
    const MAX_BLOCKS: usize = A::NUM_SIMD_REG / 4;

    #[inline(always)]
    pub fn interpolate<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        buffers: &TrilerpBuffers,
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

        let z_hop = self.grid_data.grid_size[0] * self.grid_data.grid_size[1];
        let y_hop = self.grid_data.grid_size[0];
        for z in self.z_range.start..self.z_range.end {
            let z_lerp = unsafe { self.grid_data.fade_factors[2].get_unchecked(z) };
            let z_lerp = unsafe { z_lerp.assume_init() };
            let z_lerp = Simd::splat(z_lerp);

            let mut x = range.start;
            while x < range.end {
                self.intialize_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(buffers, x, z_lerp);

                let index = z * z_hop;
                let mut y = self.y_range.start;
                while y < self.y_range.end {
                    let index = index + y * y_hop;
                    if y + 4 > self.y_range.end {
                        std::hint::cold_path();
                        process(self, state, dst, x, y, index);
                        y += 1;
                    } else {
                        process(self, state, dst, x, y, index);
                        process(self, state, dst, x, y + 1, index + y_hop);
                        process(self, state, dst, x, y + 2, index + 2 * y_hop);
                        process(self, state, dst, x, y + 3, index + 3 * y_hop);
                        y += 4;
                    }
                }
                x += self.config.block_lanes;
            }
        }
    }

    #[inline(always)]
    fn initialize_trilerp_buffers(&mut self, buffers: &mut TrilerpBuffers) {
        for x in (0..self.grid_data.grid_size[0]).step_by(Simd::<f32, A>::LANES) {
            unsafe {
                let x_lerp = self.grid_data.fade_factors[0].load_simd_aligned(x);

                let tlf = self.gradients.tlf.load_simd_aligned(x);
                let trf = self.gradients.trf.load_simd_aligned(x);
                let blf = self.gradients.blf.load_simd_aligned(x);
                let brf = self.gradients.brf.load_simd_aligned(x);
                let tlb = self.gradients.tlb.load_simd_aligned(x);
                let trb = self.gradients.trb.load_simd_aligned(x);
                let blb = self.gradients.blb.load_simd_aligned(x);
                let brb = self.gradients.brb.load_simd_aligned(x);

                let tf_base = x_lerp.mul_add(trf - tlf, tlf) * self.weight;
                let bf_base = x_lerp.mul_add(brf - blf, blf) * self.weight;

                let hi_base_dif = x_lerp.mul_add(trb - tlb, tlb).mul_sub(self.weight, tf_base);
                let lo_base_dif = x_lerp.mul_add(brb - blb, blb).mul_sub(self.weight, bf_base);

                buffers.tf_base.write_simd_aligned(x, tf_base);
                buffers.bf_base.write_simd_aligned(x, bf_base);
                buffers.top_base_dif.write_simd_aligned(x, hi_base_dif);
                buffers.bottom_base_dif.write_simd_aligned(x, lo_base_dif);
            }
        }
    }

    #[inline(always)]
    fn intialize_factors<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        buffers: &TrilerpBuffers,
        x: usize,
        z_lerp: Simd<f32, A>,
    ) {
        let num_blocks = if IS_TAIL {
            NUM_BLOCKS
        } else {
            Self::MAX_BLOCKS
        };

        // These blocked loops will get entirely unrolled by the compiler.
        for block in 0..num_blocks {
            let index = x + Simd::<f32, A>::LANES * block;
            self.initialize_factors_block::<FULL>(buffers, index, block, z_lerp);
        }

        if ACCESS_MODE != FULL {
            self.initialize_factors_block::<ACCESS_MODE>(buffers, 0, NUM_BLOCKS, z_lerp);
        }
    }

    #[inline(always)]
    fn initialize_factors_block<const ACCESS_MODE: u8>(
        &mut self,
        buffers: &TrilerpBuffers,
        index: usize,
        block: usize,
        z_lerp: Simd<f32, A>,
    ) {
        let access_mode = SimdAccessMode::from_u8(ACCESS_MODE);
        let (tf, bf, top_dif, bottom_dif) = unsafe {
            match access_mode {
                // Use unaligned load to capture 'end' of the partial vector.
                SimdAccessMode::Partial => {
                    let index = index + self.config.partial_start;
                    (
                        buffers.tf_base.load_simd(index),
                        buffers.bf_base.load_simd(index),
                        buffers.top_base_dif.load_simd(index),
                        buffers.bottom_base_dif.load_simd(index),
                    )
                }
                // Use aligned load to capture full vector for Full mode.
                // Use aligned load to capture padded vector for Scalar mode.
                _ => (
                    buffers.tf_base.load_simd_aligned(index),
                    buffers.bf_base.load_simd_aligned(index),
                    buffers.top_base_dif.load_simd_aligned(index),
                    buffers.bottom_base_dif.load_simd_aligned(index),
                ),
            }
        };

        // Base interpolation.
        unsafe {
            *self.top.get_unchecked_mut(block) = z_lerp.mul_add(top_dif, tf);
            let bottom = z_lerp.mul_add(bottom_dif, bf);
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
