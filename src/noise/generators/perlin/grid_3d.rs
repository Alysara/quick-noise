use std::array::from_fn;
use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Simd, enable_targets};

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};
use crate::noise::util::grid_data::{GridData, Lerp};
use crate::noise::util::grid_helpers::*;
use crate::{GridGenerator, Perlin};

pub const GRADIENTS_3D: [[f32; 3]; 16] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [0.0, -1.0, -1.0],
];

pub struct PerlinGradients3D<'a> {
    pub tlf: [&'a mut [MaybeUninit<f32>]; 3],
    pub trf: [&'a mut [MaybeUninit<f32>]; 3],
    pub blf: [&'a mut [MaybeUninit<f32>]; 3],
    pub brf: [&'a mut [MaybeUninit<f32>]; 3],
    pub tlb: [&'a mut [MaybeUninit<f32>]; 3],
    pub trb: [&'a mut [MaybeUninit<f32>]; 3],
    pub blb: [&'a mut [MaybeUninit<f32>]; 3],
    pub brb: [&'a mut [MaybeUninit<f32>]; 3],
    pub scratch: [&'a mut [MaybeUninit<u32>]; 2],
}

impl<'a> PerlinGradients3D<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, size: usize) -> Self {
        Self {
            tlf: from_fn(|_| arena.allocate(size)),
            trf: from_fn(|_| arena.allocate(size)),
            blf: from_fn(|_| arena.allocate(size)),
            brf: from_fn(|_| arena.allocate(size)),
            tlb: from_fn(|_| arena.allocate(size)),
            trb: from_fn(|_| arena.allocate(size)),
            blb: from_fn(|_| arena.allocate(size)),
            brb: from_fn(|_| arena.allocate(size)),
            scratch: from_fn(|_| arena.allocate(size)),
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
impl GridGenerator<3> for Perlin {
    fn sample_grid<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<3>,
        fractal_config: C::Config,
        state: &mut [f32],
        dst: &mut [f32],
    ) {
        // Validate and pad grid size.
        validate_grid_size(params.grid_size, dst.len());
        validate_state_size::<C, A, _>(params.grid_size, state.len());
        let padded_size = pad_grid_size::<A, 3>(params.grid_size);

        // Arena setup.
        let required_cache = padded_size[0] * 41 + padded_size[1] * 3 + padded_size[2] * 3;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);
        let mut data_arena = arena.allocate_arena(padded_size.iter().fold(0, |n, x| n + 3 * x));
        let mut trilerp_arena = arena.allocate_arena(padded_size[0] * 12);

        // Allocation setup.

        let num_blocks = DottedTrilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS;
        let bilerp_config = InterpolationConfig::new(num_blocks, params.grid_size[0]);
        let grid_data = GridData::new::<A, LERP>(&params, &mut data_arena, &padded_size);
        let mut trilerp_buffers = DottedTrilerpBuffers::new(&mut trilerp_arena, padded_size[0]);
        let mut gradients = PerlinGradients3D::new(&mut arena, padded_size[0]);

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

                dotted_trilerp::<A, C, INIT, FINAL>(
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
    grid_data: &GridData<3>,
    gradients: &mut PerlinGradients3D<'a>,
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

    const BYTE_SHUFFLE: [u8; 64] = [
        3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9,
        15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13, 3, 0, 2, 1, 7, 4, 6,
        5, 11, 8, 10, 9, 15, 12, 14, 13,
    ];

    let shuffle_indices = Simd::<u8, A>::from_slice(&BYTE_SHUFFLE[..]);

    let prime = Simd::splat(0x85ebca6b_u32);
    let z_shuf: [_; 2] = from_fn(|i| z_vec[i].permute_8(shuffle_indices) ^ prime);
    let y_shuf = y_vec.permute_8(shuffle_indices) ^ prime;
    let zy_mix: [_; 2] = from_fn(|i| z_shuf[i] * y_shuf);

    // Main vectorized bit mixing loop.
    let end_index = grid_data.num_loops[0] + 1;

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
            let grads: [_; 2] = from_fn(|i| (zy_mix[i] * x_shuf) >> 28);

            unsafe {
                gradients.scratch[0].write_simd_aligned(i, grads[0]);
                gradients.scratch[1].write_simd_aligned(i, grads[1]);
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
            let grads: [_; 2] = from_fn(|i| (zy_mix[i] * x_shuf) >> 28);

            unsafe {
                gradients.scratch[0].write_simd_aligned(i, grads[0]);
                gradients.scratch[1].write_simd_aligned(i, grads[1]);
            };
            x_vec += x_vec_stride;
        }
    }

    fill_gradients_3d_set_loop::<A, true>(grid_data, gradients);
    fill_gradients_3d_set_loop::<A, false>(grid_data, gradients);

    for i in (0..params.grid_size[0]).step_by(lanes) {
        unsafe {
            let cur_dist: Simd<f32, A> = grid_data.distances[0].load_simd_aligned(i);
            let lf = gradients.blf[0].load_simd_aligned(i);
            let rf = gradients.brf[0].load_simd_aligned(i);
            let lb = gradients.blb[0].load_simd_aligned(i);
            let rb = gradients.brb[0].load_simd_aligned(i);

            gradients.blf[0].write_simd_aligned(i, lf * cur_dist);
            gradients.brf[0].write_simd_aligned(i, rf.mul_sub(cur_dist, rf));
            gradients.blb[0].write_simd_aligned(i, lb * cur_dist);
            gradients.brb[0].write_simd_aligned(i, rb.mul_sub(cur_dist, rb));
        }
    }
}

#[inline(always)]
pub(super) fn fill_gradients_3d_set_loop<'a, A: Arch, const IS_FRONT: bool>(
    grid_data: &GridData<3>,
    gradients: &mut PerlinGradients3D<'a>,
) {
    let (grad_buffer, left, right) = if IS_FRONT {
        (
            &mut gradients.scratch[0],
            &mut gradients.blf,
            &mut gradients.brf,
        )
    } else {
        (
            &mut gradients.scratch[1],
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
            let l = grad_buffer.get_unchecked(x_it).assume_init() as usize;
            let r = grad_buffer.get_unchecked(x_it + 1).assume_init() as usize;

            let l = GRADIENTS_3D.get_unchecked(l);
            let r = GRADIENTS_3D.get_unchecked(r);

            let lx = Simd::<f32, A>::splat(l[0]);
            let ly = Simd::<f32, A>::splat(l[1]);
            let lz = Simd::<f32, A>::splat(l[2]);
            let rx = Simd::<f32, A>::splat(r[0]);
            let ry = Simd::<f32, A>::splat(r[1]);
            let rz = Simd::<f32, A>::splat(r[2]);

            let mut index = x_cur_index as usize;
            while amount > 0 {
                left[0].write_simd(index, lx);
                left[1].write_simd(index, ly);
                left[2].write_simd(index, lz);
                right[0].write_simd(index, rx);
                right[1].write_simd(index, ry);
                right[2].write_simd(index, rz);

                amount -= Simd::<f32, A>::LANES as isize;
                index += Simd::<f32, A>::LANES;
            }
        }

        x_cur_index = x_next_index;
    }
}

pub(crate) struct DottedTrilerpBuffers<'a> {
    y_tf_offset: &'a mut [MaybeUninit<f32>],
    y_bf_offset: &'a mut [MaybeUninit<f32>],
    y_top_offset_dif: &'a mut [MaybeUninit<f32>],
    y_bottom_offset_dif: &'a mut [MaybeUninit<f32>],
    z_tf_offset: &'a mut [MaybeUninit<f32>],
    z_bf_offset: &'a mut [MaybeUninit<f32>],
    z_top_offset_dif: &'a mut [MaybeUninit<f32>],
    z_bottom_offset_dif: &'a mut [MaybeUninit<f32>],
    tf_base: &'a mut [MaybeUninit<f32>],
    bf_base: &'a mut [MaybeUninit<f32>],
    top_base_dif: &'a mut [MaybeUninit<f32>],
    bottom_base_dif: &'a mut [MaybeUninit<f32>],
}

impl<'a> DottedTrilerpBuffers<'a> {
    #[inline(always)]
    pub fn new(arena: &'a mut Arena, x_size: usize) -> Self {
        Self {
            y_tf_offset: arena.allocate(x_size),
            y_bf_offset: arena.allocate(x_size),
            y_top_offset_dif: arena.allocate(x_size),
            y_bottom_offset_dif: arena.allocate(x_size),
            z_tf_offset: arena.allocate(x_size),
            z_bf_offset: arena.allocate(x_size),
            z_top_offset_dif: arena.allocate(x_size),
            z_bottom_offset_dif: arena.allocate(x_size),
            tf_base: arena.allocate(x_size),
            bf_base: arena.allocate(x_size),
            top_base_dif: arena.allocate(x_size),
            bottom_base_dif: arena.allocate(x_size),
        }
    }
}

/// Handles interpolation execution state and fills
/// the dst slice with interpolated values from gradient dot produtcts.
pub(crate) struct DottedTrilerpExecuter<
    'a,
    A: Arch,
    C: Combiner,
    const INIT: bool,
    const FINAL: bool,
> {
    config: &'a InterpolationConfig<A>,
    fractal_config: &'a C::Config,
    grid_data: &'a GridData<'a, 3>,
    gradients: &'a PerlinGradients3D<'a>,
    y_range: Range<usize>,
    z_range: Range<usize>,
    top: A::Block4<f32>,
    dif: A::Block4<f32>,
    top_step: A::Block4<f32>,
    dif_step: A::Block4<f32>,
    weight: Simd<f32, A>,
    y_inc_weighted: Simd<f32, A>,
    y_inc_hi: Simd<f32, A>,
    y_inc_lo: Simd<f32, A>,
    z_inc_weighted: Simd<f32, A>,
    z_inc_hi: Simd<f32, A>,
    z_inc_lo: Simd<f32, A>,
}

/// Fills the dst slice with interpolated dot products from gradients.
#[inline(always)]
pub(super) fn dotted_trilerp<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    buffers: &mut DottedTrilerpBuffers,
    config: &InterpolationConfig<A>,
    fractal_config: &C::Config,
    grid_data: &GridData<3>,
    gradients: &PerlinGradients3D,
    ranges: (Range<usize>, Range<usize>),
    output: (&mut [f32], &mut [f32]),
) {
    let y_frac_start = unsafe {
        grid_data.distances[1]
            .get_unchecked(ranges.0.start)
            .assume_init()
    };
    let z_frac_start = unsafe {
        grid_data.distances[2]
            .get_unchecked(ranges.1.start)
            .assume_init()
    };

    let mut executer = DottedTrilerpExecuter::<A, C, INIT, FINAL> {
        config,
        fractal_config,
        grid_data,
        gradients,
        y_range: ranges.0,
        z_range: ranges.1,
        top: Default::default(),
        dif: Default::default(),
        top_step: Default::default(),
        dif_step: Default::default(),
        weight: Simd::splat(grid_data.weight),
        y_inc_weighted: Simd::splat(grid_data.increment[1] * grid_data.weight),
        y_inc_hi: Simd::splat(y_frac_start),
        y_inc_lo: Simd::splat(y_frac_start - 1.0),
        z_inc_weighted: Simd::splat(grid_data.increment[2] * grid_data.weight),
        z_inc_hi: Simd::splat(z_frac_start),
        z_inc_lo: Simd::splat(z_frac_start - 1.0),
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
        match DottedTrilerpExecuter::<A, C, INIT, FINAL>::MAX_BLOCKS {
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
    DottedTrilerpExecuter<'a, A, C, INIT, FINAL>
{
    const MAX_BLOCKS: usize = A::NUM_SIMD_REG / 8;

    #[inline(always)]
    pub fn interpolate<const IS_TAIL: bool, const NUM_BLOCKS: usize, const ACCESS_MODE: u8>(
        &mut self,
        buffers: &DottedTrilerpBuffers,
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

        let mut z_cur = Simd::splat(0.0);
        let z_hop = self.grid_data.grid_size[0] * self.grid_data.grid_size[1];
        let y_hop = self.grid_data.grid_size[0];
        for z in self.z_range.start..self.z_range.end {
            let z_lerp = unsafe { self.grid_data.fade_factors[2].get_unchecked(z) };
            let z_lerp = unsafe { z_lerp.assume_init() };
            let z_lerp = Simd::splat(z_lerp);

            let mut x = range.start;
            while x < range.end {
                self.intialize_factors::<IS_TAIL, NUM_BLOCKS, ACCESS_MODE>(
                    buffers, x, z_cur, z_lerp,
                );

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
            z_cur += Simd::splat(1.0);
        }
    }

    #[inline(always)]
    fn initialize_trilerp_buffers(&mut self, buffers: &mut DottedTrilerpBuffers) {
        for x in (0..self.grid_data.grid_size[0]).step_by(Simd::<f32, A>::LANES) {
            unsafe {
                let x_lerp = self.grid_data.fade_factors[0].load_simd_aligned(x);

                let x_tlf = self.gradients.tlf[0].load_simd_aligned(x);
                let x_trf = self.gradients.trf[0].load_simd_aligned(x);
                let x_blf = self.gradients.blf[0].load_simd_aligned(x);
                let x_brf = self.gradients.brf[0].load_simd_aligned(x);
                let x_tlb = self.gradients.tlb[0].load_simd_aligned(x);
                let x_trb = self.gradients.trb[0].load_simd_aligned(x);
                let x_blb = self.gradients.blb[0].load_simd_aligned(x);
                let x_brb = self.gradients.brb[0].load_simd_aligned(x);

                let y_tlf = self.gradients.tlf[1].load_simd_aligned(x);
                let y_trf = self.gradients.trf[1].load_simd_aligned(x);
                let y_blf = self.gradients.blf[1].load_simd_aligned(x);
                let y_brf = self.gradients.brf[1].load_simd_aligned(x);
                let y_tlb = self.gradients.tlb[1].load_simd_aligned(x);
                let y_trb = self.gradients.trb[1].load_simd_aligned(x);
                let y_blb = self.gradients.blb[1].load_simd_aligned(x);
                let y_brb = self.gradients.brb[1].load_simd_aligned(x);

                let z_tlf = self.gradients.tlf[2].load_simd_aligned(x);
                let z_trf = self.gradients.trf[2].load_simd_aligned(x);
                let z_blf = self.gradients.blf[2].load_simd_aligned(x);
                let z_brf = self.gradients.brf[2].load_simd_aligned(x);
                let z_tlb = self.gradients.tlb[2].load_simd_aligned(x);
                let z_trb = self.gradients.trb[2].load_simd_aligned(x);
                let z_blb = self.gradients.blb[2].load_simd_aligned(x);
                let z_brb = self.gradients.brb[2].load_simd_aligned(x);

                let calc_prod_sum = |z_inc: Simd<f32, A>, y_inc: Simd<f32, A>, z, y, x| {
                    z_inc.mul_add(z, y_inc.mul_add(y, x))
                };

                let sum_prod_tlf = calc_prod_sum(self.z_inc_hi, self.y_inc_hi, z_tlf, y_tlf, x_tlf);
                let sum_prod_trf = calc_prod_sum(self.z_inc_hi, self.y_inc_hi, z_trf, y_trf, x_trf);
                let sum_prod_blf = calc_prod_sum(self.z_inc_hi, self.y_inc_lo, z_blf, y_blf, x_blf);
                let sum_prod_brf = calc_prod_sum(self.z_inc_hi, self.y_inc_lo, z_brf, y_brf, x_brf);
                let sum_prod_tlb = calc_prod_sum(self.z_inc_lo, self.y_inc_hi, z_tlb, y_tlb, x_tlb);
                let sum_prod_trb = calc_prod_sum(self.z_inc_lo, self.y_inc_hi, z_trb, y_trb, x_trb);
                let sum_prod_blb = calc_prod_sum(self.z_inc_lo, self.y_inc_lo, z_blb, y_blb, x_blb);
                let sum_prod_brb = calc_prod_sum(self.z_inc_lo, self.y_inc_lo, z_brb, y_brb, x_brb);

                let z_tf_offset = x_lerp.mul_add(z_trf - z_tlf, z_tlf) * self.z_inc_weighted;
                let z_bf_offset = x_lerp.mul_add(z_brf - z_blf, z_blf) * self.z_inc_weighted;
                let z_tb_offset = x_lerp.mul_add(z_trb - z_tlb, z_tlb) * self.z_inc_weighted;
                let z_bb_offset = x_lerp.mul_add(z_brb - z_blb, z_blb) * self.z_inc_weighted;

                let y_tf_offset = x_lerp.mul_add(y_trf - y_tlf, y_tlf) * self.y_inc_weighted;
                let y_bf_offset = x_lerp.mul_add(y_brf - y_blf, y_blf) * self.y_inc_weighted;
                let y_hi_offset_dif = x_lerp
                    .mul_add(y_trb - y_tlb, y_tlb)
                    .mul_sub(self.y_inc_weighted, y_tf_offset);
                let y_lo_offset_dif = x_lerp
                    .mul_add(y_brb - y_blb, y_blb)
                    .mul_sub(self.y_inc_weighted, y_bf_offset);

                let tf_base =
                    x_lerp.mul_add(sum_prod_trf - sum_prod_tlf, sum_prod_tlf) * self.weight;
                let bf_base =
                    x_lerp.mul_add(sum_prod_brf - sum_prod_blf, sum_prod_blf) * self.weight;
                let hi_base_dif = x_lerp
                    .mul_add(sum_prod_trb - sum_prod_tlb, sum_prod_tlb)
                    .mul_sub(self.weight, tf_base);
                let lo_base_dif = x_lerp
                    .mul_add(sum_prod_brb - sum_prod_blb, sum_prod_blb)
                    .mul_sub(self.weight, bf_base);

                buffers.z_tf_offset.write_simd_aligned(x, z_tf_offset);
                buffers.z_bf_offset.write_simd_aligned(x, z_bf_offset);
                buffers
                    .z_top_offset_dif
                    .write_simd_aligned(x, z_tb_offset - z_tf_offset);
                buffers
                    .z_bottom_offset_dif
                    .write_simd_aligned(x, z_bb_offset - z_bf_offset);

                buffers.y_tf_offset.write_simd_aligned(x, y_tf_offset);
                buffers.y_bf_offset.write_simd_aligned(x, y_bf_offset);
                buffers
                    .y_top_offset_dif
                    .write_simd_aligned(x, y_hi_offset_dif);
                buffers
                    .y_bottom_offset_dif
                    .write_simd_aligned(x, y_lo_offset_dif);

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
        buffers: &DottedTrilerpBuffers,
        x: usize,
        z_vec: Simd<f32, A>,
        z_lerp: Simd<f32, A>,
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
            self.initialize_factors_block::<FULL>(buffers, index, block, z_vec, z_lerp);

            // let z_tf_offset = buffers.z_tf_offset.load_simd_aligned(index);
            // let z_bf_offset = buffers.z_bf_offset.load_simd_aligned(index);
            // let z_top_offset_dif = buffers.z_top_offset_dif.load_simd_aligned(index);
            // let z_bottom_offset_dif = buffers.z_bottom_offset_dif.load_simd_aligned(index);
            //
            // let y_tf_offset = buffers.y_tf_offset.load_simd_aligned(index);
            // let y_bf_offset = buffers.y_bf_offset.load_simd_aligned(index);
            // let y_top_offset_dif = buffers.y_top_offset_dif.load_simd_aligned(index);
            // let y_bottom_offset_dif = buffers.y_bottom_offset_dif.load_simd_aligned(index);
            //
            // let tf_base_vec = buffers.tf_base.load_simd_aligned(index);
            // let bf_base_vec = buffers.bf_base.load_simd_aligned(index);
            // let top_base_dif_vec = buffers.top_base_dif.load_simd_aligned(index);
            // let bottom_base_dif_vec = buffers.bottom_base_dif.load_simd_aligned(index);
            //
            // let z_top_offset = z_lerp.mul_add(z_top_offset_dif, z_tf_offset);
            // let z_bottom_offset = z_lerp.mul_add(z_bottom_offset_dif, z_bf_offset);
            //
            // self.top[block] =
            //     z_vec.mul_add(z_top_offset, z_lerp.mul_add(top_base_dif_vec, tf_base_vec));
            // let bottom_base = z_vec.mul_add(
            //     z_bottom_offset,
            //     z_lerp.mul_add(bottom_base_dif_vec, bf_base_vec),
            // );
            // self.dif[block] = bottom_base - self.top[block];
            //
            // self.d_top[block] = z_lerp.mul_add(y_top_offset_dif, y_tf_offset);
            // let y_bottom_offset = z_lerp.mul_add(y_bottom_offset_dif, y_bf_offset);
            // self.d_dif[block] = y_bottom_offset - self.d_top[block];
        }

        if ACCESS_MODE != FULL {
            self.initialize_factors_block::<ACCESS_MODE>(buffers, 0, NUM_BLOCKS, z_vec, z_lerp);
        }
    }

    #[inline(always)]
    fn initialize_factors_block<const ACCESS_MODE: u8>(
        &mut self,
        buffers: &DottedTrilerpBuffers,
        i: usize,
        block: usize,
        z_vec: Simd<f32, A>,
        z_lerp: Simd<f32, A>,
    ) {
        let config = self.config;
        let (z_tf_offset, z_bf_offset, z_top_offset_dif, z_bottom_offset_dif) = unsafe {
            (
                buffers.z_tf_offset.ld_buf::<ACCESS_MODE>(i, config),
                buffers.z_bf_offset.ld_buf::<ACCESS_MODE>(i, config),
                buffers.z_top_offset_dif.ld_buf::<ACCESS_MODE>(i, config),
                buffers.z_bottom_offset_dif.ld_buf::<ACCESS_MODE>(i, config),
            )
        };
        let (y_tf_offset, y_bf_offset, y_top_offset_dif, y_bottom_offset_dif) = unsafe {
            (
                buffers.y_tf_offset.ld_buf::<ACCESS_MODE>(i, config),
                buffers.y_bf_offset.ld_buf::<ACCESS_MODE>(i, config),
                buffers.y_top_offset_dif.ld_buf::<ACCESS_MODE>(i, config),
                buffers.y_bottom_offset_dif.ld_buf::<ACCESS_MODE>(i, config),
            )
        };
        let (tf_base_vec, bf_base_vec, top_base_dif_vec, bottom_base_dif_vec) = unsafe {
            (
                buffers.tf_base.ld_buf::<ACCESS_MODE>(i, config),
                buffers.bf_base.ld_buf::<ACCESS_MODE>(i, config),
                buffers.top_base_dif.ld_buf::<ACCESS_MODE>(i, config),
                buffers.bottom_base_dif.ld_buf::<ACCESS_MODE>(i, config),
            )
        };

        let z_top_offset = z_lerp.mul_add(z_top_offset_dif, z_tf_offset);
        let z_bottom_offset = z_lerp.mul_add(z_bottom_offset_dif, z_bf_offset);

        unsafe {
            *self.top.as_mut().get_unchecked_mut(block) =
                z_vec.mul_add(z_top_offset, z_lerp.mul_add(top_base_dif_vec, tf_base_vec));
            let bottom_base = z_vec.mul_add(
                z_bottom_offset,
                z_lerp.mul_add(bottom_base_dif_vec, bf_base_vec),
            );
            *self.dif.as_mut().get_unchecked_mut(block) =
                bottom_base - *self.top.as_ref().get_unchecked(block);

            *self.top_step.as_mut().get_unchecked_mut(block) =
                z_lerp.mul_add(y_top_offset_dif, y_tf_offset);
            let y_bottom_offset = z_lerp.mul_add(y_bottom_offset_dif, y_bf_offset);
            *self.dif_step.as_mut().get_unchecked_mut(block) =
                y_bottom_offset - *self.top_step.as_ref().get_unchecked(block);
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

            // let output = y_lerp.mul_add(self.dif[block], self.top[block]);
            //
            // let (cur_state, mut result) = if INIT {
            //     C::initialize_sample(self.fractal_config, output)
            // } else {
            //     let mut cur_state = C::State::<A>::default();
            //     for i in 0..C::State::<A>::STATE_SIZE {
            //         let index = index + i * self.grid_data.total_size;
            //         cur_state[i] = unsafe { maybe_tail_load::<A, IS_TAIL>(index..tail_end, state) };
            //     }
            //     let cur_result = unsafe { maybe_tail_load::<A, IS_TAIL>(index..tail_end, dst) };
            //     C::apply_sample(self.fractal_config, cur_state, cur_result, output)
            // };
            //
            // // Save changes to state.
            // if !FINAL {
            //     for i in 0..C::State::<A>::STATE_SIZE {
            //         let offset = i * self.grid_data.total_size;
            //         let index = index + offset;
            //         let tail_end = tail_end + offset;
            //         unsafe { maybe_tail_store::<A, IS_TAIL>(index..tail_end, cur_state[i], state) };
            //     }
            // }
            //
            // if FINAL {
            //     result = C::finalize_sample(self.fractal_config, cur_state, result);
            // }
            //
            // unsafe { maybe_tail_store::<A, IS_TAIL>(index..tail_end, result, dst) };
            //
            // self.dif[block] += self.dif_step[block];
            // self.top[block] += self.top_step[block];
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
                *self.dif.as_ref().get_unchecked(block),
                *self.top.as_ref().get_unchecked(block),
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

        // Walk along the grid by accumulating interpolation components.
        unsafe {
            *self.dif.as_mut().get_unchecked_mut(block) +=
                *self.dif_step.as_ref().get_unchecked(block);
            *self.top.as_mut().get_unchecked_mut(block) +=
                *self.top_step.as_ref().get_unchecked(block);
        }
    }
}
