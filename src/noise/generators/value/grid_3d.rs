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
        let required_cache = padded_size[0] * 17 + padded_size[1] * 3 + padded_size[2] * 3;
        let mut cache = ArenaBuffer::<A>::with_capacity(required_cache);
        let mut arena = Arena::with_cache(&mut cache);
        let mut data_arena = arena.allocate_arena(padded_size.iter().fold(0, |n, x| n + 3 * x));
        let mut trilerp_arena = arena.allocate_arena(padded_size[0] * 4);

        // Allocation setup.
        let num_blocks = A::NUM_SIMD_REG / 4;
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
            grid_gradients_3d::<A>(&params, &grid_data, &mut gradients, 0, z_it);
            gradients.swap_top_bottom();

            let mut y_cur_index = 0;
            for y_it in 0..grid_data.num_loops[1] {
                let y_next_index =
                    unsafe { grid_data.grid_indices[1].get_unchecked(y_it).assume_init() as usize };
                let y_range = y_cur_index..y_next_index;

                // Set bottom gradients.
                grid_gradients_3d::<A>(&params, &grid_data, &mut gradients, y_it + 1, z_it);

                grid_trilerp::<A, C, INIT, FINAL>(
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
pub(super) fn grid_gradients_3d<'a, A: Arch>(
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
    let hash_mask: Simd<u32, A> = Simd::splat(0x007FFFFF);
    let exp_bits: Simd<u32, A> = Simd::splat(0x40000000);
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

    grid_gradients_3d_set_loop::<A, true>(grid_data, gradients);
    grid_gradients_3d_set_loop::<A, false>(grid_data, gradients);
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
pub(super) fn grid_trilerp<A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
    buffers: &mut TrilerpBuffers,
    config: &InterpolationConfig<A>,
    fractal_config: &C::Config,
    grid_data: &PVGridData<3>,
    gradients: &ValueGradients3D,
    ranges: (Range<usize>, Range<usize>),
    output: (&mut [f32], &mut [f32]),
) {
    let mut executer = DottedTrilerpExecuter::<A, C, INIT, FINAL> {
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
        executer.interpolate::<false>(buffers, state, dst);
    }

    if config.has_block_tail {
        executer.interpolate::<true>(buffers, state, dst);
        std::hint::cold_path();
    }
}

impl<'a, A: Arch, C: Combiner, const INIT: bool, const FINAL: bool>
    DottedTrilerpExecuter<'a, A, C, INIT, FINAL>
{
    #[inline(always)]
    pub fn interpolate<const IS_TAIL: bool>(
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

        let z_hop = self.grid_data.grid_size[0] * self.grid_data.grid_size[1];
        let y_hop = self.grid_data.grid_size[0];
        for z in self.z_range.start..self.z_range.end {
            let z_lerp = unsafe { self.grid_data.fade_factors[2].get_unchecked(z) };
            let z_lerp = unsafe { z_lerp.assume_init() };
            let z_lerp = Simd::splat(z_lerp);

            for x in range.clone().step_by(self.config.block_lanes) {
                self.intialize_factors::<IS_TAIL>(buffers, x, z_lerp);

                let index = z * z_hop + x;
                let mut y = self.y_range.start;
                while y < self.y_range.end {
                    let index = index + y * y_hop;
                    if y + 4 > self.y_range.end {
                        self.process_factors::<IS_TAIL>(index, y, state, dst);
                        y += 1;
                    } else {
                        self.process_factors::<IS_TAIL>(index, y, state, dst);
                        self.process_factors::<IS_TAIL>(index + y_hop, y + 1, state, dst);
                        self.process_factors::<IS_TAIL>(index + 2 * y_hop, y + 2, state, dst);
                        self.process_factors::<IS_TAIL>(index + 3 * y_hop, y + 3, state, dst);
                        y += 4;
                    }
                }
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
    fn intialize_factors<const IS_TAIL: bool>(
        &mut self,
        buffers: &TrilerpBuffers,
        x: usize,
        z_lerp: Simd<f32, A>,
    ) {
        let num_blocks = if IS_TAIL {
            self.config.block_tail_size
        } else {
            self.config.num_blocks
        };

        // These blocked loops will get entirely unrolled by the compiler.
        for block in 0..num_blocks {
            // Load gradients into registers.
            unsafe {
                let index = x + Simd::<f32, A>::LANES * block;
                let tf = buffers.tf_base.load_simd_aligned(index);
                let bf = buffers.bf_base.load_simd_aligned(index);
                let top_dif = buffers.top_base_dif.load_simd_aligned(index);
                let bottom_dif = buffers.bottom_base_dif.load_simd_aligned(index);

                self.top[block] = z_lerp.mul_add(top_dif, tf);
                let bottom = z_lerp.mul_add(bottom_dif, bf);
                self.dif[block] = bottom - self.top[block];
            }
        }
    }

    #[inline(always)]
    fn process_factors<const IS_TAIL: bool>(
        &mut self,
        index: usize,
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

        let tail_end = index + self.config.tail_size;
        for block in range {
            let index = index + block * Simd::<f32, A>::LANES;
            let output = y_lerp.mul_add(self.dif[block], self.top[block]);

            let (cur_state, mut result) = if INIT {
                C::initialize_sample(self.fractal_config, output)
            } else {
                let mut cur_state = C::State::default();
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
