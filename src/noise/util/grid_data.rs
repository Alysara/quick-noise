use std::array::from_fn;
use std::mem::MaybeUninit;

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::util::grid_helpers::{Arena, configure_tiling, fill_grid_indices};
use crate::simd::Arch;
use crate::simd::register::Simd;

pub(crate) struct GridData<'a, const D: usize> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; D],
    pub grid_start: [i32; D],
    pub increment: [f32; D],
    pub num_loops: [usize; D],
    pub octave_tiling: [Option<u32>; D],
    pub distances: [&'a mut [MaybeUninit<f32>]; D],
    pub fade_factors: [&'a mut [MaybeUninit<f32>]; D],
    pub grid_indices: [&'a mut [MaybeUninit<u32>]; D],
}

#[repr(u8)]
pub(crate) enum Lerp {
    Cubic = 0,
    Quintic = 1,
}

impl Lerp {
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Cubic,
            1 => Self::Quintic,
            _ => unreachable!(),
        }
    }
}

impl<'a, const D: usize> GridData<'a, D> {
    #[inline(always)]
    pub fn new<A: Arch, const LERP: u8>(
        params: &GridNoiseParams<D>,
        arena: &mut Arena<'a>,
        padded_size: &[usize; D],
    ) -> Self {
        let lerp_type = Lerp::from_u8(LERP);
        let lanes = Simd::<f32, A>::LANES;

        let total_size = params.grid_size.iter().product();
        let increment = from_fn(|i| params.frequency[i] * params.magnification);

        // Get the starting gradient coordinates and how far the first sample is to the next one.
        let grid_start: [i32; D] =
            from_fn(|i| (params.position[i] * increment[i]).floor() as i32);

        let frac_start: [f32; D] =
            from_fn(|i| (params.position[i] * increment[i] - grid_start[i] as f32).max(0.0));

        // Quintic lerp the distances to get the fade factor.
        let distances = from_fn(|i| arena.allocate(padded_size[i]));
        let fade_factors = from_fn(|i| arena.allocate(padded_size[i]));

        // Get the distances from the gradient gridpoints.
        let mut cur_dist: [_; D] = from_fn(|i| {
            Simd::<f32, A>::iota(0.0) * Simd::<f32, A>::splat(increment[i])
                + Simd::<f32, A>::splat(frac_start[i])
        });
        let chunk_increment: [_; D] =
            from_fn(|i| Simd::<f32, A>::splat(increment[i] * lanes as f32));

        for axis in 0..D {
            for i in (0..params.grid_size[axis]).step_by(lanes) {
                let fract_dist = cur_dist[axis].fract();
                let cur_lerp = match lerp_type {
                    Lerp::Cubic => fract_dist.cubic_lerp(),
                    Lerp::Quintic => fract_dist.quintic_lerp(),
                };

                unsafe {
                    fract_dist.copy_to_aligned_slice_unchecked(
                        distances[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                    cur_lerp.copy_to_aligned_slice_unchecked(
                        fade_factors[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                }
                cur_dist[axis] += chunk_increment[axis];
            }
        }

        // Identify the cutoff points between frequency-based grid boundaries .
        let mut grid_indices = from_fn(|i| arena.allocate(padded_size[i]));
        let num_loops = fill_grid_indices::<A, D>(&mut grid_indices, &distances, params.grid_size);

        // Adjust the tiling.
        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            grid_start,
            increment,
            num_loops,
            octave_tiling,
            distances,
            fade_factors,
            grid_indices,
        }
    }
}
