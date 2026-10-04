use std::array::from_fn;
use std::mem::MaybeUninit;

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::util::grid_helpers::{Arena, configure_tiling, fill_grid_indices};
use crate::simd::Arch;
use crate::simd::register::Simd;

pub(crate) struct CellularGridData<'a, const D: usize> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; D],
    pub grid_start: [i32; D],
    pub num_loops: [usize; D],
    pub octave_tiling: [Option<u32>; D],
    pub distances: [&'a mut [MaybeUninit<f32>]; D],
    pub grid_indices: [&'a mut [MaybeUninit<u32>]; D],
}

impl<'a, const D: usize> CellularGridData<'a, D> {
    #[inline(always)]
    pub fn new<A: Arch>(
        params: &GridNoiseParams<D>,
        arena: &mut Arena<'a>,
        padded_size: &[usize; D],
    ) -> Self {
        let lanes = Simd::<f32, A>::LANES;

        let total_size = params.grid_size.iter().product();
        let increment: [f32; D] = from_fn(|i| params.frequency[i] * params.magnification);

        // Get the starting gradient coordinates and how far the first sample is to the next one
        let grid_start: [i32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i]).floor() as i32);

        let frac_start: [f32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i] - grid_start[i] as f32).max(0.0));

        // Only the raw fractional distance is needed to locate cell boundaries
        let distances = from_fn(|i| arena.allocate(padded_size[i]));

        let mut cur_dist: [_; D] = from_fn(|i| {
            Simd::<f32, A>::iota(0.0) * Simd::<f32, A>::splat(increment[i])
                + Simd::<f32, A>::splat(frac_start[i])
        });
        let chunk_increment: [_; D] =
            from_fn(|i| Simd::<f32, A>::splat(increment[i] * lanes as f32));

        for axis in 0..D {
            for i in (0..params.grid_size[axis]).step_by(lanes) {
                let fract_dist = cur_dist[axis].fract();
                unsafe {
                    fract_dist.copy_to_aligned_slice_unchecked(
                        distances[axis].get_unchecked_mut(i..).assume_init_mut(),
                    );
                }
                cur_dist[axis] += chunk_increment[axis];
            }
        }

        // Identify the cutoff points between frequency-based grid boundaries.
        let mut grid_indices = from_fn(|i| arena.allocate(padded_size[i]));
        let num_loops = fill_grid_indices::<A, D>(&mut grid_indices, &distances, params.grid_size);

        // Adjust the tiling.
        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            grid_start,
            num_loops,
            octave_tiling,
            distances,
            grid_indices,
        }
    }
}


pub(crate) struct PVGridData<'a, const D: usize> {
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

impl<'a, const D: usize> PVGridData<'a, D> {
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
            from_fn(|i| (params.position[i] as f32 * increment[i]).floor() as i32);

        let frac_start: [f32; D] =
            from_fn(|i| (params.position[i] as f32 * increment[i] - grid_start[i] as f32).max(0.0));

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

const SQRT_3: f32 = 1.732_050_8;
const SKEW_2D: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW_2D: f32 = (3.0 - SQRT_3) / 6.0;

pub(crate) struct SimplexGridData<const D: usize> {
    pub total_size: usize,
    pub weight: f32,
    pub grid_size: [usize; D],
    pub increment: [f32; D],
    /// Position of the first sample in output space (`position * increment`).
    pub origin: [f32; D],
    /// Skewed lattice index `(i0, j0)` that the first sample (`origin`) falls into.
    pub grid_start: [i32; D],
    pub octave_tiling: [Option<u32>; D],
}

impl<const D: usize> SimplexGridData<D> {
    #[inline(always)]
    pub fn new(params: &GridNoiseParams<D>) -> Self {
        let total_size = params.grid_size.iter().product();
        let increment = from_fn(|i| params.frequency[i] * params.magnification);
        let origin = from_fn(|i| params.position[i] as f32 * increment[0]);

        // Skew the region's first sample to locate the enclosing lattice cell.
        let s = origin.iter().sum::<f32>() * SKEW_2D;
        let grid_start = from_fn(|i| (origin[i] + s).floor() as i32);

        let octave_tiling = configure_tiling(params);

        Self {
            total_size,
            weight: params.weight,
            grid_size: params.grid_size,
            increment,
            origin,
            grid_start,
            octave_tiling,
        }
    }

    /// Skew a sample-space coordinate into the skewed lattice coordinate space `(i, j, ...)`.
    #[inline(always)]
    pub fn skew(&self, coords: &[f32; D]) -> [f32; D] {
        let s = coords.iter().sum::<f32>() * SKEW_2D;
        from_fn(|i| coords[i] + s)
    }

    /// True sample-space position `(x, y, ...)` of a lattice corner `(i, j, ...)`.
    #[inline(always)]
    pub fn unskew(&self, coords: &[i32; D]) -> [f32; D] {
        let t = coords.iter().sum::<i32>() as f32 * UNSKEW_2D;
        from_fn(|i| coords[i] as f32 - t)
    }
}
