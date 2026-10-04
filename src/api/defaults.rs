use crate::api::configs::*;
use crate::simd::StaticSimd;

impl<const D: usize> Default for NoiseConfig<D> {
    fn default() -> Self {
        Self {
            seed: 0xD5E7B3C94F8A1E6B,
            octaves: 1,
            amplitude: 1.0,
            frequency: 0.03125,
            lacunarity: 2.0,
            persistence: 0.5,
            normalization: true,
            initialize: true,
            finalize: true,
            magnification: 1.0,
            scaling: [1.0; D],
        }
    }
}

impl<const D: usize> Default for OctaveNoiseConfig<D> {
    fn default() -> Self {
        Self {
            seed: 0xD5E7B3C94F8A1E6B,
            amplitude: 1.0,
            normalization: true,
            initialize: true,
            finalize: true,
            magnification: 1.0,
            scaling: [1.0; D],
        }
    }
}

impl<const D: usize> Default for GridConfig<D> {
    fn default() -> Self {
        Self {
            grid_size: [32; D],
            grid_seed: 0xc4ceb9fe1a85ec53,
            position: [0.0; D],
            tiling: [None; D],
        }
    }
}

/// Empty iterator for default generics.
#[derive(Default)]
pub struct EmptyIter;

impl Iterator for EmptyIter {
    type Item = StaticSimd<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

/// Zero Iter for blank noise output.
#[derive(Default)]
pub struct ZeroIter<const N: usize> {
    index: usize,
}

impl<const N: usize> Iterator for ZeroIter<N> {
    type Item = StaticSimd<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < N {
            self.index += StaticSimd::<f32>::LANES;
            Some(StaticSimd::zero())
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        const LANES: usize = StaticSimd::<f32>::LANES;
        let left = (N - self.index + LANES - 1) / StaticSimd::<f32>::LANES;
        (left, Some(left))
    }
}
