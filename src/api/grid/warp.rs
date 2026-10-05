use std::iter::zip;
use std::marker::PhantomData;

use crate::api::batch::interface::DimIter;
use crate::api::batch::multizip;
use crate::simd::{Arch, Simd};
use crate::{BatchGenerator, BatchNoiseBuilder, Combiner, Grid};

/// Creates a builder struct for warping noise.
///
/// # Type Parameters
/// - `C`: The combiner algorithm to use across octaves.
/// - `G`: The generator algorithm to sample from.
///
/// # Parameters
/// - `warp_strength`: How strongly the warp is applied.
/// - `x_iter`: An iterator of warp values to distort each sample's x input.
/// - `y_iter`: An iterator of warp values to distort each sample's y input.
impl<A: Arch> Grid<2, A> {
    pub fn warp_builder<C: Combiner, G: BatchGenerator<2>>(
        &self,
        warp_strength: f32,
        x_iter: impl Iterator<Item = Simd<f32, A>>,
        y_iter: impl Iterator<Item = Simd<f32, A>>,
    ) -> BatchNoiseBuilder<2, C, G, A, impl DimIter<A, 2>> {
        let strength = Simd::splat(warp_strength);
        let x_iter = zip(x_iter, self.x_iter()).map(move |(x, grid)| x.mul_add(strength, grid));
        let y_iter = zip(y_iter, self.y_iter()).map(move |(y, grid)| y.mul_add(strength, grid));

        BatchNoiseBuilder {
            iters: multizip((x_iter, y_iter)),
            noise_config: Default::default(),
            combiner_config: Default::default(),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }
}

/// Creates a builder struct for warping noise.
///
/// # Type Parameters
/// - `C`: The combiner algorithm to use across octaves.
/// - `G`: The generator algorithm to sample from.
///
/// # Parameters
/// - `warp_strength`: How strongly the warp is applied.
/// - `x_iter`: An iterator of warp values to distort each sample's x input.
/// - `y_iter`: An iterator of warp values to distort each sample's y input.
/// - `z_iter`: An iterator of warp values to distort each sample's z input.
impl<A: Arch> Grid<3, A> {
    pub fn warp_builder<C: Combiner, G: BatchGenerator<3>>(
        &self,
        warp_strength: f32,
        x_iter: impl Iterator<Item = Simd<f32, A>>,
        y_iter: impl Iterator<Item = Simd<f32, A>>,
        z_iter: impl Iterator<Item = Simd<f32, A>>,
    ) -> BatchNoiseBuilder<3, C, G, A, impl DimIter<A, 3>> {
        let strength = Simd::splat(warp_strength);
        let x_iter = zip(x_iter, self.x_iter()).map(move |(x, grid)| x.mul_add(strength, grid));
        let y_iter = zip(y_iter, self.y_iter()).map(move |(y, grid)| y.mul_add(strength, grid));
        let z_iter = zip(z_iter, self.z_iter()).map(move |(z, grid)| z.mul_add(strength, grid));

        BatchNoiseBuilder {
            iters: multizip((x_iter, y_iter, z_iter)),
            noise_config: Default::default(),
            combiner_config: Default::default(),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }
}
