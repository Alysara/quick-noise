use std::marker::PhantomData;

use itertools::Zip;
use simply_simd::{Arch, Simd};

use crate::api::batch::interface::{BatchGenerator, BatchNoise, DimIter};
use crate::api::configs::*;
use crate::api::parameters::*;
use crate::math::random::Random;
use crate::noise::combiners::Combiner;
use crate::{HybridMulti, PingPong, Ridged, Terrace};

#[inline(always)]
pub fn multizip<T, U>(t: U) -> Zip<T>
where
    Zip<T>: From<U> + Iterator,
{
    Zip::from(t)
}

pub struct BatchNoiseBuilder<
    const D: usize,
    C: Combiner,
    G: BatchGenerator<D>,
    A: Arch,
    I: DimIter<A, D>,
> {
    pub(crate) noise_config: NoiseConfig<D>,
    pub(crate) combiner_config: C::Config,
    pub(crate) iters: I,
    pub(crate) _noise_type: PhantomData<G>,
    pub(crate) _arch: PhantomData<A>,
}

params_noise_builder!(BatchNoiseBuilder, [const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, C, G, A, I]);
params_lacunarity_builder!(BatchNoiseBuilder, [const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, C, G, A, I]);
params_combiner_builder!(BatchNoiseBuilder, [const D: usize, C: Combiner<Config: Sized>, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, C, G, A, I]);
params_ridged_builder!(BatchNoiseBuilder, [const D: usize, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, Ridged, G, A, I]);
params_ping_pong_builder!(BatchNoiseBuilder, [const D: usize, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, PingPong, G, A, I]);
params_terrace_builder!(BatchNoiseBuilder, [const D: usize, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, Terrace, G, A, I]);
params_hybrid_multi_builder!(BatchNoiseBuilder, [const D: usize, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, HybridMulti, G, A, I]);
params_grid_seed_builder!(BatchNoiseBuilder, [const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>], [D, C, G, A, I]);
params_noise_scaling_2d!(BatchNoiseBuilder, [C: Combiner, G: BatchGenerator<2>, A: Arch, I: DimIter<A, 2>], [2, C, G, A, I]);
params_noise_scaling_3d!(BatchNoiseBuilder, [C: Combiner, G: BatchGenerator<3>, A: Arch, I: DimIter<A, 3>], [3, C, G, A, I]);

impl<C: Combiner, G: BatchGenerator<2>> BatchNoise<2, C, G> {
    /// Creates a new builder to easily configure batches of noise.
    ///
    /// # Type Parameters
    /// - `A`: The simd feature set to use.
    /// - `X`: Simd iterator type for x-inputs.
    /// - `Y`: Simd iterator type for y-inputs.
    ///
    /// # Parameters
    /// - `x_iter`: Simd iterator for x-inputs.
    /// - `y_iter`: Simd iterator for y-inputs.
    pub fn builder<A: Arch, X, Y>(
        x_iter: X,
        y_iter: Y,
    ) -> BatchNoiseBuilder<2, C, G, A, Zip<(X, Y)>>
    where
        X: Iterator<Item = Simd<f32, A>>,
        Y: Iterator<Item = Simd<f32, A>>,
        Zip<(X, Y)>: DimIter<A, 2>,
    {
        BatchNoiseBuilder::<2, C, G, A, _>::new(x_iter, y_iter)
    }
}

impl<G, C, A, X, Y> BatchNoiseBuilder<2, C, G, A, Zip<(X, Y)>>
where
    G: BatchGenerator<2>,
    C: Combiner,
    A: Arch,
    X: Iterator<Item = Simd<f32, A>>,
    Y: Iterator<Item = Simd<f32, A>>,
    Zip<(X, Y)>: DimIter<A, 2>,
{
    pub fn new(x_iter: X, y_iter: Y) -> Self {
        Self {
            noise_config: Default::default(),
            combiner_config: Default::default(),
            iters: multizip((x_iter, y_iter)),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }

    pub fn from_configs(
        noise_config: NoiseConfig<2>,
        combiner_config: C::Config,
        x_iter: X,
        y_iter: Y,
    ) -> Self {
        Self {
            noise_config,
            combiner_config,
            iters: multizip((x_iter, y_iter)),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }
}

impl<C: Combiner, G: BatchGenerator<3>> BatchNoise<3, C, G> {
    /// Creates a new builder to easily configure batches of noise.
    ///
    /// # Type Parameters
    /// - `A`: The simd feature set to use.
    /// - `X`: Simd iterator type for x-inputs.
    /// - `Y`: Simd iterator type for y-inputs.
    /// - `Z`: Simd iterator type for z-inputs.
    ///
    /// # Parameters
    /// - `x_iter`: Simd iterator for x-inputs.
    /// - `y_iter`: Simd iterator for y-inputs.
    /// - `z_iter`: Simd iterator for z-inputs.
    pub fn builder<A: Arch, X, Y, Z>(
        x_iter: X,
        y_iter: Y,
        z_iter: Z,
    ) -> BatchNoiseBuilder<3, C, G, A, Zip<(X, Y, Z)>>
    where
        X: Iterator<Item = Simd<f32, A>>,
        Y: Iterator<Item = Simd<f32, A>>,
        Z: Iterator<Item = Simd<f32, A>>,
        Zip<(X, Y, Z)>: DimIter<A, 3>,
    {
        BatchNoiseBuilder::<3, C, G, A, _>::new(x_iter, y_iter, z_iter)
    }
}

impl<G, C, A, X, Y, Z> BatchNoiseBuilder<3, C, G, A, Zip<(X, Y, Z)>>
where
    G: BatchGenerator<3>,
    C: Combiner,
    A: Arch,
    X: Iterator<Item = Simd<f32, A>>,
    Y: Iterator<Item = Simd<f32, A>>,
    Z: Iterator<Item = Simd<f32, A>>,
    Zip<(X, Y, Z)>: DimIter<A, 3>,
{
    pub fn new(x_iter: X, y_iter: Y, z_iter: Z) -> Self {
        Self {
            noise_config: Default::default(),
            combiner_config: Default::default(),
            iters: multizip((x_iter, y_iter, z_iter)),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }

    pub fn from_configs(
        noise_config: NoiseConfig<3>,
        combiner_config: C::Config,
        x_iter: X,
        y_iter: Y,
        z_iter: Z,
    ) -> Self {
        Self {
            noise_config,
            combiner_config,
            iters: multizip((x_iter, y_iter, z_iter)),
            _noise_type: PhantomData::<G>,
            _arch: PhantomData::<A>,
        }
    }
}

impl<const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch, I: DimIter<A, D>>
    BatchNoiseBuilder<D, C, G, A, I>
{
    declare_fill!(self, output, {
        if self.noise_config.initialize {
            for (i, x) in self.into_iter().enumerate() {
                x.copy_to_slice(&mut output[i * Simd::<f32, A>::LANES..]);
            }
        } else {
            let mut i = 0;
            for x in self.into_iter() {
                let cur = Simd::from_slice(&output[i..]);
                let x = cur + x;
                x.copy_to_slice(&mut output[i..]);
                i += Simd::<f32, A>::LANES;
            }
        }
    });

    declare_build!(self, { self.into_iter().collect() });

    declare_into_iter!(A, self, {
        BatchNoise::<D, C, G>::sample(self.noise_config, self.combiner_config, self.iters)
    });
}
