use std::marker::PhantomData;

use crate::api::configs::*;
use crate::api::grid::interface::GridNoise;
use crate::api::octave::Octave;
use crate::api::parameters::*;
use crate::math::random::Random;
use crate::simd::register::iters::IntoSimdIterator;
use crate::simd::{Arch, StaticArch};
use crate::{Combiner, GridGenerator, HybridMulti, PingPong, Ridged, Terrace};

/// A struct for creating noise on a uniform grid with custom octaves.
/// The most performant way to generate noise.
#[derive(Default)]
pub struct OctaveGridNoiseBuilder<
    'a,
    const D: usize,
    C: Combiner,
    G: GridGenerator<D>,
    A: Arch = StaticArch,
> {
    grid_config: GridConfig<D>,
    noise_config: NoiseConfig<D>,
    combiner_config: C::Config,
    octave_list: &'a [Octave<D>],
    _noise_type: PhantomData<G>,
    _arch: PhantomData<A>,
}

params_noise_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, C: Combiner, G: GridGenerator<D>, A: Arch], ['a, D, C, G, A]);
params_combiner_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, C: Combiner<Config: Sized>, G: GridGenerator<D>, A: Arch], ['a, D, C, G, A]);
params_ridged_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, G: GridGenerator<D>, A: Arch], ['a, D, Ridged, G, A]);
params_ping_pong_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, G: GridGenerator<D>, A: Arch], ['a, D, PingPong, G, A]);
params_terrace_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, G: GridGenerator<D>, A: Arch], ['a, D, Terrace, G, A]);
params_hybrid_multi_builder!(OctaveGridNoiseBuilder, ['a, const D: usize, G: GridGenerator<D>, A: Arch], ['a, D, HybridMulti, G, A]);
params_noise_scaling_2d!(OctaveGridNoiseBuilder, ['a, C: Combiner, G: GridGenerator<2>, A: Arch], ['a, 2, C, G, A]);
params_noise_scaling_3d!(OctaveGridNoiseBuilder, ['a, C: Combiner, G: GridGenerator<3>, A: Arch], ['a, 3, C, G, A]);

impl<'a, const D: usize, C: Combiner, G: GridGenerator<D>, A: Arch>
    OctaveGridNoiseBuilder<'a, D, C, G, A>
{
    pub(crate) fn new(grid_config: GridConfig<D>, octave_list: &'a [Octave<D>]) -> Self {
        Self {
            grid_config,
            octave_list,
            ..Default::default()
        }
    }

    declare_build!(self, {
        let size = self.grid_config.grid_size.iter().product();
        let mut result = vec![0.0; size];
        GridNoise::<D, C, G>::sample_with_octaves::<StaticArch>(
            &self.grid_config,
            &self.noise_config,
            &self.combiner_config,
            self.octave_list,
            result.as_mut_slice(),
        );
        result
    });

    declare_fill!(self, result, {
        GridNoise::<D, C, G>::sample_with_octaves::<StaticArch>(
            &self.grid_config,
            &self.noise_config,
            &self.combiner_config,
            self.octave_list,
            result,
        );
    });

    declare_into_iter!(StaticArch, self, { self.build().into_simd_iter() });
}
