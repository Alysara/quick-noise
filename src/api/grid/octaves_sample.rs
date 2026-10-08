use simply_simd::enable_targets;

use crate::api::configs::*;
use crate::api::grid::interface::{GridNoise, GridNoiseParams};
use crate::api::octave::Octave;
use crate::api::seed::gen_octave_seed;
use crate::math::random::Random;
use crate::noise::util::grid_helpers::{Arena, ArenaBuffer};
use crate::simd::Arch;
use crate::{Combiner, CombinerState, GridGenerator};

fn get_max<const D: usize>(array: [f32; D]) -> f32 {
    array.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[enable_targets(A)]
impl<const D: usize, C: Combiner, G: GridGenerator<D>> GridNoise<D, C, G> {

    /// Samples a uniform grid of noise using config structs and custom octaves.
    /// Using the builders instead is recommended for ease-of-use. 
    ///
    /// # Type Parameters:
    /// - `A`: The simd feature set to use.
    /// 
    /// # Parameters:
    /// - `grid_config`: The parameters of the grid being sampled.
    /// - `noise_config`: The parameters of the noise being generated.
    /// - `combiner_config`: Additional parameters for the combiner algorithm.
    /// - `octave_list`: List of octaves to create noise with.
    /// - `dst`: Slice to write the output noise results.
    pub fn sample_with_octaves<A: Arch>(
        grid_config: &GridConfig<D>,
        noise_config: &NoiseConfig<D>,
        combiner_config: &C::Config,
        gen_config: &G::GenConfig,
        octave_list: &[Octave<D>],
        dst: &mut [f32],
    ) {
        let seed = Random::mix_u64_pair(noise_config.seed, grid_config.grid_seed);
        let (num_octaves, total_weight): (usize, f32) = octave_list
            .iter()
            .filter(|x| get_max(x.frequency) < 1.0)
            .fold((0, 0.0), |t, x| (t.0 + 1, t.1 + x.weight));

        let weight_coef = match (noise_config.normalization, total_weight == 0.0) {
            (false, _) => noise_config.amplitude,
            (true, false) => noise_config.amplitude / total_weight,
            (true, true) => 0.0,
        };
        if weight_coef == 0.0 {
            return;
        }

        let mut params = GridNoiseParams {
            seed: 0,
            grid_size: grid_config.grid_size,
            position: grid_config.position,
            frequency: [0.0; D],
            weight: 0.0,
            magnification: noise_config.magnification,
            tiling: grid_config.tiling,
        };

        let total_size: usize = grid_config.grid_size.iter().product();
        let needed_state_size = total_size * C::State::<A>::STATE_SIZE;
        let mut state_cache = ArenaBuffer::<A>::with_capacity(needed_state_size);
        let mut arena = Arena::with_cache(&mut state_cache);
        let state = arena.allocate(needed_state_size);
        let state = unsafe { state.assume_init_mut() };
        let f_config = *combiner_config;
        let g_config = *gen_config;

        // First octave:
        let mut octave_iter = octave_list.iter().filter(|x| get_max(x.frequency) < 1.0);
        if let Some(octave) = octave_iter.next() {
            params.seed = gen_octave_seed(octave.frequency, seed);
            params.frequency = octave.frequency;
            params.weight = octave.weight * weight_coef;

            match (
                noise_config.initialize,
                noise_config.finalize && num_octaves == 1,
            ) {
                (true, true) => G::sample_grid::<A, C, true, true>(params, f_config, g_config, state, dst),
                (false, true) => G::sample_grid::<A, C, false, true>(params, f_config, g_config, state, dst),
                (true, false) => G::sample_grid::<A, C, true, false>(params, f_config, g_config, state, dst),
                (false, false) => {
                    G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst)
                }
            }
        }

        // Subsequent octaves:
        for octave in octave_iter.by_ref().take(num_octaves.saturating_sub(2)) {
            params.seed = gen_octave_seed(octave.frequency, seed);
            params.frequency = octave.frequency;
            params.weight = octave.weight * weight_coef;
            G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst);
        }

        // Final octave:
        if let Some(octave) = octave_iter.next() {
            params.seed = gen_octave_seed(octave.frequency, seed);
            params.frequency = octave.frequency;
            params.weight = octave.weight * weight_coef;
            match noise_config.finalize {
                true => G::sample_grid::<A, C, false, true>(params, f_config, g_config, state, dst),
                false => G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst),
            }
        }
    }
}
