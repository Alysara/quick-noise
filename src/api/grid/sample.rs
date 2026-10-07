use simply_simd::enable_targets;

use crate::api::configs::*;
use crate::api::grid::interface::{GridGenerator, GridNoise, GridNoiseParams};
use crate::api::seed::gen_octave_seed;
use crate::math::random::Random;
use crate::noise::util::grid_helpers::{Arena, ArenaBuffer};
use crate::simd::Arch;
use crate::{Combiner, CombinerState};

#[enable_targets(A)]
impl<const D: usize, C: Combiner, G: GridGenerator<D>> GridNoise<D, C, G> {
    /// Samples a uniform grid of noise using config structs. Using the builders
    /// instead is recommended for ease-of-use. 
    ///
    /// # Type Parameters:
    /// - `A`: The simd feature set to use.
    /// 
    /// # Parameters:
    /// - `grid_config`: The parameters of the grid being sampled.
    /// - `noise_config`: The parameters of the noise being generated.
    /// - `combiner_config`: Additional parameters for the combiner algorithm.
    /// - `dst`: Slice to write the output noise results.
    pub fn sample<A: Arch>(
        grid_config: &GridConfig<D>,
        noise_config: &NoiseConfig<D>,
        combiner_config: &C::Config,
        gen_config: &G::GenConfig,
        dst: &mut [f32],
    ) {
        let octaves = noise_config.num_grid_octaves();

        // Fill with zeroes if there are no octaves.
        if octaves == 0 {
            if noise_config.initialize {
                dst.fill(0.0)
            }
            return;
        }

        let base_seed = Random::mix_u64_pair(grid_config.grid_seed, noise_config.seed);

        // FBM algorithm:
        let frequency = std::array::from_fn(|i| noise_config.scaling[i] * noise_config.frequency);
        let weight = if noise_config.normalization && C::WEIGHT_DECAY {
            noise_config.normalize_amplitude(noise_config.amplitude)
        } else {
            noise_config.amplitude
        };

        // First octave:
        let mut params = GridNoiseParams {
            seed: gen_octave_seed(frequency, base_seed),
            grid_size: grid_config.grid_size,
            position: grid_config.position,
            magnification: noise_config.magnification,
            tiling: grid_config.tiling,
            frequency,
            weight,
        };

        let total_size: usize = grid_config.grid_size.iter().product();
        let needed_state_size = total_size * C::State::<A>::STATE_SIZE;
        let mut state_cache = ArenaBuffer::<A>::with_capacity(needed_state_size);
        let mut arena = Arena::with_cache(&mut state_cache);
        let state = arena.allocate(needed_state_size);
        let state = unsafe { state.assume_init_mut() };
        let f_config = *combiner_config;
        let g_config = *gen_config;

        match (
            noise_config.initialize,
            noise_config.finalize && octaves == 1,
        ) {
            (false, false) => G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst),
            (true, false) => G::sample_grid::<A, C, true, false>(params, f_config, g_config, state, dst),
            (false, true) => G::sample_grid::<A, C, false, true>(params, f_config, g_config, state, dst),
            (true, true) => G::sample_grid::<A, C, true, true>(params, f_config, g_config, state, dst),
        }

        // Subsequent octaves:
        for _ in 1..(octaves.saturating_sub(1)) {
            if C::WEIGHT_DECAY {
                params.weight *= noise_config.persistence;
            }
            params.frequency =
                std::array::from_fn(|i| params.frequency[i] * noise_config.lacunarity);
            params.seed = gen_octave_seed(params.frequency, base_seed);
            G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst);
        }

        if octaves > 1 {
            params.weight *= noise_config.persistence;
            params.frequency =
                std::array::from_fn(|i| params.frequency[i] * noise_config.lacunarity);
            params.seed = gen_octave_seed(params.frequency, base_seed);
            match noise_config.finalize {
                true => G::sample_grid::<A, C, false, true>(params, f_config, g_config, state, dst),
                false => G::sample_grid::<A, C, false, false>(params, f_config, g_config, state, dst),
            }
        }
    }
}
