use std::array::from_fn;

use crate::api::batch::interface::{BatchGenerator, BatchNoise, DimIter, DimTuple};
use crate::api::configs::*;
use crate::api::seed::gen_octave_seed;
use crate::noise::combiners::Combiner;
use crate::simd::{Arch, Simd};

const MAX_FBM_OCTAVES: usize = 32;

impl<const D: usize, C: Combiner, G: BatchGenerator<D>> BatchNoise<D, C, G> {
    /// Samples a batch of noise using arbitrary inputs from simd iterators.
    /// Returns a new iterator with the noise result without intermediate stores.
    /// Using the builders instead is recommended for ease-of-use. 
    ///
    /// # Type Parameters:
    /// - `A`: The simd feature set to use.
    /// - `I`: The type of the simd iterator tuple input.
    /// 
    /// # Parameters:
    /// - `noise_config`: The parameters of the noise being generated.
    /// - `combiner_config`: Additional parameters for the combiner algorithm.
    /// - `iters`: A tuple containing input simd iterators for each dimension.
    #[inline(always)]
    pub fn sample<A: Arch, I: DimIter<A, D>>(
        noise_config: NoiseConfig<D>,
        combiner_config: C::Config,
        iters: I,
    ) -> impl Iterator<Item = Simd<f32, A>> {
        let octaves = noise_config.octaves;

        let frequency: [_; D] = from_fn(|i| noise_config.scaling[i] * noise_config.frequency);
        let weight = if noise_config.normalization && octaves > 0 {
            noise_config.normalize_amplitude(noise_config.amplitude)
        } else {
            noise_config.amplitude
        };

        let mut seeds = [0u32; MAX_FBM_OCTAVES];
        let mut temp_freq = frequency;
        for seed in seeds.iter_mut().take(octaves) {
            *seed = gen_octave_seed(temp_freq, noise_config.seed);
            temp_freq
                .iter_mut()
                .for_each(|x| *x *= noise_config.lacunarity);
        }

        let lacunarity = Simd::splat(noise_config.lacunarity);
        let persistence = Simd::splat(noise_config.persistence);

        #[allow(clippy::too_many_arguments)]
        // #[enable_targets(A)]
        #[inline(always)]
        fn process_batch<const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch>(
            inputs: [Simd<f32, A>; D],
            octaves: usize,
            seeds: &[u32; 32],
            weight: f32,
            frequency: [f32; D],
            combiner_config: &C::Config,
            noise_config: &NoiseConfig<D>,
            persistence: Simd<f32, A>,
            lacunarity: Simd<f32, A>,
        ) -> Simd<f32, A> {
            if octaves == 0 {
                return Simd::zero();
            }
            let (mut state, mut sample): (C::State<A>, Simd<f32, A>) = Default::default();

            let seed = seeds[0];
            let mut weight = Simd::splat(weight);
            let mut freq = from_fn(|i| Simd::splat(frequency[i]));
            let new_sample = G::sample_batch(seed, inputs, freq) * weight;

            if noise_config.initialize {
                (state, sample) = C::initialize_sample(combiner_config, new_sample);
            } else {
                (state, sample) = C::apply_sample(combiner_config, state, sample, new_sample);
            }

            for seed in seeds.iter().take(octaves).skip(1) {
                freq.iter_mut().for_each(|x| *x *= lacunarity);
                if C::WEIGHT_DECAY {
                    weight *= persistence;
                }

                let new_sample = G::sample_batch(*seed, inputs, freq) * weight;
                (state, sample) = C::apply_sample(combiner_config, state, sample, new_sample);
            }

            if noise_config.finalize {
                C::finalize_sample(combiner_config, state, sample)
            } else {
                sample
            }
        }

        iters.map(move |x| {
            process_batch::<D, C, G, A>(
                x.into_array(),
                octaves,
                &seeds,
                weight,
                frequency,
                &combiner_config,
                &noise_config,
                persistence,
                lacunarity,
            )
        })
    }
}
