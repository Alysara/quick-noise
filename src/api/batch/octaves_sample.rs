use std::array::from_fn;

use simply_simd::enable_targets;

use crate::BatchGenerator;
use crate::api::batch::interface::{BatchNoise, DimIter, DimTuple};
use crate::api::configs::*;
use crate::api::octave::Octave;
use crate::api::seed::gen_octave_seed;
use crate::noise::combiners::Combiner;
use crate::simd::{Arch, Simd};

const MAX_CUSTOM_OCTAVES: usize = 32;

fn get_max<const D: usize>(array: [f32; D]) -> f32 {
    array.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Helper static function for custom noise.
impl<const D: usize, C: Combiner, G: BatchGenerator<D>> BatchNoise<D, C, G> {
    /// Samples a batch of noise using arbitrary inputs from simd iterators with custom octaves.
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
    /// - `octave_list`: List of octaves to create noise with.
    /// - `iters`: A tuple containing input simd iterators for each dimension.
    #[inline(always)]
    pub fn sample_with_octaves<A: Arch, I: DimIter<A, D>>(
        noise_config: OctaveNoiseConfig<D>,
        combiner_config: C::Config,
        octave_list: &[Octave<D>],
        iters: I,
    ) -> impl Iterator<Item = Simd<f32, A>> {
        let total_weight = octave_list
            .iter()
            .filter(|x| get_max(x.frequency) < 1.0)
            .fold(0.0, |acc, x| acc + x.weight);

        let weight_coef = match (noise_config.normalization, total_weight == 0.0) {
            (false, _) => noise_config.amplitude,
            (true, false) => noise_config.amplitude / total_weight,
            (true, true) => 0.0,
        };

        let mut seeds = [0u32; MAX_CUSTOM_OCTAVES];
        for (i, octave) in octave_list.iter().enumerate() {
            seeds[i] = gen_octave_seed(octave.frequency, noise_config.seed);
        }

        #[enable_targets(A)]
        fn process_batch<const D: usize, C: Combiner, G: BatchGenerator<D>, A: Arch>(
            inputs: [Simd<f32, A>; D],
            octave_list: &[Octave<D>],
            seeds: &[u32; 32],
            weight_coef: f32,
            combiner_config: &C::Config,
            noise_config: &OctaveNoiseConfig<D>,
        ) -> Simd<f32, A> {
            if octave_list.is_empty() {
                return Simd::zero();
            }

            let (mut state, mut sample): (C::State<A>, Simd<f32, A>) = Default::default();

            let freq = from_fn(|i| Simd::splat(octave_list[0].frequency[i]));
            let seed = seeds[0];
            let weight = Simd::splat(octave_list[0].weight * weight_coef);
            let new_sample = G::sample_batch(seed, inputs, freq) * weight;
            if noise_config.initialize {
                (state, sample) = C::initialize_sample(combiner_config, new_sample);
            } else {
                (state, sample) = C::apply_sample(combiner_config, state, sample, new_sample);
            }

            for (i, octave) in octave_list
                .iter()
                .enumerate()
                .skip(1)
                .take(octave_list.len().saturating_sub(2))
            {
                let freq = from_fn(|i| Simd::splat(octave.frequency[i]));
                let seed = seeds[i];
                let weight = Simd::splat(octave_list[0].weight * weight_coef);
                let new_sample = G::sample_batch(seed, inputs, freq) * weight;
                (state, sample) = C::apply_sample(combiner_config, state, sample, new_sample);
            }

            if noise_config.finalize {
                C::finalize_sample(combiner_config, state, sample)
            } else {
                sample
            }
        }

        iters.map(move |x| {
            let inputs = x.into_array();
            process_batch::<D, C, G, A>(
                inputs,
                octave_list,
                &seeds,
                weight_coef,
                &combiner_config,
                &noise_config,
            )
        })
    }
}
