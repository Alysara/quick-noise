use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray};

/// Config struct for the Ridged combiner.
#[derive(Copy, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RidgedConfig {
    pub gain: f32,
}

impl Default for RidgedConfig {
    fn default() -> Self {
        Self { gain: 2.0 }
    }
}

/// Combines noise using a signal that propagates into a running weight
/// for subsequent samples.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Ridged {}

impl Combiner for Ridged {
    const WEIGHT_DECAY: bool = false;
    type State<A: Arch> = CombinerArray<A, 1>;
    type Config = RidgedConfig;

    #[inline(always)]
    fn apply_sample<A: Arch>(
        config: &RidgedConfig,
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        let one = Simd::splat(1.0);
        let gain = Simd::splat(config.gain);
        let zero = Simd::splat(0.0);

        let weight = state[0];

        let signal = one - new_sample.abs();
        let signal = signal * signal * weight;

        let next_weight = (signal * gain).clamp(zero, one);
        let mut next_state = state;
        next_state[0] = next_weight;

        (next_state, cur_result + signal)
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        _config: &RidgedConfig,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        let one = Simd::splat(1.0);

        let signal = one - new_sample.abs();
        let signal = signal * signal;

        let mut state: Self::State<A> = Default::default();
        state[0] = signal;

        (state, signal)
    }

    #[inline(always)]
    fn finalize_sample<A: Arch>(
        _config: &RidgedConfig,
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        last
    }
}
