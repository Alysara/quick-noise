use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray};

/// Config struct for the HybridMulti combiner.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HybridMultiConfig {
    pub gain: f32,
    pub offset: f32,
}

impl Default for HybridMultiConfig {
    fn default() -> Self {
        Self {
            gain: 2.0,
            offset: 1.0,
        }
    }
}

/// Combines noise using a weighted signal that propagates into a running weight
/// for subsequent samples by multiplication.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct HybridMulti {}
impl Combiner for HybridMulti {
    const WEIGHT_DECAY: bool = false;
    type State<A: Arch> = CombinerArray<A, 1>;
    type Config = HybridMultiConfig;

    #[inline(always)]
    fn apply_sample<A: Arch>(
        config: &HybridMultiConfig,
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        let zero = Simd::splat(0.0);
        let one = Simd::splat(1.0);
        let signal = new_sample + Simd::splat(config.offset);
        let weighted_signal = state[0] * signal;

        let result = cur_result + weighted_signal;
        let weight = (weighted_signal * Simd::splat(config.gain)).clamp(zero, one);

        ([weight], result)
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        config: &HybridMultiConfig,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        let zero = Simd::splat(0.0);
        let one = Simd::splat(1.0);

        let signal = new_sample + Simd::splat(config.offset);
        let weight = (signal * Simd::splat(config.gain)).clamp(zero, one);

        ([weight], signal)
    }
}
