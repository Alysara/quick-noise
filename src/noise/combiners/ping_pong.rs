use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray};

/// Config struct for PingPong combiner.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PingPongConfig {
    pub strength: f32,
}

impl Default for PingPongConfig {
    fn default() -> Self {
        Self { strength: 2.0 }
    }
}

/// Combines noise by oscillating samples together.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct PingPong {}

impl Combiner for PingPong {
    const WEIGHT_DECAY: bool = true;
    type State<A: Arch> = CombinerArray<A, 0>;
    type Config = PingPongConfig;

    #[inline(always)]
    fn apply_sample<A: Arch>(
        config: &PingPongConfig,
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        let one = Simd::splat(1.0);
        let two = Simd::splat(2.0);
        let t = (cur_result + new_sample) * Simd::splat(config.strength);
        let sawtooth = t - (t * Simd::splat(0.5)).floor() * two;
        let folded = one - (sawtooth - one).abs();
        (state, folded)
    }
}
