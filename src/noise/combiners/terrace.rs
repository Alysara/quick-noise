use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray};

/// Config struct for the Terrace combiner.
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TerraceConfig {
    pub steps: f32,
    pub step_size: f32,
}

impl Default for TerraceConfig {
    fn default() -> Self {
        Self {
            steps: 8.0,
            step_size: 1.0 / 8.0,
        }
    }
}

/// Combines noise by adding subsequent samples together and binning the
/// final result into a specified number of steps.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Terrace {}
impl Combiner for Terrace {
    const WEIGHT_DECAY: bool = true;
    type State<A: Arch> = CombinerArray<A, 0>;
    type Config = TerraceConfig;

    #[inline(always)]
    fn apply_sample<A: Arch>(
        _config: &TerraceConfig,
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (state, cur_result + new_sample)
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        _config: &TerraceConfig,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (Default::default(), new_sample)
    }

    #[inline(always)]
    fn finalize_sample<A: Arch>(
        config: &TerraceConfig,
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        (last * Simd::splat(config.steps)).round() * Simd::splat(config.step_size)
    }
}
