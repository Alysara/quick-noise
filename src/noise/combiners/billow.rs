use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray, noise::combiners::EmptyConfig};

/// Combines noise by adding the absolute value of subsequent octaves together.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Billow {}

impl Combiner for Billow {
    const WEIGHT_DECAY: bool = true;
    type State<A: Arch> = CombinerArray<A, 0>;
    type Config = EmptyConfig;

    #[inline(always)]
    fn apply_sample<A: Arch>(
        _config: &EmptyConfig,
        state: Self::State<A>,
        _weight: Simd<f32, A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (state, cur_result + new_sample.abs())
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        _config: &EmptyConfig,
        _weight: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (Self::State::default(), new_sample.abs())
    }

    #[inline(always)]
    fn finalize_sample<A: Arch>(
        _config: &EmptyConfig,
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        last
    }
}
