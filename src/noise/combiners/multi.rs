use simply_simd::{Arch, Simd};

use crate::{Combiner, CombinerArray};

/// Combines noise by multiplying subsequent octaves together.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Multi {}

impl Combiner for Multi {
    const WEIGHT_DECAY: bool = false;
    type State<A: Arch> = CombinerArray<A, 0>;
    type Config = ();

    #[inline(always)]
    fn apply_sample<A: Arch>(
        _config: &(),
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (state, cur_result * (new_sample + Simd::splat(1.0)))
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        _config: &(),
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (Default::default(), new_sample + Simd::splat(1.0))
    }

    #[inline(always)]
    fn finalize_sample<A: Arch>(
        _config: &(),
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        last
    }
}
