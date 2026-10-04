use std::ops::{Index, IndexMut};

use simply_simd::{Arch, Simd};

pub mod billow;
pub mod fbm;
pub mod hybrid_multi;
pub mod multi;
pub mod ping_pong;
pub mod ridged;
pub mod terrace;

pub use billow::Billow;
pub use fbm::Fbm;
pub use hybrid_multi::{HybridMulti, HybridMultiConfig};
pub use multi::Multi;
pub use ping_pong::{PingPong, PingPongConfig};
pub use ridged::{Ridged, RidgedConfig};
pub use terrace::{Terrace, TerraceConfig};

pub trait CombinerState<A: Arch>:
    Copy + Index<usize, Output = Simd<f32, A>> + IndexMut<usize> + Default
{
    const STATE_SIZE: usize;
}

impl<A: Arch, const N: usize> CombinerState<A> for [Simd<f32, A>; N]
where
    [Simd<f32, A>; N]: Default,
{
    const STATE_SIZE: usize = N;
}

pub type CombinerArray<A, const N: usize> = [Simd<f32, A>; N];

pub trait Combiner: Default + Copy + Clone {
    /// Determines whether or not octave weight parameters are ignored.
    /// If this is set to false, every octave has a weight of `1.0`.
    /// If this is set to true, every subsequent octave's weight is multiplied by persistence.
    const WEIGHT_DECAY: bool;

    /// The type used for expressing State. The type `[ArchSimd<f32>; N]` can be used,
    /// where N is the number of variables tracked across samples. N does not include
    /// the running result. The type alias `FractalArray<N>` can also be used.
    ///
    /// Each additional variable tracked across samples has a signifcant performance
    /// penalty when computing grid noise. The impact is minimal for batch noise.
    type State<A: Arch>: CombinerState<A>;

    /// The config struct that is passed through noise calls to the Fractal's usage.
    /// This is used for storing new parameters specific to a custom Fractal type.
    type Config: Copy + Default;

    /// Determines how new noise samples are combined with previous samples.
    ///
    /// # Parameters
    /// - `config`: Parameters for the combiner
    /// - `state`: An array of values maintained for each sample
    /// - `current`: Existing noise value from previous samples
    /// - `output`: New sample output from the current noise pass
    fn apply_sample<A: Arch>(
        _config: &Self::Config,
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (state, cur_result + new_sample)
    }

    /// Determines how the first sample is initialized.
    ///
    /// For maximum performance and compiler optimization, it is
    /// recommended to avoid unnecessary instructions such as
    /// adding to 0.0 and multiplying by 1.0.
    ///
    /// # Parameters
    /// - `config`: Parameters for the combiner
    /// - `new_sample`: New sample output from the first noise pass
    #[inline(always)]
    fn initialize_sample<A: Arch>(
        config: &Self::Config,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        Self::apply_sample(config, Default::default(), Default::default(), new_sample)
    }

    /// Determines how the final noise sample is processed after
    /// being fully combined. This is after `sample` or `sample_first`
    /// has been called.
    ///
    /// # Parameters
    /// - `config`: Parameters for the combiner
    /// - `state`: An array of values maintained for each sample
    /// - `last`: The final noise sample after prior fractal processing
    #[inline(always)]
    fn finalize_sample<A: Arch>(
        _config: &Self::Config,
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        last
    }
}
