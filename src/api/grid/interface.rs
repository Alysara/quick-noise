use std::marker::PhantomData;

use crate::Combiner;
use crate::api::configs::GridConfig;
use crate::api::grid::builder::GridNoiseBuilder;
use crate::api::grid::octaves_builder::OctaveGridNoiseBuilder;
use crate::api::octave::Octave;
use crate::math::random::Random;
use crate::simd::{Arch, StaticArch};

/// Handles raw parameters for grid noise generators
#[derive(Copy, Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridNoiseParams<const D: usize> {
    pub seed: u32,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub grid_size: [usize; D],
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub position: [f32; D],
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub frequency: [f32; D],
    pub weight: f32,
    pub magnification: f32,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub tiling: [Option<u32>; D],
}

pub trait GridGenerator<const D: usize>: Default + Copy + Clone + PartialEq {
    /// Generates noise for a grid region.
    ///
    /// # Type Parameters
    /// - `C`: Type of combiner used to layer noise
    /// - `INIT`: Whether or not the generator should initialize dst
    /// - `FINAL`: Whether or not the generator should finalize the final octave
    ///
    /// # Runtime Parameters
    /// - `params`: Config specifying general noise parameters
    /// - `combiner_config`: Config specifying combiner parameters
    /// - `state`: Buffer containing sample information across octaves
    /// - `dst`: Buffer to insert the results into
    fn sample_grid<F: Arch, C: Combiner, const INIT: bool, const FINAL: bool>(
        params: GridNoiseParams<D>,
        combiner_config: C::Config,
        state: &mut [f32],
        dst: &mut [f32],
    );
}

/// Static struct for sampling grid noise.
pub struct GridNoise<const D: usize, C: Combiner, S: GridGenerator<D>> {
    _fractal: PhantomData<C>,
    _sampler: PhantomData<S>,
}

/// An interface struct for creating grid noise.
///
/// # Type Parameters
/// * `D: NoiseDimension` - Determines how many dimensions the grid has.
///
/// # Example
/// ```
/// use quick_noise::Grid;
///
/// // Subject to change.
/// let grid = Grid::<2>::new(32, 32)
///     .grid_position(0.0, 0.0)
///     .seed(1);
/// ```

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct Grid<const D: usize, A: Arch = StaticArch> {
    pub(crate) config: GridConfig<D>,
    pub(crate) _arch: PhantomData<A>,
}

impl<const D: usize, A: Arch> Grid<D, A> {
    /// Determines the psuedo-random values used in noise generation called
    /// on this grid. Different seeds produce different noise.
    pub fn seed(mut self, seed: i64) -> Self {
        self.config.grid_seed = Random::mix_u64(seed as u64);
        self
    }
}

impl<A: Arch> Grid<2, A> {
    /// Creates an anchor for a grid region that can be used for call noise.
    ///
    /// # Parameters
    /// -`x`: Length of the grid region along the x-axis
    /// -`y`: Length of the grid region along the y-axis
    pub fn new(x: usize, y: usize) -> Self {
        let config = GridConfig {
            grid_size: [x, y],
            ..Default::default()
        };
        Self {
            config,
            _arch: PhantomData::<A>,
        }
    }

    /// Determines the position values provided to noise calls. This value represents
    /// the position of this grid region in grid units determined by its grid_size.
    /// A 32x32 grid at position `{ 1, 2 }` covers samples in the range `{ 32..64, 64..96 }`.
    ///
    /// # Default:
    /// `0`: x
    /// `0`: y
    pub fn grid_position(mut self, x: f32, y: f32) -> Self {
        self.config.position = [
            x * self.config.grid_size[0] as f32,
            y * self.config.grid_size[1] as f32,
        ];
        self
    }

    /// Determines the position values provided to noise calls. This value represents
    /// the position of first sample in each dimension. A 32x32 given the sample position
    /// `{ 32, 16 }` covers samples in the range `{ 32..64, 16..48 }`.
    ///
    /// # Default:
    /// `0`: x
    /// `0`: y
    pub fn sample_position(mut self, x: f32, y: f32) -> Self {
        self.config.position = [x, y];
        self
    }

    /// Determines the distance the sample space has until it starts repeating noise
    /// seamlessly. When values are left as None, noise does not repeat.
    ///
    /// # Default:
    /// - `x`: None
    /// - `y`: None
    pub fn tiling(mut self, x: Option<u32>, y: Option<u32>) -> Self {
        self.config.tiling = [x, y];
        self
    }
}

impl<A: Arch> Grid<3, A> {
    /// Creates an anchor for a grid region that can be used for call noise.
    ///
    /// # Parameters
    /// -`x`: Length of the grid region along the x-axis
    /// -`y`: Length of the grid region along the y-axis
    /// -`z`: Length of the grid region along the z-axis
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        let config = GridConfig {
            grid_size: [x, y, z],
            ..Default::default()
        };
        Self {
            config,
            _arch: PhantomData::<A>,
        }
    }

    /// Determines the position values provided to noise calls. This value represents
    /// the position of this grid region in grid units determined by its grid_size.
    /// A 32x32x32 grid at position `{ 1, 2, 3 }` covers samples in the range
    /// `{ 32..64, 64..96, 96..128 }`.
    ///
    /// # Default:
    /// `0`: x
    /// `0`: y
    /// `0`: z
    pub fn grid_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.config.position = [
            x * self.config.grid_size[0] as f32,
            y * self.config.grid_size[1] as f32,
            z * self.config.grid_size[2] as f32,
        ];
        self
    }

    /// Determines the position values provided to noise calls. This value represents
    /// the position of first sample in each dimension. A 32x32x32 given the sample position
    /// `{ 32, 16, 0 }` covers samples in the range `{ 32..64, 16..48, 0..32 }`.
    ///
    /// # Default:
    /// `0`: x
    /// `0`: y
    /// `0`: z
    pub fn sample_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.config.position = [x, y, z];
        self
    }

    /// Determines the distance the sample space has until it starts repeating noise
    /// seamlessly. When values are left as None, noise does not repeat.
    ///
    /// # Default:
    /// - `x`: None
    /// - `y`: None
    /// - `z`: None
    pub fn tiling(mut self, x: Option<u32>, y: Option<u32>, z: Option<u32>) -> Self {
        self.config.tiling = [x, y, z];
        self
    }
}

impl<const D: usize, A: Arch> Grid<D, A> {
    /// Loads a config a config to create a grid.
    pub fn from_config(config: GridConfig<D>) -> Self {
        Self {
            config,
            _arch: PhantomData::<A>,
        }
    }

    /// Creates a new builder to easily configure a grid region of noise.
    pub fn builder<F: Combiner, T: GridGenerator<D>>(&self) -> GridNoiseBuilder<D, F, T, A> {
        GridNoiseBuilder::from_config(self.config)
    }

    /// Creates a new builder using a custom octave list to configure
    /// a grid region of noise.
    pub fn builder_with_octaves<'a, F: Combiner, T: GridGenerator<D>>(
        &self,
        octave_list: &'a [Octave<D>],
    ) -> OctaveGridNoiseBuilder<'a, D, F, T, A> {
        OctaveGridNoiseBuilder::new(self.config, octave_list)
    }
}
