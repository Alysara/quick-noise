use std::marker::PhantomData;

use crate::noise::combiners::Combiner;
use crate::simd::{Arch, Simd};

/// Zipped iterator type used for generically handling
/// `D` iterators.
pub trait DimTuple<A: Arch, const D: usize> {
    /// Converts the internal iterator representation
    /// into a generically sized array.
    fn into_array(self) -> [Simd<f32, A>; D];
}
type S<A> = Simd<f32, A>;

impl<A: Arch> DimTuple<A, 1> for S<A> {
    fn into_array(self) -> [Simd<f32, A>; 1] {
        [self]
    }
}

impl<A: Arch> DimTuple<A, 2> for (S<A>, S<A>) {
    fn into_array(self) -> [Simd<f32, A>; 2] {
        [self.0, self.1]
    }
}

impl<A: Arch> DimTuple<A, 3> for (S<A>, S<A>, S<A>) {
    fn into_array(self) -> [Simd<f32, A>; 3] {
        [self.0, self.1, self.2]
    }
}

impl<A: Arch> DimTuple<A, 4> for (S<A>, S<A>, S<A>, S<A>) {
    fn into_array(self) -> [Simd<f32, A>; 4] {
        [self.0, self.1, self.2, self.3]
    }
}

impl<A: Arch> DimTuple<A, 5> for (S<A>, S<A>, S<A>, S<A>, S<A>) {
    fn into_array(self) -> [Simd<f32, A>; 5] {
        [self.0, self.1, self.2, self.3, self.4]
    }
}

pub trait DimIter<A: Arch, const D: usize>: Iterator<Item: DimTuple<A, D>> {}

impl<A: Arch, const D: usize, T: DimTuple<A, D>, I: Iterator<Item = T>> DimIter<A, D> for I {}

pub trait BatchGenerator<const D: usize> {
    /// Generates noise using simd registers.
    ///
    /// # Parameters
    /// - `seed`: Configures the deterministic randomness of the noise
    /// - `input`: Array of input values for each dimension
    /// - `freq`: Array of frequency values for each dimension
    fn sample_batch<A: Arch>(
        seed: u32,
        input: [Simd<f32, A>; D],
        freq: [Simd<f32, A>; D],
    ) -> Simd<f32, A>;
}

///  struct for sampling batch noise.
///
/// # Example
/// ```
/// use quick_noise::{Grid, BatchNoise, Fbm, Perlin};
///
/// let grid = Grid::<2>::new(32, 32);
///
/// let noise = BatchNoise::<2, Fbm, Perlin>::builder(grid.x_iter(), grid.y_iter())
///     .octaves(1)
///     .frequency(1.0 / 32.0)
///     .build();
/// ```
#[derive(Default)]
pub struct BatchNoise<const D: usize, C: Combiner, G: BatchGenerator<D>> {
    _fractal: PhantomData<C>,
    _sampler: PhantomData<G>,
}
