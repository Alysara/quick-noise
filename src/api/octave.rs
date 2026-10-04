/// Configuration struct for determining the frequencies and weight of a 
/// single octave (noise pass)
///
/// # Type Parameters
/// - `D`: The dimension of the octave.
#[derive(Copy, Clone)]
pub struct Octave<const D: usize> {
    pub weight: f32,
    pub frequency: [f32; D],
}

impl<const D: usize> Octave<D> {
    /// Creates a new octave (noise pass) configuration with
    /// explicit frequencies for each dimension.
    ///
    /// # Parameters
    /// - `frequency`: An array of frequencies for each axis.
    /// - `weight`: The weight of the octave.
    pub fn new(frequency: [f32; D], weight: f32) -> Self {
        Self { frequency, weight }
    }

    /// Creates a new octave (noise pass) configuration with
    /// the same frequency in all dimensions.
    ///
    /// # Parameters
    /// - `frequency`: A float representing how quickly the noise changes.
    /// - `weight`: The weight of the octave.
    pub fn splat(frequency: f32, weight: f32) -> Self {
        let frequency = [frequency; D];
        Self { frequency, weight }
    }
}
