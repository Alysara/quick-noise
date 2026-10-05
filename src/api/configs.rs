/// Comprehensive config for noise parameters, including lacunarity
/// octave generation.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoiseConfig<const D: usize> {
    pub seed: u64,
    pub octaves: usize,
    pub frequency: f32,
    pub amplitude: f32,
    pub lacunarity: f32,
    pub persistence: f32,
    pub normalization: bool,
    pub initialize: bool,
    pub finalize: bool,
    pub magnification: f32,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub scaling: [f32; D],
}

/// Comprehensive config for noise parameters without lacunarity
/// octave generation.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OctaveNoiseConfig<const D: usize> {
    pub seed: u64,
    pub amplitude: f32,
    pub normalization: bool,
    pub initialize: bool,
    pub finalize: bool,
    pub magnification: f32,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub scaling: [f32; D],
}

/// Config specifying parameters of a grid.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GridConfig<const D: usize> {
    pub grid_seed: u64,
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub grid_size: [usize; D],
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub position: [f32; D],
    #[cfg_attr(feature = "serde", serde(with = "serde_arrays"))]
    pub tiling: [Option<u32>; D],
}

impl<const D: usize> NoiseConfig<D> {
    pub(crate) fn num_grid_octaves(&self) -> usize {
        let max_scaling = self.scaling.iter().fold(0.0, |max, x| x.max(max));

        let mut cur_freq = self.frequency * max_scaling;
        if cur_freq >= 1.0 || self.lacunarity >= 1.0 {
            for i in 0..self.octaves {
                if cur_freq >= 1.0 {
                    return i;
                }
                cur_freq *= self.lacunarity;
            }
        }

        self.octaves
    }

    pub(crate) fn normalize_amplitude(&self, amplitude: f32) -> f32 {
        let mut sum = 0.0;
        let mut cur = 1.0;
        for _ in 0..self.octaves {
            sum += cur;
            cur *= self.persistence;
        }
        amplitude / sum
    }
}
