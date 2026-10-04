#![doc = include_str!("../README.md")]

//! Blazingly fast SIMD procedural noise library for batch and uniform grid
//! sampling with runtime feature detection on stable Rust

extern crate self as quick_noise;
pub mod api;
pub mod math;
mod noise;
pub mod simd;

#[cfg(feature = "image")]
pub mod emit {
    mod grayscale;
    pub use grayscale::NoiseImageExt;
}

// pub use api::grid::interface::{GridGenerator, GridNoiseParams};
// pub use api::batch::interface::{BatchGenerator};

pub use api::batch::interface::{BatchGenerator, BatchNoise};
pub use api::defaults::*;
pub use api::grid::interface::{Grid, GridGenerator, GridNoise, GridNoiseParams};
pub use api::octave::Octave;
pub use api::{
    BatchNoiseBuilder, GridNoiseBuilder, OctaveBatchNoiseBuilder, OctaveGridNoiseBuilder,
};
pub use noise::combiners::{
    Billow, Combiner, CombinerArray, CombinerState, Fbm, HybridMulti, HybridMultiConfig, Multi,
    PingPong, PingPongConfig, Ridged, RidgedConfig, Terrace, TerraceConfig,
};
pub use noise::generators::{Cellular, Perlin, Simplex, Value};
// pub use noise::*;
