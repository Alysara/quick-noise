use std::marker::PhantomData;

/// Smooth organic gradient noise using Ken Perlin's algorithm.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Perlin {}
pub mod perlin {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
    pub mod grid_3d;
}

/// Fast blocky gradient noise that interpolates random values of each sample's cell.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Value {}
pub mod value {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
    pub mod grid_3d;
}

/// Smooth organic gradient noise using a skewed triangular grid for less artifacting.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Simplex {}
pub mod simplex {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
}

// Cellular noise distance function trait for function structs
pub trait DistanceFn: Default + Copy + Clone + PartialEq {
    type Config: Copy + Default;
}
/// Standard straight-line distance: sqrt(x^2 + y^2 + ...).
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Euclidean;
/// Squared Euclidean distance (faster, skips sqrt).
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct EuclideanSquared;
/// EuclideanSquared + Manhattan combined. Produces organic, rounded cell shapes.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Hybrid;
/// Sum of absolute axis differences: `|x| + |y| + ...`. Produces diamond-shaped cells.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Manhattan;
/// Maximum absolute difference along any single axis. Produces square-shaped cells.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct MaxAxis;
/// Generalised distance metric parameterised by P. P=1 is Manhattan, P=2 is Euclidean,
/// P=infinity is Chebyshev Distance.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Minwoski;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MinkowskiConfig {
    pub p: f32,
}

impl Default for MinkowskiConfig {
    fn default() -> Self {
        Self { p: 2.0 }
    }
}

impl DistanceFn for Euclidean {
    type Config = ();
}
impl DistanceFn for EuclideanSquared {
    type Config = ();
}
impl DistanceFn for Hybrid {
    type Config = ();
}
impl DistanceFn for Manhattan {
    type Config = ();
}
impl DistanceFn for MaxAxis {
    type Config = ();
}
impl DistanceFn for Minwoski {
    type Config = MinkowskiConfig;
}

/// Cell-like noise created by the distance between each sample and its nearest node.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Cellular<D: DistanceFn> {
    _distance: PhantomData<D>,
}

pub mod cellular {
    pub mod batch_2d_euclidean;
    pub mod batch_2d_euclidean_sq;
    pub mod batch_3d;
    pub mod grid_2d_euclidean;
    pub mod grid_2d_euclidean_sq;
}
