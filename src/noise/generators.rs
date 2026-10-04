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
}

/// Cell-like noise created by the distance between each sample and its nearest node.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Cellular {}
pub mod cellular {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
}
