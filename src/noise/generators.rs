#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Perlin {}
pub mod perlin {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
    pub mod grid_3d;
}

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Value {}
pub mod value {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
    pub mod grid_3d;
}

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Simplex {}
pub mod simplex {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
}

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Cellular {}
pub mod cellular {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
}
