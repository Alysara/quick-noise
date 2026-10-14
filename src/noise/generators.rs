use std::marker::PhantomData;

use simply_simd::{ Arch, Simd };

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

/// Cellular noise distance function trait for function structs
pub trait DistanceFn: Default + Copy + Clone + PartialEq {
    type DistanceConfig: Copy + Default;

    /// Per-row y delta, hoisted out of the loop.
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, cfg: &Self::DistanceConfig) -> Simd<f32, A>;

    /// Distance from dx and delta.
    fn dist<A: Arch, const D: usize>(
        deltas: [Simd<f32, A>; D],
        cfg: &Self::DistanceConfig
    ) -> Simd<f32, A>;

    /// Lower bound on distance for any candidate beyond `edge` boundary.
    fn gate_val<A: Arch>(edge: Simd<f32, A>, cfg: &Self::DistanceConfig) -> Simd<f32, A>;

    /// Final work on the minimum distance found.
    #[inline(always)]
    fn finalize<A: Arch>(min: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        min
    }
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
pub struct Minowski;

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
    type DistanceConfig = ();

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta * delta
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_sq: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_sq
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |sum, delta| sum + delta)
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge * edge
    }

    #[inline(always)]
    fn finalize<A: Arch>(min_sq: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        min_sq.sqrt()
    }
}

impl DistanceFn for EuclideanSquared {
    type DistanceConfig = ();

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta * delta
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_sq: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_sq
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |sum, delta| sum + delta)
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge * edge
    }
}

impl DistanceFn for Hybrid {
    type DistanceConfig = ();

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta.mul_add(delta, delta.abs())
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_h: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_h
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |sum, delta| sum + delta)
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge * edge + edge
    }
}

impl DistanceFn for Manhattan {
    type DistanceConfig = ();

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta.abs()
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_abs: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_abs
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |sum, delta| sum + delta)
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge
    }
}

impl DistanceFn for MaxAxis {
    type DistanceConfig = ();

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta.abs()
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_abs: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_abs
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |max, delta| max.max(delta))
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, _cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge
    }
}

// D(X, Y) = ( sum(|x_i - y_i|^p) )^(1/p) for i = 1 to n
// Where X and Y are n-dimensional points.
// So here it may be confusing but both x1 and y1 would be on the same axis.
impl DistanceFn for Minowski {
    type DistanceConfig = MinkowskiConfig;

    #[inline(always)]
    fn modify_delta<A: Arch>(delta: Simd<f32, A>, cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        delta.abs().pow(Simd::splat(cfg.p))
    }

    #[inline(always)]
    fn dist<A: Arch, const D: usize>(
        deltas_p: [Simd<f32, A>; D],
        _cfg: &Self::DistanceConfig
    ) -> Simd<f32, A> {
        deltas_p
            .iter()
            .copied()
            .fold(Simd::splat(0.0), |sum, delta| sum + delta)
    }

    #[inline(always)]
    fn gate_val<A: Arch>(edge: Simd<f32, A>, cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        edge.pow(Simd::splat(cfg.p))
    }

    #[inline(always)]
    fn finalize<A: Arch>(min: Simd<f32, A>, cfg: &Self::DistanceConfig) -> Simd<f32, A> {
        min.pow(Simd::splat(1.0 / cfg.p))
    }
}

/// Cell-like noise created by the distance between each sample and its nearest node.
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Cellular<D: DistanceFn = Euclidean> {
    _distance: PhantomData<D>,
}

pub mod cellular {
    pub mod batch_2d;
    pub mod batch_3d;
    pub mod grid_2d;
    pub mod grid_3d;
}
