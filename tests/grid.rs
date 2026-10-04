use std::hint::black_box;

use quick_noise::simd::Simd;
use quick_noise::*;
use simply_simd::{Arch, SimdSliceIterExt, dispatch_simd, enable_targets};

#[test]
fn tiny_grid_2d_static() {
    test_grid_2d::<10, _>(Grid::<2>::new(2, 5));
}
#[test]
fn regular_grid_2d_static() {
    test_grid_2d::<1024, _>(Grid::<2>::new(32, 32));
}
#[test]
fn irregular_grid_2d_static() {
    test_grid_2d::<1452, _>(Grid::<2>::new(33, 44));
}
#[test]
fn large_grid_2d_static() {
    test_grid_2d::<10000, _>(Grid::<2>::new(100, 100));
}
#[test]
fn long_grid_2d_static() {
    test_grid_2d::<50000, _>(Grid::<2>::new(10000, 5));
}

#[test]
#[dispatch_simd(A)]
fn tiny_grid_2d_dyn() {
    test_grid_2d::<10, _>(Grid::<2, A>::new(2, 5));
}
#[test]
#[dispatch_simd(A)]
fn regular_grid_2d_dyn() {
    test_grid_2d::<1024, _>(Grid::<2, A>::new(32, 32));
}
#[test]
#[dispatch_simd(A)]
fn irregular_grid_2d_dyn() {
    test_grid_2d::<1452, _>(Grid::<2, A>::new(33, 44));
}
#[test]
#[dispatch_simd(A)]
fn large_grid_2d_dyn() {
    test_grid_2d::<10000, _>(Grid::<2, A>::new(100, 100));
}
#[test]
#[dispatch_simd(A)]
fn long_grid_2d_dyn() {
    test_grid_2d::<50000, _>(Grid::<2, A>::new(10000, 5));
}

#[test]
fn tiny_grid_3d_static() {
    test_grid_3d::<30, _>(Grid::<3>::new(2, 5, 3));
}
#[test]
fn regular_grid_3d_static() {
    test_grid_3d::<4096, _>(Grid::<3>::new(16, 16, 16));
}
#[test]
fn irregular_grid_3d_static() {
    test_grid_3d::<5610, _>(Grid::<3>::new(17, 22, 15));
}
#[test]
fn large_grid_3d_static() {
    test_grid_3d::<125000, _>(Grid::<3>::new(50, 50, 50));
}
#[test]
fn long_grid_3d_static() {
    test_grid_3d::<90000, _>(Grid::<3>::new(10000, 3, 3));
}

#[test]
#[dispatch_simd(A)]
fn tiny_grid_3d_dyn() {
    test_grid_3d::<30, _>(Grid::<3, A>::new(2, 5, 3));
}
#[test]
#[dispatch_simd(A)]
fn regular_grid_3d_dyn() {
    test_grid_3d::<4096, _>(Grid::<3, A>::new(16, 16, 16));
}
#[test]
#[dispatch_simd(A)]
fn irregular_grid_3d_dyn() {
    test_grid_3d::<5610, _>(Grid::<3, A>::new(17, 22, 15));
}
#[test]
#[dispatch_simd(A)]
fn large_grid_3d_dyn() {
    test_grid_3d::<125000, _>(Grid::<3, A>::new(50, 50, 50));
}
#[test]
#[dispatch_simd(A)]
fn long_grid_3d_dyn() {
    test_grid_3d::<90000, _>(Grid::<3, A>::new(10000, 3, 3));
}

#[test]
fn fractional_position_matches_integer_position() {
    const CHUNK_SIZE: usize = 16;
    const MIDDLE: usize = CHUNK_SIZE / 2;

    let normal = Grid::<2>::new(CHUNK_SIZE, CHUNK_SIZE);
    let normal = normal.builder::<Fbm, Perlin>().build();

    // Magnified grid should be able to lookup fractional positions
    // and they should match with unmagnified grid values
    let zoomed_out = Grid::<2>::new(2, 2).sample_position(0.5, 0.5);
    let zommed_out = zoomed_out
        .builder::<Fbm, Perlin>()
        .magnification(CHUNK_SIZE as f32)
        .build();

    assert!((normal[MIDDLE * CHUNK_SIZE + MIDDLE] - zommed_out[0]).abs() < 1e-6);
}

#[test]
fn tiled_grid_2d() {
    let grid = Grid::<2>::new(128, 256).tiling(Some(64), None);

    let mut result = [0.0; 32768];
    grid.builder::<Fbm, Perlin>()
        .octaves(6)
        .fill(result.as_mut_slice());
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Value>()
        .octaves(6)
        .initialize(false)
        .fill(result.as_mut_slice());
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Cellular>()
        .octaves(6)
        .initialize(false)
        .fill(result.as_mut_slice());
    verify_slice(result.as_slice());
}

#[test]
fn tiled_grid_3d() {
    let grid = Grid::<3>::new(32, 48, 16).tiling(Some(32), Some(16), None);

    let mut result = [0.0; 24576];
    grid.builder::<Fbm, Perlin>()
        .frequency(1.0 / 8.0)
        .octaves(6)
        .fill(result.as_mut_slice());
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Value>()
        .frequency(1.0 / 8.0)
        .octaves(6)
        .initialize(false)
        .fill(result.as_mut_slice());
    verify_slice(result.as_slice());
}

fn verify_slice(slice: &[f32]) {
    let mut min = f32::MAX;
    let mut max = f32::NEG_INFINITY;
    let mut dif_total = 0.0;
    let mut dif_max = 0.0;

    let mut iter = slice.iter();
    let prev = iter.next().expect("Output is empty!");

    for val in slice.iter() {
        min = val.min(min);
        max = val.max(max);

        let dif = (*val - prev).abs();
        dif_total += dif;
        dif_max = dif.max(dif_max);
    }

    assert!(min > -10.0, "Minimum value of {min} was below -10!");
    assert!(max < 10.0, "Maximum value of {max} was above 10!");
    assert!(dif_total > 0.0, "Output is constant of {}!", slice[0]);
}

#[enable_targets(A)]
fn test_grid_2d<const N: usize, A: Arch>(grid: Grid<2, A>) {
    let mut result = [0.0; N];
    grid.builder::<Fbm, Perlin>().octaves(4).fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Value>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Cellular>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Billow, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Billow, Value>().octaves(2).fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Billow, Cellular>().octaves(2).fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Multi, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<HybridMulti, Perlin>().fill(&mut result);

    verify_slice(result.as_slice());
    grid.builder::<PingPong, Perlin>().fill(&mut result);

    verify_slice(result.as_slice());
    grid.builder::<Terrace, Perlin>().fill(&mut result);

    grid.builder::<Ridged, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Ridged, Value>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Ridged, Cellular>().fill(&mut result);
    verify_slice(result.as_slice());

    let noise1 = grid
        .builder::<Fbm, Perlin>()
        .scaling(0.6, 0.8)
        .frequency(0.023)
        .normalization(false)
        .build();

    let noise2 = grid
        .builder::<Ridged, Value>()
        .octaves(12)
        .gain(0.8)
        .build();

    let noise3 = grid
        .builder::<Terrace, Perlin>()
        .octaves(8)
        .frequency(1. / 128.)
        .normalization(false)
        .into_iter()
        .map(|x| x.mul_add(Simd::splat(1.5), Simd::splat(12.8)));

    let noise: Vec<f32> = grid
        .warp_builder::<Fbm, Perlin>(100.0, noise3, noise2.simd_iter())
        .octaves(4)
        .build();
    black_box(&noise);

    let noise: Vec<f32> = grid
        .warp_builder::<Billow, Cellular>(0.5, noise1.simd_iter(), grid.y_iter())
        .octaves(2)
        .build();
    black_box(&noise);

    let octave_list = [
        Octave::<2>::splat(1.0 / 100.0, 1.0),
        Octave::<2>::splat(1.0 / 30.0, 1.2),
        Octave::<2>::splat(1.0 / 1000.0, 0.8),
        Octave::<2>::new([0.01, 0.005], 0.9),
    ];

    let noise: Vec<f32> = grid
        .builder_with_octaves::<Fbm, Perlin>(&octave_list)
        .amplitude(100.0)
        .normalization(false)
        .build();

    black_box(&noise);
}

#[enable_targets(A)]
fn test_grid_3d<const N: usize, A: Arch>(grid: Grid<3, A>) {
    let mut result = [0.0; N];
    grid.builder::<Fbm, Perlin>().octaves(4).fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Fbm, Value>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Billow, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Billow, Value>().octaves(2).fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Multi, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<HybridMulti, Perlin>().fill(&mut result);

    verify_slice(result.as_slice());
    grid.builder::<PingPong, Perlin>().fill(&mut result);

    verify_slice(result.as_slice());
    grid.builder::<Terrace, Perlin>().fill(&mut result);

    grid.builder::<Ridged, Perlin>().fill(&mut result);
    verify_slice(result.as_slice());

    grid.builder::<Ridged, Value>().fill(&mut result);
    verify_slice(result.as_slice());

    let noise1 = grid
        .builder::<Fbm, Perlin>()
        .scaling(0.6, 0.8, 1.2)
        .frequency(0.023)
        .normalization(false)
        .build();

    let noise2 = grid
        .builder::<Ridged, Value>()
        .octaves(12)
        .gain(0.8)
        .build();

    let noise3 = grid
        .builder::<Terrace, Perlin>()
        .octaves(8)
        .frequency(1. / 128.)
        .normalization(false)
        .into_iter()
        .map(|x| x.mul_add(Simd::splat(1.5), Simd::splat(12.8)));

    let noise: Vec<f32> = grid
        .warp_builder::<Fbm, Perlin>(100.0, noise3, noise2.simd_iter(), noise1.simd_iter())
        .octaves(4)
        .build();
    black_box(&noise);

    let noise: Vec<f32> = grid
        .warp_builder::<Billow, Cellular>(0.5, noise1.simd_iter(), grid.x_iter(), grid.z_iter())
        .octaves(2)
        .build();
    black_box(&noise);

    let octave_list = [
        Octave::<3>::splat(1.0 / 100.0, 1.0),
        Octave::<3>::splat(1.0 / 30.0, 1.2),
        Octave::<3>::splat(1.0 / 1000.0, 0.8),
        Octave::<3>::new([0.01, 0.005, 0.008], 0.9),
    ];

    let noise: Vec<f32> = grid
        .builder_with_octaves::<Fbm, Perlin>(&octave_list)
        .amplitude(100.0)
        .normalization(false)
        .build();

    black_box(&noise);
}
