use std::hint::black_box;

use quick_noise::*;
use simply_simd::{Arch, Simd, SimdSliceIterExt, StaticArch, dispatch_simd, enable_targets};

#[test]
fn batch_2d_static() {
    batch_2d::<StaticArch>();
}
#[test]
fn batch_3d_static() {
    batch_3d::<StaticArch>();
}

#[test]
#[dispatch_simd(A)]
fn batch_2d_dyn() {
    batch_2d::<A>();
}
#[test]
#[dispatch_simd(A)]
fn batch_3d_dyn() {
    batch_3d::<A>();
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
fn batch_2d<A: Arch>() {
    let tiny_grid = Grid::<2, A>::new(1, 2);
    let medium_grid = Grid::<2, A>::new(17, 28);
    let large_grid = Grid::<2, A>::new(100, 110);

    let mut tiny_result = [0.0; 2];
    let mut medium_result = [0.0; 476];
    let mut large_result = [0.0; 11000];

    BatchNoise::<2, Fbm, Perlin>::builder(tiny_grid.x_iter(), tiny_grid.y_iter())
        .octaves(6)
        .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<2, Fbm, Value>::builder(tiny_grid.x_iter(), tiny_grid.y_iter())
        .octaves(6)
        .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<2, Fbm, Simplex>::builder(tiny_grid.x_iter(), tiny_grid.y_iter())
        .octaves(6)
        .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<2, Fbm, Cellular<Euclidean>>::builder(tiny_grid.x_iter(), tiny_grid.y_iter())
        .octaves(6)
        .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());

    BatchNoise::<2, Fbm, Perlin>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, Fbm, Value>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, Fbm, Simplex>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, Fbm, Cellular<Euclidean>>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());

    BatchNoise::<2, Fbm, Perlin>::builder(large_grid.x_iter(), large_grid.y_iter())
        .octaves(6)
        .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<2, Fbm, Value>::builder(large_grid.x_iter(), large_grid.y_iter())
        .octaves(6)
        .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<2, Fbm, Simplex>::builder(large_grid.x_iter(), large_grid.y_iter())
        .octaves(6)
        .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<2, Fbm, Cellular<Euclidean>>::builder(large_grid.x_iter(), large_grid.y_iter())
        .octaves(6)
        .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());

    BatchNoise::<2, Billow, Perlin>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, Ridged, Value>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, PingPong, Simplex>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, Multi, Cellular<Euclidean>>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<2, HybridMulti, Simplex>::builder(medium_grid.x_iter(), medium_grid.y_iter())
        .octaves(6)
        .gain(0.8)
        .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());

    let values1: [f32; 1234] = std::array::from_fn(|i| i as f32);
    let values2: [f32; 1234] = std::array::from_fn(|i| 2.0 * i as f32);

    let noise: [f32; 1234] =
        BatchNoise::<2, Terrace, Perlin>::builder(values1.simd_iter::<A>(), values2.simd_iter())
            .frequency(0.01)
            .scaling(0.5, 2.0)
            .steps(10.0)
            .normalization(false)
            .into_iter()
            .map(|x| x + Simd::splat(10.0))
            .collect();
    black_box(&noise);

    let noise: [f32; 1234] =
        BatchNoise::<2, Multi, Cellular<Euclidean>>::builder(values1.simd_iter::<A>(), values2.simd_iter())
            .octaves(10)
            .frequency(0.01)
            .scaling(0.5, 2.0)
            .normalization(false)
            .into_iter()
            .map(|x| x + Simd::splat(10.0))
            .collect();
    black_box(&noise);

    let octave_list = [
        Octave::<2>::splat(1.0 / 100.0, 1.0),
        Octave::<2>::splat(1.0 / 30.0, 1.2),
        Octave::<2>::splat(1.0 / 1000.0, 0.8),
        Octave::<2>::new([0.01, 0.005], 0.9),
    ];

    let noise: [f32; 1234] = BatchNoise::<2, Terrace, Perlin>::builder_with_octaves(
        &octave_list,
        values1.simd_iter::<A>(),
        values2.simd_iter(),
    )
    .scaling(0.5, 2.0)
    .steps(10.0)
    .normalization(false)
    .into_iter()
    .map(|x| x + Simd::splat(10.0))
    .collect();
    black_box(&noise);
}

#[enable_targets(A)]
fn batch_3d<A: Arch>() {
    let tiny_grid = Grid::<3, A>::new(1, 2, 2);
    let medium_grid = Grid::<3, A>::new(17, 28, 12);
    let large_grid = Grid::<3, A>::new(40, 40, 40);

    let mut tiny_result = [0.0; 4];
    let mut medium_result = [0.0; 5712];
    let mut large_result = [0.0; 64000];

    BatchNoise::<3, Fbm, Perlin>::builder(
        tiny_grid.x_iter(),
        tiny_grid.y_iter(),
        tiny_grid.z_iter(),
    )
    .octaves(6)
    .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<3, Fbm, Value>::builder(
        tiny_grid.x_iter(),
        tiny_grid.y_iter(),
        tiny_grid.z_iter(),
    )
    .octaves(6)
    .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<3, Fbm, Simplex>::builder(
        tiny_grid.x_iter(),
        tiny_grid.y_iter(),
        tiny_grid.z_iter(),
    )
    .octaves(6)
    .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());
    BatchNoise::<3, Fbm, Cellular<Euclidean>>::builder(
        tiny_grid.x_iter(),
        tiny_grid.y_iter(),
        tiny_grid.z_iter(),
    )
    .octaves(6)
    .fill(tiny_result.as_mut_slice());
    verify_slice(tiny_result.as_slice());

    BatchNoise::<3, Fbm, Perlin>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, Fbm, Value>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, Fbm, Simplex>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, Fbm, Cellular<Euclidean>>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());

    BatchNoise::<3, Fbm, Perlin>::builder(
        large_grid.x_iter(),
        large_grid.y_iter(),
        large_grid.z_iter(),
    )
    .octaves(6)
    .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<3, Fbm, Value>::builder(
        large_grid.x_iter(),
        large_grid.y_iter(),
        large_grid.z_iter(),
    )
    .octaves(6)
    .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<3, Fbm, Simplex>::builder(
        large_grid.x_iter(),
        large_grid.y_iter(),
        large_grid.z_iter(),
    )
    .octaves(6)
    .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());
    BatchNoise::<3, Fbm, Cellular<Euclidean>>::builder(
        large_grid.x_iter(),
        large_grid.y_iter(),
        large_grid.z_iter(),
    )
    .octaves(6)
    .fill(large_result.as_mut_slice());
    verify_slice(large_result.as_slice());

    BatchNoise::<3, Billow, Perlin>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, Ridged, Value>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, PingPong, Simplex>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, Multi, Cellular<Euclidean>>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());
    BatchNoise::<3, HybridMulti, Simplex>::builder(
        medium_grid.x_iter(),
        medium_grid.y_iter(),
        medium_grid.z_iter(),
    )
    .octaves(6)
    .gain(0.8)
    .fill(medium_result.as_mut_slice());
    verify_slice(medium_result.as_slice());

    let values1: [f32; 1234] = std::array::from_fn(|i| i as f32);
    let values2: [f32; 1234] = std::array::from_fn(|i| 2.0 * i as f32);
    let values3: [f32; 1234] = std::array::from_fn(|i| 5.0 * i as f32);

    let noise: [f32; 1234] = BatchNoise::<3, Terrace, Perlin>::builder(
        values1.simd_iter::<A>(),
        values2.simd_iter(),
        values3.simd_iter(),
    )
    .frequency(0.01)
    .scaling(0.5, 2.0, 1.0)
    .steps(10.0)
    .normalization(false)
    .into_iter()
    .map(|x| x + Simd::splat(10.0))
    .collect();
    black_box(&noise);

    let noise: [f32; 1234] = BatchNoise::<3, Multi, Cellular<Euclidean>>::builder(
        values1.simd_iter::<A>(),
        values2.simd_iter(),
        values3.simd_iter(),
    )
    .octaves(10)
    .frequency(0.01)
    .scaling(0.5, 2.0, 1.0)
    .normalization(false)
    .into_iter()
    .map(|x| x + Simd::splat(10.0))
    .collect();
    black_box(&noise);

    let octave_list = [
        Octave::<3>::splat(1.0 / 100.0, 1.0),
        Octave::<3>::splat(1.0 / 30.0, 1.2),
        Octave::<3>::splat(1.0 / 1000.0, 0.8),
        Octave::<3>::new([0.01, 0.005, 0.012], 0.9),
    ];

    let noise: [f32; 1234] = BatchNoise::<3, Terrace, Perlin>::builder_with_octaves(
        &octave_list,
        values1.simd_iter::<A>(),
        values2.simd_iter(),
        values3.simd_iter(),
    )
    .scaling(0.5, 2.0, 1.0)
    .steps(10.0)
    .normalization(false)
    .into_iter()
    .map(|x| x + Simd::splat(10.0))
    .collect();
    black_box(&noise);
}
