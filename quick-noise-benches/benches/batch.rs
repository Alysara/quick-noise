use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fastnoise2::generator::DistanceFunction;
use fastnoise2::generator::cellular::CellularDistanceReturnType;
use fastnoise2::generator::prelude::*;
use quick_noise::{BatchNoise, Cellular, Fbm, Grid, Perlin, Simplex, Value};
use quick_noise::{Euclidean};
const FREQ: f32 = 1.0 / 32.0;

fn batch_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_noise_2d");
    group.throughput(Throughput::Elements(4096));
    let grid = Grid::<2>::new(64, 64);
    let mut result = [0.0; 4096];
    group.bench_function("perlin", |b| {
        b.iter(|| {
            BatchNoise::<2, Fbm, Perlin>::builder(grid.x_iter(), grid.y_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("value", |b| {
        b.iter(|| {
            BatchNoise::<2, Fbm, Value>::builder(grid.x_iter(), grid.y_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("simplex", |b| {
        b.iter(|| {
            BatchNoise::<2, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("cellular", |b| {
        b.iter(|| {
            BatchNoise::<2, Fbm, Cellular<Euclidean>>::builder(grid.x_iter(), grid.y_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("quick_noise_3d");
    group.throughput(Throughput::Elements(32768));
    let grid = Grid::<3>::new(32, 32, 32);
    let mut result = [0.0; 32768];
    group.bench_function("perlin", |b| {
        b.iter(|| {
            BatchNoise::<3, Fbm, Perlin>::builder(grid.x_iter(), grid.y_iter(), grid.z_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("value", |b| {
        b.iter(|| {
            BatchNoise::<3, Fbm, Value>::builder(grid.x_iter(), grid.y_iter(), grid.z_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("simplex", |b| {
        b.iter(|| {
            BatchNoise::<3, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter(), grid.z_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    group.bench_function("cellular", |b| {
        b.iter(|| {
            BatchNoise::<3, Fbm, Cellular<Euclidean>>::builder(grid.x_iter(), grid.y_iter(), grid.z_iter())
                .octaves(1)
                .fill(result.as_mut_slice())
        });
    });
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("fastnoise2_2d");
    group.throughput(Throughput::Elements(4096));
    let mut result = [0.0; 4096];
    let node = perlin().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("perlin", |b| {
        b.iter(|| {
            node.gen_uniform_grid_2d(&mut result, 0.0, 0.0, 64, 64, FREQ, FREQ, 100);
        });
    });
    let node = value().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("value", |b| {
        b.iter(|| {
            node.gen_uniform_grid_2d(&mut result, 0.0, 0.0, 64, 64, FREQ, FREQ, 100);
        });
    });
    let node = simplex().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("simplex", |b| {
        b.iter(|| {
            node.gen_uniform_grid_2d(&mut result, 0.0, 0.0, 64, 64, FREQ, FREQ, 100);
        });
    });
    let node = cellular_distance(
        1.0,
        DistanceFunction::Euclidean,
        0,
        1,
        CellularDistanceReturnType::Index0,
    )
    .fbm(0.5, 0.0, 1, 2.0)
    .build();
    group.bench_function("cellular", |b| {
        b.iter(|| {
            node.gen_uniform_grid_2d(&mut result, 0.0, 0.0, 64, 64, FREQ, FREQ, 100);
        });
    });
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("fastnoise2_3d");
    group.throughput(Throughput::Elements(32768));
    let mut result = [0.0; 32768];
    let node = perlin().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("perlin", |b| {
        b.iter(|| {
            node.gen_uniform_grid_3d(
                &mut result,
                0.0,
                0.0,
                0.0,
                32,
                32,
                32,
                FREQ,
                FREQ,
                FREQ,
                100,
            );
        });
    });
    let node = value().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("value", |b| {
        b.iter(|| {
            node.gen_uniform_grid_3d(
                &mut result,
                0.0,
                0.0,
                0.0,
                32,
                32,
                32,
                FREQ,
                FREQ,
                FREQ,
                100,
            );
        });
    });
    let node = simplex().fbm(0.5, 0.0, 1, 2.0).build();
    group.bench_function("simplex", |b| {
        b.iter(|| {
            node.gen_uniform_grid_3d(
                &mut result,
                0.0,
                0.0,
                0.0,
                32,
                32,
                32,
                FREQ,
                FREQ,
                FREQ,
                100,
            );
        });
    });
    let node = cellular_distance(
        1.0,
        DistanceFunction::Euclidean,
        0,
        1,
        CellularDistanceReturnType::Index0,
    )
    .fbm(0.5, 0.0, 1, 2.0)
    .build();
    group.bench_function("cellular", |b| {
        b.iter(|| {
            node.gen_uniform_grid_3d(
                &mut result,
                0.0,
                0.0,
                0.0,
                32,
                32,
                32,
                FREQ,
                FREQ,
                FREQ,
                100,
            );
        });
    });
    black_box(&result);
    group.finish();
}

criterion_group!(benches, batch_benchmark);
criterion_main!(benches);
