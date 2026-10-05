use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use quick_noise::{Cellular, Fbm, Grid, Perlin, Simplex, Value};

const SCALES: [f32; 11] = [64.0, 48.0, 32.0, 24.0, 16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0];

fn grid_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_perlin_2d");
    group.throughput(Throughput::Elements(4096));

    let mut result = [0.0; 4096];
    for scale in SCALES {
        let grid = Grid::<2>::new(64, 64);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Perlin>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("grid_perlin_3d");
    group.throughput(Throughput::Elements(32768));

    let mut result = [0.0; 32768];
    for scale in SCALES {
        let grid = Grid::<3>::new(32, 32, 32);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Perlin>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("grid_value_2d");
    group.throughput(Throughput::Elements(4096));

    let mut result = [0.0; 4096];
    for scale in SCALES {
        let grid = Grid::<2>::new(64, 64);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Value>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("grid_value_3d");
    group.throughput(Throughput::Elements(32768));

    let mut result = [0.0; 32768];
    for scale in SCALES {
        let grid = Grid::<3>::new(32, 32, 32);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Value>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("grid_cellular_2d");
    group.throughput(Throughput::Elements(4096));

    let mut result = [0.0; 4096];
    for scale in SCALES {
        let grid = Grid::<2>::new(64, 64);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Cellular>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();

    let mut group = c.benchmark_group("grid_simplex_2d");
    group.throughput(Throughput::Elements(4096));

    let mut result = [0.0; 4096];
    for scale in SCALES {
        let grid = Grid::<2>::new(64, 64);

        group.bench_function(format!("scale: {scale}"), |b| {
            b.iter(|| {
                grid.builder::<Fbm, Simplex>()
                    .octaves(1)
                    .frequency(1.0 / scale)
                    .fill(&mut result);
            });
        });
    }
    black_box(&result);
    group.finish();
}

criterion_group!(benches, grid_benchmark);
criterion_main!(benches);
