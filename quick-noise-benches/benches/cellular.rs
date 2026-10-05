use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use quick_noise::{Cellular, Fbm, Grid};

const GRID_2D: usize = 1024;
const GRID_2D_AREA: usize = GRID_2D * GRID_2D;
const OCTAVES_2D: usize = 3;
const BASE_FREQ_2D: f64 = 1.0 / 128.0;

const GRID_3D: usize = 128;
const GRID_3D_VOLUME: usize = GRID_3D * GRID_3D * GRID_3D;
const OCTAVES_3D: usize = 3;
const BASE_FREQ_3D: f64 = 1.0 / 128.0;

fn cellular_2d_octaves_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cellular_noise_2d_3octaves_1024x1024");
    group.throughput(Throughput::Elements(GRID_2D_AREA as u64));

    // --- quick-noise grid ---
    {
        let grid = Grid::<2>::new(GRID_2D, GRID_2D);
        let mut result = vec![0.0; GRID_2D_AREA];
        group.bench_function("quick-noise grid", |b| {
            b.iter(|| {
                grid.builder::<Fbm, Cellular>()
                    .octaves(OCTAVES_2D)
                    .frequency(BASE_FREQ_2D as f32)
                    .fill(result.as_mut_slice());
                black_box(&result);
            });
        });
    }

    // --- quick-noise batch ---
    {
        use quick_noise::{BatchNoise, Cellular, Fbm, Grid};
        let grid = Grid::<2>::new(GRID_2D, GRID_2D);
        let mut result = vec![0.0; GRID_2D_AREA];
        group.bench_function("quick-noise batch", |b| {
            b.iter(|| {
                BatchNoise::<2, Fbm, Cellular>::builder(grid.x_iter(), grid.y_iter())
                    .octaves(OCTAVES_2D)
                    .frequency(BASE_FREQ_2D as f32)
                    .fill(result.as_mut_slice());
                black_box(&result);
            });
        });
    }

    // ---- fastnoise2 ----
    {
        use fastnoise2::generator::DistanceFunction;
        use fastnoise2::generator::cellular::CellularDistanceReturnType;
        use fastnoise2::generator::prelude::*;
        let node = cellular_distance(
            1.0,
            DistanceFunction::Euclidean,
            0,
            1,
            CellularDistanceReturnType::Index0,
        )
        .fbm(0.5, 0.0, OCTAVES_2D as i32, 2.0)
        .build();

        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("fastnoise2", |b| {
            b.iter(|| {
                node.gen_uniform_grid_2d(
                    &mut result,
                    0.0,
                    0.0,
                    GRID_2D as i32,
                    GRID_2D as i32,
                    BASE_FREQ_2D as f32,
                    BASE_FREQ_2D as f32,
                    1337,
                );
                black_box(&result);
            });
        });
    }

    // ---- fastnoise-lite ----
    {
        use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
        let mut noise = FastNoiseLite::with_seed(10);
        noise.set_noise_type(Some(NoiseType::Cellular));
        noise.set_frequency(Some(BASE_FREQ_2D as f32));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_octaves(Some(OCTAVES_2D as i32));
        noise.set_fractal_lacunarity(Some(2.0));
        noise.set_fractal_gain(Some(0.5));

        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("fastnoise-lite", |b| {
            b.iter(|| {
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        result[y * GRID_2D + x] = noise.get_noise_2d(x as f32, y as f32);
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- simdnoise ----
    {
        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("simdnoise", |b| {
            let mut amplitude = 1.0f32;
            let mut frequency = BASE_FREQ_2D as f32;

            b.iter(|| {
                for _ in 0..OCTAVES_2D {
                    let (octave, _, _) = simdnoise::NoiseBuilder::cellular_2d(GRID_2D, GRID_2D)
                        .with_freq(frequency)
                        .generate();

                    for (dst, src) in result.iter_mut().zip(octave.iter()) {
                        *dst += amplitude * *src;
                    }

                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
            });
        });
        black_box(&result);
    }

    // ---- noise-rs ----
    {
        use noise::{Fbm, MultiFractal, NoiseFn, Worley};
        let fbm = Fbm::<Worley>::new(0)
            .set_octaves(OCTAVES_2D)
            .set_frequency(BASE_FREQ_2D)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        let mut result = vec![0.0f64; GRID_2D_AREA];
        group.bench_function("noise-rs", |b| {
            b.iter(|| {
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        result[y * GRID_2D + x] = fbm.get([x as f64, y as f64]);
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- noiz ----
    {
        use bevy_math::prelude::*;
        use noiz::prelude::*;

        let noise = Noise::<
            LayeredNoise<
                Normed<f32>,
                Persistence,
                FractalLayers<
                    Octave<PerCellPointDistances<Voronoi, EuclideanLength, WorleyLeastDistance>>,
                >,
            >,
        >::from(LayeredNoise::new(
            Normed::default(),
            Persistence(0.5),
            FractalLayers {
                layer: Default::default(),
                lacunarity: 2.0,
                amount: OCTAVES_2D as u32,
            },
        ));

        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("noiz", |b| {
            b.iter(|| {
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        let p = Vec2::new(
                            x as f32 * BASE_FREQ_2D as f32,
                            y as f32 * BASE_FREQ_2D as f32,
                        );
                        result[y * GRID_2D + x] = noise.sample(p);
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- libnoise ----
    {
        use libnoise::prelude::*;
        let generator = Source::worley(0).fbm(OCTAVES_2D as u32, BASE_FREQ_2D, 2.0, 0.5);

        group.bench_function("libnoise", |b| {
            b.iter(|| {
                let mut result = vec![0.0f64; GRID_2D_AREA];
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        result[y * GRID_2D + x] = generator.sample([x as f64, y as f64]);
                    }
                }
                black_box(&result);
            });
        });
    }

    // --- noise functions ---
    {
        use noise_functions::{CellDistance, Noise};
        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("noise-functions", |b| {
            b.iter(|| {
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        result[y * GRID_2D + x] = CellDistance::default()
                            .fbm(OCTAVES_2D as u32, 0.5, 2.0)
                            .sample2([y as f32, x as f32]);
                    }
                }
            });
        });
        black_box(&result);
    }

    group.finish();

    let mut group = c.benchmark_group("cellular_noise_3d_3octaves_128x128x128");
    group.throughput(Throughput::Elements(GRID_3D_VOLUME as u64));

    // --- quick-noise grid ---
    // {
    //     let grid = Grid::<2>::new(GRID_2D, GRID_2D);
    //     let mut result = vec![0.0; GRID_2D_AREA];
    //     group.bench_function("quick-noise grid", |b| {
    //         b.iter(|| {
    //             grid.builder::<Fbm, Cellular>()
    //                 .octaves(OCTAVES_2D)
    //                 .frequency(BASE_FREQ_2D as f32)
    //                 .fill(result.as_mut_slice());
    //             black_box(&result);
    //         });
    //     });
    // }

    // --- quick-noise batch ---
    {
        use quick_noise::{BatchNoise, Cellular, Fbm, Grid};
        let grid = Grid::<3>::new(GRID_3D, GRID_3D, GRID_3D);
        let mut result = vec![0.0; GRID_3D_VOLUME];
        group.bench_function("quick-noise batch", |b| {
            b.iter(|| {
                BatchNoise::<3, Fbm, Cellular>::builder(
                    grid.x_iter(),
                    grid.y_iter(),
                    grid.z_iter(),
                )
                .octaves(OCTAVES_3D)
                .frequency(BASE_FREQ_3D as f32)
                .fill(result.as_mut_slice());
                black_box(&result);
            });
        });
    }

    // ---- fastnoise2 ----
    {
        use fastnoise2::generator::DistanceFunction;
        use fastnoise2::generator::cellular::CellularDistanceReturnType;
        use fastnoise2::generator::prelude::*;
        let node = cellular_distance(
            1.0,
            DistanceFunction::Euclidean,
            0,
            1,
            CellularDistanceReturnType::Index0,
        )
        .fbm(0.5, 0.0, OCTAVES_2D as i32, 2.0)
        .build();

        group.bench_function("fastnoise2", |b| {
            b.iter(|| {
                let mut result = vec![0.0f32; GRID_3D_VOLUME];
                node.gen_uniform_grid_3d(
                    &mut result,
                    0.0,
                    0.0,
                    0.0,
                    GRID_3D as i32,
                    GRID_3D as i32,
                    GRID_3D as i32,
                    BASE_FREQ_3D as f32,
                    BASE_FREQ_3D as f32,
                    BASE_FREQ_3D as f32,
                    1337,
                );
                black_box(&result);
            });
        });
    }

    // ---- fastnoise-lite ----
    {
        use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
        let mut noise = FastNoiseLite::with_seed(10);
        noise.set_noise_type(Some(NoiseType::Cellular));
        noise.set_frequency(Some(BASE_FREQ_3D as f32));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_octaves(Some(OCTAVES_3D as i32));
        noise.set_fractal_lacunarity(Some(2.0));
        noise.set_fractal_gain(Some(0.5));

        let mut result = vec![0.0f32; GRID_3D_VOLUME];
        group.bench_function("fastnoise-lite", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] =
                                noise.get_noise_3d(x as f32, y as f32, z as f32);
                        }
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- simdnoise ----
    {
        let mut result = vec![0.0f32; GRID_3D_VOLUME];

        group.bench_function("simdnoise", |b| {
            b.iter(|| {
                let mut amplitude = 1.0f32;
                let mut frequency = BASE_FREQ_3D as f32;

                for _ in 0..OCTAVES_3D {
                    let (octave, _, _) =
                        simdnoise::NoiseBuilder::cellular_3d(GRID_3D, GRID_3D, GRID_3D)
                            .with_freq(frequency)
                            .generate();

                    for (dst, src) in result.iter_mut().zip(octave.iter()) {
                        *dst += amplitude * *src;
                    }

                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
            });
        });
        black_box(&result);
    }

    // ---- noise-rs ----
    {
        use noise::{Fbm, MultiFractal, NoiseFn, Worley};
        let fbm = Fbm::<Worley>::new(0)
            .set_octaves(OCTAVES_3D)
            .set_frequency(BASE_FREQ_3D)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        let mut result = vec![0.0f64; GRID_3D_VOLUME];
        group.bench_function("noise-rs", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] =
                                fbm.get([x as f64, y as f64, z as f64]);
                        }
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- noiz ----
    {
        use bevy_math::prelude::*;
        use noiz::prelude::*;

        let noise = Noise::<
            LayeredNoise<
                Normed<f32>,
                Persistence,
                FractalLayers<
                    Octave<PerCellPointDistances<Voronoi, EuclideanLength, WorleyLeastDistance>>,
                >,
            >,
        >::from(LayeredNoise::new(
            Normed::default(),
            Persistence(0.5),
            FractalLayers {
                layer: Default::default(),
                lacunarity: 2.0,
                amount: OCTAVES_3D as u32,
            },
        ));

        let mut result = vec![0.0f32; GRID_3D_VOLUME];
        group.bench_function("noiz", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            let p = Vec3::new(
                                x as f32 * BASE_FREQ_3D as f32,
                                y as f32 * BASE_FREQ_3D as f32,
                                z as f32 * BASE_FREQ_3D as f32,
                            );
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] = noise.sample(p);
                        }
                    }
                }
                black_box(&result);
            });
        });
    }

    // ---- libnoise ----
    {
        use libnoise::prelude::*;
        let generator = Source::worley(0).fbm(OCTAVES_3D as u32, BASE_FREQ_3D, 2.0, 0.5);

        let mut result = vec![0.0f64; GRID_3D_VOLUME];
        group.bench_function("libnoise", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] =
                                generator.sample([x as f64, y as f64, z as f64]);
                        }
                    }
                }
                black_box(&result);
            });
        });
    }

    // --- noise functions ---
    {
        use noise_functions::{CellDistance, Noise};
        let mut result = vec![0.0f32; GRID_3D_VOLUME];
        group.bench_function("noise-functions", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] =
                                CellDistance::default()
                                    .fbm(OCTAVES_3D as u32, 0.5, 2.0)
                                    .sample3([z as f32, y as f32, x as f32]);
                        }
                    }
                }
            });
        });
        black_box(&result);
    }

    group.finish();
}

criterion_group!(benches, cellular_2d_octaves_benchmark);
criterion_main!(benches);
