use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use noise_functions::{Noise, Simplex};

const GRID_2D: usize = 1024;
const GRID_2D_AREA: usize = GRID_2D * GRID_2D;
const OCTAVES_2D: usize = 3;
const BASE_FREQ_2D: f64 = 1.0 / 128.0;

const GRID_3D: usize = 128;
const GRID_3D_VOLUME: usize = GRID_3D * GRID_3D * GRID_3D;
const OCTAVES_3D: usize = 3;
const BASE_FREQ_3D: f64 = 1.0 / 128.0;

fn simplex_2d_octaves_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_noise_2d_3octaves_1024x1024");
    group.throughput(Throughput::Elements(GRID_2D_AREA as u64));

    // --- quick-noise grid ---
    // {
    //     let grid = Grid::<2>::new(GRID_2D, GRID_2D);
    //     let mut result = vec![0.0; GRID_2D_AREA];
    //     group.bench_function("quick-noise grid", |b| {
    //         b.iter(|| {
    //             grid.builder::<Fbm, Simplex>()
    //                 .octaves(OCTAVES_2D)
    //                 .frequency(BASE_FREQ_2D as f32)
    //                 .fill(result.as_mut_slice());
    //             black_box(&result);
    //         });
    //     });
    // }

    // --- quick-noise batch ---
    {
        use quick_noise::{BatchNoise, Fbm, Grid, Simplex};
        let grid = Grid::<2>::new(GRID_2D, GRID_2D);
        let mut result = vec![0.0; GRID_2D_AREA];
        group.bench_function("quick-noise batch", |b| {
            b.iter(|| {
                BatchNoise::<2, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter())
                    .octaves(OCTAVES_2D)
                    .frequency(BASE_FREQ_2D as f32)
                    .fill(result.as_mut_slice());
                black_box(&result);
            });
        });
    }

    // ---- fastnoise2 ----
    {
        use fastnoise2::generator::prelude::*;
        let node = simplex().fbm(0.5, 0.0, OCTAVES_2D as i32, 2.0).build();

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
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
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
        group.bench_function("simdnoise", |b| {
            b.iter(|| {
                let (result, _min, _max) = simdnoise::NoiseBuilder::fbm_2d(GRID_2D, GRID_2D)
                    .with_freq(BASE_FREQ_2D as f32)
                    .with_octaves(OCTAVES_2D as u8)
                    .with_gain(0.5)
                    .with_lacunarity(2.0)
                    .generate();
                black_box(&result);
            });
        });
    }


    // ---- noise-rs ----
    {
        use noise::{Fbm, MultiFractal, NoiseFn, Simplex};
        let fbm = Fbm::<Simplex>::new(0)
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
                    Octave<BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>>,
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
        let generator = Source::simplex(0).fbm(OCTAVES_2D as u32, BASE_FREQ_2D, 2.0, 0.5);

        let mut result = vec![0.0f64; GRID_2D_AREA];
        group.bench_function("libnoise", |b| {
            b.iter(|| {
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
        let mut result = vec![0.0f32; GRID_2D_AREA];
        group.bench_function("noise-functions", |b| {
            b.iter(|| {
                for y in 0..GRID_2D {
                    for x in 0..GRID_2D {
                        result[y * GRID_2D + x] = Simplex
                            .fbm(OCTAVES_2D as u32, 0.5, 2.0)
                            .sample2([y as f32, x as f32]);
                    }
                }
            });
        });
        black_box(&result);
    }

    group.finish();

    let mut group = c.benchmark_group("simplex_noise_3d_3octaves_128x128x128");
    group.throughput(Throughput::Elements(GRID_3D_VOLUME as u64));

    // --- quick-noise grid ---
    // {
    //     let grid = Grid::<2>::new(GRID_2D, GRID_2D);
    //     let mut result = vec![0.0; GRID_2D_AREA];
    //     group.bench_function("quick-noise grid", |b| {
    //         b.iter(|| {
    //             grid.builder::<Fbm, Simplex>()
    //                 .octaves(OCTAVES_2D)
    //                 .frequency(BASE_FREQ_2D as f32)
    //                 .fill(result.as_mut_slice());
    //             black_box(&result);
    //         });
    //     });
    // }

    // --- quick-noise batch ---
    {
        use quick_noise::{BatchNoise, Fbm, Grid, Simplex};
        let grid = Grid::<3>::new(GRID_3D, GRID_3D, GRID_3D);
        let mut result = vec![0.0; GRID_3D_VOLUME];
        group.bench_function("quick-noise batch", |b| {
            b.iter(|| {
                BatchNoise::<3, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter(), grid.z_iter())
                    .octaves(OCTAVES_3D)
                    .frequency(BASE_FREQ_3D as f32)
                    .fill(result.as_mut_slice());
                black_box(&result);
            });
        });
    }

    // ---- fastnoise2 ----
    {
        use fastnoise2::generator::prelude::*;
        let node = simplex().fbm(0.5, 0.0, OCTAVES_3D as i32, 2.0).build();

        let mut result = vec![0.0f32; GRID_3D_VOLUME];
        group.bench_function("fastnoise2", |b| {
            b.iter(|| {
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
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
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
        group.bench_function("simdnoise", |b| {
            b.iter(|| {
                let (result, _min, _max) =
                    simdnoise::NoiseBuilder::fbm_3d(GRID_3D, GRID_3D, GRID_3D)
                        .with_freq(BASE_FREQ_3D as f32)
                        .with_octaves(OCTAVES_3D as u8)
                        .with_gain(0.5)
                        .with_lacunarity(2.0)
                        .generate();
                black_box(&result);
            });
        });
    }


    // ---- noise-rs ----
    {
        use noise::{Fbm, MultiFractal, NoiseFn, Simplex};
        let fbm = Fbm::<Simplex>::new(0)
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
                    Octave<BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>>,
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
        let generator = Source::simplex(0).fbm(OCTAVES_3D as u32, BASE_FREQ_3D, 2.0, 0.5);

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
        use noise_functions::{Noise, Simplex};
        let mut result = vec![0.0f32; GRID_3D_VOLUME];
        group.bench_function("noise-functions", |b| {
            b.iter(|| {
                for z in 0..GRID_3D {
                    for y in 0..GRID_3D {
                        for x in 0..GRID_3D {
                            result[z * GRID_3D * GRID_3D + y * GRID_3D + x] = Simplex
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

criterion_group!(benches, simplex_2d_octaves_benchmark);
criterion_main!(benches);
