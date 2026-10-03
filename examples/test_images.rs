use quick_noise::emit::NoiseImageExt;
use quick_noise::simd::{Simd, dispatch_simd};
use quick_noise::*;

#[cfg(feature = "image")]
fn main() {
    create_test_images_grid_2d();
    create_test_images_grid_3d();
    create_test_images_batch_2d();
    create_test_images_batch_3d();
}

#[dispatch_simd(A)]
fn create_test_images_grid_2d() {
    let tiny_grid_2d_1 = Grid::<2, A>::new(1, 1);
    let tiny_grid_2d_2 = Grid::<2, A>::new(7, 7);
    let tiny_grid_2d_3 = Grid::<2, A>::new(4, 3);

    tiny_grid_2d_1
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1, 1, "test_images/tiny_grid_2d_1_perlin.png");

    tiny_grid_2d_2
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_grid_2d_2_perlin.png");

    tiny_grid_2d_3
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(4, 3, "test_images/tiny_grid_2d_3_perlin.png");

    tiny_grid_2d_1
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1, 1, "test_images/tiny_grid_2d_1_value.png");

    tiny_grid_2d_2
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_grid_2d_2_value.png");

    tiny_grid_2d_3
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(4, 3, "test_images/tiny_grid_2d_3_value.png");

    tiny_grid_2d_1
        .builder::<Fbm, Cellular>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1, 1, "test_images/tiny_grid_2d_1_cellular.png");

    tiny_grid_2d_2
        .builder::<Fbm, Cellular>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_grid_2d_2_cellular.png");

    tiny_grid_2d_3
        .builder::<Fbm, Cellular>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(4, 3, "test_images/tiny_grid_2d_3_cellular.png");

    BatchNoise::<2, Fbm, Perlin>::builder(tiny_grid_2d_2.x_iter(), tiny_grid_2d_2.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_batch_perlin.png");

    let grid_2d = Grid::<2, A>::new(1000, 1000).sample_position(-500, -500);
    let grid_2d_tiled = Grid::<2, A>::new(1024, 1024).tiling(Some(128), Some(256));

    grid_2d
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_single_pass_perlin.png");

    grid_2d
        .builder::<Fbm, Value>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_single_pass_value.png");

    grid_2d
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_perlin.png");

    grid_2d
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_value.png");

    grid_2d
        .builder::<Ridged, Perlin>()
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_ridged_perlin.png");

    grid_2d
        .builder::<HybridMulti, Value>()
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/grid_2d_hybrid_multi_value.png");

    grid_2d_tiled
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_2d_perlin_tiled.png");

    grid_2d_tiled
        .builder::<Fbm, Value>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_2d_value_tiled.png");

    grid_2d_tiled
        .builder::<Billow, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_2d_billow_perlin_tiled.png");

    grid_2d_tiled
        .builder::<PingPong, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_2d_ping_pong_perlin_tiled.png");

    grid_2d_tiled
        .builder::<HybridMulti, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(
            1024,
            1024,
            "test_images/grid_2d_hybrid_multi_perlin_tiled.png",
        );

    let grid_2d_long = Grid::<2, A>::new(1024, 2048);

    grid_2d_long
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .frequency(1.0 / 512.0)
        .into_iter()
        .to_grayscale_image(1024, 2048, "test_images/grid_2d_long_perlin.png");

    // Cellular Grid Test
    let grid = Grid::<2>::new(256, 256).seed(42).sample_position(-128, -128);

    grid.builder::<Fbm, Cellular>()
        .frequency(1.0 / 32.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_cellular_seeded.png");

    grid.builder::<Fbm, Cellular>()
        .octaves(2)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_fbm_cellular_seeded.png");

    grid.builder::<PingPong, Cellular>()
        .octaves(2)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_ping_pong_cellular_seeded.png");

    grid.builder::<Ridged, Cellular>()
        .octaves(2)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_ridged_cellular_seeded.png");

    grid.builder::<Billow, Cellular>()
        .octaves(2)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_billow_cellular_seeded.png");

    grid.builder::<HybridMulti, Cellular>()
        .octaves(1)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_hybrid_multi_cellular_seeded.png");

    grid.builder::<Multi, Cellular>()
        .octaves(1)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_multi_cellular_seeded.png");

    grid.builder::<Terrace, Cellular>()
        .octaves(1)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/grid_2d_terrace_cellular_seeded.png");

    let grid_2d_tiled = Grid::<2>::new(1024, 1024).tiling(Some(128), Some(256));

    grid_2d_tiled
        .builder::<Fbm, Cellular>()
        .octaves(4)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_2d_cellular_tiled.png");
}

#[dispatch_simd(A)]
fn create_test_images_grid_3d() {
    let tiny_grid_3d_1 = Grid::<3, A>::new(1, 1, 1);
    let tiny_grid_3d_2 = Grid::<3, A>::new(7, 7, 7);
    let tiny_grid_3d_3 = Grid::<3, A>::new(4, 3, 2);
    tiny_grid_3d_1
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1, 1, "test_images/tiny_grid_3d_1_perlin.png");
    tiny_grid_3d_2
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_grid_3d_2_perlin.png");
    tiny_grid_3d_3
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(4, 3, "test_images/tiny_grid_3d_3_perlin.png");
    tiny_grid_3d_1
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1, 1, "test_images/tiny_grid_3d_1_value.png");
    tiny_grid_3d_2
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(7, 7, "test_images/tiny_grid_3d_2_value.png");
    tiny_grid_3d_3
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(4, 3, "test_images/tiny_grid_3d_3_value.png");
    BatchNoise::<3, Fbm, Perlin>::builder(
        tiny_grid_3d_2.x_iter(),
        tiny_grid_3d_2.y_iter(),
        tiny_grid_3d_2.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(7, 7, "test_images/tiny_batch_3d_perlin.png");
    let grid_3d = Grid::<3, A>::new(1000, 1000, 1).sample_position(-500, -500, 0);
    let grid_3d_tiled = Grid::<3, A>::new(1024, 1024, 1).tiling(Some(128), Some(256), None);
    grid_3d
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_single_pass_perlin.png");
    grid_3d
        .builder::<Fbm, Value>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_single_pass_value.png");
    grid_3d
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_perlin.png");
    grid_3d
        .builder::<Fbm, Value>()
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_value.png");
    grid_3d
        .builder::<Ridged, Perlin>()
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_ridged_perlin.png");
    grid_3d
        .builder::<HybridMulti, Value>()
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/grid_3d_hybrid_multi_value.png");
    grid_3d_tiled
        .builder::<Fbm, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_3d_perlin_tiled.png");
    grid_3d_tiled
        .builder::<Fbm, Value>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_3d_value_tiled.png");
    grid_3d_tiled
        .builder::<Billow, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_3d_billow_perlin_tiled.png");
    grid_3d_tiled
        .builder::<PingPong, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "test_images/grid_3d_ping_pong_perlin_tiled.png");
    grid_3d_tiled
        .builder::<HybridMulti, Perlin>()
        .octaves(6)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(
            1024,
            1024,
            "test_images/grid_3d_hybrid_multi_perlin_tiled.png",
        );

    let grid_3d_full = Grid::<3, A>::new(1024, 1024, 32).grid_position(-50, 100, 123);
    grid_3d_full
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1024, 1024 * 32, "test_images/grid_3d_perlin_full.png");
}

#[dispatch_simd(A)]
fn create_test_images_batch_2d() {
    let grid_2d = Grid::<2, A>::new(1000, 1000);

    BatchNoise::<2, Fbm, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(1)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_single_pass_perlin.png",
        );
    BatchNoise::<2, Fbm, Value>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(1)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_single_pass_value.png",
        );
    BatchNoise::<2, Fbm, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_perlin.png");
    BatchNoise::<2, Fbm, Value>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_value.png");
    BatchNoise::<2, Fbm, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_simplex.png");
    BatchNoise::<2, Fbm, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_cellular.png");

    BatchNoise::<2, Ridged, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_ridged_perlin.png");
    BatchNoise::<2, Ridged, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_ridged_simplex.png");
    BatchNoise::<2, Ridged, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_ridged_cellular.png");

    BatchNoise::<2, HybridMulti, Value>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_hybrid_multi_value.png",
        );
    BatchNoise::<2, HybridMulti, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_hybrid_multi_simplex.png",
        );
    BatchNoise::<2, HybridMulti, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_hybrid_multi_cellular.png",
        );

    BatchNoise::<2, Billow, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_billow_perlin.png");
    BatchNoise::<2, Billow, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_billow_simplex.png");
    BatchNoise::<2, Billow, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_billow_cellular.png");

    BatchNoise::<2, PingPong, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_ping_pong_perlin.png");
    BatchNoise::<2, PingPong, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_ping_pong_simplex.png",
        );
    BatchNoise::<2, PingPong, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_2d_ping_pong_cellular.png",
        );

    BatchNoise::<2, Multi, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_multi_perlin.png");
    BatchNoise::<2, Multi, Value>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_multi_value.png");
    BatchNoise::<2, Multi, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_multi_simplex.png");
    BatchNoise::<2, Multi, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_multi_cellular.png");

    BatchNoise::<2, Terrace, Perlin>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_terrace_perlin.png");
    BatchNoise::<2, Terrace, Value>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_terrace_value.png");
    BatchNoise::<2, Terrace, Simplex>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_terrace_simplex.png");
    BatchNoise::<2, Terrace, Cellular>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_2d_terrace_cellular.png");
    
    // Cellular Batch Test
    let grid = Grid::<2>::new(256, 256).seed(42).sample_position(-128, -128);

    BatchNoise::<2, Fbm, Cellular>::builder(grid.x_iter(), grid.y_iter())
        .seed_with_grid(42, 42)
        .frequency(1.0 / 32.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/batch_2d_cellular_seeded.png");

    BatchNoise::<2, Fbm, Cellular>::builder(grid.x_iter(), grid.y_iter())
        .seed_with_grid(42, 42)
        .octaves(2)
        .frequency(1.0 / 64.0)
        .into_iter()
        .map(|x| x * Simd::splat(1.4) - Simd::splat(1.0))
        .to_grayscale_image(256, 256, "test_images/batch_2d_fbm_cellular_seeded.png");
}

#[dispatch_simd(A)]
fn create_test_images_batch_3d() {
    let grid_3d = Grid::<3, A>::new(1000, 1000, 1);

    BatchNoise::<3, Fbm, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(1)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_3d_single_pass_perlin.png",
        );
    BatchNoise::<3, Fbm, Value>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(1)
        .into_iter()
        .to_grayscale_image(
            1000,
            1000,
            "test_images/batch_grid_3d_single_pass_value.png",
        );
    BatchNoise::<3, Fbm, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_perlin.png");
    BatchNoise::<3, Fbm, Value>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_value.png");
    BatchNoise::<3, Fbm, Simplex>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(1)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_simplex.png");
    BatchNoise::<3, Fbm, Cellular>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_cellular.png");

    BatchNoise::<3, Ridged, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_ridged_perlin.png");
    BatchNoise::<3, Ridged, Simplex>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_ridged_simplex.png");
    BatchNoise::<3, Ridged, Cellular>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
    .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_ridged_cellular.png");

    BatchNoise::<3, HybridMulti, Value>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
    .to_grayscale_image(
        1000,
        1000,
        "test_images/batch_grid_3d_hybrid_multi_value.png",
    );
    BatchNoise::<3, HybridMulti, Simplex>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
    .to_grayscale_image(
        1000,
        1000,
        "test_images/batch_grid_3d_hybrid_multi_simplex.png",
    );
    BatchNoise::<3, HybridMulti, Cellular>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .map(|x| x * Simd::splat(0.25) - Simd::splat(1.0))
    .to_grayscale_image(
        1000,
        1000,
        "test_images/batch_grid_3d_hybrid_multi_cellular.png",
    );

    BatchNoise::<3, Billow, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_billow_perlin.png");
    BatchNoise::<3, Billow, Simplex>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_billow_simplex.png");
    BatchNoise::<3, Billow, Cellular>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_billow_cellular.png");

    BatchNoise::<3, PingPong, Perlin>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_ping_pong_perlin.png");
    BatchNoise::<3, PingPong, Simplex>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(
        1000,
        1000,
        "test_images/batch_grid_3d_ping_pong_simplex.png",
    );
    BatchNoise::<3, PingPong, Cellular>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(
        1000,
        1000,
        "test_images/batch_grid_3d_ping_pong_cellular.png",
    );

    BatchNoise::<3, Multi, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_multi_perlin.png");
    BatchNoise::<3, Multi, Value>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_multi_value.png");
    BatchNoise::<3, Multi, Simplex>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_multi_simplex.png");
    BatchNoise::<3, Multi, Cellular>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .map(|x| x * Simd::splat(0.5) - Simd::splat(1.0))
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_multi_cellular.png");

    BatchNoise::<3, Terrace, Perlin>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_terrace_perlin.png");
    BatchNoise::<3, Terrace, Value>::builder(grid_3d.x_iter(), grid_3d.y_iter(), grid_3d.z_iter())
        .octaves(6)
        .into_iter()
        .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_terrace_value.png");
    BatchNoise::<3, Terrace, Simplex>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_terrace_simplex.png");
    BatchNoise::<3, Terrace, Cellular>::builder(
        grid_3d.x_iter(),
        grid_3d.y_iter(),
        grid_3d.z_iter(),
    )
    .octaves(6)
    .into_iter()
    .to_grayscale_image(1000, 1000, "test_images/batch_grid_3d_terrace_cellular.png");
}
