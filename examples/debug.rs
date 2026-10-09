use quick_noise::emit::NoiseImageExt;
use quick_noise::{
    BatchNoise, Billow, Combiner, Fbm, Grid, Octave, Perlin, PingPong, PingPongConfig, Value,
};
use simply_simd::{ScalarArch, Simd, StaticArch};
#[cfg(feature = "image")]
fn main() {
    let debug_grid = Grid::<2>::new(1024, 1024).seed(0);

    debug_grid
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .frequency(1.0 / 64.0)
        .seed(0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "noise_images/perlin_debug_2d.png");

    BatchNoise::<2, Fbm, Perlin>::builder(debug_grid.x_iter(), debug_grid.y_iter())
        .octaves(1)
        .frequency(1.0 / 64.0)
        .seed_with_grid(0, 0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "noise_images/perlin_debug_2d_truth.png");

    // let mut result = vec![0.0; 243 * 243];
    // BatchNoise::<2, Fbm, Perlin>::builder(debug_grid.x_iter(), debug_grid.y_iter())
    //     .octaves(1)
    //     .frequency(1.0 / 32.0)
    //     .fill(result.as_mut_slice());
    //
    // std::hint::black_box(&result);

    // let cfg = PingPongConfig { strength: 2.0 };
    // for v in [-0.69f32, -0.5, 0.0, 0.5, 0.69] {
    //     let out = PingPong::finalize_sample::<StaticArch>(&cfg, Default::default(), Simd::splat(v));
    //     eprintln!("{v} -> {}", out.to_array()[0]);
    // }
    // let octave_list = [
    //     Octave::<2>::splat(0.002, 0.2),
    //     Octave::<2>::splat(0.004, 0.4),
    //     Octave::<2>::splat(0.008, 0.8),
    // ];
    //
    // debug_grid
    //     .builder_with_octaves::<Fbm, Perlin>(octave_list.as_slice())
    //     .into_iter()
    //     .to_grayscale_image(243, 243, "noise_images/perlin_debug_2d.png");

    // let debug_grid_3d = Grid::<3>::new(32, 32, 32);
    //
    // let mut result = [0.0; 32768];
    //
    // for _ in 0..1000000 {
    //     debug_grid_3d
    //         .builder::<Fbm, Perlin>()
    //         .octaves(1)
    //         .frequency(1.0 / 32.0)
    //         .fill(result.as_mut_slice())
    // }
    //
    // std::hint::black_box(&result);

    // .into_iter()
    // .to_grayscale_image(256, 256 * 10, "noise_images/single_pass_perlin_3d.png");
}
