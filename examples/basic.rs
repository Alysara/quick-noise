use quick_noise::emit::NoiseImageExt;
use quick_noise::simd::StaticSimd;
use quick_noise::{
    BatchNoise, Cellular, Fbm, Grid, HybridMulti, Octave, Perlin, PingPong, Ridged, Simplex, Value,
};
use quick_noise::Euclidean;

#[cfg(feature = "image")]
fn main() {

    let grid_2d = Grid::<2>::new(2048, 2048);
    let grid_3d = Grid::<3>::new(256, 256, 256);

    // Basic grid generation
    grid_2d
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .into_iter()
        .to_grayscale_image(256, 256, "noise_images/single_pass_perlin.png");

    // Domain warping
    let iter1 = grid_2d
        .builder::<Fbm, Value>()
        .seed(0)
        .octaves(6)
        .frequency(1.0 / 512.0)
        .into_iter();
    let iter2 = grid_2d
        .builder::<Fbm, Perlin>()
        .seed(1)
        .octaves(6)
        .frequency(1.0 / 512.0)
        .into_iter();

    grid_2d
        .warp_builder::<HybridMulti, Simplex>(1000.0, iter1, iter2)
        .octaves(6)
        .frequency(1.0 / 512.0)
        .into_iter()
        .map(|x| StaticSimd::splat(0.25) * x - StaticSimd::splat(1.0))
        .to_grayscale_image(2048, 2048, "noise_images/warped.png");

    // Cellular batch noise.
    BatchNoise::<2, Ridged, Cellular<Euclidean>>::builder(grid_2d.x_iter(), grid_2d.y_iter())
        .octaves(6)
        .frequency(1.0 / 512.0)
        .gain(1.5)
        .into_iter()
        .map(|x| StaticSimd::splat(0.4) * x - StaticSimd::splat(1.0))
        .to_grayscale_image(2048, 2048, "noise_images/ridged_cellular.png");

    // 3D grid + Custom octaves
    let octave_list = [
        Octave::<3>::new([0.01, 0.02, 0.01], 1.0),
        Octave::<3>::new([0.02, 0.01, 0.02], 1.0),
        Octave::<3>::new([0.03, 0.01, 0.01], 1.0),
        Octave::<3>::new([0.005, 0.01, 0.005], 0.3),
        Octave::<3>::splat(0.002, 0.2),
    ];

    grid_3d
        .builder_with_octaves::<PingPong, Value>(octave_list.as_slice())
        .into_iter()
        .to_grayscale_image(256, 256, "noise_images/custom_value.png");
}
