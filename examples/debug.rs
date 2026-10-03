use quick_noise::emit::NoiseImageExt;
use quick_noise::{Fbm, Grid, Value};

#[cfg(feature = "image")]
fn main() {
    use quick_noise::Perlin;

    let debug_grid = Grid::<2>::new(243, 243);
    debug_grid
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .frequency(1.0 / 16.0)
        .into_iter()
        .to_grayscale_image(243, 243, "noise_images/single_pass_perlin.png");

    let debug_grid_3d = Grid::<3>::new(256, 256, 256);

    debug_grid_3d
        .builder::<Fbm, Perlin>()
        .octaves(1)
        .frequency(1.0 / 16.0)
        .into_iter()
        .to_grayscale_image(256, 256 * 10, "noise_images/single_pass_perlin_3d.png");
}
