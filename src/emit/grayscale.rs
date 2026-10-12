#[cfg(feature = "image")]
use std::fs;
use std::path::Path;

use crate::simd::{Arch, Simd};

// TODO: Add error handling here.
pub trait NoiseImageExt<A: Arch>: Iterator<Item = Simd<f32, A>> + Sized {
    // /// Writes the values from the iterator into a png file with a path you specify.
    // ///
    // /// # Parameters
    // /// - `x`: Width of the image.
    // /// - `y`: Height of the image.
    // /// - `path`: The file location where the image gets stored.
    // fn to_grayscale_image_raw(mut self, x: usize, y: usize, path: impl AsRef<Path>) {
    //     if let Some(parent) = path.as_ref().parent()
    //         && !parent.exists()
    //     {
    //         fs::create_dir_all(parent).expect("Failed to create parent");
    //     }
    //
    //     let size = x * y;
    //     let mut pixels = vec![0; size];
    //
    //     for i in (0..size).step_by(Simd::<f32, A>::LANES) {
    //         let cur = self
    //             .next()
    //             .expect("Given iterator did not fit image dimensions!");
    //
    //         let adj = (cur + Simd::splat(1.0)) * Simd::splat(127.5);
    //
    //         // TODO: Equip simd to do this in register
    //         let slice = adj.to_array();
    //         let upper_index = min(size, i + Simd::<f32, A>::LANES);
    //         let slice_bound = upper_index - i;
    //         for m in 0..slice_bound {
    //             pixels[i + m] = slice[m] as u8;
    //         }
    //     }
    //
    //     image::save_buffer(&path, &pixels, x as u32, y as u32, image::ColorType::L8)
    //         .expect("Failed to write height map!");
    // }

    /// Writes the values from the iterator into a png file with a path you specify.
    /// Automatically normalizes the result.
    ///
    /// # Parameters
    /// - `x`: Width of the image.
    /// - `y`: Height of the image.
    /// - `path`: The file location where the image gets stored.
    fn to_grayscale_image(self, x: usize, y: usize, path: impl AsRef<Path>) {
        if let Some(parent) = path.as_ref().parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent).expect("Failed to create parent");
        }

        let size = x * y;
        let mut pixels = vec![0; size];

        let result: Vec<f32> = self.collect();

        if result.len() < size {
            panic!("Given iterator did not fit image dimensions!");
        }

        let (min, max) = result
            .iter()
            .take(size)
            .copied()
            .fold((f32::MAX, f32::MIN), |(min, max), x| {
                (min.min(x), max.max(x))
            });

        let dif = max - min;
        let coef = 255.0 / dif;

        let offset = -min * coef;

        for (pixel, val) in pixels.iter_mut().zip(result.iter()) {
            *pixel = val.mul_add(coef, offset) as u8;
        }

        image::save_buffer(&path, &pixels, x as u32, y as u32, image::ColorType::L8)
            .expect("Failed to write height map!");
    }
}

impl<A: Arch, I> NoiseImageExt<A> for I where I: Iterator<Item = Simd<f32, A>> + Sized {}
