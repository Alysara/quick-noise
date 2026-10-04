use quick_noise::Grid;

fn is_approx_eq(left: f32, right: f32, epsilon: f32) -> bool {
    let dif = left - right;
    dif.abs() <= epsilon
}

fn validate_iter_output(slice: &[f32], start: f32, stride: usize, rows: usize, slices: usize) {
    let mut matches = true;
    let mut vec = Vec::new();
    let mut i = 0;
    for _ in 0..slices {
        let mut cur = start;
        for _ in 0..rows {
            for _ in 0..stride {
                matches &= is_approx_eq(cur, slice[i], 0.001);
                vec.push(cur);
                i += 1;
            }
            cur += 1.0;
        }
    }

    if !matches {
        panic!(
            "Iter does not match!\nReceived: {:?}\n\nExpected: {:?}",
            slice, vec
        );
    }
}

macro_rules! validate_2d_iter {
    ($x:expr, $y:expr) => {
        paste::paste! {
            #[test]
            fn [<grid_ $x x $y>]() {
                let grid = Grid::<2>::new($x, $y);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                validate_iter_output(x.as_slice(), 0.0, 1, $x, $y);
                validate_iter_output(y.as_slice(), 0.0, $x, $y, 1);

                let grid = grid.grid_position(12.0, 34.0);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                validate_iter_output(x.as_slice(), 12.0 * $x as f32, 1, $x, $y);
                validate_iter_output(y.as_slice(), 34.0 * $y as f32, $x, $y, 1);

                let grid = grid.sample_position(12.0, 34.0);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                validate_iter_output(x.as_slice(), 12.0, 1, $x, $y);
                validate_iter_output(y.as_slice(), 34.0, $x, $y, 1);
            }
        }
    };
}

macro_rules! validate_3d_iter {
    ($x:expr, $y:expr, $z:expr) => {
        paste::paste! {
            #[test]
            fn [<grid_ $x x $y x $z>]() {
                let grid = Grid::<3>::new($x, $y, $z);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                let z: Vec<f32> = grid.z_iter().collect();
                validate_iter_output(x.as_slice(), 0.0, 1, $x, $y * $z);
                validate_iter_output(y.as_slice(), 0.0, $x, $y, $z);
                validate_iter_output(z.as_slice(), 0.0, $x * $y, $z, 1);

                let grid = grid.grid_position(12.0, 34.0, -77.0);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                let z: Vec<f32> = grid.z_iter().collect();
                validate_iter_output(x.as_slice(), 12.0 * $x as f32, 1, $x, $y * $z);
                validate_iter_output(y.as_slice(), 34.0 * $y as f32, $x, $y, $z);
                validate_iter_output(z.as_slice(), -77.0 * $z as f32, $x * $y, $z, 1);

                let grid = grid.sample_position(12.0, 34.0, -77.0);
                let x: Vec<f32> = grid.x_iter().collect();
                let y: Vec<f32> = grid.y_iter().collect();
                let z: Vec<f32> = grid.z_iter().collect();
                validate_iter_output(x.as_slice(), 12.0, 1, $x, $y * $z);
                validate_iter_output(y.as_slice(), 34.0, $x, $y, $z);
                validate_iter_output(z.as_slice(), -77.0, $x * $y, $z, 1);
            }
        }
    };
}

validate_2d_iter!(1, 1);
validate_2d_iter!(1, 2);
validate_2d_iter!(2, 3);
validate_2d_iter!(1, 7);
validate_2d_iter!(7, 1);
validate_2d_iter!(3, 5);
validate_2d_iter!(4, 6);
validate_2d_iter!(7, 7);
validate_2d_iter!(15, 18);
validate_2d_iter!(22, 37);
validate_2d_iter!(16, 16);
validate_2d_iter!(32, 32);
validate_2d_iter!(32, 64);
validate_2d_iter!(64, 64);
validate_2d_iter!(100, 100);
validate_2d_iter!(72, 133);
validate_2d_iter!(256, 256);
validate_2d_iter!(255, 255);
validate_2d_iter!(1023, 1833);

validate_3d_iter!(1, 1, 1);
validate_3d_iter!(1, 2, 3);
validate_3d_iter!(3, 2, 1);
validate_3d_iter!(3, 4, 8);
validate_3d_iter!(18, 4, 12);
validate_3d_iter!(1, 14, 3);
validate_3d_iter!(7, 7, 7);
validate_3d_iter!(8, 8, 8);
validate_3d_iter!(32, 32, 32);
validate_3d_iter!(24, 58, 72);
validate_3d_iter!(64, 64, 64);
validate_3d_iter!(100, 100, 100);
validate_3d_iter!(123, 99, 177);
