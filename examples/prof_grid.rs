use quick_noise::{Fbm, Grid, Simplex};

fn main() {
    let size = 64usize;
    let mut result = vec![0.0f32; size * size];

    let scale = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(64.0);
    let iters: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(u64::MAX);

    let grid = Grid::<2>::new(size, size);
    let source = grid.builder::<Fbm, Simplex>();

    let mut sink = 0u64;
    let mut i = 0u64;
    while i < iters {
        source
            .octaves(1)
            .frequency(1.0 / scale)
            .fill(result.as_mut_slice());
        sink = sink.wrapping_add(
            (result[0] as i32 as u32 as u64) ^ (result[size * size - 1] as i32 as u32 as u64),
        );
        i += 1;
    }
    std::process::exit((sink & 1) as i32);
}