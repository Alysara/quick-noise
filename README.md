[![Crates.io](https://img.shields.io/crates/v/quick-noise.svg)](https://crates.io/crates/quick-noise)
[![Documentation](https://img.shields.io/docsrs/quick-noise)](https://docs.rs/quick-noise)
[![License](https://img.shields.io/crates/l/quick-noise.svg)](https://github.com/Alysara/quick-noise#license)
[![Build Status](https://github.com/Alysara/quick-noise/actions/workflows/rust.yml/badge.svg)](https://github.com/Alysara/quick-noise/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/Alysara/quick-noise/branch/main/graph/badge.svg)](https://codecov.io/gh/Alysara/quick-noise)
[![MSRV](https://img.shields.io/badge/MSRV-1.95-orange.svg)](https://www.rust-lang.org)

Blazingly fast SIMD procedural noise library for batch and uniform grid sampling with runtime feature detection on stable Rust.

# Performance

### 2D Noise
Time taken to produce 3 octaves of FBM noise for 1024x1024 (1,048,576) samples.
| Library              | Perlin  |  Value  | Simplex | Cellular |
|----------------------|---------|---------|---------|----------|
| quick-noise (grid)   | 0.82 ms | 0.60 ms |    X    | 0.85 ms  |
| quick-noise (batch)  | 2.65 ms | 2.47 ms | 4.11 ms | 5.30 ms  |
| fastnoise2           | 4.30 ms | 3.67 ms | 5.15 ms | 16.2 ms  |
| fastnoise-lite       | 28.4 ms | 27.4 ms | 35.6 ms | 87.7 ms  |
| simd-noise           |    X    |    X    | 7.79 ms | 11.3 ms  |
| noise-rs             | 26.2 ms | 25.0 ms | 41.5 ms | 86.0 ms  |
| noiz                 | 25.4 ms | 20.9 ms | 36.3 ms | 74.0 ms  |
| libnoise             | 26.9 ms | 23.5 ms | 52.1 ms | 126 ms   |
| noise-functions      | 7.70 ms | 4.69 ms | 35.2 ms | 41.6 ms  |

### 3D Noise
Time taken to produce 3 octaves of FBM noise for 128x128x128 (2,097,152) samples.
| Library              | Perlin  |  Value  | Simplex | Cellular |
|----------------------|---------|---------|---------|----------|
| quick-noise (grid)   | 0.73 ms | 0.65 ms |    X    |    X     |
| quick-noise (batch)  | 18.8 ms | 8.83 ms | 18.8 ms | 30.6 ms  |
| fastnoise2           | 22.4 ms | 11.9 ms | 20.0 ms | 137 ms   |
| fastnoise-lite       | 102 ms  | 83.7 ms | 133 ms  | 545 ms   |
| simd-noise           |    X    |    X    | 22.0 ms | 78.0 ms  |
| noise-rs             | 80.5 ms | 167 ms  | 153 ms  | 328 ms   |
| noiz                 | 102 ms  | 85.2 ms | 132 ms  | 390 ms   |
| libnoise             | 100 ms  | 74.8 ms | 155 ms  | 406 ms   |
| noise-functions      | 83.6 ms | 43.7 ms | 267 ms  | 249 ms   |

* X signifies the noise type is not supported or readily exposed
* Grid path performance degrades for very high frequencies, and cannot support
frequencies >= 1.0. Grid noise. However, it can generate 10+ billion samples per second
at smaller grid sizes (64x64, 32x32x32) where memory transfer is a smaller barrier.
More detailed benchmarks below.
* This performance is achieved with static dispatch using the `target-cpu=native` flag.
Without this flag, runtime feature detection (dynamic dispatch) is needed to achieve similar performance. See below
for guidance on runtime feature detection.

# Usage

quick-noise offers two public facing interfaces. Grid noise samples a squared (2D)
or cubed (3D) region uniformly while batch noise samples points at arbitrary inputs.
The performance of grid noise is often magnitudes higher than batch noise, and the
recommended path for high-performance procedural generation.

## Builders

Builders are used to offer extensive options while remaining approachable.
Every builder can be executed with one of three methods: `build()`, `fill()`, and `into_iter()`.
- `build()`: returns a new Vec of the noise result directly
- `fill()`: fills a slice that you provide, potentially saving costly memory copies
- `into_iter()`: returns an iterator containing simd registers of the output

Iterators allow multiple steps of the noise pipeline to fuse together, providing speedups by keeping data in registers directly.
Note that grid noise is an exception to this rule, but makes up for it many times over in speed.


## Combiners and Generators

Generators are structs that define how to generate noise. This includes `Perlin`, `Value`, `Simplex`, and `Cellular`.
Combiners specify *how* that noise is applied across multiple octaves (noise passes). This includes
`Fbm`, `Billow`, `Ridged`, `Multi`, `HybridMulti`, `Terrace`, and `PingPong`. Combiners apply to both batch and grid noise.

## Grid Noise

Grid noise is called through a grid region. All of the samples inside of this region are
generated all at once in a single call.

```rust
use quick_noise::{Grid, Fbm, Perlin};

// Creates an anchor into a region of sample space.
let grid = Grid::<2>::new(1000, 1000); // Specify a 2D 1000x1000 grid.
    
// Creates 1,000,000 samples.
let results: Vec<f32> = grid.builder::<Fbm, Perlin>()
    .octaves(6)      // Number of noise passes.
    .frequency(0.01) // How fast the noise changes across sample space.
    .build();
```

Each builder has sensible defaults allowing you to be as simple or customized as
you like:

```rust
use quick_noise::{Grid, Fbm, Perlin};

let grid = Grid::<2>::new(32, 32)
    .grid_position(0.0, 0.0) // Position of the grid region in sample space.
    .seed(102);              // Seed for deterministic random results.

let mut result = [0.0; 1024];

// FBM Grid noise with all parameters.
let noise = grid.builder::<Fbm, Perlin>()
    .seed(0)             // Acts a second seed in combination with the grid's seed.
    .octaves(1)          // Number of noise passes.
    .frequency(0.03125)  // How fast the noise changes across sample space.
    .lacunarity(2.0)     // Multiplier to the frequency after each subsequent octave.
    .persistence(0.5)    // Multiplier to the weight after each subsequent octave.
    .amplitude(1.0)      // The maximum absolute value the noise can produce (For Fbm).
    .normalization(true) // Whether or not to normalize the output.
    .scaling(1.0, 1.0)   // How much is the noise stretched in each dimension.
    .initialize(true)    // Setting to false adds noise to current values.
    .finalize(true)      // Some combiners have a finalization stage.
    .fill(result.as_mut_slice());
```

Currently, only Perlin and Value is supported for grid noise. For octave sequences more complicated than FBM noise,
`builder_with_octaves` can be used for granular control over frequencies and weights.

```rust
use quick_noise::{Octave, Grid, Billow, Value};

let grid = Grid::<2>::new(200, 200);

// Custom list of octaves that can't be easily described by FBM noise.
let octave_list = [
	// Creates octaves from frequency and weight.
	Octave::<2>::splat(0.05, 7.0),
	Octave::<2>::splat(0.02, 4.0),
	Octave::<2>::splat(0.03, 15.0),
	Octave::<2>::splat(0.04, 9.0),
	Octave::<2>::splat(0.05, 15.0),

	// Allows axis-specific granularity for frequency.
	// This creates 'stretched' noise.
	Octave::<2>::new([0.01, 0.015], 50.0),
];

let mut result = vec![0.0; 40000];
let noise = grid.builder_with_octaves::<Billow, Value>(octave_list.as_slice())
	.seed(1000)
	.amplitude(2.0)
	.fill(result.as_mut_slice());
```

quick-noise makes warped noise convenient through a dedicated grid method.
It internally adds the values of the grid to the offset iterators you provide it.
This can be chained together for complex warp configurations. Since it uses batch noise,
Perlin, Value, Simplex, and Cellular can all be used here.

```rust

#[cfg(feature = "image")]
{
    use quick_noise::{Grid, Fbm, Perlin};
    use quick_noise::emit::NoiseImageExt;
    
    let grid = Grid::<2>::new(1024, 512);
    
    // Create noise offsets to warp by with fast grid noise.
    let noise1 = grid.builder::<Fbm, Perlin>().octaves(6).seed(0).into_iter();
    let noise2 = grid.builder::<Fbm, Perlin>().octaves(6).seed(1).into_iter();
    
    grid.warp_builder::<Fbm, Perlin>(100.0, noise1, noise2)
        .octaves(2) // Cheap two octaves for expensive batch noise call.
        .frequency(1. / 32.0)
        .into_iter()
        .to_grayscale_image(1024, 512, "noise_images/perlin_warp_2d.png");
}

```

![Warped Perlin Noise](images/perlin_warp_2d.png)

You can also set custom tiling parameters to the grid to generate noise that
wraps around and repeats. Unlike other methods, this method does not
require a higher dimension and operates natively in that algorithm.
However, frequencies must align with the given tiling. For example,
a frequency of `1 / 1000` would not work with a tiling of `(2048, 2048)`.
Frequencies of `1 / 1024` and `1 / 512` would. Tiling is only supported
for grid_noise currently.

You can choose to only enable tiling for specific axes and can specify the
size of the tiles for each axis specifically.

```rust
#[cfg(feature = "image")]
{
    use quick_noise::{Grid, Fbm, Perlin};
    use quick_noise::emit::NoiseImageExt;
    
    let grid = Grid::<2>::new(1024, 1024)
        .grid_position(0.0, 0.0)
        .seed(100)
        .tiling(Some(128), Some(128)); // Put None to disable tiling for that axis.
    
    grid.builder::<Fbm, Perlin>()
        .octaves(6)
        .frequency(1.0 / 128.0)
        .into_iter()
        .to_grayscale_image(1024, 1024, "noise_images/perlin_tiles.png");
}
```

![Tiled Perlin Noise](images/perlin_tiles_2d.png)

## Batch Noise

Batch noise operates directly on static methods and takes iterators as inputs. Perlin, Value, Simplex, and Cellular all support Batch noise.

```rust
use quick_noise::{Grid, BatchNoise, Fbm, Simplex};

// Use grid for generating iters.
let grid = Grid::<2>::new(100, 100).grid_position(0.0, 0.0);

let noise = BatchNoise::<2, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter())
	.octaves(6)
	.frequency(0.2)
	.lacunarity(0.4)
	.persistence(0.6)
	.scaling(1.0, 0.5)
	.build();
```

Batch noise allows for arbitrary input coordinates, enabling techniques such as domain warping.
In this example, a uniform grid is being generated manually for demonstration purposes. Using grid noise is much faster for this use case.
Since simplex and cellular do not currently support grid noise, this method can be used to generate
them on a grid. Batch noise also supports custom octaves.

## Simd

quick-noise uses a custom simd module purpose-built for noise. Unlike std::simd, it works on stable.
However, only SSE, AVX2, AVX512, and NEON are supported currently. For other systems, a scalar fallback exists,
but the performance is much worse. Luckily the vast majority of computers used today support one of these instruction sets.

This simd module can support most basic operations, and can be used directly to benefit from it:

```rust
use quick_noise::{Grid, BatchNoise, Ridged, Cellular};
use quick_noise::simd::StaticSimd;
use std::iter::zip;

let grid = Grid::<2>::new(128, 128).grid_position(0.0, 0.0);

let iter_1 = BatchNoise::<2, Ridged, Cellular>::builder(grid.x_iter(), grid.y_iter())
    .seed(0)
    .into_iter();

let iter_2 = BatchNoise::<2, Ridged, Cellular>::builder(grid.x_iter(), grid.y_iter())
    .seed(1)
    .into_iter();

let iter_3 = zip(iter_1, iter_2).map(|(x, y)| x * y);

// You also get access to simd registers directly.
let simd = StaticSimd::<f32>::splat(1.0);
```

Using these iterators can fuse operations and avoid multiple vertical passes, particularly for batch noise.
`Simd` represents a raw simd register for a given architecture. Unlike std::simd which abstracts these architecture details,
this simd module offers you the ability to explicitly control loops that work best for your CPU.

## Dynamic Dispatch

The examples shown so far use static dispatch, which identifies which simd feature set to use at compile time.
This requires compiler flags up-front and reduces the portability of your program. However, quick-noise allows you
to compile for multiple targets and identify which target to use at runtime. The primary method is with
the attribute macro `dispatch_simd`:

```rust
use quick_noise::{Grid, Fbm, Perlin, BatchNoise};
use quick_noise::simd::{Simd, dispatch_simd};

#[dispatch_simd(A)]
pub fn generate_noise() {
    // Include A from the macro when specifying the grid.
    let grid = Grid::<2, A>::new(128, 128);

    // Now all grid methods and builders use that architecture.
    let x_iter = grid.x_iter();
    let y_iter = grid.y_iter();
    let noise = grid.builder::<Fbm, Perlin>().build();

    // Batch noise will infer that it's using the specified architecture.
    let result = BatchNoise::<2, Fbm, Perlin>::builder(x_iter, y_iter).into_iter();

    // You can use simd registers dynamically as well.
    let simd = Simd::<f32, A>::splat(1.0);
}

fn main() {
    // Function can be called like normal.
    generate_noise()
}
```

Since the dispatch requires branching at runtime,
it is best to do this outside of hot loops.

Rust will not emit SIMD instructions inline if it does not have
the associated feature flags. Functions with generic type `A: Arch`
must either inline or use the `enable_targets` macro. If neither
of these are done, Rust may not inline the SIMD instructions and
performance will significantly degrade, up to 50x times slower.

```rust
use quick_noise::simd::{Arch, dispatch_simd, enable_targets};

// Dispatch simd here to avoid repeated dispatching
// in every iteration.
#[dispatch_simd(A)]
pub fn simd_entry() {
    for _ in 0..1024 {
        simd_work_1::<A>();
        simd_work_2::<A>();
        broken_simd_work::<A>();
    }
}

// Avoid using #[dispatch_simd(A)] again here.
// Instead, use #[enable_targets(A)] with <A: Arch>.
#[enable_targets(A)]
pub fn simd_work_1<A: Arch>() {
    // ...
}

// #[inline(always)] can be used as well.
#[inline(always)]
pub fn simd_work_2<A: Arch>() {
    // ...
}

// No inline or enable_targets,
// compiler may not optimize or inline 
// SIMD instructions.
pub fn broken_simd_work<A: Arch>() {
    // ...
}

```

`dispatch_simd` and `enable_targets` can both be used on impl blocks as well. This is necessary
when `A: Arch` is generic across a struct. It will also apply to every function that contains
a generic `A: Arch` (or other identifier you specify). This method must be used for trait implementations:

```rust
use quick_noise::simd::{Arch, enable_targets};

trait SimdTask {
    fn simd_work<A: Arch>();
}
struct SimdWorker {}

#[enable_targets(A)]
impl SimdTask for SimdWorker {
    fn simd_work<A: Arch>() {}
}
```

If `#[dispatch_simd(A)]` is applied to an associated function using generic
parameters from its impl block, the macro will not have enough information to tell it's an
associated function and requires an additional flag: `#[dispatch_simd(A, associated)]`.
For the majority of cases, this flag can be omitted.

Unfortunately, these restrictions make it impossible to ensure other functions that you do not own
use the dispatched target. As a result, adapaters on iterators do not work with dynamic dispatch
and result in non-inlined intrinsics.

## Loading Simd

`simd_iter` and `simd_iter_mut` are exposed by the `SimdSliceIterExt` to create these iters from slices.
These require the architecture to be known. For static dispatch, `simd_iter_static` and `simd_iter_mut_static`
mut be used.

```rust
use quick_noise::{Fbm, Perlin, BatchNoise};
use quick_noise::simd::{SimdSliceIterExt, dispatch_simd};

#[dispatch_simd(A)]
pub fn generate_noise() {
    // Say we have buffers of arbitrary inputs we want to query noise results.
    let x_buffer: [f32; 1024] = std::array::from_fn(|i| i as f32);
    let y_buffer: [f32; 1024] = std::array::from_fn(|i| i as f32);
    let mut result: [f32; 1024] = [0.0; 1024];

    // We can use the statically dispatched simd feature set, which does not require
    // dynamic dispatch.
    let x_iter = x_buffer.as_slice().simd_iter_static();
    let y_iter = y_buffer.as_slice().simd_iter_static();
    BatchNoise::<2, Fbm, Perlin>::builder(x_iter, y_iter).fill(result.as_mut_slice());

    // We can also use dynamic dispatch.
    let x_iter = x_buffer.as_slice().simd_iter::<A>();
    let y_iter = y_buffer.as_slice().simd_iter::<A>();
    BatchNoise::<2, Fbm, Perlin>::builder(x_iter, y_iter).fill(result.as_mut_slice());

    // BatchNoise is the same in both cases.
}

```

## Extensibility

quick-noise allows you to implement your own custom combiners and generators.
They are defined once in one place and work for both grid and batch noise.
For example, the Fbm combiner is defined as:

```rust
use quick_noise::{Combiner, CombinerArray};
use quick_noise::simd::{Arch, Simd};

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Fbm {}
impl Combiner for Fbm {
    const WEIGHT_DECAY: bool = true;
    type State<A: Arch> = CombinerArray<A, 0>;
    type Config = ();

    #[inline(always)]
    fn apply_sample<A: Arch>(
        _config: &(),
        state: Self::State<A>,
        cur_result: Simd<f32, A>,
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (state, cur_result + new_sample)
    }

    #[inline(always)]
    fn initialize_sample<A: Arch>(
        _config: &(),
        new_sample: Simd<f32, A>,
    ) -> (Self::State<A>, Simd<f32, A>) {
        (Default::default(), new_sample)
    }

    #[inline(always)]
    fn finalize_sample<A: Arch>(
        _config: &(),
        _state: Self::State<A>,
        last: Simd<f32, A>,
    ) -> Simd<f32, A> {
        last
    }
}
```
(See Combiner trait documentation for more details)

Custom noise generators can be created by implementing the `GridGenerator` and `BatchGenerator`
traits.

To sample directly, `GridNoise` and `BatchNoise` both have `sample` and `sample_with_octaves`.
Structs that implement `GridGenerator` and `BatchGenerator` support `sample_grid` and `sample_batch`.

## Feature Flags

quick-noise offers a couple of utility features. These are disabled by default to keep compilation lean.

### image
The image feature flag uses the `image` crate and enables the usage of `to_grayscale_image` for generating
grayscale images of your noise.

### serde
The serde feature flag dervies `Serialize` and `Deserialize` for config structs.


# Detailed Performance

## Grid Noise

Grid noise shares computations across samples to achieve greater performance.
As a result, lower frequencies have greater performance than higher frequencies.
Additionally, the dimensions of a noise call impact performance as well. Array sizes
that are multiples of 16 offer the best SIMD usage. When sizes get very large, memory 
transfer and cache intermediaries becomes more expensive. For maximum performance, 32x32 and 64x64
is recommended. However, it is better to use larger calls directly than to transfer
memory from smaller calls onto a larger noise map.

Results are measured in billions of points per second single-threaded for one noise pass
over a 64x64 grid (2D) and 32x32x32 grid (3D).
- AVX2: I7-13700H | XPS 15 9530 Laptop | Linux
- AVX512: Ryzen 7 9800X3D | Linux

### Perlin
| Frequency | 2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-----------|----------|----------|-----------|-----------|
| 1 / 64    | 16.2 B/s | 15.6 B/s | 32.7 B/s  | 32.4 B/s  |
| 1 / 48    | 14.3 B/s | 15.6 B/s | 27.5 B/s  | 32.4 B/s  |
| 1 / 32    | 14.1 B/s | 15.6 B/s | 26.3 B/s  | 32.4 B/s  |
| 1 / 24    | 12.6 B/s | 12.9 B/s | 22.9 B/s  | 22.3 B/s  |
| 1 / 16    | 11.5 B/s | 13.3 B/s | 21.5 B/s  | 22.4 B/s  |
| 1 / 8     | 8.55 B/s | 9.64 B/s | 11.6 B/s  | 14.9 B/s  |
| 1 / 4     | 4.52 B/s | 3.96 B/s | 5.32 B/s  | 5.33 B/s  |

### Value
| Frequency | 2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-----------|----------|----------|-----------|-----------|
| 1 / 64    | 28.0 B/s | 18.2 B/s | 33.6 B/s  | 40.0 B/s  |
| 1 / 48    | 25.0 B/s | 18.0 B/s | 30.7 B/s  | 39.9 B/s  |
| 1 / 32    | 25.2 B/s | 18.2 B/s | 30.1 B/s  | 39.9 B/s  |
| 1 / 24    | 22.1 B/s | 16.7 B/s | 28.6 B/s  | 30.0 B/s  |
| 1 / 16    | 20.1 B/s | 16.8 B/s | 27.0 B/s  | 31.8 B/s  |
| 1 / 8     | 14.6 B/s | 15.0 B/s | 18.6 B/s  | 24.5 B/s  |
| 1 / 4     | 7.77 B/s | 7.87 B/s | 10.2 B/s  | 11.6 B/s  |

## Batch Noise

Batch noise processing is much more flexible than uniform grid, allowing for any arbitrary input and enabling
techniques such as domain warping, but at the cost of performance. Results are measured in millions of points per second.

|   Perlin    |  2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-------------|-----------|----------|-----------|-----------|
| quick-noise | 1,097 M/s | 328 M/s  | 3,076 M/s | 1,133 M/s |
| FastNoise2  | 586 M/s   | 248 M/s  | 960 M/s   | 651 M/s   |

|    Value    |  2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-------------|-----------|----------|-----------|-----------|
| quick-noise | 1,224 M/s | 641 M/s  | 4,109 M/s | 2,134 M/s |
| FastNoise2  | 679 M/s   | 440 M/s  | 1,117 M/s | 831 M/s   |

|   Simplex   |  2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-------------|-----------|----------|-----------|-----------|
| quick-noise | 764 M/s   | 325 M/s  | 2,007 M/s | 1,033 M/s |
| FastNoise2  | 514 M/s   | 284 M/s  | 840 M/s   | 637 M/s   |

|   Cellular  |  2D AVX2  | 3D AVX2  | 2D AVX512 | 3D AVX512 |
|-------------|-----------|----------|-----------|-----------|
| quick-noise | 698 M/s   | 211 M/s  | 1,882 M/s | 537 M/s   |
| FastNoise2  | 185 M/s   | 56.5 M/s | 398 M/s   | 149 M/s   |

# Running

Height maps can be generated in `examples/basic.rs`. To run these examples, use:

> cargo run --example basic --release --features="image"

It is important that `RUSTFLAGS='-C target-cpu=native'` and `--release` is used for the best performance.
If this flag is not used, runtime feature detection (dynamic dispatch) can be used to achieve
similar performance.

Criterion benches can be run with:

> cargo bench -p quick-noise-benches

If you use Linux, it's recommended to run with `taskset -c 0` to ensure
the same cpu core is used across tests.

> taskset -c 0 cargo bench -p quick-noise-benches

Test suite can be run with:

> cargo test
