All changes in `quick-noise` are documented here.

## Unreleased

## 0.3.0 - In development

### Added
- Generator for 2D Cellular grid noise.

### Changed
- Performance improvements to Value and Perlin grid noise.

### Fixed
- Incorrect cellular noise result for scalar fallback.

## 0.2.0 - 2026-07-25

### Added
- Runtime feature selection support.

### Changed
- Breaking change: Combiner implementations now use generics for SIMD features.
- Breaking change: ArchSimd changed to StaticSimd, and other SIMD changes.
- Simd module split into the simply-simd crate. Simd can still be used like before through quick-noise.

## 0.1.1 - 2026-07-19

### Added
- Benchmark details for the `noise-functions` crate.

### Fixed
- False panic on debug builds in scalar fallback that checked for incorrect alignment. ([#6](https://github.com/Alysara/quick-noise/issues/6) by [@ReCore-sys](https://github.com/ReCore-sys)) 
- Typos in README.
- Interface structs `GridNoise` and `GridNoiseParams` wrongfully kept private.

## 0.1.0 - 2026-07-16

Initial release.

