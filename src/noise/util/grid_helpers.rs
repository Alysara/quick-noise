use std::array::from_fn;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Mask, Simd, SimdElement};

use crate::api::grid::interface::GridNoiseParams;
use crate::noise::combiners::{Combiner, CombinerState};

const STACK_SIZE: usize = 8192;
pub struct ArenaBuffer<F: Arch> {
    heap: Vec<f32>,
    stack: [MaybeUninit<f32>; STACK_SIZE],
    _family: PhantomData<F>,
}

impl<F: Arch> ArenaBuffer<F> {
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity + Simd::<f32, F>::LANES; // Add LANES for alignment padding.
        let heap = if capacity > STACK_SIZE {
            Vec::with_capacity(capacity)
        } else {
            Vec::new()
        };

        let stack: [MaybeUninit<f32>; STACK_SIZE] = std::array::from_fn(|_| MaybeUninit::uninit());

        Self {
            heap,
            stack,
            _family: PhantomData::<F>,
        }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [MaybeUninit<f32>] {
        let slice = if self.heap.capacity() > 0 {
            self.heap.spare_capacity_mut()
        } else {
            self.stack.as_mut_slice()
        };

        let offset = slice.as_ptr().align_offset(F::SIMD_WIDTH);
        unsafe { slice.get_unchecked_mut(offset..) }
    }
}

pub struct Arena<'a> {
    slice: &'a mut [MaybeUninit<f32>],
}

impl<'a> Arena<'a> {
    #[inline(always)]
    pub fn with_cache<F: Arch>(cache: &'a mut ArenaBuffer<F>) -> Self {
        let slice = cache.as_mut_slice();
        Self { slice }
    }

    #[inline(always)]
    pub fn allocate<T>(&mut self, capacity: usize) -> &'a mut [MaybeUninit<T>] {
        const {
            assert!(size_of::<T>() == size_of::<f32>());
        }

        let whole = std::mem::take(&mut self.slice);

        let (buf, rem) = whole.split_at_mut(capacity);
        self.slice = rem;
        unsafe { std::mem::transmute(buf) }
    }

    #[inline(always)]
    pub fn allocate_arena(&mut self, capacity: usize) -> Self {
        let whole = std::mem::take(&mut self.slice);

        let (slice, rem) = whole.split_at_mut(capacity);
        self.slice = rem;
        Self { slice }
    }
}

pub struct InterpolationConfig<A: Arch> {
    pub num_blocks: usize,
    pub block_lanes: usize,
    pub has_block_head: bool,
    pub has_block_tail: bool,
    pub tail_size: usize,
    pub block_tail_size: usize,
    pub block_tail_start: usize,
    pub _family: PhantomData<A>,
}

impl<F: Arch> InterpolationConfig<F> {
    pub fn new(num_blocks: usize, x_dim: usize) -> Self {
        let lanes: usize = Simd::<f32, F>::LANES;
        let block_lanes: usize = num_blocks * lanes;
        Self {
            num_blocks,
            block_lanes,
            has_block_head: x_dim >= block_lanes,
            has_block_tail: !x_dim.is_multiple_of(block_lanes),
            tail_size: x_dim % block_lanes,
            block_tail_size: (x_dim % block_lanes).div_ceil(lanes),
            block_tail_start: (x_dim / block_lanes) * block_lanes,
            _family: PhantomData::<F>,
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn maybe_tail_load<A: Arch, const IS_TAIL: bool>(
    range: Range<usize>,
    slice: &[f32],
) -> Simd<f32, A> {
    unsafe {
        if IS_TAIL {
            let lanes = Simd::<f32, A>::LANES;
            let start = range.start;
            if start + lanes <= slice.len() {
                // Whole vector fits: plain unaligned load.
                Simd::from_slice_unchecked(slice.get_unchecked(start..))
            } else {
                // End of buffer: masked load, fault-suppressed on inactive lanes.
                let rem = range.end - start;
                let mask = Mask::<f32, A>::first_n_true(rem as u32);
                Simd::masked_load(slice.get_unchecked(start..), mask)
            }
        } else {
            Simd::from_slice_unchecked(slice.get_unchecked(range.start..))
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn maybe_tail_store<A: Arch, const IS_TAIL: bool>(
    range: Range<usize>,
    simd: Simd<f32, A>,
    slice: &mut [f32],
) {
    unsafe {
        if IS_TAIL {
            let rem = range.end - range.start;
            let mask = Mask::<f32, A>::first_n_true(rem as u32);
            simd.masked_store(slice.get_unchecked_mut(range.start..), mask);
        } else {
            simd.copy_to_slice_unchecked(slice.get_unchecked_mut(range.start..));
        }
    }
}

pub trait MaybeUninitSliceSimdExt<T: SimdElement, F: Arch> {
    /// # Safety
    /// - The range `index..index + ArchSimd::<T>::LANES` must be in bounds.
    /// - Data in range `index..index + ArchSimd::<T>::LANES` must be initialized.
    unsafe fn load_simd(&self, index: usize) -> Simd<T, F>;

    /// # Safety
    /// - The range `index..index + ArchSimd::<T>::LANES` must be in bounds.
    /// - Data in range `index..index + ArchSimd::<T>::LANES` must be initialized.
    /// - `index` must be aligned according to `SIMD_WIDTH`.
    unsafe fn load_simd_aligned(&self, index: usize) -> Simd<T, F>;

    /// # Safety
    /// - The range `index..index + ArchSimd::<T>::LANES` must be in bounds.
    unsafe fn write_simd(&mut self, index: usize, simd: Simd<T, F>);

    /// # Safety
    /// - The range `index..index + ArchSimd::<T>::LANES` must be in bounds.
    /// - `index` must be aligned according to `SIMD_WIDTH`.
    unsafe fn write_simd_aligned(&mut self, index: usize, simd: Simd<T, F>);
}

impl<T: SimdElement, F: Arch> MaybeUninitSliceSimdExt<T, F> for [MaybeUninit<T>] {
    unsafe fn load_simd(&self, index: usize) -> Simd<T, F> {
        unsafe { Simd::from_slice_unchecked(self.get_unchecked(index..).assume_init_ref()) }
    }

    unsafe fn load_simd_aligned(&self, index: usize) -> Simd<T, F> {
        unsafe { Simd::from_aligned_slice_unchecked(self.get_unchecked(index..).assume_init_ref()) }
    }

    unsafe fn write_simd(&mut self, index: usize, simd: Simd<T, F>) {
        unsafe { simd.copy_to_slice_unchecked(self.get_unchecked_mut(index..).assume_init_mut()) }
    }

    unsafe fn write_simd_aligned(&mut self, index: usize, simd: Simd<T, F>) {
        unsafe {
            simd.copy_to_aligned_slice_unchecked(self.get_unchecked_mut(index..).assume_init_mut())
        }
    }
}

#[inline(always)]
pub fn validate_grid_size<const D: usize>(grid_size: [usize; D], slice_len: usize) {
    let num_samples = grid_size.iter().product();
    assert!(
        slice_len >= num_samples,
        "Uniform grid with dimensions {:?} has a size of {num_samples}, which is more than the given slice length of {slice_len}",
        grid_size
    );
}

#[inline(always)]
pub fn validate_state_size<C: Combiner, F: Arch, const D: usize>(
    grid_size: [usize; D],
    slice_len: usize,
) {
    if C::State::<F>::STATE_SIZE > 0 {
        let total_size: usize = grid_size.iter().product();
        let required_size = total_size * C::State::<F>::STATE_SIZE;
        assert!(
            slice_len >= required_size,
            "Uniform grid with dimensions {:?} with {} state variables requires a state size of{required_size}, which is more than the given slice length of {slice_len}",
            required_size,
            C::State::<F>::STATE_SIZE,
        );
    }
}

#[inline(always)]
pub fn pad_grid_size<F: Arch, const D: usize>(grid_size: [usize; D]) -> [usize; D] {
    let lanes: usize = Simd::<f32, F>::LANES;
    from_fn(|i| lanes - grid_size[i] % lanes + grid_size[i] + lanes)
}

// SAFETY: caller/invariant of this type guarantees these slices are
// fully initialized by the time Debug is used. If that's not
// guaranteed, this is unsound — see note below.
pub(crate) unsafe fn assume_init_slice<T>(s: &[MaybeUninit<T>]) -> &[T] {
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), s.len()) }
}

#[inline(always)]
pub fn fill_grid_indices<A: Arch, const D: usize>(
    grid_indices: &mut [&mut [MaybeUninit<u32>]; D],
    distances: &[&mut [MaybeUninit<f32>]; D],
    distances_len: [usize; D],
) -> [usize; D] {
    std::array::from_fn(|i| {
        let mut write_idx = 0usize;
        let indices_ptr = grid_indices[i].as_mut_ptr();

        let last_valid = distances_len[i] - 1;
        let full_block_end = last_valid - last_valid % 64;
        for base_index in (1..=full_block_end).step_by(64) {
            let mut bits = 0u64;
            for bit_index in (0..64).step_by(Simd::<f32, A>::LANES) {
                let cur_index = base_index + bit_index;
                let cur: Simd<f32, A> = unsafe { distances[i].load_simd(cur_index) };
                let prev: Simd<f32, A> = unsafe { distances[i].load_simd_aligned(cur_index - 1) };

                let mask_bits = prev.simd_gt(cur).to_bits();
                bits |= mask_bits << bit_index;
            }

            while bits != 0 {
                let cur_index = base_index as u32 + bits.trailing_zeros();
                unsafe {
                    indices_ptr
                        .add(write_idx)
                        .write(MaybeUninit::new(cur_index))
                };
                write_idx += 1;
                bits &= bits - 1;
            }
        }

        let tail_len = last_valid - full_block_end;
        let mut bits = 0u64;
        for bit_index in (0..tail_len).step_by(Simd::<f32, A>::LANES) {
            let cur_index = bit_index + full_block_end + 1;
            let cur: Simd<f32, A> = unsafe { distances[i].load_simd(cur_index) };
            let prev: Simd<f32, A> = unsafe { distances[i].load_simd_aligned(cur_index - 1) };

            let mask_bits = prev.simd_gt(cur).to_bits();
            bits |= mask_bits << bit_index;
        }
        bits &= (1u64 << tail_len) - 1;

        while bits != 0 {
            let cur_index = full_block_end as u32 + bits.trailing_zeros() + 1;
            unsafe {
                indices_ptr
                    .add(write_idx)
                    .write(MaybeUninit::new(cur_index))
            };
            write_idx += 1;
            bits &= bits - 1;
        }

        // Write sentinel.
        unsafe {
            indices_ptr
                .add(write_idx)
                .write(MaybeUninit::new(distances_len[i] as u32))
        };
        write_idx + 1
    })
}

#[inline(always)]
pub(crate) fn configure_tiling<const D: usize>(params: &GridNoiseParams<D>) -> [Option<u32>; D] {
    std::array::from_fn(|i| {
        if let Some(val) = params.tiling[i] {
            let float = val as f32 * params.frequency[i];
            let nearness = (float - float.round()).abs();
            assert!(
                nearness < 0.001,
                "frequency does not align with the tiling of {val} (frequency={}, nearness={nearness})!",
                params.frequency[i]
            );
            Some(float.round() as u32)
        } else {
            None
        }
    })
}

#[inline(always)]
pub(crate) fn simd_rem_euclid_i32<A: Arch>(x: Simd<i32, A>, t: i32) -> Simd<i32, A> {
    let t_f = Simd::<f32, A>::splat(t as f32);
    let x_f = x.cast_float();
    (x_f - (x_f / t_f).floor() * t_f).cast_int_trunc()
}
