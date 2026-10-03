use std::array::from_fn;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

use simply_simd::{Arch, Mask, Simd, SimdElement, SimdToArray};

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
    pub block_lanes: usize,
    pub has_block_head: bool,
    pub has_block_tail: bool,
    pub block_tail_size: usize,
    pub block_tail_start: usize,
    pub has_partial: bool,
    pub partial_mask: Mask<f32, A>,
    pub partial_start: usize,
    pub partial_size: usize,
    pub _family: PhantomData<A>,
}

impl<F: Arch> InterpolationConfig<F> {
    pub fn new(num_blocks: usize, x_dim: usize) -> Self {
        let lanes: usize = Simd::<f32, F>::LANES;
        let block_lanes: usize = num_blocks * lanes;
        Self {
            block_lanes,
            has_block_head: x_dim >= block_lanes,
            has_block_tail: !x_dim.is_multiple_of(block_lanes),
            block_tail_size: (x_dim % block_lanes) / lanes,
            block_tail_start: (x_dim / block_lanes) * block_lanes,
            has_partial: !x_dim.is_multiple_of(lanes),
            partial_mask: Mask::first_n_false(lanes as u32 - (x_dim % lanes) as u32),
            partial_start: x_dim.saturating_sub(lanes),
            partial_size: x_dim % lanes,
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

    /// Loads from padded buffers starting from `index`, except in the access mode
    /// case of partial, where it instead loads starting from config's partial_start.
    ///
    /// # Safety
    /// - The range `index..index + ArchSimd::<T>::LANES must be in bounds.
    /// - Index must be aligned to ArchSimd::<T>::LANES for access mode full and scalar.
    /// - For partial access mode, config's partial_start..partial_start + Arch::<T>::LANES must be
    ///   in bounds.
    unsafe fn ld_buf<const ACCESS_MODE: u8>(
        &self,
        index: usize,
        config: &InterpolationConfig<F>,
    ) -> Simd<T, F>;
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

    #[inline(always)]
    unsafe fn ld_buf<const ACCESS_MODE: u8>(
        &self,
        index: usize,
        config: &InterpolationConfig<F>,
    ) -> Simd<T, F> {
        unsafe {
            match SimdAccessMode::from_u8(ACCESS_MODE) {
                SimdAccessMode::Partial => self.load_simd(config.partial_start),
                _ => self.load_simd_aligned(index),
            }
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

#[inline(always)]
fn fill_grid_indices_single<A: Arch>(
    indices: &mut [MaybeUninit<u32>],
    distances: &[MaybeUninit<f32>],
    distances_len: usize,
) -> usize {
    let mut write_idx = 0usize;
    let indices_ptr = indices.as_mut_ptr();

    let last_valid = distances_len - 1;
    let full_block_end = last_valid - last_valid % 64;
    for base_index in (1..=full_block_end).step_by(64) {
        let mut bits = 0u64;
        for bit_index in (0..64).step_by(Simd::<f32, A>::LANES) {
            let cur_index = base_index + bit_index;
            let cur: Simd<f32, A> = unsafe { distances.load_simd(cur_index) };
            let prev: Simd<f32, A> = unsafe { distances.load_simd_aligned(cur_index - 1) };
            bits |= prev.simd_gt(cur).to_bits() << bit_index;
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
        let cur: Simd<f32, A> = unsafe { distances.load_simd(cur_index) };
        let prev: Simd<f32, A> = unsafe { distances.load_simd_aligned(cur_index - 1) };
        bits |= prev.simd_gt(cur).to_bits() << bit_index;
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

    unsafe {
        indices_ptr
            .add(write_idx)
            .write(MaybeUninit::new(distances_len as u32))
    };
    write_idx + 1
}

pub fn fill_grid_indices<A: Arch, const D: usize>(
    grid_indices: &mut [&mut [MaybeUninit<u32>]; D],
    distances: &[&mut [MaybeUninit<f32>]; D],
    distances_len: [usize; D],
) -> [usize; D] {
    let mut out = [0usize; D];
    for i in 0..D {
        out[i] =
            fill_grid_indices_single::<A>(&mut *grid_indices[i], &*distances[i], distances_len[i]);
    }
    out
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

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub(crate) enum SimdAccessMode {
    Full = 0,
    Partial = 1,
    Scalar = 2,
}

pub(crate) const FULL: u8 = SimdAccessMode::Full as u8;
pub(crate) const PARTIAL: u8 = SimdAccessMode::Partial as u8;
pub(crate) const SCALAR: u8 = SimdAccessMode::Scalar as u8;


impl SimdAccessMode {
    #[inline(always)]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Full,
            1 => Self::Partial,
            2 => Self::Scalar,
            _ => unreachable!(),
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn load_simd_rw<A: Arch>(
    slice: &[f32],
    index: usize,
    config: &InterpolationConfig<A>,
    access_mode: SimdAccessMode,
) -> Simd<f32, A> {
    unsafe {
        match access_mode {
            SimdAccessMode::Full => Simd::from_slice_unchecked(slice.get_unchecked(index..)),
            SimdAccessMode::Partial => {
                let slice = slice.get_unchecked(index + config.partial_start..);
                Simd::from_slice_unchecked(slice)
            }
            SimdAccessMode::Scalar => {
                let slice = slice.get_unchecked(index..index + config.partial_size);
                Simd::from_slice(slice)
            }
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn write_simd_rw<A: Arch>(
    slice: &mut [f32],
    index: usize,
    simd: Simd<f32, A>,
    config: &InterpolationConfig<A>,
    access_mode: SimdAccessMode,
) {
    unsafe {
        match access_mode {
            SimdAccessMode::Full => {
                simd.copy_to_slice_unchecked(slice.get_unchecked_mut(index..));
            }
            SimdAccessMode::Partial => {
                let slice = slice.get_unchecked_mut(index + config.partial_start..);
                simd.masked_store(slice, config.partial_mask);
            }
            SimdAccessMode::Scalar => {
                let vals = simd.to_array();

                let slice = slice.get_unchecked_mut(index..index + config.partial_size);
                for (x, v) in slice.iter_mut().zip(vals.iter()) {
                    *x = *v;
                }
            }
        };
    }
}
