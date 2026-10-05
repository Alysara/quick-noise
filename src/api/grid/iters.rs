use std::marker::PhantomData;

use simply_simd::{Arch, Mask, Simd};

use crate::Grid;

/// Upper bound on f32 lanes for any supported arch (used for the
/// scalar fallback scratch buffer).
const MAX_LANES: usize = 32;

impl<A: Arch> Grid<2, A> {
    /// Creates a simd iterator of the x values when iterating
    /// through the grid sample by sample.
    #[inline(always)]
    pub fn x_iter(&self) -> RowIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        RowIter::new(dim[0], dim[1], pos[0])
    }

    /// Creates a simd iterator of the y values when iterating
    /// through the grid sample by sample.
    #[inline(always)]
    pub fn y_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0], dim[1], 1, pos[1])
    }
}

impl<A: Arch> Grid<3, A> {
    /// Creates a simd iterator of the x values when iterating
    /// through the grid sample by sample.
    #[inline(always)]
    pub fn x_iter(&self) -> RowIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        RowIter::new(dim[0], dim[1] * dim[2], pos[0])
    }

    /// Creates a simd iterator of the y values when iterating
    /// through the grid sample by sample.
    #[inline(always)]
    pub fn y_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0], dim[1], dim[2], pos[1])
    }

    /// Creates a simd iterator of the z values when iterating
    /// through the grid sample by sample.
    #[inline(always)]
    pub fn z_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0] * dim[1], dim[2], 1, pos[2])
    }
}

#[derive(Clone, Copy)]
struct RowScalar {
    row_size: usize,
    left_in_row: usize,
    rows_left: usize,
    cur: f32,
    start: f32,
}

/// Fills `buf[..lanes]`. Returns false if the iterator was already finished.
#[cold]
#[inline(never)]
fn row_scalar_fill(s: &mut RowScalar, buf: &mut [f32; MAX_LANES], lanes: usize) -> bool {
    if s.left_in_row == 0 && s.rows_left == 0 {
        return false;
    }

    for slot in buf[..lanes].iter_mut() {
        *slot = if s.left_in_row > 0 {
            let next = s.cur;
            s.cur += 1.0;
            s.left_in_row -= 1;
            next
        } else if s.rows_left > 0 {
            s.cur = s.start + 1.0;
            s.rows_left -= 1;
            s.left_in_row = s.row_size - 1;
            s.start
        } else {
            0.0
        };
    }
    true
}

#[derive(Clone, Copy)]
struct SliceScalar {
    row_size: usize,
    slice_size: usize,
    left_in_row: usize,
    left_in_slice: usize,
    slices_left: usize,
    cur_val: f32,
    start_val: f32,
}

/// Fills `buf[..lanes]`. Returns false if the iterator was already finished.
#[cold]
#[inline(never)]
fn slice_scalar_fill(s: &mut SliceScalar, buf: &mut [f32; MAX_LANES], lanes: usize) -> bool {
    if s.left_in_row == 0 && s.left_in_slice == 0 && s.slices_left == 0 {
        return false;
    }

    for slot in buf[..lanes].iter_mut() {
        *slot = if s.left_in_row > 0 {
            s.left_in_row -= 1;
            s.cur_val
        } else if s.left_in_slice > 0 {
            s.cur_val += 1.0;
            s.left_in_row = s.row_size - 1;
            s.left_in_slice -= 1;
            s.cur_val
        } else if s.slices_left > 0 {
            s.cur_val = s.start_val;
            s.left_in_row = s.row_size - 1;
            s.left_in_slice = s.slice_size - 1;
            s.slices_left -= 1;
            s.start_val
        } else {
            0.0
        };
    }
    true
}

/// A simd iterator that iterates through the fastest changing
/// value (x axis). These values change after every subsequent
/// sample.
#[derive(Debug)]
pub struct RowIter<A: Arch> {
    row_size: usize,
    left_in_row: usize,
    rows_left: usize,
    cur_vec: Simd<f32, A>,
    start_vec: Simd<f32, A>,
    // Only used by the scalar fallback.
    cur_scalar: f32,
    start_scalar: f32,
    _arch: PhantomData<A>,
}

impl<A: Arch> RowIter<A> {
    #[inline(always)]
    fn new(row_size: usize, num_rows: usize, start_val: f32) -> Self {
        Self {
            row_size,
            left_in_row: row_size,
            rows_left: num_rows - 1,
            cur_vec: Simd::iota(start_val),
            start_vec: Simd::iota(start_val),
            cur_scalar: start_val,
            start_scalar: start_val,
            _arch: PhantomData::<A>,
        }
    }

    #[inline(always)]
    fn next_scalar(&mut self) -> Option<Simd<f32, A>> {
        // Copy the state out by value so no reference to `self`'s fields
        // escapes into the non-inlined call.
        let mut s = RowScalar {
            row_size: self.row_size,
            left_in_row: self.left_in_row,
            rows_left: self.rows_left,
            cur: self.cur_scalar,
            start: self.start_scalar,
        };
        let mut buf = [0.0f32; MAX_LANES];

        if !row_scalar_fill(&mut s, &mut buf, Simd::<f32, A>::LANES) {
            return None;
        }

        self.left_in_row = s.left_in_row;
        self.rows_left = s.rows_left;
        self.cur_scalar = s.cur;
        Some(Simd::from_slice(&buf[..Simd::<f32, A>::LANES]))
    }
}

impl<A: Arch> Iterator for RowIter<A> {
    type Item = Simd<f32, A>;

    #[inline(always)]
    fn next(&mut self) -> Option<Simd<f32, A>> {
        let lanes = Simd::<f32, A>::LANES;

        // Scalar case (row narrower than one vector).
        if self.row_size < lanes {
            std::hint::cold_path();
            return self.next_scalar();
        }

        // Full columns: the overwhelmingly common case.
        if self.left_in_row >= lanes {
            let next = self.cur_vec;
            self.cur_vec += Simd::splat(lanes as f32);
            self.left_in_row -= lanes;
            return Some(next);
        }

        // Partial/Transition columns.
        if self.rows_left > 0 {
            let mask = Mask::first_n_true(self.left_in_row as u32);
            let old = self.cur_vec;
            let next = self.start_vec - Simd::splat(self.left_in_row as f32);
            self.left_in_row += self.row_size - lanes;
            self.rows_left -= 1;
            self.cur_vec = next + Simd::splat(lanes as f32);
            return Some(mask.select(old, next));
        }

        // Tail.
        if self.left_in_row > 0 {
            let mask = Mask::first_n_true(self.left_in_row as u32);
            let vec = self.cur_vec;
            self.left_in_row = 0;
            return Some(mask.select(vec, Simd::zero()));
        }

        // Finished iter.
        None
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left_in_row + self.rows_left * self.row_size;
        let chunks_left = left.div_ceil(Simd::<f32, A>::LANES);
        (chunks_left, Some(chunks_left))
    }
}

/// A simd iterator that iterates through the non-fastest changing
/// values (all axes other than the x axis). These values usually do
/// not change after every subsequent sample.
#[derive(Debug)]
pub struct SliceIter<A: Arch> {
    row_size: usize,
    slice_size: usize,
    left_in_row: usize,
    left_in_slice: usize,
    slices_left: usize,
    cur_val: f32,
    start_val: f32,
    _arch: PhantomData<A>,
}

impl<A: Arch> SliceIter<A> {
    #[inline(always)]
    pub fn new(row_size: usize, slice_size: usize, num_slices: usize, start_val: f32) -> Self {
        Self {
            row_size,
            slice_size,
            left_in_row: row_size,
            left_in_slice: slice_size - 1,
            slices_left: num_slices - 1,
            cur_val: start_val,
            start_val,
            _arch: PhantomData::<A>,
        }
    }

    #[inline(always)]
    fn next_scalar(&mut self) -> Option<Simd<f32, A>> {
        let mut s = SliceScalar {
            row_size: self.row_size,
            slice_size: self.slice_size,
            left_in_row: self.left_in_row,
            left_in_slice: self.left_in_slice,
            slices_left: self.slices_left,
            cur_val: self.cur_val,
            start_val: self.start_val,
        };
        let mut buf = [0.0f32; MAX_LANES];

        if !slice_scalar_fill(&mut s, &mut buf, Simd::<f32, A>::LANES) {
            return None;
        }

        self.left_in_row = s.left_in_row;
        self.left_in_slice = s.left_in_slice;
        self.slices_left = s.slices_left;
        self.cur_val = s.cur_val;
        Some(Simd::from_slice(&buf[..Simd::<f32, A>::LANES]))
    }
}

impl<A: Arch> Iterator for SliceIter<A> {
    type Item = Simd<f32, A>;

    #[inline(always)]
    fn next(&mut self) -> Option<Simd<f32, A>> {
        let lanes = Simd::<f32, A>::LANES;

        // Scalar case (row narrower than one vector).
        if self.row_size < lanes {
            std::hint::cold_path();
            return self.next_scalar();
        }

        // Full rows: the overwhelmingly common case.
        if self.left_in_row >= lanes {
            self.left_in_row -= lanes;
            return Some(Simd::splat(self.cur_val));
        }

        // Transition within a slice.
        if self.left_in_slice > 0 {
            let old = Simd::splat(self.cur_val);
            self.cur_val += 1.0;
            let new = Simd::splat(self.cur_val);

            let mask = Mask::first_n_true(self.left_in_row as u32);
            self.left_in_row += self.row_size - lanes;
            self.left_in_slice -= 1;
            return Some(mask.select(old, new));
        }

        // Transition to next slice.
        if self.slices_left > 0 {
            let old = Simd::splat(self.cur_val);
            let new = Simd::splat(self.start_val);
            self.cur_val = self.start_val;

            let mask = Mask::first_n_true(self.left_in_row as u32);
            self.left_in_row += self.row_size - lanes;
            self.left_in_slice = self.slice_size - 1;
            self.slices_left -= 1;
            return Some(mask.select(old, new));
        }

        // Tail.
        if self.left_in_row > 0 {
            let old = Simd::splat(self.cur_val);
            let mask = Mask::first_n_true(self.left_in_row as u32);
            self.left_in_row = 0;
            return Some(mask.select(old, Simd::zero()));
        }

        // Finished iter.
        None
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_in_slice = self.left_in_row + self.left_in_slice * self.row_size;
        let left_after_slice = self.slices_left * self.row_size * self.slice_size;
        let left = left_in_slice + left_after_slice;
        let chunks_left = left.div_ceil(Simd::<f32, A>::LANES);
        (chunks_left, Some(chunks_left))
    }
}
