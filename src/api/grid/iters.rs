use std::marker::PhantomData;

use simply_simd::{Arch, Mask, Simd, SimdToArray, enable_targets};

use crate::Grid;

impl<A: Arch> Grid<2, A> {
    #[inline(always)]
    pub fn x_iter(&self) -> RowIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        RowIter::new(dim[0], dim[1], pos[0])
    }

    #[inline(always)]
    pub fn y_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0], dim[1], 1, pos[1])
    }
}

impl<A: Arch> Grid<3, A> {
    #[inline(always)]
    pub fn x_iter(&self) -> RowIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        RowIter::new(dim[0], dim[1] * dim[2], pos[0])
    }

    #[inline(always)]
    pub fn y_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0], dim[1], dim[2], pos[1])
    }

    #[inline(always)]
    pub fn z_iter(&self) -> SliceIter<A> {
        let pos = self.config.position;
        let dim = self.config.grid_size;
        SliceIter::new(dim[0] * dim[1], dim[2], 1, pos[2])
    }
}

#[derive(Debug)]
pub struct RowIter<A: Arch> {
    row_size: usize,
    left_in_row: usize,
    rows_left: usize,
    cur_vec: Simd<f32, A>,
    start_vec: Simd<f32, A>,
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
            _arch: PhantomData::<A>,
        }
    }
}

#[enable_targets(A)]
impl<A: Arch> Iterator for RowIter<A> {
    type Item = Simd<f32, A>;

    fn next(&mut self) -> Option<Simd<f32, A>> {
        // Scalar case.
        if self.row_size < Simd::<f32, A>::LANES {
            if self.left_in_row == 0 && self.rows_left == 0 {
                return None;
            }

            let mut cur = self.cur_vec.to_array()[0];
            let start = self.start_vec.to_array()[0];

            let array = <A as Arch>::Array32::<f32>::from_fn(|_| {
                if self.left_in_row > 0 {
                    let next = cur;
                    cur += 1.0;
                    self.left_in_row -= 1;
                    next
                } else if self.rows_left > 0 {
                    cur = start + 1.0;
                    self.rows_left -= 1;
                    self.left_in_row = self.row_size - 1;
                    start
                } else {
                    0.0
                }
            });

            self.cur_vec = Simd::iota(cur);
            return Some(Simd::from_slice(array.as_slice()));
        }

        // Full columns.
        if self.left_in_row >= Simd::<f32, A>::LANES {
            let next = self.cur_vec;
            self.cur_vec += Simd::splat(Simd::<f32, A>::LANES as f32);
            self.left_in_row -= Simd::<f32, A>::LANES;
            return Some(next);
        }

        // Partial/Transition columns.
        if self.rows_left > 0 {
            let mask = Mask::first_n_true(self.left_in_row as u32);
            let old = self.cur_vec;
            let next = self.start_vec - Simd::splat(self.left_in_row as f32);
            self.left_in_row += self.row_size - Simd::<f32, A>::LANES;
            self.rows_left -= 1;
            self.cur_vec = next + Simd::splat(Simd::<f32, A>::LANES as f32);
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left_in_row + self.rows_left * self.row_size;
        let chunks_left = left.div_ceil(Simd::<f32, A>::LANES);
        (chunks_left, Some(chunks_left))
    }
}

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
}

#[enable_targets(A)]
impl<A: Arch> Iterator for SliceIter<A> {
    type Item = Simd<f32, A>;

    #[inline(always)]
    fn next(&mut self) -> Option<Simd<f32, A>> {
        // Scalar case.
        if self.row_size < Simd::<f32, A>::LANES {
            if self.left_in_row == 0 && self.left_in_slice == 0 && self.slices_left == 0 {
                return None;
            }

            let array = <A as Arch>::Array32::<f32>::from_fn(|_| {
                if self.left_in_row > 0 {
                    self.left_in_row -= 1;
                    self.cur_val
                } else if self.left_in_slice > 0 {
                    self.cur_val += 1.0;
                    self.left_in_row = self.row_size - 1;
                    self.left_in_slice -= 1;
                    self.cur_val
                } else if self.slices_left > 0 {
                    self.cur_val = self.start_val;
                    self.left_in_row = self.row_size - 1;
                    self.left_in_slice = self.slice_size - 1;
                    self.slices_left -= 1;
                    self.start_val
                } else {
                    0.0
                }
            });
            return Some(Simd::from_slice(array.as_slice()));
        }

        // Full rows.
        if self.left_in_row >= Simd::<f32, A>::LANES {
            self.left_in_row -= Simd::<f32, A>::LANES;
            return Some(Simd::splat(self.cur_val));
        }

        if self.left_in_slice > 0 {
            let old = Simd::splat(self.cur_val);
            self.cur_val += 1.0;
            let new = Simd::splat(self.cur_val);

            let mask = Mask::first_n_true(self.left_in_row as u32);
            self.left_in_row += self.row_size - Simd::<f32, A>::LANES;
            self.left_in_slice -= 1;
            return Some(mask.select(old, new));
        }

        if self.slices_left > 0 {
            let old = Simd::splat(self.cur_val);
            let new = Simd::splat(self.start_val);
            self.cur_val = self.start_val;

            let mask = Mask::first_n_true(self.left_in_row as u32);
            self.left_in_row += self.row_size - Simd::<f32, A>::LANES;
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
