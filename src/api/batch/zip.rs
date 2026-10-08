/// Drop-in replacement for `itertools::Zip<(A, B, ..)>`
/// where `next` is force-inlined.
#[derive(Debug, Clone)]
pub struct Zip<T>(pub T);

#[inline(always)]
pub fn multizip<T>(t: T) -> Zip<T> {
    Zip(t)
}

macro_rules! impl_zip {
    ($($T:ident $idx:tt),+) => {
        impl<$($T: Iterator),+> Iterator for Zip<($($T,)+)> {
            type Item = ($($T::Item,)+);

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                // Tuple elements evaluate left to right, and `?`
                // short-circuits as soon as one iterator is exhausted.
                Some(($(self.0.$idx.next()?,)+))
            }

            #[inline(always)]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let mut lower = usize::MAX;
                let mut upper: Option<usize> = None;
                $(
                    let (l, u) = self.0.$idx.size_hint();
                    lower = lower.min(l);
                    upper = match (upper, u) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };
                )+
                (lower, upper)
            }
        }
    };
}

impl_zip!(A 0, B 1);
impl_zip!(A 0, B 1, C 2);
impl_zip!(A 0, B 1, C 2, D 3);
