//! A slotbkvec, a vector-like container with unique indices, but
//! distribute the elements in the inner vector of the outer buckets.
//!
//! See the documentation of [`SlotBkVec`] for details.
//!
//! [`SlotBkVec`]: struct.SlotBkVec.html
use core::ops::{Index, IndexMut};

use super::{ManagedSlice as Slice, SlotVec};

/// Provides a slotbkvec based on external memory.
///
/// `N` buckets are provided, each bucket is a slice of element `T`,
/// and each indice carries bucket and slice index information, so that
/// elements in the slice can be reversely located through the buckets.
///
/// By default its bucket number `N` is 1, which looks like a [`SlotVec`].
///
/// The bucket parameter of the public API function refers to the bucket
/// number of buckets minus one, the range is 0..=N-1, that is, the index.
///
/// # Panics
/// If `N` is provided is not in the range of 1 to 16, a compilation error
/// will occur. The range of 1 to 16 is only an empirical magic number.
#[derive(Debug)]
pub struct SlotBkVec<'a, T, const N: usize = 1> {
    /// Buckets for each slotvec.
    buckets: [SlotVec<'a, T>; N],
    /// Bucket status bits.
    bucket_bits: usize,
    /// Bucket bit mask.
    bucket_mask: usize,
}

impl<'a, T, const N: usize> SlotBkVec<'a, T, N> {
    const MIN_BUCKET_COUNT: usize = 1;
    const MAX_BUCKET_COUNT: usize = 16;
    /// Compiler will check const N must be in range [1..16].
    const __COMPILER_CHECK__: () =
        assert!(N >= Self::MIN_BUCKET_COUNT && N <= Self::MAX_BUCKET_COUNT);

    /// Returns the bucket bit numbers.
    fn bucket_bits() -> usize {
        // N <= 2:  last bit        mask: 0b1    , buckets: (0,1)
        // N <= 4:  last tow bits   mask: 0b11   , buckets: (0,1,2,3)
        // N <= 8:  last three bits mask: 0b111  , buckets: (0,1,2,3,4,5,6,7)
        // N <= 16: last four bits  mask: 0b1111 , buckets: (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        // this calculation method only works in std: `(N as f64).log2().ceil() as usize`
        match N {
            1..=2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            9..=16 => 4,
            _ => unreachable!(),
        }
    }

    /// Returns the bucket mask from bucket bit numbers.
    fn bucket_mask() -> usize {
        let bucket_bits = Self::bucket_bits();
        (1 << bucket_bits) - 1
    }

    /// Returns a unmasked index and buket from a masked index.
    fn index_unmask(&self, index: usize) -> (usize, usize) {
        Self::calculate_index_unmask(self.bucket_bits, self.bucket_mask, index)
    }

    /// Returns a masked index with a unmasked index and buket.
    fn masked_index(&self, index: usize, buket: usize) -> usize {
        Self::calculate_masked_index(self.bucket_bits, self.bucket_mask, index, buket)
    }

    /// Returns a unmasked index and buket from a masked index.
    #[inline]
    fn calculate_index_unmask(
        bucket_bits: usize,
        bucket_mask: usize,
        index: usize,
    ) -> (usize, usize) {
        (index >> bucket_bits, index & bucket_mask)
    }

    /// Returns a masked index with a unmasked index and buket.
    #[inline]
    fn calculate_masked_index(
        bucket_bits: usize,
        bucket_mask: usize,
        index: usize,
        buket: usize,
    ) -> usize {
        (index << bucket_bits) | (buket & bucket_mask)
    }

    /// Iterates every element of all slices of the buckets, as mutable.
    #[auto_enums::auto_enum(DoubleEndedIterator)]
    fn iter_mut_impl<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> {
        // This implementation does not affect performance, because N is determined
        // at compile time, and a specific N will only take a specific branch.
        // If there is a const-match syntax, we can save the runtime pattern-matching,
        // and we don't need auto_enums to solve multi-branch impl opaque types.
        macro_rules! iters_chain {
            ($n:literal, $first:ident $(,$rest:ident)*) => {{
                // SAFETY: Type casting is safe after the const generic parameter N is asserted.
                let _self_: &mut SlotBkVec<'_, T, $n> = unsafe { ::core::mem::transmute(self) };
                let [$first$(,$rest)*] = _self_.each_mut();
                $first$(.chain($rest))*
            }};
        }
        // buckets[0].iter_mut().chain(b1.iter_mut()) ... .chain(buckets[N-1].iter_mut())
        match N {
            01 => iters_chain!(01, a),
            02 => iters_chain!(02, a, b),
            03 => iters_chain!(03, a, b, c),
            04 => iters_chain!(04, a, b, c, d),
            05 => iters_chain!(05, a, b, c, d, e),
            06 => iters_chain!(06, a, b, c, d, e, f),
            07 => iters_chain!(07, a, b, c, d, e, f, g),
            08 => iters_chain!(08, a, b, c, d, e, f, g, h),
            09 => iters_chain!(09, a, b, c, d, e, f, g, h, i),
            10 => iters_chain!(10, a, b, c, d, e, f, g, h, i, j),
            11 => iters_chain!(11, a, b, c, d, e, f, g, h, i, j, k),
            12 => iters_chain!(12, a, b, c, d, e, f, g, h, i, j, k, l),
            13 => iters_chain!(13, a, b, c, d, e, f, g, h, i, j, k, l, m),
            14 => iters_chain!(14, a, b, c, d, e, f, g, h, i, j, k, l, m, n),
            15 => iters_chain!(15, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o),
            16 => iters_chain!(16, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p),
            _ => ::core::iter::empty(), // unreachable: compile-checking 1 <= N <= 16
        }
    }
}

impl<'a, T, const N: usize> SlotBkVec<'a, T, N> {
    /// Creates a slot bucket vec, `Option` is used to mark whether the slot has been used.
    pub fn new(slices: [Slice<'a, Option<T>>; N]) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::__COMPILER_CHECK__;
        Self {
            buckets: slices.map(SlotVec::new),
            bucket_bits: Self::bucket_bits(),
            bucket_mask: Self::bucket_mask(),
        }
    }

    /// Pushes an element to the back of the bucket slice in the buckets,
    /// an element should be generated by the function `elem_fn` calling.
    ///
    /// Returns None if the bucket number is overflow ( > `N` ),
    /// or if the slice is fixed-size (not a `Vec`) and is full.
    pub fn push_with(&mut self, bucket: usize, elem_fn: impl FnOnce(usize) -> T) -> Option<usize> {
        if bucket >= N {
            None
        } else {
            let (bucket_bits, bucket_mask) = (self.bucket_bits, self.bucket_mask);
            let mask_fn =
                |index| Self::calculate_masked_index(bucket_bits, bucket_mask, index, bucket);
            let index = self.buckets[bucket].push_with(|index| {
                let index = mask_fn(index);
                elem_fn(index)
            })?;
            Some(self.masked_index(index, bucket))
        }
    }

    /// Pushes an element to the back of the bucket slice in the buckets.
    ///
    /// Returns None if the bucket number is overflow ( > `N` ),
    /// or if the slice is fixed-size (not a `Vec`) and is full.
    pub fn push(&mut self, bucket: usize, elem: T) -> Option<usize> {
        if bucket >= N {
            None
        } else {
            let index = self.buckets[bucket].push(elem)?;
            Some(self.masked_index(index, bucket))
        }
    }

    /// Gets an element from the bucket slice by its index, as immutable.
    ///
    /// Returns `None` if the index did not refer to a valid element.
    pub fn get(&self, index: usize) -> Option<&T> {
        let (index, bucket) = self.index_unmask(index);
        self.buckets[bucket].get(index)
    }

    /// Gets an element from the bucket slice by its index, as mutable.
    ///
    /// Returns `None` if the index did not refer to a valid element.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let (index, buket) = self.index_unmask(index);
        self.buckets[buket].get_mut(index)
    }

    /// Removes an element from the bucket slice, without changing it.
    ///
    /// Returns the removed element that could be freed if successful,
    /// returns `None` if the index did not refer to a valid element.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        let (index, buket) = self.index_unmask(index);
        self.buckets[buket].remove(index)
    }

    /// Returns the number of valid elements of all slices of the buckets.
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Returns true if all slices of the buckets contains no valid elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates every element of the bucket slice, as immutable.
    ///
    /// # Panics
    /// This function panics if the bucket number is overflow (>N).
    pub fn bucket_iter<'s>(&'s self, bucket: usize) -> impl DoubleEndedIterator<Item = &'s T> {
        if bucket >= N {
            None
        } else {
            Some(self.buckets[bucket].iter())
        }
        .into_iter()
        .flatten()
    }

    /// Iterates every element of the bucket slice, as mutable.
    ///
    /// # Panics
    /// This function panics if the bucket number is overflow (>N).
    pub fn bucket_iter_mut<'s>(
        &'s mut self,
        bucket: usize,
    ) -> impl DoubleEndedIterator<Item = &'s mut T> {
        if bucket >= N {
            None
        } else {
            Some(self.buckets[bucket].iter_mut())
        }
        .into_iter()
        .flatten()
    }

    /// Iterates every element of all slices of the buckets, as immutable.
    pub fn iter<'s>(&'s self) -> impl DoubleEndedIterator<Item = &'s T> {
        self.buckets.iter().flat_map(|slice| slice.iter())
    }

    /// Iterates every element of all slices of the buckets, as mutable.
    pub fn iter_mut<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> {
        // NB: In the future we may use a more concise Rust API like this.
        // Although it is possible to use the Captures trick here,
        // the Captures trick signature will infect the upper APIs,
        // so we use array pattern-matching to avoid the Captures trick.
        //
        // See: https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
        // pub trait Captures<'a> {}
        // impl<'a, T: ?Sized> Captures<'a> for T {}
        //
        // ```
        // pub fn iter_mut<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> + Captures<'a> {
        //     self.buckets.iter_mut().flat_map(|slice| slice.iter_mut())
        // }
        // ```

        // This won't compile due to opaque type lifetime capture 'a:
        // self.buckets.iter_mut().flat_map(|slice| slice.iter_mut())
        self.iter_mut_impl() // using array pattern-matching to solve
    }

    /// Borrows each bucket element mutably and returns an array of immutable references.
    pub fn each_ref<'s>(&'s self) -> [impl DoubleEndedIterator<Item = &'s T>; N] {
        self.buckets.each_ref().map(|x| x.iter())
    }

    /// Borrows each bucket element mutably and returns an array of mutable references.
    pub fn each_mut<'s>(&'s mut self) -> [impl DoubleEndedIterator<Item = &'s mut T>; N] {
        self.buckets.each_mut().map(|x| x.iter_mut())
    }
}

impl<'a, T, const N: usize> Index<usize> for SlotBkVec<'a, T, N> {
    type Output = T;

    /// Returns a immutable reference to the value corresponding to the supplied index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not present in the `SlotBkVec`.
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a, T, const N: usize> IndexMut<usize> for SlotBkVec<'a, T, N> {
    /// Returns a mutable reference to the value corresponding to the supplied index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not present in the `SlotBkVec`.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{Slice, SlotBkVec};
    use std::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn test_compile_error() {
        macro_rules! repeat_owned {
            ($n:literal) => {
                [0; $n].map(|_| Slice::Owned(vec![]))
            };
        }
        macro_rules! repeat_borrowed {
            ($n:literal) => {
                [0; $n].map(|_| Slice::Borrowed(&mut []))
            };
        }
        // compile error
        // let _ = SlotBkVec::<i32, 0>::new([]);
        // compile ok
        let _ = SlotBkVec::<i32, 1>::new(repeat_borrowed!(1));
        let _ = SlotBkVec::<i32, 15>::new(repeat_borrowed!(15));
        let _ = SlotBkVec::<i32, 16>::new(repeat_owned!(16));
        // compile error
        // let _ = SlotBkVec::<i32, 17>::new(repeat_owned!(17));
    }

    #[test]
    fn simple() {
        let (slice1, slice1_bucket) = (vec![], 0);
        let (slice2, slice2_bucket) = (vec![], 1);
        let (slice3, slice3_bucket) = (vec![], 2);
        let mut slot_bkvec: SlotBkVec<&str, 3> = SlotBkVec::new([
            Slice::Owned(slice1),
            Slice::Owned(slice2),
            Slice::Owned(slice3),
        ]);

        // nonexistent buckets, existent buckets: 0,1,2 < 3
        assert_eq!(slot_bkvec.push(3, "hello"), None);
        assert_eq!(slot_bkvec.push(4, "world"), None);

        let index = slot_bkvec.push(slice1_bucket, "hello").unwrap();
        assert_eq!(slot_bkvec.get(index), Some("hello").as_ref());
        assert_eq!(slot_bkvec[index], "hello");

        let elem = slot_bkvec.get_mut(index).unwrap();
        *elem = "world";
        assert_eq!(slot_bkvec.get(index), Some("world").as_ref());
        assert_eq!(slot_bkvec[index], "world");

        let index = slot_bkvec.push(slice2_bucket, "hello").unwrap();
        assert_eq!(slot_bkvec.get(index), Some("hello").as_ref());
        assert_eq!(slot_bkvec[index], "hello");

        let index = slot_bkvec.push(slice2_bucket, "jacky").unwrap();
        assert_eq!(slot_bkvec.get(index), Some("jacky").as_ref());
        assert_eq!(slot_bkvec[index], "jacky");

        let index = slot_bkvec.push(slice3_bucket, "noise").unwrap();
        assert_eq!(slot_bkvec.get(index), Some("noise").as_ref());
        assert_eq!(slot_bkvec[index], "noise");

        assert_eq!(
            slot_bkvec.iter().collect::<Vec<&&str>>(),
            [&"world", &"hello", &"jacky", &"noise"]
        );
    }

    #[test]
    fn retained() {
        let (slice1, slice1_bucket) = (vec![], 0);
        let (slice2, slice2_bucket) = (vec![], 1);
        let mut slot_bkvec: SlotBkVec<&str, 2> =
            SlotBkVec::new([Slice::Owned(slice1), Slice::Owned(slice2)]);

        let idx1 = slot_bkvec.push(slice1_bucket, "hello").unwrap();
        let idx2 = slot_bkvec.push(slice2_bucket, "world").unwrap();
        let idx3 = slot_bkvec.push(slice2_bucket, "jacky").unwrap();

        assert_eq!(slot_bkvec.remove(idx1), Some("hello"));
        assert_eq!(slot_bkvec.get(idx1), None);
        // nonexistent index, panics
        // slot_bkvec[idx1];

        assert_eq!(slot_bkvec.remove(idx3), Some("jacky"));
        assert_eq!(slot_bkvec.get(idx3), None);

        assert_eq!(slot_bkvec.iter().count(), 1);

        assert_eq!(slot_bkvec.remove(idx2), Some("world"));
        assert_eq!(slot_bkvec.get(idx2), None);
        assert_eq!(slot_bkvec.iter().count(), 0);

        let idx4 = slot_bkvec.push(slice1_bucket, "small").unwrap();
        assert_eq!(slot_bkvec.get(idx4), Some(&"small"));
        assert_eq!(slot_bkvec.iter().count(), 1);
    }

    #[test]
    fn complex() {
        let (slice1, slice1_bucket) = (&mut [Default::default(); 1], 0);
        let (slice2, slice2_bucket) = (vec![Default::default(); 2], 1);
        let mut slot_bkvec: SlotBkVec<String, 2> =
            SlotBkVec::new([Slice::Borrowed(slice1), Slice::Owned(slice2)]);

        let index = slot_bkvec.push(slice1_bucket, "hello".to_string()).unwrap();
        assert_eq!(slot_bkvec.get(index), Some("hello".to_string()).as_ref());

        let elem = slot_bkvec.get_mut(index).unwrap();
        *elem = "world".to_string();
        assert_eq!(slot_bkvec.get(index), Some("world".to_string()).as_ref());

        // the slice1 is full, the length of the slice1 borrowed is 1.
        assert_eq!(slot_bkvec.push(slice1_bucket, "messy".to_string()), None);

        let index = slot_bkvec.push(slice2_bucket, "hello".to_string()).unwrap();
        assert_eq!(slot_bkvec.get(index), Some("hello".to_string()).as_ref());

        let index = slot_bkvec.push(slice2_bucket, "jacky".to_string()).unwrap();
        assert_eq!(slot_bkvec.get(index), Some("jacky".to_string()).as_ref());

        // Even though the slice2 is 2 in length,
        // it can still be pushed because it is an owned vector.
        assert!(slot_bkvec
            .push(slice2_bucket, "noise".to_string())
            .is_some());

        let index = slot_bkvec
            .push_with(slice2_bucket, |index| index.to_string())
            .unwrap();
        let rusty = index.to_string();

        assert_eq!(
            slot_bkvec
                .bucket_iter(slice1_bucket)
                .collect::<Vec<&String>>(),
            [&"world"]
        );

        assert_eq!(
            slot_bkvec
                .bucket_iter(slice2_bucket)
                .collect::<Vec<&String>>(),
            [&"hello", &"jacky", &"noise", &rusty.as_str()]
        );

        assert_eq!(
            slot_bkvec.iter().collect::<Vec<&String>>(),
            [&"world", &"hello", &"jacky", &"noise", &rusty.as_str()]
        );
    }
}
