//! A slotdeque, a vector-deque-like container with unique indices.
//!
//! See the documentation of [`SlotDeque`] for details.
//!
//! [`SlotDeque`]: struct.SlotDeque.html
use core::ops::{Index, IndexMut};

use super::{ManagedSlice as Slice, SlotBkVec};

/// Provides a slotdeque based on external memory.
///
/// A slotdeque provides a `VecDeque`-like interface where each entry is
/// associated with a stable index. Lookup with the index will detect if
/// an entry has been removed but does not require a lifetime relation.
/// It will allocate memory internally when adding elements to slotdeque
/// if the [`Slice`] argument to the constructor is a [`Slice::Owned`] Vec.
/// It only replaces the Option element value with the value that actually
/// if the [`Slice`] argument to the constructor is a [`Slice::Borrowed`] slice.
///
/// [`push_front`] is logically like [`VecDeque::push_front`],
/// [`push_back`] is logically like [`VecDeque::push_back`].
///
/// [`VecDeque::push_front`]: std::collections::VecDeque::push_front
/// [`VecDeque::push_back`]: std::collections::VecDeque::push_back
#[derive(Debug)]
pub struct SlotDeque<'a, T> {
    /// The first bucket slice is logically similar to the front slice of
    /// the [`VecDeque`] after being reversed, and the second bucket slice
    /// directly is like the back slice of the [`VecDeque`].
    ///
    /// Pushes on the first bucket slice is equivalent to [`VecDeque::push_front`],
    /// and pushes on the second bucket slice is equivalent to [`VecDeque::push_back`].
    ///
    /// [`VecDeque`]: std::collections::VecDeque
    /// [`VecDeque::push_front`]: std::collections::VecDeque::push_front
    /// [`VecDeque::push_back`]: std::collections::VecDeque::push_back
    slices: SlotBkVec<'a, T, 2>,
}

impl<'a, T> SlotDeque<'a, T> {
    const BUCKET_FRONT: usize = 0;
    const BUCKET_BACK: usize = 1;

    /// Creates a slot deque, `Option` is used to mark whether the slot has been used.
    pub fn new(front: Slice<'a, Option<T>>, back: Slice<'a, Option<T>>) -> Self {
        Self {
            slices: SlotBkVec::new([front, back]),
        }
    }

    /// Prepends an element to the front of deque.
    ///
    /// Returns None if the front slice is fixed-size (not a `Vec`) and is full.
    pub fn push_front_with(&mut self, f: impl FnOnce(usize) -> T) -> Option<usize> {
        self.slices.push_with(Self::BUCKET_FRONT, f)
    }

    /// Prepends an element to the front of deque.
    ///
    /// Returns None if the front slice is fixed-size (not a `Vec`) and is full.
    pub fn push_front(&mut self, elem: T) -> Option<usize> {
        self.slices.push(Self::BUCKET_FRONT, elem)
    }

    /// Appends an element to the back of deque.
    ///
    /// Returns None if the back slice is fixed-size (not a `Vec`) and is full.
    pub fn push_back_with(&mut self, f: impl FnOnce(usize) -> T) -> Option<usize> {
        self.slices.push_with(Self::BUCKET_BACK, f)
    }

    /// Appends an element to the back of the deque.
    ///
    /// Returns None if the back slice is fixed-size (not a `Vec`) and is full.
    pub fn push_back(&mut self, elem: T) -> Option<usize> {
        self.slices.push(Self::BUCKET_BACK, elem)
    }

    /// Gets an element from the deque by its index, as immutable.
    ///
    /// Returns `None` if the index did not refer to a valid element.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slices.get(index)
    }

    /// Gets an element from the deque by its index, as mutable.
    ///
    /// Returns `None` if the index did not refer to a valid element.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.slices.get_mut(index)
    }

    /// Removes an element from the deque, without changing it.
    ///
    /// Returns the removed element that could be freed if successful,
    /// returns `None` if the index did not refer to a valid element.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.slices.remove(index)
    }

    /// Returns the number of valid elements in the deque.
    pub fn len(&self) -> usize {
        self.iter().count()
    }

    /// Returns true if the deque contains no valid elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates every element in the front of this deque, as immutable.
    pub fn front_iter<'s>(&'s self) -> impl DoubleEndedIterator<Item = &'s T> {
        self.slices.bucket_iter(Self::BUCKET_FRONT).rev()
    }

    /// Iterates every element in the front of this deque, as mutable.
    pub fn front_iter_mut<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> {
        self.slices.bucket_iter_mut(Self::BUCKET_FRONT).rev()
    }

    /// Iterates every element in the back of this deque, as immutable.
    pub fn back_iter<'s>(&'s self) -> impl DoubleEndedIterator<Item = &'s T> {
        self.slices.bucket_iter(Self::BUCKET_BACK)
    }

    /// Iterates every element in the back of this deque, as mutable.
    pub fn back_iter_mut<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> {
        self.slices.bucket_iter_mut(Self::BUCKET_BACK)
    }

    /// Iterates every element of this deque, as immutable.
    pub fn iter<'s>(&'s self) -> impl DoubleEndedIterator<Item = &'s T> {
        // Compile ok because that borrow immutable on self.
        // self.front_iter().chain(self.back_iter())
        let [front, back] = self.slices.each_ref();
        front.rev().chain(back)
    }

    /// Iterates every element of this deque, as mutable.
    pub fn iter_mut<'s>(&'s mut self) -> impl DoubleEndedIterator<Item = &'s mut T> {
        // Compile error because that borrow twice mutable on self.
        // self.front_iter_mut().chain(self.back_iter_mut())
        let [front, back] = self.slices.each_mut();
        front.rev().chain(back)
    }
}

impl<'a, T> Index<usize> for SlotDeque<'a, T> {
    type Output = T;

    /// Returns a immutable reference to the value corresponding to the supplied index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not present in the `SlotDeque`.
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a, T> IndexMut<usize> for SlotDeque<'a, T> {
    /// Returns a mutable reference to the value corresponding to the supplied index.
    ///
    /// # Panics
    ///
    /// Panics if the index is not present in the `SlotDeque`.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{Slice, SlotDeque};
    use std::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[test]
    fn simple() {
        let front = Slice::Owned(vec![]);
        let back = Slice::Owned(vec![]);
        let mut slot_deque: SlotDeque<&str> = SlotDeque::new(front, back);

        let index = slot_deque.push_front("hello").unwrap();
        assert_eq!(slot_deque.get(index), Some(&"hello"));
        assert_eq!(slot_deque[index], "hello");
        let index = slot_deque.push_front("world").unwrap();
        assert_eq!(slot_deque.get(index), Some(&"world"));
        assert_eq!(slot_deque[index], "world");

        let index = slot_deque.push_back("jacky").unwrap();
        assert_eq!(slot_deque.get(index), Some(&"jacky"));
        assert_eq!(slot_deque[index], "jacky");
        let index = slot_deque.push_back("messy").unwrap();
        assert_eq!(slot_deque.get(index), Some(&"messy"));
        assert_eq!(slot_deque[index], "messy");

        assert_eq!(
            slot_deque.front_iter().collect::<Vec<&&str>>(),
            [&"world", &"hello"]
        );

        assert_eq!(
            slot_deque.back_iter().collect::<Vec<&&str>>(),
            [&"jacky", &"messy"]
        );

        assert_eq!(
            slot_deque.iter().collect::<Vec<&&str>>(),
            [&"world", &"hello", &"jacky", &"messy"]
        );
    }

    #[test]
    fn retained() {
        let front = Slice::Owned(vec![]);
        let back = Slice::Owned(vec![]);
        let mut slot_deque: SlotDeque<&str> = SlotDeque::new(front, back);

        let idx1 = slot_deque.push_front("hello").unwrap();
        let idx2 = slot_deque.push_front("world").unwrap();
        let idx3 = slot_deque.push_front("jacky").unwrap();

        assert_eq!(slot_deque.remove(idx1), Some("hello"));
        assert_eq!(slot_deque.get(idx1), None);
        // nonexistent index, panics
        // slot_deque[idx1];

        assert_eq!(slot_deque.remove(idx3), Some("jacky"));
        assert_eq!(slot_deque.get(idx3), None);

        assert_eq!(slot_deque.iter().count(), 1);

        assert_eq!(slot_deque.remove(idx2), Some("world"));
        assert_eq!(slot_deque.get(idx2), None);
        assert_eq!(slot_deque.iter().count(), 0);

        let idx4 = slot_deque.push_back("small").unwrap();
        assert_eq!(slot_deque.get(idx4), Some(&"small"));
        assert_eq!(slot_deque.iter().count(), 1);
    }

    #[test]
    fn complex() {
        let front = &mut [Default::default(); 1];
        let back = vec![Default::default(); 2];
        let mut slot_deque: SlotDeque<String> =
            SlotDeque::new(Slice::Borrowed(front), Slice::Owned(back));

        let index = slot_deque.push_front("hello".to_string()).unwrap();
        assert_eq!(slot_deque.get(index).map(|x| x.as_str()), Some("hello"));

        let elem = slot_deque.get_mut(index).unwrap();
        *elem = "world".to_string();
        assert_eq!(slot_deque.get(index).map(|x| x.as_str()), Some("world"));

        // the front slice is full, the length of the front slice borrowed is 1.
        assert_eq!(slot_deque.push_front("messy".to_string()), None);

        let index = slot_deque.push_back("hello".to_string()).unwrap();
        assert_eq!(slot_deque.get(index).map(|x| x.as_str()), Some("hello"));

        let index = slot_deque.push_back("jacky".to_string()).unwrap();
        assert_eq!(slot_deque.get(index).map(|x| x.as_str()), Some("jacky"));

        // Even though the back slice is 2 in length,
        // it can still be pushed because it is an owned vector.
        let noise_index = slot_deque
            .push_back_with(|index| index.to_string())
            .unwrap();
        let noise = noise_index.to_string();

        assert_eq!(
            slot_deque.iter().collect::<Vec<&String>>(),
            [&"world", &"hello", &"jacky", &noise.as_str()]
        );
    }
}
