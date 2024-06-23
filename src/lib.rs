#![no_std]

//! A library that provides a way to logically own objects, whether or not
//! heap allocation is available.

#[cfg(feature = "std")]
extern crate std;
#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

mod object;
mod slice;
mod slotvec;
mod slotbkvec;
mod slotdeque;
mod slotmap;
#[cfg(feature = "map")]
mod map;

pub use object::Managed;
pub use slice::ManagedSlice;
pub use slotvec::SlotVec;
pub use slotbkvec::SlotBkVec;
pub use slotdeque::SlotDeque;
pub use slotmap::{
    Key as SlotKey,
    Slot as SlotIndex,
    SlotMap,
};
#[cfg(feature = "map")]
pub use map::{ManagedMap,
              Iter as ManagedMapIter,
              IterMut as ManagedMapIterMut};
