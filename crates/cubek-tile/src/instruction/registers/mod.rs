//! Register-resident loop nests: the schedules between a leaf dispatcher and the instructions it
//! issues.
//!
//! Not instructions ([`instruction`](crate::instruction) holds those, one hardware op each) and
//! not verbs (`ops` walks levels and stages memory). Each nest here walks a comptime-shaped loop
//! over register values, one [`Monoid`](crate::Monoid) or `fma` per step, only at a final tile.
//!
//! - [`horizontal`]: 1-D folds over a vector's lanes or an array's elements.
//! - [`block`]: the `mr × nr` register block: seed it, contract into it, commit it back.
//!
//! - [`contract`]: `acc += lhs · rhs` into a memory accumulator, 2-D and N-D.
//! - [`reduce`]: fold one operand's contracted axes into an accumulator cell.

pub mod block;
pub mod contract;
pub mod horizontal;
pub mod reduce;
