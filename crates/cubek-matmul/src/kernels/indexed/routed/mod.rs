//! The routed product: each lhs row against the slabs of a stacked rhs its routes name, launched as
//! the dense product of its batches of one row, the stack read through the routes. A caller that
//! measures it races the candidates listed for it ([`RoutedMatmulCandidates`]) under a key of its
//! own ([`RoutedMatmulKey`]).

mod candidate;
mod key;
mod settings;
mod tensors;

#[cfg(test)]
mod tests;

pub use candidate::*;
pub use key::*;
pub(crate) use settings::RoutedFloors;
pub use settings::RoutedMatmulCandidates;
pub use tensors::*;
