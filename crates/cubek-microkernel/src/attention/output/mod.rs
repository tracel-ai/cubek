//! The output of the queries over the keys and values, `softmax(query · keyᵀ · scale) · value`:
//! its microkernels, a folder per scope ([`unit`](mod@unit), [`plane`], [`cube`],
//! [`hypercube`]).

pub mod cube;
pub mod hypercube;
pub mod plane;
pub mod unit;
