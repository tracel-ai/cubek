//! Every microkernel launched alone on the test device, one work item in one cube, against a
//! reference written for what it computes alone: a folder per problem, laid out as its source, a
//! file per microkernel.

#[cfg(feature = "attention")]
mod attention;
#[cfg(feature = "matmul")]
mod matmul;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "reduce")]
mod reduce;
#[cfg(feature = "swiglu")]
mod swiglu;
