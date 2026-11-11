//! Every microkernel launched alone on the test device, one work item in one cube, against a
//! reference written for what it computes alone: a folder per problem, laid out as its source, a
//! file per microkernel.

mod launch;

#[cfg(feature = "attention")]
mod attention;
#[cfg(feature = "attention-backward")]
mod attention_backward;
#[cfg(feature = "interpolate")]
mod interpolate;
#[cfg(feature = "linear-recurrence")]
mod linear_recurrence;
#[cfg(feature = "matmul")]
mod matmul;
#[cfg(feature = "pool")]
mod pool;
#[cfg(feature = "random")]
mod random;
#[cfg(feature = "reduce")]
mod reduce;
#[cfg(feature = "rms-norm")]
mod rms_norm;
#[cfg(feature = "attention")]
mod split_keys_merge;
#[cfg(feature = "swiglu")]
mod swiglu;
