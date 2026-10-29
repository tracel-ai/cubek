//! The matmul family's autotunes: **few-rows**, **many-rows** and **block-scaled**, which
//! [`MatmulAutotune`] names.
//!
//! Every candidate is a routine of this crate, at one of its settings. The selector's rule says
//! which autotune a product races on: a routine that does not stage and one that does are different
//! kernel bodies, so one autotune would be a list whose rows mostly decline. Float and packed products
//! race the same two autotunes, each candidate listed for one kind and declining the other before
//! compiling anything. Under each autotune sits a floor that serves any product: the unit routine on
//! every packed input decoded first, which asks nothing of the device.
//!
//! An autotune's groups run in the order its caller gives them, a **cutoff** rather than a tie-break:
//! the tuner measures the highest group whose candidates launch, and stops there. Inside a group a
//! round tries the candidates in the order their cuts suit the key, and the first one close enough
//! to the device's roofline ends it before the ones after it are compiled.
//!
//! A cached winner is an index into one autotune's candidates under its name, so each autotune has its
//! own tuner, named rather than taken from the module path.
//!
//! # The autotunes, and the line between them
//!
//! A product races on a **few-rows** autotune or a **many-rows** one, the [`MatmulAutotune`]s, whatever its
//! inputs. The few-rows and many-rows autotunes share no candidate but the unit routine, their floor:
//! the two regimes are different kernel bodies, so a single autotune would be one whose rows mostly
//! decline, paying a compile each to say so. Within an autotune a
//! float product and a packed one race the same list: each candidate is listed for one kind of
//! product and declines the other before compiling anything, so a key races what its inputs are
//! listed for. A packed product races the settings worth trying on its packed inputs, and the
//! float routines on them decoded first.
//!
//! # The block-scaled autotune
//!
//! A third autotune, [`MatmulAutotune::BlockScaled`], is reached only by naming
//! [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores). Its kernel
//! multiplies two packed inputs as they are stored on the device's block-scaled instruction of
//! their format: NVFP4's 4-bit by 4-bit, or any other the device has. Autotune never sends a
//! product there: a packed product under a policy goes where every other packed product goes. The
//! autotune races only that kernel's settings, which all give the same numbers.
//!
//! The line is [`FLOAT_FEW_ROWS`] for a float product and [`QUANTIZED_FEW_ROWS`] for a packed
//! one, each a **rule** rather than a measurement: a staged kernel can be competitive on
//! time at a handful of rows while reading a whole input through a tiling nobody needed, so
//! which autotune a row count belongs to is decided by what the kernels do. A rule tuned to one
//! device's sweep instead is a number every other device has to live with.

mod base;
mod block_scaled;
mod few_rows;
mod many_rows;
#[cfg(test)]
mod tests;

pub use base::*;
