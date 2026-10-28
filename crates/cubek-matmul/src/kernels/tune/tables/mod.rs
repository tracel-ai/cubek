//! The matmul family's autotune tables: **few-rows**, **many-rows** and **NVFP4**, which
//! [`MatmulTable`](crate::kernels::select::selector::MatmulTable) names.
//!
//! Every candidate is a routine of this crate, at one of its settings. The selector's rule says
//! which table a product races on: a routine that does not stage and one that does are different
//! kernel bodies, so one table would be a list whose rows mostly decline. Float and packed products
//! race the same two tables, each candidate listed for one kind and declining the other before
//! compiling anything. Under each table sits a floor that serves any product: the unit routine on
//! every packed input decoded first, which asks nothing of the device.
//!
//! A table's groups run in the order its caller gives them, a **cutoff** rather than a tie-break:
//! the tuner measures the highest group whose candidates launch, and stops there. Inside a group a
//! round tries the candidates in the order their cuts suit the key, and the first one close enough
//! to the device's roofline ends it before the ones after it are compiled.
//!
//! A cached winner is an index into one table's candidates under its name, so each table has its
//! own tuner, named rather than taken from the module path.

mod base;
mod few_rows;
mod many_rows;
mod nvfp4;
#[cfg(test)]
mod tests;

pub use base::*;
