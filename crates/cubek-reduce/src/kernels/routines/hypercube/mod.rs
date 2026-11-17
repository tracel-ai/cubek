//! The hypercube routines: a row split across cubes, the parts meeting either in the last cube to
//! arrive or by atomic adds into the output.

mod atomic;
mod last_arrival;

pub use atomic::*;
pub use last_arrival::*;
