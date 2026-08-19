//! The hardware contraction leaves: CMMA/WMMA fragments and manual MMA, plus the dispatcher that
//! picks between them and the software microkernel in
//! [`microkernel::contract`](crate::microkernel::contract).

mod base;
mod cmma;
mod manual;

pub use base::*;
