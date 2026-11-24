//! What the Component Tests of the reduce microkernels share: the rows they reduce and what those
//! write ([`HostReduce`]), and how one is launched alone ([`MicrokernelLauncher`]).

#[macro_use]
mod cases;
mod checker;
mod host;
mod launch;
mod launcher;
mod rows;

pub(crate) use checker::*;
pub(crate) use host::*;
pub(crate) use launch::*;
pub(crate) use launcher::*;
pub(crate) use rows::*;
