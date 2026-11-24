//! What the Component Tests of the reduce microkernels share: the rows they reduce and what those
//! write ([`HostReduce`]), and how one is launched alone ([`MicrokernelLauncher`]).

#[macro_use]
mod cases;
mod both_halves;
mod checker;
mod cube_columns;
mod cube_line;
mod cube_partial;
mod cube_staged;
mod cube_strided;
mod host;
mod hypercube_atomic;
mod hypercube_last_arrival;
mod launch;
mod launcher;
mod plane_columns;
mod plane_line;
mod plane_shuffle;
mod rows;
mod unit_carrier;
mod unit_direct;
mod unit_group_column;
mod unit_lines;
mod unit_partial;
mod unit_partials;
mod unit_region;
mod unit_strided;
mod unit_walk;

pub(crate) use both_halves::*;
pub(crate) use checker::*;
pub(crate) use cube_columns::*;
pub(crate) use cube_line::*;
pub(crate) use cube_partial::*;
pub(crate) use cube_staged::*;
pub(crate) use cube_strided::*;
pub(crate) use host::*;
pub(crate) use hypercube_atomic::*;
pub(crate) use hypercube_last_arrival::*;
pub(crate) use launch::*;
pub(crate) use launcher::*;
pub(crate) use plane_columns::*;
pub(crate) use plane_line::*;
pub(crate) use plane_shuffle::*;
pub(crate) use rows::*;
pub(crate) use unit_carrier::*;
pub(crate) use unit_direct::*;
pub(crate) use unit_group_column::*;
pub(crate) use unit_lines::*;
pub(crate) use unit_partial::*;
pub(crate) use unit_partials::*;
pub(crate) use unit_region::*;
pub(crate) use unit_strided::*;
pub(crate) use unit_walk::*;
