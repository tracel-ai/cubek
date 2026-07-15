//! Launchable routines over the tile engine: a host entry per algorithm that adapts a
//! caller's [`Partitioner`](crate::Partitioner) plan to the device (widths, geometry,
//! argument wiring) and launches the engine's DSL kernel — the layer between a backend's
//! selector and the `#[cube]` code, so clients own *no* kernels.
//!
//! Per the blueprint/routine split (`GUIDE.md`): the partitioner and the operands' comptime
//! wiring are the blueprint (a different plan is a different kernel); the routine derives
//! everything else — vector widths from the bindings and device, cube geometry from the
//! concrete space — and validates host-side what the engine can only refuse at expand time.

mod matmul;

pub use matmul::*;
