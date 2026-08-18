//! Device-derived lowering knobs for the register leaf.

use cubecl::ir::DeviceProperties;

/// Everything about the register leaf's *lowering* that is a device decision rather than a
/// shape decision. Shape belongs to the [`Space`](crate::Space) and is resolved at the launcher;
/// this is only how the fixed shape gets inlined.
///
/// Fetched once per kernel via `comptime::device_properties()` and carried as a comptime value,
/// so a new knob is a field here rather than a parameter at every microkernel call site. The
/// leaf's kernels are compiled per device, so branching on the device costs nothing at runtime.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct RegisterTuning {
    /// Cells past which the `mr × nr` block stops fully inlining.
    pub unroll_block: usize,
    /// Split a checked edge block into an in-bounds unrolled body and a checked fallback.
    pub split_edge: bool,
}

impl RegisterTuning {
    pub fn new(props: &DeviceProperties) -> Self {
        // `num_cpu_cores` is the runtime's own statement that it is the CPU backend, and the only
        // property that states it. Plane size, tensor cores and SM count describe hardware a
        // backend may legitimately report oddly: a plane size is explicitly undefined at compile
        // time on the devices that vary it, so none of them may stand in for this test.
        match props.hardware.num_cpu_cores {
            Some(_) => Self {
                unroll_block: CPU_UNROLL_BLOCK,
                split_edge: true,
            },
            None => Self {
                unroll_block: SHADER_UNROLL_BLOCK,
                split_edge: false,
            },
        }
    }
}

/// Unrolling is what makes the accumulator block registers at all: a rolled loop indexes the block
/// at runtime, and a runtime-indexed local array is unconditionally memory on every backend. So the
/// cap is not a performance trade-off, it is a budget on emitted straight-line code, and it differs
/// by backend only because the compilers behind them differ.
///
/// The shader compilers walk a fully inlined block through recursive per-block passes that grow
/// superlinearly with its size; a few hundred cells is where that stops terminating in reasonable
/// time, which is what this value has always guarded. Each cell is also real instruction memory on
/// a GPU, competing with everything else resident.
const SHADER_UNROLL_BLOCK: usize = 64;

/// The CPU backend lowers through LLVM, whose block passes are not the ones that overflow above,
/// so the budget is set by emitted code size alone: `unroll_block` cells cost that many `fma`s per
/// K step plus twice that in the load/store prologue and epilogue, so this bounds one block at a
/// few thousand instructions. Nothing here is measured off a particular CPU; it is a property of
/// the compiler on that path, not of the part it targets.
const CPU_UNROLL_BLOCK: usize = 256;
