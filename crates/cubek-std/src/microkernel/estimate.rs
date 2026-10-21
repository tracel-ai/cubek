use cubecl::tune::Work;

/// A kernel's performance, estimated: what the kernel a routine prepared asks of a device,
/// counted from its own walk, before any device is known.
///
/// One vocabulary for every routine of every family, so kernels of different routines are
/// compared on the same quantities. It starts from the kernel's grid (its `From<&Grid>`), and
/// each part has one owner, which sets it whole: the problem its [`demand`](Self::demand), the
/// component placing the stage's reads its [`staging`](Self::staging), the microkernel holding
/// the work in its units its [`work`](Self::work). A routine sets what it owns and hands the
/// estimate to its microkernel, which does the same, as the kernel hands down its walk.
///
/// Counts only: every rate, latency and capacity is the device's. A part left unset is a routine
/// that forgot its own.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelEstimate {
    /// Cubes the grid spawns; `None` where the count is only known at launch.
    pub cubes: Option<u64>,
    /// What one cube holds of its worker.
    pub cube: CubeFootprint,
    /// What the problem asks for, whichever routine runs it: its arithmetic, and the bytes its
    /// operands hold, what a kernel reading each byte once would move. The same [`Work`] an
    /// autotune table's roofline is drawn from.
    pub demand: Option<Work>,
    pub staging: Option<CubeStaging>,
    pub work: Option<UnitWork>,
}

/// What one cube holds of its worker, from the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeFootprint {
    pub units: u32,
    /// Bytes of shared memory.
    pub shared_memory: usize,
}

/// How one cube walks its work through global memory: in stages, each loaded while the stages
/// before it compute, then drained.
#[derive(Debug, Clone, PartialEq)]
pub struct CubeStaging {
    /// Arithmetic operations one cube performs, the cells it covers past the problem's edge
    /// included.
    pub operations: f64,
    /// What one cube reads over its walk.
    pub reads: Vec<GlobalStream>,
    /// What one cube writes once its walk ends.
    pub writes: Vec<GlobalStream>,
    /// Stages one cube walks.
    pub stages: u64,
    /// Stages one cube holds at once: one where it loads a stage only once the last is computed,
    /// more where it loads ahead.
    pub stages_in_flight: u32,
    /// Atomic adds into shared memory one cube issues as it drains.
    pub drain_atomics: u64,
}

/// Bytes one cube moves through one operand, and the stretch of them a plane touches contiguously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalStream {
    pub bytes: u64,
    /// Consecutive bytes a plane touches over the cube's walk: what a memory transaction is filled
    /// from, a line a later stage finishes reading counted as filled.
    pub run_bytes: usize,
}

/// What one unit holds of the work and how it computes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitWork {
    /// The instructions it runs on, which decide the rate.
    pub instruction: ArithmeticInstruction,
    /// 32-bit registers it keeps its work in: a lower bound, counting what the microkernel keeps
    /// live and not what the compiler adds around it.
    pub registers: u32,
}

impl UnitWork {
    /// This work and `other` held by one unit at once, as a microkernel built from two runs them
    /// one after the other within each step: their registers added, on the instruction both run.
    pub fn beside(self, other: UnitWork) -> UnitWork {
        debug_assert_eq!(
            self.instruction, other.instruction,
            "work held beside other work runs on the same instruction"
        );
        UnitWork {
            instruction: self.instruction,
            registers: self.registers + other.registers,
        }
    }
}

/// Which of the device's arithmetic rates a kernel runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmeticInstruction {
    /// Vector fused multiply-adds, one unit at a time.
    Vector,
    /// A matrix instruction a plane issues together.
    Matrix,
}
