use crate::kernels::{MatmulRoutineProblem, PlaneDequantizeMatmulRoutine};
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` against a packed weight in two launches: the weight decoded into a float one
/// in the activation's element ([`PlaneDequantizeMatmulRoutine`]), then the float product's routine
/// `R` on it. The float weight is allocated and decoded on every launch, so an autotune that races
/// it measures what it costs; nothing is cached.
///
/// Over the float product's routine that asks nothing of the hardware, it serves every packed
/// weight in rows: the floor of a packed weight's autotune.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DequantizedMatmulRoutine<R> {
    pub dense: R,
}

/// The float routine's settings; the decode runs at its own lead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DequantizedMatmulRoutineSettings<S> {
    pub dense: S,
}

impl<S: core::fmt::Display> core::fmt::Display for DequantizedMatmulRoutineSettings<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "dequantized {}", self.dense)
    }
}

impl<R> DequantizedMatmulRoutine<R> {
    /// The float product the decoded weight runs: `problem`'s rows against a float `[k, n]` weight
    /// in the activation's element, laid with `k` contiguous where the packed words run along it.
    /// A packed problem states every row, its batches merged.
    fn dense_problem(problem: &MatmulRoutineProblem) -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(
            problem.stated_rows,
            problem.n,
            problem.k,
            problem.elements.get(MatmulOperand::Lhs),
            problem.layouts.rhs,
        )
    }
}

impl<R> Routine for DequantizedMatmulRoutine<R>
where
    R: Routine<Problem = MatmulRoutineProblem>,
{
    type Problem = MatmulRoutineProblem;
    type Microkernel = R::Microkernel;
    type Settings = DequantizedMatmulRoutineSettings<R::Settings>;
    /// The float routine's: the decode is prepared again from the problem wherever it is needed,
    /// at its lead, so it is no decision of this routine's.
    type Blueprint = R::Blueprint;
    type Partitioning = R::Partitioning;

    /// The float routine's settings, each after the decode at its lead.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        self.dense
            .settings(hw)
            .into_iter()
            .map(|dense| DequantizedMatmulRoutineSettings { dense })
            .collect()
    }

    /// The float routine's: the decode's launch makes its own accesses.
    fn vectorize(
        &self,
        microkernel: &R::Microkernel,
        partitioning: &R::Partitioning,
        vectorization: &mut Vectorization<<R::Microkernel as Microkernel>::Operand>,
    ) {
        self.dense
            .vectorize(microkernel, partitioning, vectorization);
    }

    /// The float routine's launch, which this kernel's grid is: the decode's is a launch of its
    /// own.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        self.dense
            .estimate(&Self::dense_problem(problem), blueprint, grid, estimate);
    }

    /// The float routine at `settings` over the float weight, where the decode at its lead
    /// prepares too. The grid is the float routine's.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        PlaneDequantizeMatmulRoutine.prepare(
            hw,
            problem,
            &PlaneDequantizeMatmulRoutine::SETTINGS,
        )?;
        self.dense
            .prepare(hw, &Self::dense_problem(problem), &settings.dense)
    }
}
