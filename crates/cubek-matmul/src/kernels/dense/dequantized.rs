use crate::kernels::{MatmulRoutineProblem, PlaneDequantizeMatmulRoutine};
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// `out = lhs · rhs` of packed inputs in launches of their own: each packed input decoded into a
/// float one ([`PlaneDequantizeMatmulRoutine`]), then the float product's routine `R` on them. The
/// float inputs are allocated and decoded on every launch, so an autotune that races it measures
/// what it costs; nothing is cached.
///
/// Over the float product's routine that asks nothing of the hardware, it serves every packed
/// input in rows: the floor of a packed product's autotune.
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
    /// The float product the decoded inputs run: `problem`'s rows against `n` columns, `k` deep, at
    /// the element the packed inputs decode into, each decoded input laid as the decode writes it.
    /// A packed problem states every row, its batches merged.
    fn dense_problem(problem: &MatmulRoutineProblem) -> MatmulRoutineProblem {
        let element = MatmulOperand::INPUTS
            .into_iter()
            .find(|&input| problem.packings.get(input).is_none())
            .map_or(problem.elements.out, |input| problem.elements.get(input));
        MatmulRoutineProblem::dense(problem.stated_rows, problem.n, problem.k, element)
            .laid(MatmulOperand::Lhs, problem.layouts.lhs)
            .laid(MatmulOperand::Rhs, problem.layouts.rhs)
    }

    /// Each packed input of `problem` alone, as the decode places it: the `[k, n]` input of a
    /// product of one row, a packed `[m, k]` input's rows as its `N`.
    fn decodes(problem: &MatmulRoutineProblem) -> impl Iterator<Item = MatmulRoutineProblem> + '_ {
        let element = Self::dense_problem(problem).elements.out;
        MatmulOperand::INPUTS.into_iter().filter_map(move |input| {
            let packing = problem.packings.get(input)?;
            let across = match input {
                MatmulOperand::Lhs => problem.stated_rows,
                _ => problem.n,
            };
            Some(MatmulRoutineProblem::packed(
                1,
                across,
                problem.k,
                element,
                MatmulOperands {
                    rhs: Some(packing),
                    ..MatmulOperands::all(None)
                },
            ))
        })
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

    /// The float routine at `settings` over the decoded inputs, where the decode of each packed
    /// input at its lead prepares too. The grid is the float routine's.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        for decode in Self::decodes(problem) {
            PlaneDequantizeMatmulRoutine.prepare(
                hw,
                &decode,
                &PlaneDequantizeMatmulRoutine::SETTINGS,
            )?;
        }
        self.dense
            .prepare(hw, &Self::dense_problem(problem), &settings.dense)
    }
}
