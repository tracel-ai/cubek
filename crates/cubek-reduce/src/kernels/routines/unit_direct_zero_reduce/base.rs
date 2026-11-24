use super::kernel::unit_direct_zero_reduce_kernel;
use crate::kernels::{ReduceBlueprint, ReduceRoutineProblem, ReduceTensors, unit_launch};
use cubecl::tile::launch::{Decline, VectorRequirement, Vectorization};
use cubek_microkernel::zero::ZeroOperand;
use cubek_microkernel::zero::unit::{UnitDirectZeroMicrokernel, UnitDirectZeroMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// A reduction's output zeroed, a unit per box of rows' cells: what a reduction that adds its
/// parts into the output starts from ([`HypercubeAtomicReduceRoutine`](crate::kernels::HypercubeAtomicReduceRoutine)).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectZeroReduceRoutine;

/// The zero microkernel's setting, and how many planes each cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroReduceSettings {
    pub microkernel: UnitDirectZeroMicrokernelSettings,
    pub plane_count: usize,
}

impl core::fmt::Display for UnitDirectZeroReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.plane_count)
    }
}

pub type UnitDirectZeroReduceBlueprint = ReduceBlueprint<UnitDirectZeroMicrokernel>;

impl Routine for UnitDirectZeroReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitDirectZeroMicrokernel;
    type Settings = UnitDirectZeroReduceSettings;
    type Blueprint = UnitDirectZeroReduceBlueprint;
    type Partitioning = ();

    /// Cubes of four planes: a zeroing is one write a unit, which no setting speeds up.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        UnitDirectZeroMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| UnitDirectZeroReduceSettings {
                microkernel,
                plane_count: 4,
            })
            .collect()
    }

    fn vectorize(
        microkernel: &UnitDirectZeroMicrokernel,
        _partitioning: &(),
        vectorization: &mut Vectorization<ZeroOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Placed as the unit reduce places its owners, its lines laid as a reduction lays them, as wide
    /// as the output alone takes: each unit zeroes the cells of its rows.
    fn prepare(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        let microkernel = UnitDirectZeroMicrokernel::best(hw, &(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        Self::vectorize(&microkernel, &(), &mut vectorization);
        let [_, alignment] = problem.elem_alignments;
        vectorization.require(VectorRequirement::Aligned {
            operand: ZeroOperand::Out,
            alignment,
        });
        vectorization.require(VectorRequirement::Access {
            operand: ZeroOperand::Out,
            elements: problem.box_row_count_at_lane_count(problem.line_stays_in()),
        });
        let ceiling = hw.vector_ceiling(problem.elements.out);
        let lane_count = vectorization.width(ZeroOperand::Out, |_| ceiling);
        let (partitioning, grid) = unit_launch(hw, problem, lane_count, settings.plane_count)?;
        Ok((
            ReduceBlueprint {
                microkernel,
                partitioning,
                lane_count,
            },
            grid,
        ))
    }

    fn estimate(
        _problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&(), &(), grid.planes(), estimate);
    }
}

impl UnitDirectZeroReduceRoutine {
    /// The output of `tensors` zeroed on `grid`, as `blueprint` states it: their input is never read.
    ///
    /// # Errors
    ///
    /// As [`ReduceTensors::bind_out`].
    pub fn launch<In>(
        tensors: &ReduceTensors<In>,
        blueprint: UnitDirectZeroReduceBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
        let (launcher, out) = tensors.bind_out(&blueprint, &grid)?;
        // SAFETY: the output is bound through the launcher, which masks the writes of a tile the
        // problem's edge cuts short, and the kernel indexes no buffer but through its tile; it holds
        // no shared memory.
        unsafe {
            unit_direct_zero_reduce_kernel::launch_unchecked(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                out.arg(),
                launcher.partitioning_arg(),
                blueprint.microkernel,
                tensors.problem().elements.out,
            )
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{Reduction, unwritten_output, written_output, zeros};
    use cubecl::client::Client;
    use cubecl::zspace::Shape;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_test_utils::{HostData, HostDataVec, ScopeChecker, Written};

    prepares_lacking_every_feature!(UnitDirectZeroReduceRoutine);
    names_its_setting!(UnitDirectZeroReduceRoutine, "default planes=4");
    estimates_its_microkernel!(UnitDirectZeroReduceRoutine, |_: &ReduceRoutineProblem| ());

    /// Launches the zero routine alone and checks that a reduction's output, which starts away from
    /// zero, holds only zeros after it.
    struct ZeroScopeChecker;

    impl ScopeChecker for ZeroScopeChecker {
        type Case = Reduction;

        fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
            let out = unwritten_output(client, reduction);
            let tensors = ReduceTensors::new(
                client,
                zeros(client, &reduction.shape),
                out.clone().binding(),
                reduction.axis,
                reduction.operation,
                reduction.elements(),
            );
            let hw = Hardware::from_client(client);
            let (blueprint, grid) =
                UnitDirectZeroReduceRoutine::prepare_first_served(&hw, tensors.problem())?;
            UnitDirectZeroReduceRoutine::launch(&tensors, blueprint, grid)?;
            Ok(vec![written_output(out, reduction)])
        }

        /// Zeros, the output's shape.
        fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
            let out_shape = reduction.out_shape();
            vec![HostData::row_major(
                HostDataVec::F32(vec![0.0; out_shape.iter().product()]),
                Shape::from(out_shape),
            )]
        }

        fn tolerance(&self, _reduction: &Reduction) -> f32 {
            0.0
        }
    }

    /// The output of `[5, 300, 7]` summed along its middle dim, 35 cells, all zeroed.
    #[test]
    fn zeroes_every_cell_of_the_output() {
        ZeroScopeChecker.check([Reduction {
            shape: vec![5, 300, 7],
            axis: 1,
            operation: ReduceOperation::Sum,
        }]);
    }
}
