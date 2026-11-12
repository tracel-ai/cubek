//! The hypercube matmul microkernel on one box whose `K` its cubes share: each cube walks its run
//! of the box's stages into one sum, and the box's last arrival adds the runs' parts up.

use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Grid, LastArrival, Refusal, VectorRequirement, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{
    Launcher, Partitioning, Space, StageStorage, TileSpec, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::matmul::cube::CubeStagedWalk;
use cubek_microkernel::matmul::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
    hypercube_staged_last_arrival_microkernel,
};
use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::microkernel::Microkernel;

use super::{MatmulElements, MatmulLaunch, OneCube};
use crate::launch::Lines;

/// The cubes sharing the boxes' stages of `K`, laid end to end. Over [`BOXES`] boxes of one stage
/// each, two cubes hold a box whole and two walk nothing; of three, every run but one crosses from
/// one box into the next; of five, runs of three cut both boxes unevenly.
const CUBES: usize = 4;

/// The boxes of the output, side by side along `n`.
const BOXES: usize = 2;

/// One hypercube staged microkernel ready to launch on [`BOXES`] boxes, `steps` stages of `K` deep,
/// their stages shared by [`CUBES`] cubes, its operands and the boxes' slots `widths` values a line.
#[derive(Debug, Clone)]
pub struct HypercubeStagedLaunch {
    pub microkernel: HypercubeStagedMicrokernel,
    pub elements: MatmulElements,
    widths: MatmulOperands<usize>,
    storage: StageStorage,
    plane_width: u32,
    shape: [usize; 3],
    partitioning: Partitioning,
}

impl HypercubeStagedLaunch {
    /// # Errors
    ///
    /// Why the device runs none: no cube staged microkernel at `settings`, stages past its shared
    /// memory, or cubes whose stores do not reach one another within a launch.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &HypercubeStagedMicrokernelSettings,
        lines: Lines,
        steps: usize,
    ) -> Result<Self, Refusal> {
        if !hw.properties.features.device_memory_scope {
            return Err(Refusal::Declined {
                why: "the last arrival hands one cube's stores to another within a launch",
            });
        }
        let item = elements.work_item();
        HypercubeStagedMicrokernel::admits(&item)?;
        // A stage the fill decodes binds its input as words, which this launch's plain bindings
        // do not: the first candidate staging its inputs as they lie, as the cube's launch takes.
        let microkernel = HypercubeStagedMicrokernel::candidates(hw, &item, settings)
            .find(|candidate| candidate.cube.walk().decoded().is_none())
            .ok_or(Refusal::Declined {
                why: "every hypercube staged candidate decodes its stages, which bind words",
            })?;
        let one_box = OneCube::new(
            microkernel.cube.tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        let resolved = microkernel.cube.resolve(
            hw,
            &microkernel.cube.contracted(&item),
            &one_box.cube_space(),
        )?;
        if resolved.shared_memory > hw.shared_memory_budget() {
            return Err(Refusal::Declined {
                why: "the cube's stages hold more shared memory than the device has",
            });
        }
        let microkernel = HypercubeStagedMicrokernel {
            cube: resolved.microkernel,
        };
        let one_box = OneCube::new(
            microkernel.cube.launch_tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        // The slots hold the sum's element at the output's width, as the routine binds them.
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&one_box.cube_space(), &mut vectorization);
        vectorization.require(VectorRequirement::Access {
            operand: MatmulOperand::Out,
            elements: hw.vector_ceiling(elements.sum),
        });
        let [m, n, k] = one_box.shape;
        let shape = [m, n * BOXES, k];
        let partitioning = Partitioning::new(
            Space::new(&[(M, shape[0]), (N, shape[1]), (K, shape[2])]),
            microkernel
                .cube
                .launch_tiling()
                .cubes(&[M, N])
                .shared_by(CUBES)
                .build(),
        );
        Ok(Self {
            microkernel,
            elements,
            widths: elements.widths(hw, lines, &vectorization),
            storage: resolved.storage,
            plane_width: resolved.plane_width,
            shape,
            partitioning,
        })
    }
}

impl MatmulLaunch for HypercubeStagedLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Twice on the same arrival counters, first on zeros into an output of its own, then on
    /// `lhs` and `rhs` into `out`: the boxes' last arrivals must leave the counters at zero, or the
    /// second launch hands its parts on early or never.
    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let arrivals = TensorHandle::zeros(
            client,
            [self.partitioning.arrival_counters()],
            u32::elem_type_native(),
        );
        let [m, n, k] = self.shape;
        let rhs_dims = match self.rhs_layout() {
            MatmulLayout::Col => [n, k],
            _ => [k, n],
        };
        let [elements, input] = [self.elements.out, self.elements.input];
        let scratch = [([m, k], input), (rhs_dims, input), ([m, n], elements)]
            .map(|(dims, element)| TensorHandle::zeros(client, dims, element).into_arg());
        self.dispatch(client, scratch, &arrivals)?;
        self.dispatch(client, [lhs, rhs, out], &arrivals)
    }
}

impl HypercubeStagedLaunch {
    /// The kernel on `operands` (`lhs`, `rhs`, `out`), the boxes counting in at `arrivals`.
    fn dispatch(
        &self,
        client: &Client,
        operands: [TensorArg; 3],
        arrivals: &TensorHandle,
    ) -> Result<(), Refusal> {
        let partitioning = &self.partitioning;
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let launcher = Launcher::new(client, partitioning.clone(), partitioning.space(), grid)?;
        let [lhs, rhs, out] = self
            .elements
            .bound(operands, self.rhs_layout(), self.widths);
        let slots = TensorHandle::empty(
            client,
            partitioning.arrival_slots(&[M, N]),
            self.elements.sum,
        );
        let slots = TileTensorLaunch::new(
            slots.into_arg(),
            self.elements.sum,
            self.widths.out,
            TileSpec::direct(&[M, N]),
        );
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // kernel indexes no buffer but through its tiles and the arrival counter of each box, which
        // holds zero; its shared memory is its stages.
        unsafe {
            hypercube_staged_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                self.widths.out,
                lhs,
                rhs,
                out,
                slots,
                arrivals.clone().into_arg().into_buffer_arg(),
                launcher.partitioning_arg(),
                self.microkernel.cube.walk(),
                self.storage.clone(),
                [self.elements.sum, self.elements.input],
            );
        }
        Ok(())
    }
}

/// Every cube walking its run of the box's stages, handing its sum on through the box's last
/// arrival.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn hypercube_staged_kernel<EA: Numeric, EL: Numeric, VO: Size>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    slots: &TileTensor<'_>,
    arrivals: &[Atomic<u32>],
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    let stages = comptime!(partitioning.levels()[1].clone());
    let portion = partitioning.walk().portion(stages.clone());
    for i in 0..portion.touched() {
        let tile = portion.region(i);
        let (from, steps) = portion.steps(i);
        let merge: LastArrival<'_> = slots.last_arrival::<VO>(&partitioning, &portion, i, arrivals);
        hypercube_staged_last_arrival_microkernel::<EA>(
            tile.over(&stages).range(from, steps),
            &a.at(&tile),
            &b.at(&tile),
            merge,
            &c,
            walk,
            storage.clone(),
        );
    }
}
