//! The cube matmul microkernels on one cube's box: planes sharing `K` and meeting in shared memory,
//! or contracting stages the cube's units fill.

use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Axis, Partitioning, SoftwareMatmul, StageStorage, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::matmul::cube::{
    CubeMeet, CubeShuffleMicrokernel, CubeShuffleMicrokernelSettings, CubeStagedMicrokernel,
    CubeStagedMicrokernelSettings, CubeStagedWalk, cube_shuffle_microkernel,
    cube_staged_microkernel,
};
use cubek_microkernel::matmul::{
    MatmulLayout, MatmulMicrokernel, MatmulOperands, MatmulRead, MatmulRows,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::Microkernel;

use super::{MatmulElements, MatmulLaunch, OneCube, WholeRows};

/// One cube shuffle microkernel ready to launch on one cube's box, every row held whole, `steps`
/// of the cube's step of `K` deep.
#[derive(Debug, Clone)]
pub struct CubeShuffleLaunch {
    pub microkernel: CubeShuffleMicrokernel,
    pub elements: MatmulElements,
    rows: WholeRows,
    one_cube: OneCube,
}

impl CubeShuffleLaunch {
    /// A cube shuffle sums a float rhs laid in columns, a decode step's weight.
    const RHS: MatmulLayout = MatmulLayout::Col;

    /// The cube shuffle microkernel `hw` runs at `settings` over `rows`.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &CubeShuffleMicrokernelSettings,
        rows: WholeRows,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item_of(MatmulRows::Whole(rows.0), Self::RHS);
        let microkernel = CubeShuffleMicrokernel::best(hw, &item, settings)?;
        let levels = microkernel.tiling().cubes(&[M, N]).build();
        Ok(Self {
            microkernel,
            elements,
            rows,
            one_cube: OneCube::new(levels, ComputeScope::Cube, steps),
        })
    }
}

impl MatmulLaunch for CubeShuffleLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn rhs_layout(&self) -> MatmulLayout {
        Self::RHS
    }

    fn shape(&self) -> [usize; 3] {
        self.one_cube.shape
    }

    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let microkernel = &self.microkernel;
        let launcher = self
            .one_cube
            .launcher(client, microkernel.plane.plane_width() as u32)?;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; its shared memory is the cube's one
        // accumulator, which its planes meet in.
        unsafe {
            cube_shuffle_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                lhs,
                rhs,
                out,
                launcher.partitioning_arg(),
                microkernel.plane.software_matmul(),
                microkernel.meet,
                microkernel.split(),
                microkernel.planes_along_k.get(),
                microkernel.plane.reads(
                    &self
                        .elements
                        .work_item_of(MatmulRows::Whole(self.rows.0), Self::RHS),
                ),
                [self.elements.sum, self.elements.out],
            );
        }
        Ok(())
    }
}

/// The one cube's planes taking `K` in turns and meeting in one accumulator.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
fn cube_shuffle_kernel<EA: Numeric, E: Numeric>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] block: Option<SoftwareMatmul>,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    for cube in &partitioning {
        cube_shuffle_microkernel::<EA>(
            &cube,
            &c.at(&cube),
            &a.at(&cube),
            &b.at(&cube),
            block,
            meet,
            split,
            planes,
            reads,
        );
    }
}

/// One cube staged microkernel ready to launch on one cube's box, `steps` stages of `K` deep: its
/// units fill the stages, its planes drain their fragments into the output.
#[derive(Debug, Clone)]
pub struct CubeStagedLaunch {
    pub microkernel: CubeStagedMicrokernel,
    pub elements: MatmulElements,
    storage: StageStorage,
    plane_width: u32,
    one_cube: OneCube,
}

impl CubeStagedLaunch {
    /// # Errors
    ///
    /// Why the device runs none, or cannot hold its stages in its shared memory.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &CubeStagedMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item();
        CubeStagedMicrokernel::admits(&item)?;
        // A stage the fill decodes binds its input as words, which this launch's plain bindings
        // do not: the first candidate staging its inputs as they lie.
        let microkernel = CubeStagedMicrokernel::candidates(hw, &item, settings)
            .find(|candidate| candidate.walk().decoded().is_none())
            .ok_or(Refusal::Declined {
                why: "every cube staged candidate decodes its stages, which bind words",
            })?;
        let contracted = microkernel.contracted(&item);
        let placed = OneCube::new(
            microkernel.tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        let resolved = microkernel.resolve(hw, &contracted, &placed.cube_space())?;
        if resolved.shared_memory > hw.shared_memory_budget() {
            return Err(Refusal::Declined {
                why: "the cube's stages hold more shared memory than the device has",
            });
        }
        let microkernel = resolved.microkernel;
        let one_cube = OneCube::new(
            microkernel.launch_tiling().cubes(&[M, N]).build(),
            ComputeScope::Cube,
            steps,
        );
        Ok(Self {
            microkernel,
            elements,
            storage: resolved.storage,
            plane_width: resolved.plane_width,
            one_cube,
        })
    }
}

impl MatmulLaunch for CubeStagedLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn shape(&self) -> [usize; 3] {
        self.one_cube.shape
    }

    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let launcher = self.one_cube.launcher(client, self.plane_width)?;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; its shared memory is its stages.
        unsafe {
            cube_staged_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                lhs,
                rhs,
                out,
                launcher.partitioning_arg(),
                self.microkernel.walk(),
                self.storage.clone(),
                [self.elements.sum, self.elements.input],
            );
        }
        Ok(())
    }
}

/// The one cube walking its box's `K` a stage at a time.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "each element is defined from its operand at launch, which the kernel is compiled for"
)]
#[cube(launch_unchecked)]
fn cube_staged_kernel<EA: Numeric, EL: Numeric>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    for cube in &partitioning {
        let b = b.at(&cube);
        cube_staged_microkernel::<EA>(
            cube.walk_within(&b),
            &a.at(&cube),
            &b,
            &c.at(&cube),
            walk,
            storage.clone(),
        );
    }
}
