use crate::microkernel::MatmulPartitioning;
use crate::microkernel::dense::cube::CubeMeet;
use crate::microkernel::{
    MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulRead, MatmulRowsKind,
    MatmulWorkItem,
};
use core::fmt::{Debug, Display};
use core::num::NonZeroUsize;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};
use cubek_tile::launch::Refusal;
use cubek_tile::launch::Vectorization;
use cubek_tile::{Axis, Instruction, Levels, Space};

/// Every row of a run of the output's columns against a packed weight, owned by one cube: its
/// planes, each a `P`, run on their own share of `K`'s scale blocks, and add their partial sums
/// into one accumulator in shared memory as the device lets them meet ([`CubeMeet`]): `K` shared
/// inside a cube, for a product whose runs of columns are too few to keep the device's loads in
/// flight with one plane each.
///
/// Its planes are the float product's: the software plane shuffle
/// ([`PlaneShuffleMicrokernel`](crate::microkernel::dense::plane::PlaneShuffleMicrokernel)),
/// whose units sum their share in registers, or the tensor-core plane fragment
/// ([`PlaneFragmentMicrokernel`](crate::microkernel::dense::plane::PlaneFragmentMicrokernel)),
/// which contracts fragments over windows of `K`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeShuffleMatmulQRhsMicrokernel<P> {
    pub plane: P,
    /// Planes sharing one run of columns, each on its own share of `K`.
    pub planes_along_k: NonZeroUsize,
    /// How its planes add their partial sums into one.
    pub meet: CubeMeet,
}

/// The settings of the plane a cube shuffle's planes run, and how many planes share a run of
/// columns: a target the planes' shares of `K` cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleMatmulQRhsMicrokernelSettings<S> {
    pub plane: S,
    pub planes_along_k: NonZeroUsize,
}

/// A float product's plane microkernel a packed cube shuffle's planes run: what the cube asks of
/// its plane, and the settings and candidates of a cube of it.
pub trait CubeShuffleQRhsPlane:
    Microkernel<WorkItem = MatmulWorkItem, Operand = MatmulOperand>
    + Debug
    + Clone
    + PartialEq
    + Display
{
    /// What a cube of this plane declines.
    const CUBE_DECLINE: &'static str;

    /// The cube settings worth trying on `hw`, the lead first.
    fn cube_settings(
        hw: &Hardware,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>>;

    /// What a cube of this plane admits.
    ///
    /// # Errors
    ///
    /// Why it declines `item`.
    fn cube_admits(item: &MatmulWorkItem) -> Result<(), Refusal>;

    /// The cubes of this plane `settings` names on `hw` for `item`, their planes meeting as `hw`'s
    /// cubes do ([`CubeMeet::of`]).
    fn cubes(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernel<Self>>;

    /// The levels of one plane's steps through `K`.
    fn step_levels(&self) -> Levels;

    /// The digit of `N` the plane's columns land on.
    fn column_axis(&self) -> Axis;

    /// Columns one plane covers.
    fn plane_columns(&self) -> usize;

    /// Values of `K` one step of the plane covers, `leaf` its step's box.
    fn step_depth(&self, leaf: &Space) -> usize;

    /// The plane width the plane is compiled for.
    fn plane_width(&self) -> usize;

    /// The instruction the plane contracts through.
    fn tile_instruction(&self) -> Instruction;

    /// How the plane's units read each input.
    fn reads(&self, item: &MatmulWorkItem) -> MatmulOperands<MatmulRead>;

    /// What the plane's accesses require of the widths, inside a cube placed by `partitioning`.
    fn vectorize_in_cube(
        &self,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    );

    /// The plane's work, inside a cube placed by `partitioning`.
    fn estimate_in_cube(
        &self,
        item: &MatmulWorkItem,
        partitioning: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    );
}

impl<S> CubeShuffleMatmulQRhsMicrokernelSettings<S> {
    /// `plane` on each of `planes_along_k` planes sharing a run.
    ///
    /// # Panics
    ///
    /// On no planes.
    pub const fn new(plane: S, planes_along_k: usize) -> Self {
        Self {
            plane,
            planes_along_k: NonZeroUsize::new(planes_along_k).unwrap(),
        }
    }
}

impl<S: Display> Display for CubeShuffleMatmulQRhsMicrokernelSettings<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_along_k={}", self.plane, self.planes_along_k)
    }
}

impl<P: CubeShuffleQRhsPlane> CubeShuffleMatmulQRhsMicrokernel<P> {
    /// The levels inside one cube's run, which a routine states the cubes around: its planes'
    /// steps through `K`, dealt to the planes in runs, so the planes need not divide the steps.
    pub fn tiling(&self) -> Levels {
        self.plane
            .step_levels()
            .planes_across(self.split(), self.planes_along_k.get())
    }

    /// The axis its planes split `K` along: the scale blocks.
    pub fn split(&self) -> Axis {
        blocked(K).0
    }

    /// Bytes of shared memory one cube holds over its run: the accumulator, one sum a cell of
    /// the cube's box of the output for each copy its planes meet in ([`CubeMeet::copies`]),
    /// which cubek sizes itself.
    pub fn shared_memory(&self, item: &MatmulWorkItem, partitioning: &MatmulPartitioning) -> usize {
        let cells = (partitioning.rows() * partitioning.columns()) as usize;
        cells * self.meet.copies(self.planes_along_k.get()) * item.sum.size()
    }
}

impl<P: CubeShuffleQRhsPlane> Microkernel for CubeShuffleMatmulQRhsMicrokernel<P> {
    type WorkItem = MatmulWorkItem;
    type Settings = CubeShuffleMatmulQRhsMicrokernelSettings<P::Settings>;
    type Operand = MatmulOperand;
    type Partitioning = MatmulPartitioning;

    const DECLINE: &'static str = P::CUBE_DECLINE;

    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        P::cube_settings(hw)
    }

    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        P::cube_admits(item)
    }

    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        P::cubes(hw, item, settings).into_iter()
    }

    /// Its planes read as the plane microkernel's.
    fn vectorize(
        &self,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        self.plane.vectorize_in_cube(partitioning, vectorization);
    }

    /// Its planes' work: the plane microkernel's.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        partitioning: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.plane
            .estimate_in_cube(item, partitioning, planes, estimate);
    }
}

/// Its plane microkernel, and how many planes share a run.
impl<P: Display> Display for CubeShuffleMatmulQRhsMicrokernel<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_along_k={}", self.plane, self.planes_along_k)
    }
}

impl<P: CubeShuffleQRhsPlane> MatmulMicrokernel for CubeShuffleMatmulQRhsMicrokernel<P> {
    /// Its planes hold every row.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    /// The packed weight is bound over the axes the product names, its words where they lie.
    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        MatmulOperands::all(MatmulOrder::Stated)
    }
}
