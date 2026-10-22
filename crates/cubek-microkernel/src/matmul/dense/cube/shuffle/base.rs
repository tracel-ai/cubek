use crate::matmul::MatmulPartitioning;
use crate::matmul::dense::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings, PlaneShuffleMicrokernel,
    PlaneShuffleMicrokernelSettings,
};
use crate::matmul::{
    MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulRead,
    MatmulRowsKind, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Instruction, Levels, Space};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Every row of a run of the output's columns, owned by one cube: its planes each run their plane
/// microkernel's step on their own slice of every step of `K`, and add their partial sums into
/// one accumulator in shared memory, as the device lets them meet ([`CubeMeet`]).
///
/// What a product whose columns give too few runs to fill the device with planes wants: each run
/// gets a cube's planes rather than one. Its planes are a [`CubeShufflePlane`]: the software plane
/// shuffle, whose units sum their slice in registers, or, against a packed weight, the
/// tensor-core plane fragment, which contracts fragments over windows of `K`, decoding the weight
/// as each fragment loads.
///
/// It sums a float weight in cols only: a float row weight's step lies along `N`, so each of a
/// unit's cells is a vector of the weight, and the shared accumulator the planes add their partial
/// sums into takes one element a cell, which cubek does not drain a wider vector into. A plane
/// shuffle reads a float row weight whole.
///
/// Against a packed weight its planes split `K`'s scale blocks, each taking whole turns of its
/// plane's units, or whole windows of its fragments, so no two planes share a block's scale. A
/// unit's cell is then a read of the weight's values along `K` whichever way its words lie, folded
/// on drain as a col weight's is, so a packed weight in rows sums too.
///
/// Atomic adds land in no fixed order, and `f32` addition does not associate, so two launches on
/// the same operands may differ in the last bits of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleMicrokernel {
    /// What each of its planes runs on its slice of a step.
    pub plane: CubeShufflePlane,
    /// Planes sharing one run of columns, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
    /// How its planes add their partial sums into one.
    pub meet: CubeMeet,
}

/// The plane microkernel a cube shuffle's planes run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeShufflePlane {
    /// The software plane: its units take `K` in turns, each summing its share in registers.
    Shuffle(PlaneShuffleMicrokernel),
    /// The tensor-core plane, against a packed weight: it contracts fragments over windows of
    /// `K`, decoding the weight as each fragment loads.
    Fragment(PlaneFragmentMicrokernel),
}

/// How a cube's planes add their partial sums into one accumulator in shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeMeet {
    /// Every plane adds its partial atomically, in no fixed order: where the device adds an `f32`
    /// atomically in shared memory.
    Atomic,
    /// Every plane lands its partial in a copy of the box of its own at once, the cube
    /// synchronizes once, and the output adds the copies in plane order: wherever else.
    InSlots,
}

impl CubeMeet {
    /// How `hw`'s cubes meet: atomically where it adds an `f32` atomically in shared memory.
    pub fn of(hw: &Hardware) -> Self {
        match hw.shared_f32_atomic_add() {
            true => CubeMeet::Atomic,
            false => CubeMeet::InSlots,
        }
    }

    /// Copies of the box the planes meeting this way hold in shared memory, of `planes`.
    pub fn copies(self, planes: usize) -> usize {
        match self {
            CubeMeet::Atomic => 1,
            CubeMeet::InSlots => planes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleMicrokernelSettings {
    pub plane: CubeShufflePlaneSettings,
    /// Planes sharing one run of columns, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
}

/// The settings of the plane microkernel a cube shuffle's planes run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeShufflePlaneSettings {
    Shuffle(PlaneShuffleMicrokernelSettings),
    Fragment(PlaneFragmentMicrokernelSettings),
}

impl CubeShuffleMicrokernelSettings {
    /// The plane microkernel's default at one read deep, four planes a cube. The cube's step of
    /// `K` is its planes' step times their count, and at the plane default's four reads it would
    /// be 8192 `f16` elements on a wave64 plane, which divides no decode projection's `K`; at one
    /// it is 1024 on wave32 and 2048 on wave64.
    pub const DEFAULT: Self = Self::new(
        CubeShufflePlaneSettings::Shuffle(PlaneShuffleMicrokernelSettings::ONE_READ),
        4,
    );

    /// The lead against a packed weight: software planes one read a step, four planes a cube.
    pub const PACKED: Self = Self::new(
        CubeShufflePlaneSettings::Shuffle(PlaneShuffleMicrokernelSettings::stepping(1, None)),
        4,
    );

    /// `plane` on each of `planes_along_k` planes sharing a run.
    ///
    /// # Panics
    ///
    /// On no planes.
    pub const fn new(plane: CubeShufflePlaneSettings, planes_along_k: usize) -> Self {
        Self {
            plane,
            planes_along_k: NonZeroUsize::new(planes_along_k).unwrap(),
        }
    }
}

impl core::fmt::Display for CubeShuffleMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_along_k={}", self.plane, self.planes_along_k)
    }
}

impl core::fmt::Display for CubeShufflePlaneSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CubeShufflePlaneSettings::Shuffle(settings) => settings.fmt(f),
            CubeShufflePlaneSettings::Fragment(settings) => settings.fmt(f),
        }
    }
}

impl CubeShuffleMicrokernel {
    /// What a row weight is declined with: its step lies along `N`, so each of a unit's cells is
    /// a vector of the weight, and the shared accumulator the planes add their partial sums into
    /// takes one element a cell, which cubek does not drain a wider vector into.
    pub const ROW_RHS_UNSUPPORTED: Refusal = Refusal::Declined {
        why: "the cube shuffle sums a float weight in cols only: a row weight's vectors along `N` do not drain into the shared accumulator's one-element cells",
    };

    /// What a cube of tensor-core planes declines.
    pub const FRAGMENT_DECLINE: &'static str = "the cube shuffle's plane fragment needs the device's 16-bit 16x8x16 manual mma on a 32-unit plane, a packed weight, columns whole fragments of eight, and K whole windows each unit reads in whole loads";

    /// The fewest planes a cube holds: a run one plane owns is the plane routine's.
    const FEWEST_PLANES: usize = 2;

    /// Steps of `K` a tensor-core plane's window holds: eight of the decoding instruction's, so
    /// each unit's run of it is thirty-two values, sixteen bytes of codes, and a tile of tile
    /// order's depth.
    const WINDOW_STEPS: usize = 8;

    /// The settings worth racing against a packed weight, the lead first. The software planes:
    /// the lead, then the counts of planes along `K` measured best, or within five percent of it,
    /// on the Qwen3-4B projections at one, four and eight rows on an RTX PRO 6000, the weight in
    /// rows and in tile order. A cube's planes split `K` for the few long columns a plane alone
    /// would leave the device idle on, and its units take more columns and fewer rows as the rows
    /// grow. Sixteen planes along `K` split the narrow projections of a 4B model at one row best
    /// on an M2 Pro.
    ///
    /// Then the tensor-core planes, on four, eight and sixteen planes, which split the Qwen3-4B
    /// projections best from one to sixteen rows on the RTX PRO 6000: one fragment a plane nearly
    /// everywhere, two at eight planes beside it. None on a device that does not offer the plane's
    /// decoding instruction: there they would only be tried and declined.
    pub fn packed_settings(hw: &Hardware) -> Vec<CubeShuffleMicrokernelSettings> {
        use CubeShufflePlaneSettings::{Fragment, Shuffle};
        use PlaneShuffleMicrokernelSettings as Plane;
        let cube = CubeShuffleMicrokernelSettings::new;
        let fragments = |along_n| {
            Fragment(PlaneFragmentMicrokernelSettings::windowed(
                along_n,
                Self::WINDOW_STEPS,
            ))
        };
        let mut settings = vec![
            CubeShuffleMicrokernelSettings::PACKED,
            cube(Shuffle(Plane::stepping(1, None)), 8),
            cube(Shuffle(Plane::stepping(2, Some(2))), 2),
            cube(Shuffle(Plane::stepping(2, Some(2))), 4),
            cube(Shuffle(Plane::stepping(4, None)), 8),
            cube(Shuffle(Plane::stepping(4, Some(2))), 4),
            cube(Shuffle(Plane::stepping(4, Some(2))), 8),
            cube(Shuffle(Plane::stepping(1, None)), 16),
        ];
        if PlaneFragmentMicrokernel::decodes_on(hw) {
            settings.extend([
                cube(fragments(1), 8),
                cube(fragments(2), 8),
                cube(fragments(1), 4),
                cube(fragments(2), 4),
                cube(fragments(1), 16),
            ]);
        }
        settings
    }

    /// The levels inside one cube's run, which a routine states the cubes around: its planes'
    /// steps through all of `K`, dealt to the planes in runs along [`split`](Self::split), so the
    /// planes need not divide them.
    pub fn tiling(&self) -> Levels {
        self.plane
            .step_levels()
            .planes_across(self.split(), self.planes_along_k.get())
    }

    /// The axis its planes split `K` along: `K` itself, or a packed weight's scale blocks.
    pub fn split(&self) -> Axis {
        match self.plane {
            CubeShufflePlane::Shuffle(plane) => match plane.blocks {
                None => plane.axes.k,
                Some(_) => blocked(K).0,
            },
            CubeShufflePlane::Fragment(_) => blocked(K).0,
        }
    }

    /// Bytes of shared memory one cube holds over its box in `partitioning`: the accumulator over
    /// its run, one sum per cell of each copy its planes meet in ([`CubeMeet::copies`]). cubek
    /// sizes that buffer itself, dense over the cube's box of the output at one `sum` element a
    /// cell, which is what this counts; its sizing is not exposed to read.
    pub fn shared_memory(&self, item: &MatmulWorkItem, partitioning: &MatmulPartitioning) -> usize {
        let cells = (partitioning.rows() * partitioning.columns()) as usize;
        cells * self.meet.copies(self.planes_along_k.get()) * item.sum.size()
    }

    /// The cubes of software planes `settings` names, for a col float weight or a packed one.
    /// Against a packed weight the planes are capped at the turns its plane's units take
    /// ([`PlaneShuffleMicrokernel::turns`]), and a plane whose units take no turns, or too few
    /// for two planes, is not offered.
    fn shuffle_cubes(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneShuffleMicrokernelSettings,
        target: usize,
    ) -> impl Iterator<Item = Self> {
        let col_weight = item.layouts.rhs == MatmulLayout::Col;
        let meet = CubeMeet::of(hw);
        let planes = PlaneShuffleMicrokernel::admits(item)
            .is_ok()
            .then(|| PlaneShuffleMicrokernel::candidates(hw, item, settings));
        planes.into_iter().flatten().filter_map(move |plane| {
            let planes = match plane.blocks {
                None => col_weight.then_some(target)?,
                Some(_) => target.min(plane.turns()?),
            };
            NonZeroUsize::new(planes)
                .filter(|planes| planes.get() >= Self::FEWEST_PLANES)
                .map(|planes_along_k| Self {
                    plane: CubeShufflePlane::Shuffle(plane),
                    planes_along_k,
                    meet,
                })
        })
    }

    /// The cubes of tensor-core planes `settings` names, against a packed weight its plane
    /// admits, the planes capped at the windows of `K` it takes.
    fn fragment_cubes(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &PlaneFragmentMicrokernelSettings,
        target: usize,
    ) -> impl Iterator<Item = Self> {
        let meet = CubeMeet::of(hw);
        let planes = PlaneFragmentMicrokernel::admits(item)
            .is_ok()
            .then(|| PlaneFragmentMicrokernel::candidates(hw, item, settings));
        planes.into_iter().flatten().filter_map(move |plane| {
            let windows = item.blocks?.k / plane.tile_extent(plane.axes.k);
            Some(Self {
                plane: CubeShufflePlane::Fragment(plane),
                planes_along_k: NonZeroUsize::new(target.min(windows))?,
                meet,
            })
        })
    }
}

impl CubeShufflePlane {
    /// The levels of one plane's steps through `K`.
    pub fn step_levels(&self) -> Levels {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.step_levels(),
            CubeShufflePlane::Fragment(plane) => plane.step_levels(),
        }
        .expect("candidates hold a plane whose steps walk K")
    }

    /// The digit of `N` the plane's columns land on.
    pub fn column_axis(&self) -> Axis {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.column_cut().0,
            CubeShufflePlane::Fragment(plane) => plane.axes.n,
        }
    }

    /// Columns one plane covers.
    pub fn columns(&self) -> usize {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.columns(),
            CubeShufflePlane::Fragment(plane) => plane.tile_extent(plane.axes.n),
        }
    }

    /// Values of `K` one step of the plane covers, `leaf` its step's box: a unit's slice by the
    /// units sharing a step, over both digits of a packed weight's `K`, or a fragment's depth.
    pub fn step_depth(&self, leaf: &Space) -> usize {
        match self {
            CubeShufflePlane::Shuffle(plane) => {
                let depth = match plane.blocks {
                    None => leaf.extent(plane.axes.k),
                    Some(_) => {
                        let (blocks, inside) = blocked(K);
                        leaf.extent(blocks) * leaf.extent(inside)
                    }
                };
                depth * plane.units_along_k
            }
            CubeShufflePlane::Fragment(plane) => plane.tile_extent(plane.axes.k),
        }
    }

    /// Whether the operands bound at `widths` are read in vectors the plane folds: the software
    /// plane's check ([`PlaneShuffleMicrokernel::check_vectors`]); a tensor-core plane states its
    /// widths as it vectorizes.
    ///
    /// # Errors
    ///
    /// The software plane's refusal.
    pub fn check_vectors(&self, widths: &MatmulOperands<usize>) -> Result<(), Refusal> {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.check_vectors(widths),
            CubeShufflePlane::Fragment(_) => Ok(()),
        }
    }

    /// The plane width the plane is compiled for: the tensor-core plane's, the only width its
    /// decoding instruction's lane runs are laid over.
    pub fn plane_width(&self) -> usize {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.plane_width(),
            CubeShufflePlane::Fragment(_) => PlaneFragmentMicrokernel::DECODING_PLANE_UNITS,
        }
    }

    /// The instruction the plane contracts through.
    pub fn tile_instruction(&self) -> Instruction {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.tile_instruction(),
            CubeShufflePlane::Fragment(plane) => plane.tile_instruction(),
        }
    }

    /// How the plane's units read each input: as bound in software, the units summing a packed
    /// weight decoded at `f32`; every packed input dequantized at the element it decodes into on
    /// the tensor cores, a fragment's two operands being one element
    /// ([`MatmulWorkItem::decoding_reads`]).
    pub fn reads(&self, item: &MatmulWorkItem) -> MatmulOperands<MatmulRead> {
        match self {
            CubeShufflePlane::Shuffle(_) => MatmulOperands::all(MatmulRead::AsBound),
            CubeShufflePlane::Fragment(_) => item.decoding_reads(),
        }
    }
}

impl core::fmt::Display for CubeShufflePlane {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CubeShufflePlane::Shuffle(plane) => plane.fmt(f),
            CubeShufflePlane::Fragment(plane) => plane.fmt(f),
        }
    }
}

impl Microkernel for CubeShuffleMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = CubeShuffleMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the cube shuffle needs a plane width the device commits to, and against a packed weight one whose scales group K and two turns of K or more for its planes";

    /// Never one plane: a run one plane owns is the plane routine's. Every setting's step of `K`,
    /// its planes' step times their count, divides a decode projection's `K` (2048, 6144) on a
    /// wave32 plane and on a wave64 one: the default, one column one read deep across four planes;
    /// two reads deep across two, the same step of `K` in longer runs; and a transaction's worth of
    /// units beside more columns a cube, with four planes along `K` and eight. Against a packed
    /// weight a routine races [`packed_settings`](Self::packed_settings).
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        use CubeShufflePlaneSettings::Shuffle;
        use PlaneShuffleMicrokernelSettings as Plane;
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::new(Shuffle(Plane::TWO_READS), 2),
            Self::Settings::new(Shuffle(Plane::ONE_READ_ONE_TRANSACTION), 4),
            Self::Settings::new(Shuffle(Plane::ONE_READ_ONE_TRANSACTION), 8),
        ]
    }

    /// Its planes' items: of a float weight on matmul's own axes and in cols
    /// ([`ROW_RHS_UNSUPPORTED`](Self::ROW_RHS_UNSUPPORTED)), or of a packed weight on its
    /// block-split space, either way its words lie, which one of its planes admits.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        let None = item.blocks else {
            let shuffle = PlaneShuffleMicrokernel::admits(item);
            return match PlaneFragmentMicrokernel::admits(item) {
                Ok(()) => Ok(()),
                Err(_) => shuffle,
            };
        };
        MatmulPartitioning::admits(item)?;
        PlaneShuffleMicrokernel::admits(item)?;
        match item.layouts.rhs {
            MatmulLayout::Row => Err(Self::ROW_RHS_UNSUPPORTED),
            MatmulLayout::Col | MatmulLayout::Strided => Ok(()),
        }
    }

    /// Why a setting's cubes come back empty: its plane's reason.
    fn decline(settings: &Self::Settings) -> &'static str {
        match settings.plane {
            CubeShufflePlaneSettings::Shuffle(_) => Self::DECLINE,
            CubeShufflePlaneSettings::Fragment(_) => Self::FRAGMENT_DECLINE,
        }
    }

    /// Its plane microkernel's at the settings' plane count, its planes meeting as `hw`'s cubes
    /// do ([`CubeMeet::of`]).
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let target = settings.planes_along_k.get();
        let (shuffle, fragment) = match &settings.plane {
            CubeShufflePlaneSettings::Shuffle(plane) => {
                (Some(Self::shuffle_cubes(hw, item, plane, target)), None)
            }
            CubeShufflePlaneSettings::Fragment(plane) => {
                (None, Some(Self::fragment_cubes(hw, item, plane, target)))
            }
        };
        shuffle
            .into_iter()
            .flatten()
            .chain(fragment.into_iter().flatten())
    }

    /// Its planes read as the plane microkernel's: a col weight's step down `K`, folded with the
    /// lhs's a vector of each at a time. The cube copies the output out of the accumulator's
    /// atomics one element at a time.
    ///
    /// Nothing against a packed weight under software planes: every width follows from its
    /// packing. A tensor-core plane states its own.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        let plane = match self.plane {
            CubeShufflePlane::Shuffle(plane) if plane.blocks.is_none() => plane,
            CubeShufflePlane::Shuffle(_) => return,
            CubeShufflePlane::Fragment(plane) => return plane.vectorize(&(), vectorization),
        };
        for operand in [MatmulOperand::Lhs, MatmulOperand::Rhs] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: plane.leaf_extent(plane.orders.innermost(operand)),
            });
        }
        vectorization.require(VectorRequirement::Together {
            operand: MatmulOperand::Rhs,
            with: MatmulOperand::Lhs,
        });
        vectorization.require(VectorRequirement::Elementwise {
            operand: MatmulOperand::Out,
        });
    }

    /// Its planes' microkernel, for the work. The reads and the drain are its routine's to count,
    /// which places its cubes.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        _: &(),
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        match &self.plane {
            CubeShufflePlane::Shuffle(plane) => plane.estimate(item, &(), planes, estimate),
            CubeShufflePlane::Fragment(plane) => plane.estimate(item, &(), planes, estimate),
        }
    }
}

/// Its planes' microkernel, and how many share a run.
impl core::fmt::Display for CubeShuffleMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_along_k={}", self.plane, self.planes_along_k)
    }
}

impl MatmulMicrokernel for CubeShuffleMicrokernel {
    /// Its planes each hold every row.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    /// Its planes' orders: a float weight's as the plane shuffle binds it, a packed weight's
    /// words over the axes the product names.
    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        match self.plane {
            CubeShufflePlane::Shuffle(plane) => plane.orders,
            CubeShufflePlane::Fragment(_) => MatmulOperands::all(MatmulOrder::Stated),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::devices;

    /// The tensor-core plane's packed settings are listed only where the device offers its
    /// decoding instruction: elsewhere they would only be tried and declined.
    #[test]
    fn tensor_core_planes_only_where_the_instruction_is_offered() {
        for hw in [
            devices::strix_halo(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::cpu_12(),
        ] {
            let fragments = CubeShuffleMicrokernel::packed_settings(&hw)
                .into_iter()
                .any(|settings| matches!(settings.plane, CubeShufflePlaneSettings::Fragment(_)));
            assert_eq!(
                fragments,
                PlaneFragmentMicrokernel::decodes_on(&hw),
                "{hw:?}"
            );
        }
    }
}
