use crate::microkernel::MatmulPartitioning;
use crate::microkernel::dense::plane::{PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings};
use crate::microkernel::{
    MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulRowsKind,
    MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Levels};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Every row of a run of the output's columns, owned by one cube: its planes each run the plane
/// microkernel's step on their own slice of every step of `K`, and add their partial sums into
/// one accumulator in shared memory, as the device lets them meet ([`CubeMeet`]).
///
/// What a product whose columns give too few runs to fill the device with planes wants: each run
/// gets a cube's planes rather than one. It sums a float weight in cols only: a float row
/// weight's step lies along `N`, so each of a unit's cells is a vector of the weight, and the
/// shared accumulator the planes add their partial sums into takes one element a cell, which cubek
/// does not drain a wider vector into. A plane shuffle reads a float row weight whole.
///
/// Against a packed weight (its plane's [`blocks`](PlaneShuffleMicrokernel::blocks)) its planes split
/// `K`'s scale blocks, each taking whole turns of its plane's units, so no two planes share a
/// block's scale. A unit's cell is then a read of the weight's values along `K` whichever way its
/// words lie, folded on drain as a col weight's is, so a packed weight in rows sums too.
///
/// Atomic adds land in no fixed order, and `f32` addition does not associate, so two launches on
/// the same operands may differ in the last bits of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeShuffleMicrokernel {
    /// What each of its planes runs on its slice of a step.
    pub plane: PlaneShuffleMicrokernel,
    /// Planes sharing one run of columns, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
    /// How its planes add their partial sums into one.
    pub meet: CubeMeet,
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
    pub plane: PlaneShuffleMicrokernelSettings,
    /// Planes sharing one run of columns, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
}

impl CubeShuffleMicrokernelSettings {
    /// The plane microkernel's default at one read deep, four planes a cube. The cube's step of
    /// `K` is its planes' step times their count, and at the plane default's four reads it would
    /// be 8192 `f16` elements on a wave64 plane, which divides no decode projection's `K`; at one
    /// it is 1024 on wave32 and 2048 on wave64.
    pub const DEFAULT: Self = Self::new(PlaneShuffleMicrokernelSettings::ONE_READ, 4);

    /// `plane` on each of `planes_along_k` planes sharing a run.
    ///
    /// # Panics
    ///
    /// On no planes.
    pub const fn new(plane: PlaneShuffleMicrokernelSettings, planes_along_k: usize) -> Self {
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

impl CubeShuffleMicrokernel {
    /// What a row weight is declined with: its step lies along `N`, so each of a unit's cells is
    /// a vector of the weight, and the shared accumulator the planes add their partial sums into
    /// takes one element a cell, which cubek does not drain a wider vector into.
    pub const ROW_RHS_UNSUPPORTED: Refusal = Refusal::Declined {
        why: "the cube shuffle sums a float weight in cols only: a row weight's vectors along `N` do not drain into the shared accumulator's one-element cells",
    };

    /// The fewest planes a cube holds: a run one plane owns is the plane routine's.
    const FEWEST_PLANES: usize = 2;

    /// The levels inside one cube's run, which a routine states the cubes around: its planes'
    /// steps through all of `K`, dealt to the planes in runs along [`split`](Self::split), so the
    /// planes need not divide them.
    pub fn tiling(&self) -> Levels {
        self.plane
            .step_levels()
            .expect("candidates hold a plane whose units take K in turns")
            .planes_across(self.split(), self.planes_along_k.get())
    }

    /// The axis its planes split `K` along: `K` itself, or a packed weight's scale blocks.
    pub fn split(&self) -> Axis {
        match self.plane.blocks {
            None => self.plane.axes.k,
            Some(_) => blocked(K).0,
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
}

impl Microkernel for CubeShuffleMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = CubeShuffleMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = ();

    const DECLINE: &'static str = "the cube shuffle needs a plane width the device commits to";

    /// Never one plane: a run one plane owns is the plane routine's. Every setting's step of `K`,
    /// its planes' step times their count, divides a decode projection's `K` (2048, 6144) on a
    /// wave32 plane and on a wave64 one: the default, one column one read deep across four planes;
    /// two reads deep across two, the same step of `K` in longer runs; and a transaction's worth of
    /// units beside more columns a cube, with four planes along `K` and eight.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        use PlaneShuffleMicrokernelSettings as Plane;
        vec![
            Self::Settings::DEFAULT,
            Self::Settings::new(Plane::TWO_READS, 2),
            Self::Settings::new(Plane::ONE_READ_ONE_TRANSACTION, 4),
            Self::Settings::new(Plane::ONE_READ_ONE_TRANSACTION, 8),
        ]
    }

    /// Its planes' items: of a float weight on matmul's own axes and in cols
    /// ([`ROW_RHS_UNSUPPORTED`](Self::ROW_RHS_UNSUPPORTED)), or of a packed weight on its block-split
    /// space, either way its words lie.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        let None = item.blocks else {
            return PlaneShuffleMicrokernel::admits(item);
        };
        MatmulPartitioning::admits(item)?;
        PlaneShuffleMicrokernel::admits(item)?;
        match item.layouts.rhs {
            MatmulLayout::Row => Err(Self::ROW_RHS_UNSUPPORTED),
            MatmulLayout::Col | MatmulLayout::Strided => Ok(()),
        }
    }

    /// Its plane microkernel's at the settings' plane count, for a col float weight or a packed
    /// one, its planes meeting as `hw`'s cubes do ([`CubeMeet::of`]). Against a packed weight the
    /// planes are capped at the turns its plane's units take ([`PlaneShuffleMicrokernel::turns`]),
    /// and a plane whose units take no turns, or too few for two planes, is not offered.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let col_weight = item.layouts.rhs == MatmulLayout::Col;
        let (meet, target) = (CubeMeet::of(hw), settings.planes_along_k.get());
        PlaneShuffleMicrokernel::candidates(hw, item, &settings.plane).filter_map(move |plane| {
            let planes = match plane.blocks {
                None => col_weight.then_some(target)?,
                Some(_) => target.min(plane.turns()?),
            };
            NonZeroUsize::new(planes)
                .filter(|planes| planes.get() >= Self::FEWEST_PLANES)
                .map(|planes_along_k| Self {
                    plane,
                    planes_along_k,
                    meet,
                })
        })
    }

    /// Its planes read as the plane microkernel's: a col weight's step down `K`, folded with the
    /// lhs's a vector of each at a time. The cube copies the output out of the accumulator's
    /// atomics one element at a time.
    ///
    /// Nothing against a packed weight, as its plane's: every width follows from its packing.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<MatmulOperand>) {
        if self.plane.blocks.is_some() {
            return;
        }
        for operand in [MatmulOperand::Lhs, MatmulOperand::Rhs] {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.plane.leaf_extent(self.plane.orders.innermost(operand)),
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
        self.plane.estimate(item, &(), planes, estimate);
    }
}

/// Its planes' microkernel, and how many share a run.
impl core::fmt::Display for CubeShuffleMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes_along_k={}", self.plane, self.planes_along_k)
    }
}

impl MatmulMicrokernel for CubeShuffleMicrokernel {
    /// Its planes are plane shuffles, each holding every row.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Whole;

    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.plane.orders
    }
}
