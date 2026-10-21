use crate::microkernel::MatmulPartitioning;
use crate::microkernel::dense::cube::{
    CubeStagedDrain, CubeStagedMicrokernel, CubeStagedMicrokernelSettings, CubeStagedRows,
};
use crate::microkernel::dense::plane::PlaneFragmentMicrokernelSettings;
use crate::microkernel::staged_walk::MatmulPlaneLayout;
use crate::microkernel::staging::StageRing;
use crate::microkernel::{
    MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulRowsKind, MatmulWorkItem,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::{CubePlanes, GlobalStream, KernelEstimate, Microkernel};
use cubek_tile::Scratch;
use cubek_tile::launch::Delivery;
use cubek_tile::launch::Refusal;
use cubek_tile::launch::Vectorization;

/// A box of the output several cubes own together: each cube runs a [`CubeStagedMicrokernel`]
/// over its own run of `K`, and the parts reach the output once, as the routine's
/// [`MatmulCubeWrite`](crate::microkernel::staged_walk::MatmulCubeWrite) says: in turns
/// at a carry, or into slots the last cube to arrive adds up. Which cubes share which box is the
/// routine's too, so one microkernel serves every way a hypercube's cubes share their boxes.
///
/// Its settings, with the cube's settings replaced by the cube microkernel the hardware settled;
/// the cubes sharing a box are the routine's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeStagedMicrokernel {
    /// What each of its cubes runs.
    pub cube: CubeStagedMicrokernel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeStagedMicrokernelSettings {
    pub cube: CubeStagedMicrokernelSettings,
}

impl HypercubeStagedMicrokernelSettings {
    /// A smaller box than a staged cube's own default: what a product with too few output tiles to fill
    /// the device and a long contraction wants, which is the problem a hypercube exists for. Two by
    /// four fragments on each of two by two planes.
    pub const DEFAULT: Self = Self::boxed(PlaneFragmentMicrokernelSettings::grid(2, 4, 1), (2, 2));

    /// The staged cube's own box ([`CubeStagedMicrokernelSettings::DEFAULT`]).
    pub const CUBE_BOX: Self = Self {
        cube: CubeStagedMicrokernelSettings::DEFAULT,
    };

    /// The staged cube's box two instruction steps of `K` a stage.
    pub const CUBE_BOX_DEEP: Self =
        Self::boxed(PlaneFragmentMicrokernelSettings::grid(4, 4, 2), (2, 4));

    /// Eight by two fragments two steps deep on each of two by two planes.
    pub const SQUARE: Self = Self::boxed(PlaneFragmentMicrokernelSettings::grid(8, 2, 2), (2, 2));

    /// The default plane's grid two steps deep on four planes along `m`.
    pub const DEEP_ALONG_M: Self =
        Self::boxed(PlaneFragmentMicrokernelSettings::grid(4, 4, 2), (4, 1));

    /// `plane` on each of `planes` (along `m` by along `n`), the cube's other settings
    /// its default's.
    pub const fn boxed(plane: PlaneFragmentMicrokernelSettings, planes: (usize, usize)) -> Self {
        Self {
            cube: CubeStagedMicrokernelSettings::boxed(
                plane,
                MatmulPlaneLayout::new(planes.0, planes.1),
            ),
        }
    }

    /// The default's grid `depth` instruction steps of `K` a stage on each of `planes`: a col
    /// weight runs along `K`, which a stage one fragment deep reads in short runs.
    pub const fn deep(depth: usize, planes: (usize, usize)) -> Self {
        Self::boxed(PlaneFragmentMicrokernelSettings::grid(2, 4, depth), planes)
    }

    /// These settings with each cube holding its stages in `ring`.
    pub const fn ringed(self, ring: StageRing) -> Self {
        Self {
            cube: self.cube.ringed(ring),
        }
    }

    /// These settings with each plane's fragments bouncing through `scratch`.
    pub const fn with_scratch(self, scratch: Scratch) -> Self {
        Self {
            cube: CubeStagedMicrokernelSettings {
                scratch,
                ..self.cube
            },
        }
    }

    /// These settings with each stage's rows in order ([`CubeStagedRows::InOrder`]).
    pub const fn rows_in_order(self) -> Self {
        Self {
            cube: CubeStagedMicrokernelSettings {
                stage_rows: CubeStagedRows::InOrder,
                ..self.cube
            },
        }
    }
}

impl core::fmt::Display for HypercubeStagedMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.cube.fmt(f)
    }
}

impl Microkernel for HypercubeStagedMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = HypercubeStagedMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = MatmulPartitioning;

    const DECLINE: &'static str = "the hypercube's cubes meet at their box, which needs a device where one cube's \
         stores reach another within a launch, and planes that each drain their own tile with \
         none set aside to fill";

    /// The boxes worth sharing between cubes: the default, the staged cube's box, the default's
    /// box with its whole partition resident in the scratch, a second stage in flight, filled in
    /// its slot and through registers; the square and deep boxes of four planes; the default's grid
    /// two and four fragments deep on two by two and two by one planes, and the deeper on two by
    /// two with its rows in order; the deep box along `m` with its rows in order; and the staged
    /// cube's box with a second stage in flight and two steps deep. None where one cube's stores
    /// do not reach another within a launch, which is how the cubes meet.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        if !hw.properties.features.device_memory_scope {
            return Vec::new();
        }
        let lead = Self::Settings::DEFAULT;
        let deep = [2, 4]
            .into_iter()
            .flat_map(|depth| [(2, 2), (2, 1)].map(|planes| Self::Settings::deep(depth, planes)));
        [
            lead,
            Self::Settings::CUBE_BOX,
            lead.with_scratch(Scratch::WholeGrid),
            lead.ringed(StageRing::DOUBLE),
            lead.ringed(StageRing::DOUBLE_THROUGH_REGISTERS),
            Self::Settings::SQUARE,
            Self::Settings::DEEP_ALONG_M,
        ]
        .into_iter()
        .chain(deep)
        .chain([
            Self::Settings::deep(4, (2, 2)).rows_in_order(),
            Self::Settings::DEEP_ALONG_M.rows_in_order(),
            Self::Settings::CUBE_BOX.ringed(StageRing::DOUBLE),
            Self::Settings::CUBE_BOX_DEEP,
        ])
        .collect()
    }

    /// Its cubes' items.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        CubeStagedMicrokernel::admits(item)
    }

    /// Every cube microkernel this device offers whose planes each drain their own tile, with
    /// none set aside to fill: a cube walks its run into one sum, which those planes hold until
    /// the cubes meet. None where one cube's stores do not reach another within a launch, which
    /// is how the cubes meet.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let meets = hw.properties.features.device_memory_scope;
        CubeStagedMicrokernel::candidates(hw, item, &settings.cube)
            .filter(move |cube| {
                let walk = cube.walk();
                meets && walk.drain == CubeStagedDrain::EachPlane && walk.filling_planes == 0
            })
            .map(|cube| Self { cube })
    }

    /// Its cubes' microkernel's: each plane drains its tile in lines to where its box meets the
    /// others, as into an output it alone writes.
    fn vectorize(
        &self,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        self.cube.vectorize(partitioning, vectorization);
    }

    /// Its cubes' microkernel's, each walking its run of `K` as a staged cube walks a whole box,
    /// and each also reading its box of the carry the cubes meet at and writing it back, in the
    /// sum's element, in its planes' runs of the output.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        partitioning: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.cube.estimate(item, partitioning, planes, estimate);
        let cube = partitioning.cube_box();
        let size = item.sum.size();
        let carry = GlobalStream {
            bytes: (cube.extent(M) * cube.extent(N) * size) as u64,
            run_bytes: self.cube.plane.tile_extent(N) * size,
        };
        if let Some(staging) = &mut estimate.staging {
            staging.reads.push(carry);
            staging.writes.push(carry);
        }
    }
}

/// Its cubes' microkernel.
impl core::fmt::Display for HypercubeStagedMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.cube.fmt(f)
    }
}

impl MatmulMicrokernel for HypercubeStagedMicrokernel {
    /// Its cubes own a box of the rows together.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    /// Its cubes' order: their planes read the rhs's tiles out of their stages.
    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.cube.orders()
    }

    fn delivery(&self) -> Delivery {
        self.cube.delivery()
    }
}
