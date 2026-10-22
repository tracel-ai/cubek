use crate::matmul::dense::cube::{CubeStagedDelivery, CubeStagedFill, CubeStagedStore};
use crate::matmul::dense::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentTransport};
use crate::matmul::staged_walk::MatmulPlaneLayout;
use crate::staging::StageRing;
use core::num::NonZeroUsize;
use cubecl::tile::{Scratch, StageStorage, stage::RowChunks};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedMicrokernelSettings {
    pub plane: PlaneFragmentMicrokernelSettings,
    /// How its planes are laid out over the box, `rows` along `m`, `columns` along `n`.
    pub planes: MatmulPlaneLayout,
    /// Planes sharing each tile, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
    /// How much of a plane's fragments the window it bounces them through holds at once, where
    /// they bounce: one fragment, three cube barriers each, or all of them, two barriers for the
    /// whole drain at as many times the bytes.
    pub scratch: Scratch,
    /// How it holds the stages its planes contract.
    pub ring: StageRing,
    /// Who fills them.
    pub fill: CubeStagedFill,
    /// How a stage lays out the rows its fragments load.
    pub stage_rows: CubeStagedRows,
    /// Who stores its box of the output.
    pub store: CubeStagedStore,
}

impl CubeStagedMicrokernelSettings {
    /// A box of four by four fragments on each of two by four planes, one stage at a time, filled
    /// by the cube's units.
    pub const DEFAULT: Self = Self {
        plane: PlaneFragmentMicrokernelSettings::DEFAULT,
        planes: MatmulPlaneLayout::new(2, 4),
        planes_along_k: NonZeroUsize::MIN,
        scratch: Scratch::OneTile,
        ring: StageRing::SINGLE,
        fill: CubeStagedFill::BY_UNITS,
        stage_rows: CubeStagedRows::Spread,
        store: CubeStagedStore::Fragments,
    };

    /// One tile a cube, four planes splitting its `K`: what a problem with too few output tiles to
    /// fill the device wants, each tile getting a cube's planes rather than one plane. The tile is
    /// half as high as the default's plane's, for what the scratch and the accumulator leave of
    /// shared memory.
    pub const SPLIT_ALONG_K: Self = Self {
        plane: PlaneFragmentMicrokernelSettings::grid(2, 4, 1),
        planes: MatmulPlaneLayout::new(1, 1),
        planes_along_k: NonZeroUsize::new(4).unwrap(),
        ..Self::DEFAULT
    };

    /// The tile DSL's multi-row box: one column of four planes, each holding two rows of fragments
    /// across the cube's `n`, reached through the tile DSL's transport. Under `cmma` a row weight's
    /// `B` fragment is read in scalar loads and spills two steps deep, where the tile DSL's
    /// transport does not.
    pub const MULTI_ROW: Self = Self::boxed(
        PlaneFragmentMicrokernelSettings::grid(2, 8, 2).through(PlaneFragmentTransport::Mma),
        MatmulPlaneLayout::new(4, 1),
    );

    /// A box against a packed weight, counted in steps of the packed product
    /// ([`MatmulStep`](crate::matmul::MatmulStep)): `steps` of `M` and `N` on each
    /// plane, on `planes` (along `m` by along `n`), holding its stages in `ring`. A step's box is
    /// whole loads of the weight's words and scales, so every load filling a stage is whole; the
    /// stage holds the step's scale blocks, decoded by the units that fill it, and each plane drains
    /// its own tile.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn in_steps(steps: (usize, usize), planes: (usize, usize), ring: StageRing) -> Self {
        Self {
            plane: PlaneFragmentMicrokernelSettings {
                along_m: NonZeroUsize::new(steps.0).unwrap(),
                along_n: NonZeroUsize::new(steps.1).unwrap(),
                ..PlaneFragmentMicrokernelSettings::DEFAULT
            },
            planes: MatmulPlaneLayout::new(planes.0, planes.1),
            planes_along_k: NonZeroUsize::MIN,
            scratch: Scratch::None,
            ring,
            fill: CubeStagedFill::BY_UNITS,
            stage_rows: CubeStagedRows::Spread,
            store: CubeStagedStore::Fragments,
        }
    }

    /// Eight by one steps on each of one by two planes, one stage at a time: on the M2, eight by
    /// two instructions a plane in tile order, the float staged routine's fastest box on the
    /// Qwen3-4B projections at a prompt's rows, whose planes contract the same stages these hold.
    pub const PACKED: Self = Self::in_steps((8, 1), (1, 2), StageRing::SINGLE);

    /// These settings with `planes` planes sharing each box along `K`.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn split_along_k(self, planes: usize) -> Self {
        Self {
            planes_along_k: NonZeroUsize::new(planes).unwrap(),
            ..self
        }
    }

    /// `plane` on each of `planes`, the default's other settings.
    pub const fn boxed(plane: PlaneFragmentMicrokernelSettings, planes: MatmulPlaneLayout) -> Self {
        Self {
            plane,
            planes,
            ..Self::DEFAULT
        }
    }

    /// These settings holding their stages in `ring`.
    pub const fn ringed(self, ring: StageRing) -> Self {
        Self { ring, ..self }
    }

    /// These settings with their planes reaching the matrix instruction through `transport`.
    pub const fn through(self, transport: PlaneFragmentTransport) -> Self {
        Self {
            plane: self.plane.through(transport),
            ..self
        }
    }

    /// These settings with their stages filled by `delivery`, `filling_planes` set aside for it.
    pub const fn filled(self, delivery: CubeStagedDelivery, filling_planes: usize) -> Self {
        Self {
            fill: CubeStagedFill {
                delivery,
                filling_planes,
            },
            ..self
        }
    }

    /// Whether these settings hold the same box of the output as `other`: the same plane grid on
    /// the same layout of planes, whatever their stages.
    pub fn same_box(&self, other: &Self) -> bool {
        (self.plane, self.planes) == (other.plane, other.planes)
    }
}

/// Nothing for one plane along `K`, a one-fragment scratch or spread rows, so the settings that
/// hold them keep their names.
impl core::fmt::Display for CubeStagedMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.plane, self.planes)?;
        if self.planes_along_k.get() > 1 {
            write!(f, " planes_along_k={}", self.planes_along_k)?;
        }
        if self.scratch != Scratch::OneTile {
            write!(f, " scratch={}", self.scratch)?;
        }
        write!(f, "{}{}", self.ring, self.fill)?;
        if self.stage_rows == CubeStagedRows::InOrder {
            write!(f, " rows=in-order")?;
        }
        write!(f, "{}", self.store)
    }
}

/// How a stage lays out the rows a matrix instruction's fragments load from it.
///
/// Spread is the instruction's own layout, which keeps the rows one load reads on distinct
/// shared memory banks: `cmma`'s rows padded by a chunk, `mma`'s chunks swizzled
/// ([`PlaneFragmentInstruction::stage_storage`](crate::matmul::dense::plane::PlaneFragmentInstruction::stage_storage)).
/// In order is the rows as they lie, which costs bank conflicts and saves the padding's bytes and
/// strides. Which reads faster depends on the device and on which way the operands run, so it is a
/// setting a table races rather than a rule. Under `mma` it races the swizzle the same way. The
/// software instruction reads a strided stage either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedRows {
    Spread,
    InOrder,
}

impl CubeStagedRows {
    /// `storage`, the instruction's stage, with its rows laid out this way.
    pub fn lay(self, storage: StageStorage) -> StageStorage {
        match (self, storage) {
            (Self::InOrder, StageStorage::Tiled { block, .. }) => StageStorage::Tiled {
                block,
                chunks: RowChunks::InOrder,
            },
            (Self::Spread, storage) | (Self::InOrder, storage) => storage,
        }
    }
}
