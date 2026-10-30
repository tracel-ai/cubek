use crate::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
    PlaneShuffleAttentionMicrokernel, PlaneShuffleAttentionMicrokernelSettings,
};
use crate::attention::{AttentionMicrokernel, AttentionOperand, AttentionWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::{Levels, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel, StageLayout};

/// The plane microkernel a cube runs on each of its planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(
    clippy::large_enum_variant,
    reason = "a setting a kernel compiles, held by value as the cube's other knobs are; the \
              fragment plane holds its two products' planes"
)]
pub enum CubeAttentionPlane {
    /// Fragments of a square matrix instruction, against the keys the cube stages.
    Fragment(PlaneFragmentAttentionMicrokernel),
    /// Teams of units scoring each key where it lies, summing across the plane.
    Shuffle(PlaneShuffleAttentionMicrokernel),
}

/// The settings of the plane microkernel a cube runs on each of its planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeAttentionPlaneSettings {
    Fragment(PlaneFragmentAttentionMicrokernelSettings),
    Shuffle(PlaneShuffleAttentionMicrokernelSettings),
}

/// Where a plane microkernel's steps read their keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubePlaneReads {
    /// The cube's stage: the cube copies each step of keys and values into shared memory once
    /// for all its planes, and stages its query before the walk.
    Stage,
    /// Where they lie: each plane reads its own keys of each step, and the cube stages nothing.
    Global,
}

/// How a cube lays its planes over its box: a plane for every head it holds, along every plane of
/// a head's queries, along every plane of the same rows' keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePlaneLayout {
    pub heads: usize,
    pub planes_along_queries: usize,
    pub planes_along_keys: usize,
}

impl CubePlaneLayout {
    /// Planes one cube holds.
    pub fn planes(&self) -> usize {
        self.heads * self.planes_along_queries * self.planes_along_keys
    }

    /// Groups of planes holding the same rows, one for every head along every plane of its
    /// queries.
    pub fn row_groups(&self) -> usize {
        self.heads * self.planes_along_queries
    }
}

impl CubeAttentionPlaneSettings {
    /// What the routines whose cubes run planes of this kind are named after, as a cube of
    /// fragments is `cube-fragment-attention`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fragment(_) => PlaneFragmentAttentionMicrokernel::NAME,
            Self::Shuffle(_) => PlaneShuffleAttentionMicrokernel::NAME,
        }
    }

    /// Why a device runs no plane at these settings.
    pub fn decline(&self) -> &'static str {
        match self {
            Self::Fragment(_) => PlaneFragmentAttentionMicrokernel::DECLINE,
            Self::Shuffle(_) => PlaneShuffleAttentionMicrokernel::DECLINE,
        }
    }
}

impl core::fmt::Display for CubeAttentionPlaneSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fragment(settings) => settings.fmt(f),
            Self::Shuffle(settings) => settings.fmt(f),
        }
    }
}

impl CubeAttentionPlane {
    /// Every plane this device can run `item` on at `settings`, best first.
    pub fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &CubeAttentionPlaneSettings,
    ) -> Vec<Self> {
        match settings {
            CubeAttentionPlaneSettings::Fragment(settings) => {
                PlaneFragmentAttentionMicrokernel::candidates(hw, item, settings)
                    .map(Self::Fragment)
                    .collect()
            }
            CubeAttentionPlaneSettings::Shuffle(settings) => {
                PlaneShuffleAttentionMicrokernel::candidates(hw, item, settings)
                    .map(Self::Shuffle)
                    .collect()
            }
        }
    }

    /// Where its steps read their keys and values: the cube's stage for fragments, which load each
    /// stage from shared memory, and where they lie for a shuffle, each team reading its keys of
    /// every step.
    pub fn reads(&self) -> CubePlaneReads {
        match self {
            Self::Fragment(_) => CubePlaneReads::Stage,
            Self::Shuffle(_) => CubePlaneReads::Global,
        }
    }

    /// The levels inside one plane's box of one step of keys, which the cube states its planes and
    /// its walk around.
    pub fn tiling(&self) -> Levels {
        match self {
            Self::Fragment(plane) => plane.tiling(),
            Self::Shuffle(plane) => plane.tiling(),
        }
    }

    /// Query positions one plane owns, of one head.
    pub fn seq_q(&self) -> usize {
        match self {
            Self::Fragment(plane) => plane.seq_q(),
            Self::Shuffle(plane) => plane.seq_q(),
        }
    }

    /// Keys one plane takes in one step.
    pub fn seq_kv(&self) -> usize {
        match self {
            Self::Fragment(plane) => plane.seq_kv(),
            Self::Shuffle(plane) => plane.seq_kv(),
        }
    }

    /// Bytes of shared memory one plane holds while the cube walks, beside the cube's own.
    pub fn window_bytes(&self, item: &AttentionWorkItem) -> usize {
        match self {
            Self::Fragment(plane) => plane.window_bytes(item),
            Self::Shuffle(plane) => plane.window_bytes(item),
        }
    }

    /// Bytes of shared memory a cube laid out as `layout` holds for its planes that split the same
    /// rows' keys to meet in once the walk ends; none where no planes split them.
    ///
    /// Fragments meet in the sum of the products of the planes holding the same rows and every
    /// plane's softmax state, beside the stages: the compiler does not hand a freed stage's bytes
    /// to what is declared after it. A shuffle meets in a slot for every plane of every row: a
    /// line of each element of a value, and its running max and sum
    /// ([`SlotShape`](crate::split_keys_merge::SlotShape)).
    pub fn merge_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        if layout.planes_along_keys == 1 {
            return 0;
        }
        match self {
            Self::Fragment(_) => {
                let rows = layout.row_groups() * self.seq_q();
                (rows * item.val_dim + layout.planes() * self.seq_q()) * item.sum.size()
            }
            Self::Shuffle(_) => {
                layout.planes() * self.seq_q() * (item.val_dim + 2) * item.sum.size()
            }
        }
    }

    /// Bytes of shared memory a cube laid out as `layout` holds for its planes to leave their rows
    /// in slots.
    ///
    /// Fragments drain through a window of every row of the cube's box over the value dim, and two
    /// of one cell a row, their softmax state's max and sum. A shuffle's rows leave their slots
    /// from its units' registers, after its planes that split the same rows' keys meet
    /// ([`merge_bytes`](Self::merge_bytes)).
    pub fn partial_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        let rows = layout.row_groups() * self.seq_q();
        match self {
            Self::Fragment(_) => rows * (item.val_dim + 2) * item.sum.size(),
            Self::Shuffle(_) => self.merge_bytes(item, layout),
        }
    }

    /// The storage a stage over `stage` of the keys and values its steps read is held in, laid
    /// out as `layout`: in the tiles a fragment's step reads
    /// ([`stage_tile`](PlaneFragmentAttentionMicrokernel::stage_tile)) where the layout blocks it,
    /// and row-major for a shuffle, which reads no stage.
    pub fn stage_storage(&self, layout: StageLayout, stage: &Space) -> StageStorage {
        match self {
            Self::Fragment(plane) => layout.storage(stage, plane.stage_tile()),
            Self::Shuffle(_) => StageStorage::Strided,
        }
    }

    /// The storage its query, staged once over `stage`, is held in, laid out as `layout` lays the
    /// keys and values: in the tiles a fragment of the score reads
    /// ([`query_tile`](PlaneFragmentAttentionMicrokernel::query_tile)) where the layout blocks it,
    /// so a fragment's rows of the query meet the banks as a step's rows of the keys do, and
    /// row-major for a shuffle, which reads no stage.
    pub fn query_storage(&self, layout: StageLayout, stage: &Space) -> StageStorage {
        match self {
            Self::Fragment(plane) => layout.storage(stage, plane.query_tile()),
            Self::Shuffle(_) => StageStorage::Strided,
        }
    }

    /// Whether its rows' softmax state merges with another plane's of the same rows, as planes that
    /// split a cube's keys merge it.
    ///
    /// # Errors
    ///
    /// Why it keeps that state where no other holder reaches it.
    pub fn merges_rows(&self) -> Result<(), Refusal> {
        match self {
            Self::Fragment(plane) => plane.merges_rows(),
            Self::Shuffle(_) => Ok(()),
        }
    }

    /// The element a stage of its query, keys and values holds where the stage's fill decodes
    /// them, as it does an operand bound as its words: `f32` where `None`, the element the tile DSL
    /// decodes a packed operand at unless the kernel names another.
    pub fn decoded_stage(&self) -> Option<ElemType> {
        match self {
            Self::Fragment(plane) => plane.decoded_stage(),
            Self::Shuffle(_) => None,
        }
    }

    /// Whether it reads a stage laid out as `layout`.
    ///
    /// # Errors
    ///
    /// Why it cannot.
    pub fn reads_stage(&self, layout: StageLayout) -> Result<(), Refusal> {
        match self {
            Self::Fragment(plane) => plane.reads_stage(layout),
            Self::Shuffle(_) => Ok(()),
        }
    }

    /// Its accesses' requirements of the operands' widths, its cube covering `cube_space`.
    pub fn vectorize(
        &self,
        cube_space: &Space,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        match self {
            Self::Fragment(plane) => plane.vectorize(cube_space, vectorization),
            Self::Shuffle(plane) => plane.vectorize(cube_space, vectorization),
        }
    }

    /// Its parts of a kernel running `item`, its cube covering `cube_space` on cubes of `planes`.
    pub fn estimate(
        &self,
        item: &AttentionWorkItem,
        cube_space: &Space,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        match self {
            Self::Fragment(plane) => plane.estimate(item, cube_space, planes, estimate),
            Self::Shuffle(plane) => plane.estimate(item, cube_space, planes, estimate),
        }
    }

    /// The order its score reads the keys in ([`AttentionMicrokernel::key_order`]).
    pub fn key_order(&self) -> BindingOrder {
        match self {
            Self::Fragment(plane) => plane.key_order(),
            Self::Shuffle(plane) => plane.key_order(),
        }
    }

    /// `item` as its steps read it ([`AttentionMicrokernel::served`]).
    pub fn served(&self, item: &AttentionWorkItem) -> AttentionWorkItem {
        match self {
            Self::Fragment(plane) => plane.served(item),
            Self::Shuffle(plane) => plane.served(item),
        }
    }
}

impl core::fmt::Display for CubeAttentionPlane {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fragment(plane) => plane.fmt(f),
            Self::Shuffle(plane) => plane.fmt(f),
        }
    }
}
