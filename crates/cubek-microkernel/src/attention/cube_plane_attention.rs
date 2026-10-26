use crate::attention::CubeAttentionStageLayout;
use crate::attention::output::cube::CubeAttentionPlaneSettings;
use crate::attention::{AttentionMicrokernel, AttentionWorkItem};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Levels, Region, Space, StageStorage, Tile, Walk};
use cubecl::{self as cubecl, cube, ir::ElemType, prelude::*};

/// A plane microkernel as the cube holds it: the box one plane owns, where its steps read their
/// keys and values, and the shared memory each plane holds beside the cube's.
///
/// The levels above a plane are written once for every plane microkernel of this kind through
/// this trait; what a plane runs while the cube walks is its [`CubePlaneAttention`].
pub trait CubePlaneMicrokernel:
    AttentionMicrokernel<
        Settings: Copy + Eq + core::hash::Hash + Send + Sync + 'static + CubeAttentionPlaneSettings,
    > + Copy
    + Eq
    + core::hash::Hash
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// What the routines placing it are named after, as a cube of fragments is
    /// `cube-fragment-attention`.
    const NAME: &'static str;

    /// Where its steps read their keys and values, which decides whether the cube stages them.
    const READS: CubePlaneReads;

    /// How planes holding the same rows may take their keys.
    const KEYS: CubePlaneKeys;

    /// The levels inside one plane's box of one step of keys, which the cube states its planes and
    /// its walk around.
    fn tiling(&self) -> Levels;

    /// Query positions one plane owns, of one head.
    fn seq_q(&self) -> usize;

    /// Keys one plane takes in one step.
    fn seq_kv(&self) -> usize;

    /// Bytes of shared memory one plane holds while the cube walks, beside the cube's own.
    fn window_bytes(&self, item: &AttentionWorkItem) -> usize;

    /// Bytes of shared memory a cube laid out as `layout` holds for its planes that split the same
    /// rows' keys to meet in once the walk ends; none where no planes split them.
    fn merge_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize;

    /// The storage the cube holds a stage over `stage` of the keys and values its steps read in,
    /// laid out as `layout`: blocked into the tiles its steps read.
    fn stage_storage(&self, layout: CubeAttentionStageLayout, stage: &Space) -> StageStorage;

    /// Bytes of shared memory a cube laid out as `layout` holds for its planes to leave their
    /// rows in slots ([`CubePlaneAttention::store_partial`]).
    fn partial_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize;

    /// Whether its rows' softmax state merges with another plane's of the same rows, as planes that
    /// split a cube's keys merge it.
    ///
    /// # Errors
    ///
    /// Why it keeps that state where no other holder reaches it.
    fn merges_rows(&self) -> Result<(), Refusal> {
        Ok(())
    }

    /// The element a stage of its query, keys and values holds where the stage's fill decodes
    /// them, as it does an operand bound as its words: `f32` where `None`, the element the tile DSL
    /// decodes a packed operand at unless the kernel names another.
    fn decoded_stage(&self) -> Option<ElemType> {
        None
    }

    /// Whether it reads a stage laid out as `layout`.
    ///
    /// # Errors
    ///
    /// Why it cannot.
    fn reads_stage(&self, _layout: CubeAttentionStageLayout) -> Result<(), Refusal> {
        Ok(())
    }
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

/// How planes of a plane microkernel holding the same rows may take their keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubePlaneKeys {
    /// Split between them, each taking its block of every step, their rows merged once the walk
    /// ends.
    Split,
    /// Whole: a plane walks every key of its rows, so no two planes hold the same rows.
    Whole,
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

/// What a cube's planes run while the cube walks its keys: each plane's rows opened before the
/// walk, a step of keys taken into them, and the rows ended as the level above asks once the walk
/// ends: normalized into the output where the cube owns its rows alone, or left undivided in
/// slots where other cubes hold the same rows
/// ([`AttentionSlotShape`](crate::attention::AttentionSlotShape)).
///
/// `EA` is the element the rows' sums are kept in, `E` the query's and the output's; `K`, `V` and
/// `M` are where a step reads its keys, its values and which keys each row sees: a plane that reads
/// the cube's stage takes tiles of it, the bias among them, and a plane that reads keys where they
/// lie takes any [`StepLines`](crate::attention::StepLines) and the
/// [`VisibleKeys`](crate::attention::VisibleKeys) rule. A cube compiles only with the
/// sources its planes take, so a packed cache, which only lines decode, never reaches a plane
/// that reads a stage.
#[cube]
pub trait CubePlaneAttention<EA: Float, E: Float, K: CubeType, V: CubeType, M: CubeType>:
    CubeType + Sized
{
    /// The plane microkernel it runs.
    type Microkernel: CubePlaneMicrokernel;

    /// What its rows are opened over before the walk.
    type Opening: CubeType;

    /// One line of a row's output as it leaves it in a slot.
    type Line: CubePrimitive;

    /// Every plane's rows over `walk`, the cube's walk through the keys, before any key: opened
    /// over `opening` as `microkernel` runs them, the cube's planes laid out as `layout`.
    fn open(
        walk: &Walk,
        opening: &Self::Opening,
        #[comptime] microkernel: Self::Microkernel,
        #[comptime] layout: CubePlaneLayout,
    ) -> Self;

    /// The step of keys `plane`, this plane's region of one step of the walk, taken into its rows:
    /// its keys read from `keys` and `values`, each row seeing those `visible` admits, scaled by
    /// `scale`.
    fn step(&mut self, plane: &Region, keys: &K, values: &V, visible: &M, scale: EA);

    /// Every plane's rows normalized and written into `out`, `cube`'s window of the output. Planes
    /// that read the cube's stage and split the same rows' keys sum their outputs here, a meeting
    /// of the cube; planes that read their keys where they lie each write their own rows, and the
    /// cube merges the ones that split keys from their slots instead
    /// ([`store_partial`](Self::store_partial)).
    fn store_normalized(&mut self, cube: &Region, out: &Tile);

    /// `plane`'s own rows of the region of `walk`'s first step left undivided in slots, the first
    /// `rows` of them: row `r`'s running max, running sum and output at slot `first_slot + r ·
    /// row_stride` of `slot_max`, `slot_sum` and `slot_out`, laid out as
    /// [`AttentionSlotShape`](crate::attention::AttentionSlotShape) states. A plane
    /// meets no other here: planes that split the same rows' keys are merged from their slots by
    /// the level above.
    #[allow(clippy::too_many_arguments)]
    fn store_partial(
        &mut self,
        walk: &Walk,
        plane: &Region,
        slot_out: &mut [Self::Line],
        slot_max: &mut [EA],
        slot_sum: &mut [EA],
        first_slot: usize,
        row_stride: usize,
        rows: usize,
    );
}

/// The attention the planes of a plane microkernel that reads the cube's stage run: what a kernel
/// generic over the plane microkernel names its planes' attention by, where the elements are the
/// kernel's.
pub trait CubeStagedPlane {
    type Attention<EA: Float, E: Float, ES: Float>: CubePlaneAttention<
            EA,
            E,
            Tile,
            Tile,
            Tile,
            Microkernel = Self,
            Opening = StagedPlaneOperands,
        >;
}

/// What a plane that reads the cube's stage opens its rows over: the query the cube staged for
/// all its planes, and the cube's windows of the output, the values and the bias, which shape
/// what its rows sum into. The query and the values are read in `ES`, the output stored in `E`.
#[derive(CubeType)]
pub struct StagedPlaneOperands {
    pub query: Tile,
    pub out: Tile,
    pub value: Tile,
    pub bias: Tile,
}

/// What a plane that reads its keys where they lie opens its rows over: the cube's window of the
/// query, from which each plane loads its rows, and the elements of a value.
#[derive(CubeType)]
pub struct GlobalPlaneOperands {
    pub query: Tile,
    #[cube(comptime)]
    pub val_dim: usize,
}
