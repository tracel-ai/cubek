use super::microkernel::PlaneUnitGroupAttention;
use crate::attention::CubeAttentionStageLayout;
use crate::attention::output::unit::{UnitAttentionMicrokernel, UnitAttentionMicrokernelSettings};
use crate::attention::{
    AttentionMicrokernel, AttentionOperand, AttentionPartitioning, AttentionWorkItem,
    CubePlaneKeys, CubePlaneLayout, CubePlaneMicrokernel, CubePlaneReads, CubeStagedPlane, Q,
};
use crate::matmul::MatmulOrder;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Levels, Space, StageStorage};
use cubecl::{self as cubecl, prelude::*};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A plane whose units each own rows of one head whole, against the cube's stage: every unit takes
/// each step of keys into its rows with the unit microkernel's step, in its own registers, the
/// keys and values read from shared memory, where the cube staged them once for all its planes.
///
/// What a cube asks of its planes where the device has no square matrix instruction for the
/// element, as `bf16` on Metal: scalar math, but on keys each cube reads once, where a unit of the
/// unit routine reads them again for every row it owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneUnitGroupAttentionMicrokernel {
    pub unit: UnitAttentionMicrokernel,
    /// Units the plane holds, a block of rows each: the device's plane width, a count a blueprint
    /// holds, an exception to the crate's rule as the plane shuffle's teams are, since the plane's
    /// rows are exactly its units' blocks.
    pub units: usize,
}

impl PlaneUnitGroupAttentionMicrokernel {
    /// The rows one unit owns.
    pub fn unit_seq_q(&self) -> usize {
        self.unit.settings.seq_q.get()
    }
}

/// Its unit's rows and keys.
impl core::fmt::Display for PlaneUnitGroupAttentionMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unit group {}", self.unit)
    }
}

impl Microkernel for PlaneUnitGroupAttentionMicrokernel {
    type WorkItem = AttentionWorkItem;
    type Settings = UnitAttentionMicrokernelSettings;
    type Operand = AttentionOperand;
    type Partitioning = AttentionPartitioning;

    const DECLINE: &'static str = "the plane unit group lays a block of rows on each of its units, \
         which needs a plane width the device commits to";

    /// The unit microkernel's default, then a row a unit against sixteen and thirty-two keys, and
    /// two rows a unit against eight.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        [(1, 8), (1, 16), (1, 32), (2, 8)]
            .map(|(rows, keys)| UnitAttentionMicrokernelSettings::new(rows, keys))
            .into()
    }

    /// The unit microkernel's candidates, each on every unit of a plane where the device commits
    /// to a plane width.
    fn candidates(
        hw: &Hardware,
        item: &AttentionWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let units = hw.plane_dim().ok().map(|width| width as usize);
        UnitAttentionMicrokernel::candidates(hw, item, settings).filter_map(move |unit| {
            Some(Self {
                unit,
                units: units?,
            })
        })
    }

    /// Its unit's.
    fn vectorize(
        &self,
        partitioning: &AttentionPartitioning,
        vectorization: &mut Vectorization<AttentionOperand>,
    ) {
        self.unit.vectorize(partitioning, vectorization);
    }

    /// Its unit's: every unit holds its blocks, beside the softmax state of its rows.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        partitioning: &AttentionPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.unit.estimate(item, partitioning, planes, estimate);
    }
}

impl AttentionMicrokernel for PlaneUnitGroupAttentionMicrokernel {
    /// Its unit's.
    fn key_order(&self) -> MatmulOrder {
        self.unit.key_order()
    }
}

impl CubePlaneMicrokernel for PlaneUnitGroupAttentionMicrokernel {
    const NAME: &'static str = "unit-group";

    /// Its units read each stage from shared memory.
    const READS: CubePlaneReads = CubePlaneReads::Stage;

    /// A unit's rows are its own: no plane holds another's to merge with.
    const KEYS: CubePlaneKeys = CubePlaneKeys::Whole;

    /// Its unit's step, a block of rows on each of its units.
    fn tiling(&self) -> Levels {
        self.unit.step_tiling().units(&[(Q, self.units)])
    }

    fn seq_q(&self) -> usize {
        self.unit_seq_q() * self.units
    }

    fn seq_kv(&self) -> usize {
        self.unit.seq_kv()
    }

    /// None: its rows and their sums stay in registers.
    fn window_bytes(&self, _item: &AttentionWorkItem) -> usize {
        0
    }

    /// None: its planes split no keys.
    fn merge_bytes(&self, _item: &AttentionWorkItem, _layout: CubePlaneLayout) -> usize {
        0
    }

    /// Row-major, whatever the layout: each unit reads its keys and values a row at a time, in no
    /// instruction's tiles.
    fn stage_storage(&self, _layout: CubeAttentionStageLayout, _stage: &Space) -> StageStorage {
        StageStorage::Strided
    }

    /// A window of every row of the cube's box over the value dim, its units' sums drain into.
    fn partial_bytes(&self, item: &AttentionWorkItem, layout: CubePlaneLayout) -> usize {
        layout.row_groups() * self.seq_q() * item.val_dim * item.sum.size()
    }
}

impl CubeStagedPlane for PlaneUnitGroupAttentionMicrokernel {
    type Attention<EA: Float, E: Float, ES: Float> = PlaneUnitGroupAttention<EA, ES>;
}
