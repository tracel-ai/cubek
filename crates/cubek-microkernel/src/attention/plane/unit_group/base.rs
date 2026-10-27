use crate::attention::unit::{UnitAttentionMicrokernel, UnitAttentionMicrokernelSettings};
use crate::attention::{AttentionMicrokernel, AttentionOperand, AttentionWorkItem, Q};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Levels, Space};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
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
    type CubeSpace = Space;

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
    fn vectorize(&self, cube_space: &Space, vectorization: &mut Vectorization<AttentionOperand>) {
        self.unit.vectorize(cube_space, vectorization);
    }

    /// Its unit's: every unit holds its blocks, beside the softmax state of its rows.
    fn estimate(
        &self,
        item: &AttentionWorkItem,
        cube_space: &Space,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.unit.estimate(item, cube_space, planes, estimate);
    }
}

impl AttentionMicrokernel for PlaneUnitGroupAttentionMicrokernel {
    /// Its unit's.
    fn key_order(&self) -> BindingOrder {
        self.unit.key_order()
    }
}

/// What a cube holding it on its planes reads of it.
impl PlaneUnitGroupAttentionMicrokernel {
    /// What the routines whose cubes hold it are named after.
    pub const NAME: &'static str = "unit-group";

    /// Its unit's step, a block of rows on each of its units.
    pub fn tiling(&self) -> Levels {
        self.unit.step_tiling().units(&[(Q, self.units)])
    }

    pub fn seq_q(&self) -> usize {
        self.unit_seq_q() * self.units
    }

    pub fn seq_kv(&self) -> usize {
        self.unit.seq_kv()
    }

    /// None: its rows and their sums stay in registers.
    pub fn window_bytes(&self, _item: &AttentionWorkItem) -> usize {
        0
    }
}
