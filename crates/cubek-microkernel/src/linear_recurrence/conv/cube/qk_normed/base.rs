use crate::linear_recurrence::RecurrenceHeads;
use crate::linear_recurrence::conv::QkNormedConvOperand;
use crate::linear_recurrence::conv::cube::{ChannelBox, PastKind};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// What a conv is: the heads whose channels it mixes, the filter's taps and where its past is read
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QkNormedConvWorkItem {
    pub heads: RecurrenceHeads,
    pub taps: usize,
    pub past: PastKind,
}

impl QkNormedConvWorkItem {
    /// Whether the heads hold no key head, value head or feature: nothing to convolve.
    pub fn is_empty(&self) -> bool {
        let heads = &self.heads;
        [
            heads.key_heads,
            heads.heads_per_group,
            heads.key_features,
            heads.value_features,
        ]
        .contains(&0)
    }

    /// Operations `channels` channels take.
    pub fn operations(&self, channels: usize) -> usize {
        channels * (2 * self.taps + 6)
    }
}

/// The conv has nothing to measure: one rule places its units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct QkNormedConvSettings;

impl core::fmt::Display for QkNormedConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cube-qk-normed-conv")
    }
}

/// A cube's key head, its channels dealt between `units` units. It reads no plane instruction, only
/// shared memory to meet the norms in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeQkNormedConvMicrokernel {
    pub units: usize,
    pub taps: usize,
    pub heads: RecurrenceHeads,
    pub past: PastKind,
}

impl CubeQkNormedConvMicrokernel {
    /// Bytes of shared memory for the head's convolved queries and keys and their sums of squares,
    /// at the narrowest width.
    pub fn shared_memory(&self) -> usize {
        4 * self.heads.key_features * size_of::<f32>()
    }
}

impl Microkernel for CubeQkNormedConvMicrokernel {
    type WorkItem = QkNormedConvWorkItem;
    type Settings = QkNormedConvSettings;
    type Operand = QkNormedConvOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the device spawns no cube of units for the conv";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![QkNormedConvSettings]
    }

    fn candidates(
        hw: &Hardware,
        item: &QkNormedConvWorkItem,
        _: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let units = ChannelBox::most_units(hw).min(item.heads.channels_per_key_head());
        (units > 0)
            .then_some(Self {
                units,
                taps: item.taps,
                heads: item.heads,
                past: item.past,
            })
            .into_iter()
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<QkNormedConvOperand>) {
        QkNormedConvOperand::require(
            self.heads.key_features,
            self.heads.value_features,
            vectorization,
        );
    }

    fn estimate(
        &self,
        _item: &QkNormedConvWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: (2 * self.taps + 8) as u32,
        });
    }
}
