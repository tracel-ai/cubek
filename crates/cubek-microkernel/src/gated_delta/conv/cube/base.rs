use crate::gated_delta::GatedDeltaHeads;
use crate::gated_delta::conv::GatedDeltaConvOperand;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// The units a cube deals a key head's channels between, at most.
const MOST_UNITS: usize = 64;

/// What a conv is: the heads whose channels it mixes, the filter's taps and the operands its case
/// reads along their channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GatedDeltaConvWorkItem {
    pub heads: GatedDeltaHeads,
    pub taps: usize,
    pub channels: &'static [GatedDeltaConvOperand],
}

impl GatedDeltaConvWorkItem {
    /// Whether the heads hold no key head, value head or feature: nothing to convolve.
    pub fn is_empty(&self) -> bool {
        let heads = &self.heads;
        [
            heads.key_heads,
            heads.group,
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
pub struct GatedDeltaConvSettings;

impl core::fmt::Display for GatedDeltaConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cube-gated-delta-conv")
    }
}

/// A cube's key head, its channels dealt between `units` units. It reads no plane instruction, only
/// shared memory to meet the norms in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGatedDeltaConvMicrokernel {
    pub units: usize,
    pub taps: usize,
    pub heads: GatedDeltaHeads,
    pub channels: &'static [GatedDeltaConvOperand],
}

impl CubeGatedDeltaConvMicrokernel {
    /// Bytes of shared memory for the head's convolved queries and keys and their sums of squares,
    /// at the narrowest width.
    pub fn shared_memory(&self) -> usize {
        4 * self.heads.key_features * size_of::<f32>()
    }
}

impl Microkernel for CubeGatedDeltaConvMicrokernel {
    type WorkItem = GatedDeltaConvWorkItem;
    type Settings = GatedDeltaConvSettings;
    type Operand = GatedDeltaConvOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the device spawns no cube of units for the conv";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![GatedDeltaConvSettings]
    }

    fn candidates(
        hw: &Hardware,
        item: &GatedDeltaConvWorkItem,
        _: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let hardware = &hw.properties.hardware;
        let units = MOST_UNITS
            .min(item.heads.channels_per_key_head())
            .min(hardware.max_units_per_cube as usize)
            .min(hardware.max_cube_dim.0 as usize);
        (units > 0)
            .then_some(Self {
                units,
                taps: item.taps,
                heads: item.heads,
                channels: item.channels,
            })
            .into_iter()
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<GatedDeltaConvOperand>) {
        GatedDeltaConvOperand::require(
            self.channels,
            self.heads.key_features,
            self.heads.value_features,
            vectorization,
        );
    }

    fn estimate(
        &self,
        _item: &GatedDeltaConvWorkItem,
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
