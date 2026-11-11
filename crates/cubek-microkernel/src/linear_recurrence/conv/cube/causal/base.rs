use crate::linear_recurrence::conv::CausalConvOperand;
use crate::linear_recurrence::conv::cube::{CausalFilters, ChannelBox, PastKind};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// What a causal conv is: the channels it convolves, the filter's taps and where its past is read
/// from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalConvWorkItem {
    pub channels: usize,
    pub taps: usize,
    pub past: PastKind,
}

impl CausalConvWorkItem {
    /// Operations `channels` channels take.
    pub fn operations(&self, channels: usize) -> usize {
        channels * (2 * self.taps + 7)
    }
}

/// The conv has nothing to measure: one rule places its units.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CausalConvSettings;

impl core::fmt::Display for CausalConvSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cube-causal-conv")
    }
}

/// A cube's box of channel lines, a line to each of `units` units. It reads no plane instruction
/// and holds no shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeCausalConvMicrokernel {
    pub units: usize,
    pub taps: usize,
    pub channels: usize,
    pub past: PastKind,
}

impl CubeCausalConvMicrokernel {
    /// The boxes the cube deals its units' lines between.
    pub fn channel_box(&self) -> ChannelBox {
        ChannelBox {
            units: self.units,
            channels: self.channels,
        }
    }
}

impl Microkernel for CubeCausalConvMicrokernel {
    type WorkItem = CausalConvWorkItem;
    type Settings = CausalConvSettings;
    type Operand = CausalConvOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the device spawns no cube of units for the causal conv";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![CausalConvSettings]
    }

    fn candidates(
        hw: &Hardware,
        item: &CausalConvWorkItem,
        _: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let units = ChannelBox::most_units(hw).min(item.channels);
        (units > 0)
            .then_some(Self {
                units,
                taps: item.taps,
                channels: item.channels,
                past: item.past,
            })
            .into_iter()
    }

    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<CausalConvOperand>) {
        CausalConvOperand::require(self.channels, vectorization);
    }

    fn estimate(
        &self,
        _item: &CausalConvWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(CausalFilters::unit_work(self.taps));
    }
}
