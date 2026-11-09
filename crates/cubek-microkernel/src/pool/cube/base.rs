use crate::pool::{PoolOperand, PoolWorkItem};
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// One block of a cube's walk: every output cell of it the largest tap of its window.
///
/// Every device runs it, and it holds nothing a device changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeMaxPoolMicrokernel;

/// A block's microkernel has nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeMaxPoolMicrokernelSettings;

impl core::fmt::Display for CubeMaxPoolMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("reduce")
    }
}

impl Microkernel for CubeMaxPoolMicrokernel {
    type WorkItem = PoolWorkItem;
    type Settings = CubeMaxPoolMicrokernelSettings;
    type Operand = PoolOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube max pool runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![CubeMaxPoolMicrokernelSettings]
    }

    fn candidates(
        _hw: &Hardware,
        _item: &PoolWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        core::iter::once(Self)
    }

    /// Its accesses are its work item's: nothing in the block's walk reads past a unit's run.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<PoolOperand>) {}

    /// It holds the outputs it reduces into: a lower bound, the registers it keeps live.
    fn estimate(
        &self,
        item: &PoolWorkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        estimate.work = Some(UnitWork {
            instruction: ArithmeticInstruction::Vector,
            registers: item.outputs as u32,
        });
    }
}
