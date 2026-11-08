use crate::pool::{PoolOperand, PoolReduction, PoolWorkItem};
use cubecl::tile::RegisterBlock;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// One block of a cube's walk: every output cell of it pools its window, an average in registers
/// laid out as `register_block` states.
///
/// Every device runs it: the register block is the one the device's units hold best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePoolMicrokernel {
    pub reduction: PoolReduction,
    pub register_block: RegisterBlock,
}

/// A block's microkernel has nothing a measurement settles: what it makes of a window is the
/// problem's, handed in its work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubePoolMicrokernelSettings;

impl core::fmt::Display for CubePoolMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("registers")
    }
}

impl Microkernel for CubePoolMicrokernel {
    type WorkItem = PoolWorkItem;
    type Settings = CubePoolMicrokernelSettings;
    type Operand = PoolOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube pool runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![CubePoolMicrokernelSettings]
    }

    /// A CPU unit is a core, which holds a wide block and splits it along its edge; a GPU unit
    /// holds a narrow one and spreads its components.
    fn candidates(
        hw: &Hardware,
        item: &PoolWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let register_block = match hw.properties.hardware.num_cpu_cores {
            Some(_) => RegisterBlock::new(256).split_edge(),
            None => RegisterBlock::new(64).component_fanout(),
        };
        core::iter::once(Self {
            reduction: item.reduction,
            register_block,
        })
    }

    /// Its accesses are its work item's: nothing in the block's walk reads past a unit's run.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<PoolOperand>) {}

    /// It holds the outputs it pools: a lower bound, the registers it keeps live.
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
