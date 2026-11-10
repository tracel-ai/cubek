use crate::interpolate::{InterpolateOperand, InterpolateWorkItem};
use cubecl::tile::RegisterBlock;
use cubecl::tile::launch::Vectorization;
use cubek_std::Hardware;
use cubek_std::microkernel::{
    ArithmeticInstruction, CubePlanes, KernelEstimate, Microkernel, UnitWork,
};

/// One block of a cube's walk: every output cell of it contracts its tap window, weights by
/// samples, in registers laid out as `register_block` states.
///
/// Every device runs it: the register block is the one the device's units hold best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeInterpolateMicrokernel {
    pub register_block: RegisterBlock,
}

/// A block's microkernel has nothing a measurement settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeInterpolateMicrokernelSettings;

impl core::fmt::Display for CubeInterpolateMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("registers")
    }
}

impl Microkernel for CubeInterpolateMicrokernel {
    type WorkItem = InterpolateWorkItem;
    type Settings = CubeInterpolateMicrokernelSettings;
    type Operand = InterpolateOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = "the cube interpolate runs on every device";

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![CubeInterpolateMicrokernelSettings]
    }

    /// A CPU unit is a core, which holds a wide block and splits it along its edge; a GPU unit
    /// holds a narrow one and spreads its components.
    fn candidates(
        hw: &Hardware,
        _item: &InterpolateWorkItem,
        _settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let register_block = match hw.properties.hardware.num_cpu_cores {
            Some(_) => RegisterBlock::new(256).split_edge(),
            None => RegisterBlock::new(64).component_fanout(),
        };
        core::iter::once(Self { register_block })
    }

    /// Its accesses are its work item's: nothing in the block's walk reads past a unit's run.
    fn vectorize(&self, _: &(), _vectorization: &mut Vectorization<InterpolateOperand>) {}

    /// It holds the outputs it accumulates: a lower bound, the registers it keeps live.
    fn estimate(
        &self,
        item: &InterpolateWorkItem,
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
