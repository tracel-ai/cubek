use crate::matmul::{MatmulOperand, MatmulOperands, MatmulWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::Levels;
use cubecl::tile::launch::Delivery;
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::BlockScales;
use cubek_std::microkernel::Microkernel;

/// A microkernel of every matmul product: handed a [`MatmulWorkItem`], and read by a routine for
/// the order it binds each operand in and how it moves its stages. A cube microkernel that stages
/// is handed the space its cube covers ([`MatmulCubeSpace`](crate::matmul::cube::MatmulCubeSpace));
/// a cube shuffle, a plane or a unit one reads only its own tiling, so its
/// `CubeSpace` is `()`.
pub trait MatmulMicrokernel:
    Microkernel<WorkItem = MatmulWorkItem, Operand = MatmulOperand> + core::fmt::Display + 'static
{
    /// The levels inside one owner, which its routine lays the rest of the partitioning around.
    fn tiling(&self) -> Levels;

    /// The order it reads each operand's tiles in, which the launch binds the operand in.
    fn orders(&self) -> MatmulOperands<BindingOrder>;

    /// How the lhs and the rhs it binds are moved into its stages, which the launch states on
    /// both: [`SyncPerUnit`](Delivery::SyncPerUnit) for one that stages nothing.
    fn delivery(&self) -> Delivery {
        Delivery::SyncPerUnit
    }

    /// Whether the TMA engine fills its stages, from descriptors its routine binds in place of the
    /// operands the launcher binds: false for one whose units move them.
    fn fills_by_tma(&self) -> bool {
        false
    }

    /// How a packed input's words are moved into its stages: as [`delivery`](Self::delivery)
    /// moves the float inputs, except where its fill decodes them
    /// ([`decoded_stage`](Self::decoded_stage)), which passes them through the units' registers,
    /// where the copy engine's lines never land.
    fn words_delivery(&self) -> Delivery {
        match self.decoded_stage() {
            Some(_) => Delivery::SyncPerUnit,
            None => self.delivery(),
        }
    }

    /// How it reads a packed input's block scales, which the launch binds them as:
    /// [`Fields`](BlockScales::Fields) for one that decodes them.
    fn block_scales(&self) -> BlockScales {
        BlockScales::Fields
    }

    /// Whether it reads a packed rhs a scale block at a time
    /// ([`Operand::by_block`](cubek_std::layout::stored::Operand::by_block)) rather than as the
    /// rhs is stored: its routine plans and binds the rhs so. `false` for one that reads it as
    /// stored.
    fn reads_rhs_by_block(&self) -> bool {
        false
    }

    /// The element its fill decodes its inputs into, where its stages hold another than they are
    /// stored in: a float input read so, an `f32` one read at tf32, is bound as its words, which
    /// the fill decodes. `None` for one that stages its inputs as they lie.
    fn decoded_stage(&self) -> Option<ElemType> {
        None
    }
}
