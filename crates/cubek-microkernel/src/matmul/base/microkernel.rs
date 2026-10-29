use crate::matmul::{MatmulOperand, MatmulOperands, MatmulRowsKind, MatmulWorkItem};
use cubecl::ir::ElemType;
use cubecl::tile::launch::Delivery;
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::BlockScales;
use cubek_std::microkernel::Microkernel;

/// A microkernel of every matmul product: handed a [`MatmulWorkItem`], and read by a routine for
/// the order it binds each operand in and how it moves its stages. A cube microkernel that stages
/// is handed the space its cube covers ([`MatmulCubeSpace`](crate::matmul::cube::MatmulCubeSpace));
/// a cube shuffle, a tile group, a plane or a unit one reads only its own tiling, so its
/// `CubeSpace` is `()`.
pub trait MatmulMicrokernel:
    Microkernel<WorkItem = MatmulWorkItem, Operand = MatmulOperand> + core::fmt::Display + 'static
{
    /// How its launches hold a product's rows, which decides how their tensors and their view read
    /// the problem and what their kernels compile in.
    const ROWS_KIND: MatmulRowsKind;

    /// The order it reads each operand's tiles in, which the launch binds the operand in.
    fn orders(&self) -> MatmulOperands<BindingOrder>;

    /// How the lhs and the rhs it binds are moved into its stages, which the launch states on
    /// both: [`SyncPerUnit`](Delivery::SyncPerUnit) for one that stages nothing.
    fn delivery(&self) -> Delivery {
        Delivery::SyncPerUnit
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

    /// The element its fill decodes its inputs into, where its stages hold another than they are
    /// stored in: a float input read so, an `f32` one read at tf32, is bound as its words, which
    /// the fill decodes. `None` for one that stages its inputs as they lie.
    fn decoded_stage(&self) -> Option<ElemType> {
        None
    }
}
