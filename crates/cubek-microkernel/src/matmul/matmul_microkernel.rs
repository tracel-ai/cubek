use crate::matmul::stored::BlockScales;
use crate::matmul::{MatmulOperand, MatmulOperands, MatmulOrder, MatmulRowsKind, MatmulWorkItem};
use cubecl::tile::launch::Delivery;
use cubek_std::microkernel::Microkernel;

/// A microkernel of every matmul product: handed a [`MatmulWorkItem`], and read by a routine for
/// the order it binds each operand in and how it moves its stages. A cube microkernel that stages is
/// placed in a [`MatmulPartitioning`](crate::matmul::MatmulPartitioning); a
/// cube shuffle, a plane or a unit one reads only its own tiling, so its partitioning is `()`.
pub trait MatmulMicrokernel:
    Microkernel<WorkItem = MatmulWorkItem, Operand = MatmulOperand> + core::fmt::Display + 'static
{
    /// How its launches hold a product's rows, which decides how their tensors and their view read
    /// the problem and what their kernels compile in.
    const ROWS_KIND: MatmulRowsKind;

    /// The order it reads each operand's tiles in, which the launch binds the operand in.
    fn orders(&self) -> MatmulOperands<MatmulOrder>;

    /// How the lhs and the rhs it binds are moved into its stages, which the launch states on
    /// both: [`SyncPerUnit`](Delivery::SyncPerUnit) for one that stages nothing.
    fn delivery(&self) -> Delivery {
        Delivery::SyncPerUnit
    }

    /// How it reads a packed input's block scales, which the launch binds them as:
    /// [`Fields`](BlockScales::Fields) for one that decodes them.
    fn block_scales(&self) -> BlockScales {
        BlockScales::Fields
    }
}
