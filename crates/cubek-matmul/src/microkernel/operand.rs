use crate::microkernel::{MatmulAxes, MatmulDim, MatmulOrder};
use cubecl::tile::{Axis, Tile};
use cubecl::{self as cubecl, cube, ir::ElemType, prelude::*};

/// One of the three operands of `out = lhs · rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulOperand {
    /// `[.., m, k]`.
    Lhs,
    /// `[.., k, n]`.
    Rhs,
    /// `[.., m, n]`.
    Out,
}

impl MatmulOperand {
    pub const ALL: [Self; 3] = [Self::Lhs, Self::Rhs, Self::Out];

    /// The dimensions the product names it by, in that order: `[m, k]`, `[k, n]` or `[m, n]`.
    pub fn dims(self) -> [MatmulDim; 2] {
        match self {
            Self::Lhs => [MatmulDim::M, MatmulDim::K],
            Self::Rhs => [MatmulDim::K, MatmulDim::N],
            Self::Out => [MatmulDim::M, MatmulDim::N],
        }
    }

    /// The axes the product names it by, in that order.
    pub fn axes(self) -> [Axis; 2] {
        self.dims().map(|dim| MatmulAxes::MNK.of(dim))
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MatmulOperands<T> {
    pub lhs: T,
    pub rhs: T,
    pub out: T,
}

impl<T: Copy> MatmulOperands<T> {
    /// `value` for every operand.
    pub fn all(value: T) -> Self {
        Self {
            lhs: value,
            rhs: value,
            out: value,
        }
    }

    pub fn get(&self, operand: MatmulOperand) -> T {
        match operand {
            MatmulOperand::Lhs => self.lhs,
            MatmulOperand::Rhs => self.rhs,
            MatmulOperand::Out => self.out,
        }
    }

    /// A value per operand, computed from each operand and its value here.
    pub fn map<U>(&self, f: impl Fn(MatmulOperand, T) -> U) -> MatmulOperands<U> {
        MatmulOperands {
            lhs: f(MatmulOperand::Lhs, self.lhs),
            rhs: f(MatmulOperand::Rhs, self.rhs),
            out: f(MatmulOperand::Out, self.out),
        }
    }
}

impl MatmulOperands<MatmulOrder> {
    /// The dimension `operand` is bound innermost along, in the order it is bound in.
    pub fn innermost(&self, operand: MatmulOperand) -> MatmulDim {
        self.get(operand).innermost(operand)
    }
}

/// How a matmul microkernel's leaves read an operand they are handed: as it is bound, or, for a
/// packed operand, dequantized at the element its leaves contract.
///
/// A packed operand read as bound decodes at `f32`. A product contracting at another element
/// names it here, where the kernel is built: each leaf then dequantizes its window of the operand
/// into the fragments it contracts, and a stage the fill decodes holds that element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulRead {
    /// As bound: a float operand as it is, a packed one decoded at `f32`.
    AsBound,
    /// Dequantized at this element.
    Dequantized(ElemType),
}

impl MatmulRead {
    /// The element a stage of the operand decoded by its fill holds: `f32` where `None`.
    pub fn staged(self) -> Option<ElemType> {
        match self {
            Self::AsBound => None,
            Self::Dequantized(element) => Some(element),
        }
    }
}

#[cube]
impl MatmulRead {
    /// `operand`, a leaf's window of it, as the leaf `sum` contracts it read as `read` says.
    pub fn leaf(sum: &Tile, operand: &Tile, #[comptime] read: MatmulRead) -> Tile {
        match comptime!(read) {
            MatmulRead::AsBound => operand.clone(),
            MatmulRead::Dequantized(element) => sum.dequantized(operand, element),
        }
    }
}
