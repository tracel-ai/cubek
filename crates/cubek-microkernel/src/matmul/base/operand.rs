use crate::matmul::{
    MATMUL_PACKED_READS_SPLIT, MATMUL_READS_UNCLOSED, MatmulDim, MatmulDimensions,
};
use cubecl::ir::FloatKind;
use cubecl::tile::launch::Decline;
use cubecl::tile::{BaseRegion, Dimension, Space, Tile, VectorRegion};
use cubecl::{self as cubecl, cube, ir::ElemType, prelude::*};
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_dimensions::{K, M, N};
use cubek_std::{Hardware, Instructions, block_scaled_mma};

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

    /// The product's two inputs.
    pub const INPUTS: [Self; 2] = [Self::Lhs, Self::Rhs];

    /// The dimensions the product names it by, in that order: `[m, k]`, `[k, n]` or `[m, n]`.
    pub fn dims(self) -> [MatmulDim; 2] {
        match self {
            Self::Lhs => [MatmulDim::M, MatmulDim::K],
            Self::Rhs => [MatmulDim::K, MatmulDim::N],
            Self::Out => [MatmulDim::M, MatmulDim::N],
        }
    }

    /// The dimension it is bound innermost along in `order`: the one its lines run along, the
    /// last the product names it by, or the first where it is bound as its buffer steps.
    pub fn innermost(self, order: BindingOrder) -> MatmulDim {
        let [outer, inner] = self.dims();
        match order {
            BindingOrder::Stated => inner,
            BindingOrder::InBufferOrder => outer,
        }
    }

    /// The dimensions the product names it by, in that order.
    pub fn dimensions(self) -> [Dimension; 2] {
        self.dims().map(|dim| MatmulDimensions::MNK.of(dim))
    }

    /// The region a load stores this input in on `hw`: the input read as `operand`, its values
    /// `values`, against the other input at `contracted`, the element the product is contracted
    /// at, under the largest instruction `hw` offers to contract it, its stages filled by the TMA
    /// engine where `hw` has one. The decode routines take no instruction, and their vector region
    /// divides this one, so one stored tile serves both.
    ///
    /// Packed codes on a device that contracts them as stored, under a block-scaled instruction
    /// that reads them for this input at their block along `k`, take the largest such instruction:
    /// NVFP4's `128 × 8`. Under an `8 × 8 × 8` instruction NVFP4 against an `f16` input takes
    /// `64 × 8`, and a float input `8 × 8`. `None` on a device that offers no instruction to
    /// contract it: there a tile would feed no fragment, and the input stays as it was laid down.
    pub fn stored_region(
        self,
        hw: &Hardware,
        operand: Operand,
        values: ElemType,
        contracted: ElemType,
    ) -> Option<BaseRegion> {
        let f32 = ElemType::Float(FloatKind::F32);
        let block_scaled = operand.scale(K).and_then(|block| {
            block_scaled_mma(hw)
                .filter(|instruction| {
                    instruction.shape.cd_type == f32
                        && instruction.holds(self == Self::Lhs, values, block)
                })
                .map(|instruction| instruction.shape)
                .largest()
        });
        let shape = match block_scaled {
            Some(shape) => Some(shape),
            None => hw
                .properties
                .features
                .matmul
                .cmma
                .iter()
                .copied()
                .for_triple(contracted, contracted, f32)
                .largest(),
        }?;
        let operands = MatmulOperands::all(Operand::plain()).with(self, operand);
        let elements = MatmulOperands::all(contracted).with(self, values);
        let region = BaseRegion::new(
            &operands.vector_region(hw, &elements),
            Space::new(&[
                (M, shape.m as usize),
                (N, shape.n as usize),
                (K, shape.k as usize),
            ]),
        );
        Some(match hw.bulk_tensor_copy() {
            true => region.divided_by(operand.tma_row(values)),
            false => region,
        })
    }
}

/// One value per operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MatmulOperands<T> {
    pub lhs: T,
    pub rhs: T,
    pub out: T,
}

impl<T> MatmulOperands<T> {
    /// The value of `operand`, where `T` is not `Copy`.
    pub fn get_ref(&self, operand: MatmulOperand) -> &T {
        match operand {
            MatmulOperand::Lhs => &self.lhs,
            MatmulOperand::Rhs => &self.rhs,
            MatmulOperand::Out => &self.out,
        }
    }
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
        *self.get_ref(operand)
    }

    /// These values with `operand`'s replaced by `value`.
    pub fn with(self, operand: MatmulOperand, value: T) -> Self {
        self.map(|other, current| if other == operand { value } else { current })
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

impl MatmulOperands<Operand> {
    /// The vector region of a product of inputs read so and loaded at `elements` on `hw`: each
    /// packed input's read and its scales' as its operand states them, and each float input one
    /// of the device's loads along the dimension the other input's reads run along where it has
    /// that dimension, one value where it does not and each of its values meets a whole line of
    /// the other.
    ///
    /// NVFP4 in tile order against an `f16` input on a device of 128-bit loads: the `f16` input
    /// loaded eight of `k` at a time, a word of four scales covering sixty-four of `k`, and the
    /// values two words, one block, down `k` a read: one column by sixty-four of `k`.
    pub fn vector_region(
        &self,
        hw: &Hardware,
        elements: &MatmulOperands<ElemType>,
    ) -> VectorRegion {
        let [first, second] = MatmulOperand::INPUTS;
        let mut vector_tiles = Vec::new();
        for (input, other) in [(first, second), (second, first)] {
            let operand = self.get(input);
            let lined = self.get(other).lined();
            match operand.scaled_along() {
                Some(_) => vector_tiles.extend(
                    [operand.read_space()]
                        .into_iter()
                        .chain(operand.scale_read_space()),
                ),
                None if input.dimensions().contains(&lined) => {
                    vector_tiles.push(Space::new(&[(lined, hw.load_values(elements.get(input)))]))
                }
                None => {}
            }
        }
        VectorRegion::new(vector_tiles)
    }

    /// Whether a product of `n` columns and `k` deep, its inputs read so and loaded at `elements`,
    /// is whole reads along both, each read of a packed input inside one scale tile, and two
    /// packed inputs' reads along `K` whole words of each other.
    ///
    /// # Errors
    ///
    /// [`MATMUL_READS_UNCLOSED`], and [`MATMUL_PACKED_READS_SPLIT`].
    pub fn check_reads(
        &self,
        hw: &Hardware,
        elements: &MatmulOperands<ElemType>,
        n: usize,
        k: usize,
    ) -> Result<(), Decline> {
        let straddles = MatmulOperand::INPUTS.into_iter().any(|input| {
            let operand = self.get(input);
            let lined = operand.lined();
            operand
                .scale(lined)
                .is_some_and(|tile| !tile.is_multiple_of(operand.read(lined)))
        });
        let vectors = self.vector_region(hw, elements);
        let closes = n.is_multiple_of(vectors.along(N)) && k.is_multiple_of(vectors.along(K));
        if !closes || straddles {
            return Err(MATMUL_READS_UNCLOSED);
        }
        let [first, second] = MatmulOperand::INPUTS.map(|input| self.get(input));
        let both_packed = first.scaled_along().is_some() && second.scaled_along().is_some();
        let [short, long] = [
            first.read(K).min(second.read(K)),
            first.read(K).max(second.read(K)),
        ];
        match both_packed && !long.is_multiple_of(short) {
            true => Err(MATMUL_PACKED_READS_SPLIT),
            false => Ok(()),
        }
    }
}

impl MatmulOperands<BindingOrder> {
    /// The dimension `operand` is bound innermost along, in the order it is bound in.
    pub fn innermost(&self, operand: MatmulOperand) -> MatmulDim {
        operand.innermost(self.get(operand))
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

impl MatmulOperands<MatmulRead> {
    /// Every operand read as bound.
    pub const AS_BOUND: Self = Self {
        lhs: MatmulRead::AsBound,
        rhs: MatmulRead::AsBound,
        out: MatmulRead::AsBound,
    };
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
        match read {
            MatmulRead::AsBound => operand.clone(),
            MatmulRead::Dequantized(element) => sum.dequantized(operand, element),
        }
    }
}
