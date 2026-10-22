use crate::microkernel::{MatmulAxes, MatmulLayout, MatmulOperand, MatmulOperands, MatmulRead};
use crate::stored::Operand;
use crate::stored::Packing;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::quant::scheme::QuantValue;
use cubecl::tile::Axis;
use cubek_std::matmul_axes::{K, M, N};

/// What a microkernel is handed to choose by: the elements it reads and writes and the one it
/// sums in, how each operand lies and the space their scales cut, how many rows it holds, and
/// where its product lands.
///
/// Its extents are the routine's: a routine tiles the problem into work items of the size its
/// microkernel states and masks the ones the problem's edge cuts short, so no choice depends on
/// them, but for the rows a microkernel holds whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulWorkItem {
    pub elements: MatmulOperands<ElemType>,
    pub sum: ElemType,
    pub contraction: MatmulContraction,
    pub layouts: MatmulOperands<MatmulLayout>,
    /// The block-split space a packed operand puts the product in, where the microkernel
    /// multiplies by its scales as it reads it: none for float operands, or packed ones decoded
    /// before the microkernel sees them.
    pub blocks: Option<MatmulBlocks>,
    pub rows: MatmulRows,
    pub destination: MatmulDestination,
    pub axes: MatmulAxes,
}

/// The block-split space packed operands put a product in: how each packed operand is stored, its
/// reads, its scales and their scheme, whose scales cut the space, and the extents the space
/// compiles in.
///
/// The extents are here because the blocks are compiled into the space: a float product's frees
/// `n` and `k`, and a packed one's cannot without a block-split space whose extents are dynamic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulBlocks {
    pub packings: MatmulOperands<Option<Packing>>,
    /// Rows of the lhs, all of them: a microkernel that does not stage holds every one, so the
    /// weight is read once; one that stages cuts them into boxes.
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatmulBlocks {
    /// What one read of `operand` and of its scales brings: a float one's a value, no scales.
    pub fn operand(&self, operand: MatmulOperand) -> Operand {
        self.packings
            .get(operand)
            .map_or(Operand::plain(), |packing| packing.operand)
    }

    /// The two axes `axis` becomes: the blocks the scales of the operand grouping it make of it,
    /// and the position inside one. An axis no scale groups is one block over the whole of it.
    pub fn axes(&self, axis: Axis) -> [(Axis, usize); 2] {
        let extent = self.extent(axis);
        let grouping = [MatmulOperand::Lhs, MatmulOperand::Rhs]
            .map(|operand| self.operand(operand))
            .into_iter()
            .find(|operand| operand.grouping(axis, extent).is_some());
        grouping.unwrap_or(Operand::plain()).axes(axis, extent)
    }

    /// What `axis`, one of the product's, measures.
    pub fn extent(&self, axis: Axis) -> usize {
        match axis {
            M => self.m,
            N => self.n,
            K => self.k,
            _ => panic!("{axis:?} is not one of the product's axes"),
        }
    }

    /// The dims dense rows along `rows` by `columns`, `step` apart, are restated in over the
    /// block-split space: the same bytes, one dim for each axis, `(axis, extent, stride)`,
    /// coarsest first. A dim's blocks are left out where there is one, the tensor constant along
    /// them; its inside always stays, so the tensor is a matrix.
    pub fn restated_dims(
        &self,
        rows: Axis,
        columns: Axis,
        step: usize,
    ) -> Vec<(Axis, usize, usize)> {
        let mut dims = Vec::new();
        for (dim, dim_step) in [(rows, step), (columns, 1)] {
            let [(blocks, block_count), (inside, inside_count)] = self.axes(dim);
            if block_count > 1 {
                dims.push((blocks, block_count, inside_count * dim_step));
            }
            dims.push((inside, inside_count, dim_step));
        }
        dims
    }
}

/// The elements a matrix instruction multiplies a product's operands at, where one of them is not
/// contracted at the element it is stored in. A 16-bit activation against an NVFP4 weight is
/// either product, so the caller names which: the elements cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulContraction {
    /// Each operand at the element it is stored in.
    AsStored,
    /// The lhs, a 16-bit activation, rounded to NVFP4 as each plane loads it, against an NVFP4
    /// rhs: the 4-bit product, on the device's block-scaled instruction.
    LhsRoundedToNvfp4,
    /// The rhs, NVFP4, decoded into the lhs's element as each plane loads it: the lhs's product,
    /// on the instruction the lhs's element runs, the weight read in a quarter of the bytes.
    RhsDecodedToLhs,
}

impl MatmulWorkItem {
    /// The element types a matrix instruction multiplies this item at, as `(lhs, rhs, sum)`: the
    /// stored ones, but for the operand its [`contraction`](MatmulContraction) loads at another.
    pub fn instruction_elements(&self) -> (ElemType, ElemType, ElemType) {
        let MatmulOperands { lhs, rhs, .. } = self.elements;
        match self.contraction {
            MatmulContraction::AsStored => (lhs, rhs, self.sum),
            MatmulContraction::LhsRoundedToNvfp4 => {
                (ElemType::Float(FloatKind::E2M1), rhs, self.sum)
            }
            MatmulContraction::RhsDecodedToLhs => (lhs, lhs, self.sum),
        }
    }

    /// The element this item's packed inputs are decoded into to be contracted: the lhs's where
    /// it is float, the output's where it is packed too.
    pub fn decoded_element(&self) -> ElemType {
        match self.blocks.and_then(|blocks| blocks.packings.lhs) {
            Some(_) => self.elements.out,
            None => self.elements.lhs,
        }
    }

    /// How a microkernel that decodes this item's packed inputs reads each: a packed one
    /// dequantized at [`decoded_element`](Self::decoded_element), a float one as bound.
    pub fn decoding_reads(&self) -> MatmulOperands<MatmulRead> {
        let element = self.decoded_element();
        let packings = self
            .blocks
            .map_or(MatmulOperands::all(None), |blocks| blocks.packings);
        packings.map(|_, packing| match packing {
            Some(_) => MatmulRead::Dequantized(element),
            None => MatmulRead::AsBound,
        })
    }

    /// The product a cube's planes contract once it has decoded this item's packed inputs into
    /// its stages: both inputs at [`decoded_element`](Self::decoded_element), over the axes
    /// inside a block, the rows tiled. A packed rhs runs along `K`, and so does the stage it
    /// decodes into: a col rhs.
    pub fn decoded(&self) -> Self {
        let element = self.decoded_element();
        Self {
            contraction: MatmulContraction::AsStored,
            elements: MatmulOperands {
                lhs: element,
                rhs: element,
                out: self.elements.out,
            },
            sum: self.sum,
            layouts: MatmulOperands {
                rhs: MatmulLayout::Col,
                ..self.layouts
            },
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::INSIDE_BLOCKS,
        }
    }

    /// This item with both inputs read as they are stored, where both are packed into values that
    /// name an element (`e2m1`): what an instruction that reads packed values and their scales as
    /// they lie contracts, the block-scaled one. `None` where an input is float, or packed into
    /// values no element names.
    pub fn as_stored(&self) -> Option<Self> {
        let blocks = self.blocks?;
        let stored = |packing: Option<Packing>| match packing?.scheme.value {
            QuantValue::E2M1 => Some(ElemType::Float(FloatKind::E2M1)),
            _ => None,
        };
        Some(Self {
            elements: MatmulOperands {
                lhs: stored(blocks.packings.lhs)?,
                rhs: stored(blocks.packings.rhs)?,
                out: self.elements.out,
            },
            ..*self
        })
    }

    /// Whether a stage of this item keeps an operand's block scales beside its values: an operand
    /// stored in `e2m1`, staged as its words are stored with the words of its scales.
    pub fn stages_scales_beside(&self) -> bool {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        self.elements.lhs == e2m1 || self.elements.rhs == e2m1
    }

    /// The elements a kernel of every matmul product defines, in the order it names them: the sum,
    /// the lhs, the rhs, the output.
    pub fn kernel_elements(&self) -> [ElemType; 4] {
        [
            self.sum,
            self.elements.lhs,
            self.elements.rhs,
            self.elements.out,
        ]
    }
}

/// How a launch holds a product's rows: every row whole, compiled into the kernel beside a weight
/// read once, or tiled, the rows one more extent the launch reads: the kind of rows a work item
/// holds ([`MatmulRows`]). What a launch compiles in and how it binds the activation both follow
/// from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulRowsKind {
    /// Every row held whole: the rows compiled in, every other extent read at launch.
    Whole,
    /// The rows tiled: every extent read at launch.
    Tiled,
}

impl MatmulRowsKind {
    /// The product's axes a launch holding its rows this way reads at launch rather than compiling
    /// in, so one kernel serves every product they tell apart. A batch or a case's own axis is the
    /// launch's to add.
    pub fn dynamic_axes(self) -> &'static [Axis] {
        match self {
            Self::Whole => &[N, K],
            Self::Tiled => &[M, N, K],
        }
    }
}

/// How many rows of the lhs a work item holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulRows {
    /// As many as the routine's box cuts: the work item is a tile of them.
    Tiled,
    /// Every row, this many of them, held in registers beside each step of the weight.
    Whole(usize),
}

/// Where a product lands.
///
/// A microkernel reads it for what a sink cannot take: a fragment's store intrinsic, which writes
/// through an address, and an add, which reads the cell back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulDestination {
    /// A buffer: an address every write lands at.
    Buffer,
    /// A call each line of the product is handed to, with no address and nothing to read back: a
    /// fused element-wise tail runs in it. It is written a line at a time, each cell once.
    Sink,
}
