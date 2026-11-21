use crate::matmul::{MatmulDim, MatmulDimensions, MatmulOperand, MatmulOperands, MatmulRead};
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Dimension, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::microkernel::StagedOperand;

/// The box one step of a walk through `K` stages of both operands: a cube's, or a plane's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulStage {
    space: Space,
    /// The dimensions the product's dimensions lie along in `space`.
    dimensions: MatmulDimensions,
}

impl MatmulStage {
    /// The box `space`, the product's dimensions along `dimensions`.
    pub fn new(space: Space, dimensions: MatmulDimensions) -> Self {
        Self { space, dimensions }
    }

    /// The box as a space, for what the tile engine reads of it.
    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn extent(&self, dimension: Dimension) -> usize {
        self.space.extent(dimension)
    }

    /// The part of the box `operand` stages, its dimensions in the order `orders` binds it in.
    pub fn operand_space(
        &self,
        operand: MatmulOperand,
        orders: MatmulOperands<BindingOrder>,
    ) -> Space {
        let [outer, inner] = operand.dims();
        let dims = match orders.get(operand) {
            BindingOrder::Stated => [outer, inner],
            BindingOrder::InBufferOrder => [inner, outer],
        };
        self.space
            .subspace(&dims.map(|dim| self.dimensions.of(dim)))
    }

    /// Elements of `operand` a stage holds consecutively, bound in the order `orders` says: the
    /// stage's extent along the dimension the operand is bound innermost along.
    pub fn run(&self, operand: MatmulOperand, orders: MatmulOperands<BindingOrder>) -> usize {
        self.extent(self.dimensions.of(orders.innermost(operand)))
    }

    /// Require of `vectorization` the reads a ring makes of both operands' tiles a stage at a
    /// time, bound in the order `orders` says: stretches of the stage's extent along each one's
    /// contiguous dimension.
    pub fn require_runs(
        &self,
        orders: MatmulOperands<BindingOrder>,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        for operand in MatmulOperand::INPUTS {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.run(operand, orders),
            });
        }
    }

    /// Elements of `operand` a plane reads contiguously over a cube's whole walk of the space
    /// `cube_space`, bound in the order `orders` says: the whole of `K` a cube walks for an operand
    /// bound innermost along it, whose stages continue each other's rows, and one stage's extent
    /// otherwise.
    pub fn walk_run(
        &self,
        cube_space: &Space,
        operand: MatmulOperand,
        orders: MatmulOperands<BindingOrder>,
    ) -> usize {
        match orders.innermost(operand) {
            MatmulDim::K => cube_space.extent(self.dimensions.k),
            dim => self.extent(self.dimensions.of(dim)),
        }
    }

    /// Both operands' boxes of one stage, each stored in its own element of `elements` and bound
    /// in the order `orders` says, for a fill that reads them as `reads` say: an operand it reads
    /// at another element is decoded on its way into the stage where `packed` says it is a packed
    /// input, and converted otherwise (rounded to tf32, read at another float).
    pub fn operands(
        &self,
        elements: MatmulOperands<ElemType>,
        orders: MatmulOperands<BindingOrder>,
        reads: MatmulOperands<MatmulRead>,
        packed: MatmulOperands<bool>,
    ) -> [StagedOperand; 2] {
        MatmulOperand::INPUTS.map(|operand| {
            let staged = StagedOperand::new(
                elements.get(operand),
                self.space
                    .subspace(&operand.dims().map(|dim| self.dimensions.of(dim))),
                self.dimensions.of(orders.innermost(operand)),
            );
            match (reads.get(operand).staged(), packed.get(operand)) {
                (Some(_), true) => staged.decoded_by_fill(),
                (Some(_), false) => staged.converted_by_fill(),
                (None, _) => staged,
            }
        })
    }

    /// Bytes one stage holds of both operands on `hw`, each stored in its own element of
    /// `elements`, laid out as `storage` and bound in the order `orders` says: every block it
    /// holds, each laid out alike over the dimensions inside one.
    pub fn bytes(
        &self,
        hw: &Hardware,
        elements: MatmulOperands<ElemType>,
        storage: &StageStorage,
        orders: MatmulOperands<BindingOrder>,
    ) -> usize {
        let block: usize = self
            .operands(
                elements,
                orders,
                MatmulOperands::all(MatmulRead::AsBound),
                MatmulOperands::all(false),
            )
            .iter()
            .map(|operand| operand.bytes(hw, storage))
            .sum();
        block * self.blocks()
    }

    /// Blocks the stage holds: its extent along every dimension it lies [`beside`](Self::beside).
    fn blocks(&self) -> usize {
        self.beside()
            .map(|dimension| self.space.extent(dimension))
            .product()
    }

    /// The dimensions of the box the stage does not lie along: a batch's, or a block-split space's
    /// blocks, each laid out one deep.
    pub fn beside(&self) -> impl Iterator<Item = Dimension> + '_ {
        let inside = [self.dimensions.m, self.dimensions.n, self.dimensions.k];
        self.space
            .dimensions()
            .filter(move |dimension| !inside.contains(dimension))
    }

    /// The dimensions the stage lies along.
    pub fn dimensions(&self) -> MatmulDimensions {
        self.dimensions
    }

    /// The box `operand`'s stage lands over `dimensions`, bound in the order `orders` says: its
    /// part's extent along each dimension it names, the whole box's along one it lies beside (a
    /// block-split space's blocks of `K`), and one along a dimension the box does not hold. What
    /// one bulk copy moves of the operand a stage.
    pub fn landed_space(
        &self,
        operand: MatmulOperand,
        orders: MatmulOperands<BindingOrder>,
        dimensions: &[Dimension],
    ) -> Space {
        let own = self.operand_space(operand, orders);
        let extents: Vec<_> = dimensions
            .iter()
            .map(|&dimension| {
                let extent = match (own.contains(dimension), self.space.contains(dimension)) {
                    (true, _) => own.extent(dimension),
                    (false, true) => self.space.extent(dimension),
                    (false, false) => 1,
                };
                (dimension, extent)
            })
            .collect();
        Space::new(&extents)
    }
}
