use crate::matmul::MatmulPartitioning;
use crate::matmul::{MatmulAxes, MatmulDim, MatmulOperand, MatmulOperands, MatmulOrder};
use crate::staging::StagedOperand;
use cubecl::ir::ElemType;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Axis, Space, StageStorage};
use cubek_std::Hardware;

/// The box one step of a walk through `K` stages of both operands: a cube's, or a plane's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulStage {
    space: Space,
    /// The axes the product's dimensions lie along in `space`.
    axes: MatmulAxes,
}

impl MatmulStage {
    /// The box `space`, the product's dimensions along `axes`.
    pub fn new(space: Space, axes: MatmulAxes) -> Self {
        Self { space, axes }
    }

    /// The box as a space, for what the tile engine reads of it.
    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn extent(&self, axis: Axis) -> usize {
        self.space.extent(axis)
    }

    /// The part of the box `operand` stages, its axes in the order `orders` binds it in.
    pub fn operand_space(
        &self,
        operand: MatmulOperand,
        orders: MatmulOperands<MatmulOrder>,
    ) -> Space {
        let [outer, inner] = operand.dims();
        let dims = match orders.get(operand) {
            MatmulOrder::Stated => [outer, inner],
            MatmulOrder::InBufferOrder => [inner, outer],
        };
        self.space.subspace(&dims.map(|dim| self.axes.of(dim)))
    }

    /// Elements of `operand` a stage holds consecutively, bound in the order `orders` says: the
    /// stage's extent along the axis the operand is bound innermost along.
    pub fn run(&self, operand: MatmulOperand, orders: MatmulOperands<MatmulOrder>) -> usize {
        self.extent(self.axes.of(orders.innermost(operand)))
    }

    /// Require of `vectorization` the reads a ring makes of both operands' tiles a stage at a
    /// time, bound in the order `orders` says: stretches of the stage's extent along each one's
    /// contiguous axis.
    pub fn require_runs(
        &self,
        orders: MatmulOperands<MatmulOrder>,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        for operand in MatmulOperand::INPUTS {
            vectorization.require(VectorRequirement::Access {
                operand,
                elements: self.run(operand, orders),
            });
        }
    }

    /// Elements of `operand` a plane reads contiguously over a cube's whole walk of `cube`, bound
    /// in the order `orders` says: the whole of `K` a cube walks for an operand bound innermost
    /// along it, whose stages continue each other's rows, and one stage's extent otherwise.
    pub fn walk_run(
        &self,
        partitioning: &MatmulPartitioning,
        operand: MatmulOperand,
        orders: MatmulOperands<MatmulOrder>,
    ) -> usize {
        match orders.innermost(operand) {
            MatmulDim::K => partitioning.cube_box().extent(self.axes.k),
            dim => self.extent(self.axes.of(dim)),
        }
    }

    /// Both operands' boxes of one stage, each stored in its own element of `elements` and bound
    /// in the order `orders` says.
    pub fn operands(
        &self,
        elements: MatmulOperands<ElemType>,
        orders: MatmulOperands<MatmulOrder>,
    ) -> [StagedOperand; 2] {
        MatmulOperand::INPUTS.map(|operand| StagedOperand {
            element: elements.get(operand),
            space: self
                .space
                .subspace(&operand.dims().map(|dim| self.axes.of(dim))),
            innermost: self.axes.of(orders.innermost(operand)),
        })
    }

    /// Bytes one stage holds of both operands on `hw`, each stored in its own element of
    /// `elements`, laid out as `storage` and bound in the order `orders` says: every block it
    /// holds, each laid out alike over the axes inside one.
    pub fn bytes(
        &self,
        hw: &Hardware,
        elements: MatmulOperands<ElemType>,
        storage: &StageStorage,
        orders: MatmulOperands<MatmulOrder>,
    ) -> usize {
        let block: usize = self
            .operands(elements, orders)
            .iter()
            .map(|operand| operand.bytes(hw, storage))
            .sum();
        block * self.blocks()
    }

    /// Blocks the stage holds: its extent along every axis it lies [`beside`](Self::beside).
    fn blocks(&self) -> usize {
        self.beside().map(|axis| self.space.extent(axis)).product()
    }

    /// The axes of the box the stage does not lie along: a batch's, or a block-split space's
    /// blocks, each laid out one deep.
    pub fn beside(&self) -> impl Iterator<Item = Axis> + '_ {
        let inside = [self.axes.m, self.axes.n, self.axes.k];
        self.space.axes().filter(move |axis| !inside.contains(axis))
    }

    /// The axes the stage lies along.
    pub fn axes(&self) -> MatmulAxes {
        self.axes
    }
}
