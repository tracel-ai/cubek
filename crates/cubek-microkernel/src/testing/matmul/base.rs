//! What every matmul microkernel launched alone shares: `out = lhs · rhs` over one cube's box, `lhs`
//! row-major `m × k`, `rhs` `k × n` row-major or its transpose and `out` `m × n`, every operand
//! bound one element a line, and the box one cube of the microkernel's levels covers.

use crate::matmul::cube::MatmulCubeSpace;
use crate::matmul::{
    MatmulAxes, MatmulAxesKind, MatmulDestination, MatmulLayout, MatmulOperands, MatmulReading,
    MatmulRows, MatmulWorkItem,
};
use core::fmt::Debug;
use cubecl::prelude::*;
use cubecl::tile::kind::Boundary;
use cubecl::tile::launch::{BoundaryPolicy, Grid, Refusal};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Launcher, Level, Partitioning, Space, TileSpec, TileTensorLaunch};
use cubek_std::matmul_axes::{K, M, N};

/// The elements of a float product: what both inputs are stored in, what it sums in and what it
/// writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulElements {
    pub input: ElemType,
    pub sum: ElemType,
    pub out: ElemType,
}

impl MatmulElements {
    /// A float product of row-major inputs into a buffer, its tiles cut by the launch.
    pub fn work_item(&self) -> MatmulWorkItem {
        self.work_item_of(MatmulRows::Tiled, MatmulLayout::Row)
    }

    /// [`work_item`](Self::work_item) with its rows held as `rows` says and its rhs laid as `rhs`.
    pub fn work_item_of(&self, rows: MatmulRows, rhs: MatmulLayout) -> MatmulWorkItem {
        MatmulWorkItem {
            elements: MatmulOperands {
                lhs: self.input,
                rhs: self.input,
                out: self.out,
            },
            sum: self.sum,
            layouts: MatmulOperands {
                lhs: MatmulLayout::Row,
                rhs,
                out: MatmulLayout::Row,
            },
            blocks: None,
            rows,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
            reading: MatmulReading::AsStored,
        }
    }
}

/// The rows a product holds whole, every one beside each step of the rhs: a decode step's few.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholeRows(pub usize);

/// A matmul microkernel ready to launch alone on one cube's box.
pub trait MatmulLaunch: Debug {
    /// The elements of the product.
    fn elements(&self) -> MatmulElements;

    /// `m`, `n` and `k` of the problem.
    fn shape(&self) -> [usize; 3];

    /// How the rhs is laid: `[k, n]` row-major, or its transpose, `[n, k]`.
    fn rhs_layout(&self) -> MatmulLayout {
        MatmulLayout::Row
    }

    /// `out = lhs · rhs`, `lhs` row-major `m × k`, `rhs` laid as [`rhs_layout`](Self::rhs_layout)
    /// says and `out` `m × n`.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal>;
}

/// The problem one instance of a launch's outermost scope covers (one cube, or the one plane or
/// unit a cube holds), `steps` steps of `K` deep, and the microkernel's levels over it.
#[derive(Debug, Clone)]
pub struct OneCube {
    /// `m`, `n` and `k`.
    pub shape: [usize; 3],
    pub partitioning: Partitioning,
}

impl OneCube {
    /// Extents no one cube reaches, to read its box off.
    const UNBOUNDED: usize = 1 << 20;

    /// The box `levels` give one instance of `scope`, the launch's outermost, `steps` of its
    /// leaf's `K` deep.
    pub fn new(levels: Vec<Level>, scope: ComputeScope, steps: usize) -> Self {
        let unbounded = Space::new(&[
            (M, Self::UNBOUNDED),
            (N, Self::UNBOUNDED),
            (K, Self::UNBOUNDED),
        ]);
        let probe = Partitioning::new(unbounded, levels.clone());
        let instance = probe.box_of(scope);
        let shape = [
            instance.extent(M),
            instance.extent(N),
            probe.leaf().extent(K) * steps,
        ];
        let [m, n, k] = shape;
        let partitioning = Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), levels);
        Self {
            shape,
            partitioning,
        }
    }

    /// The box as a matmul microkernel reads it: every axis plain, no batch, no overhang.
    pub fn cube_space(&self) -> MatmulCubeSpace {
        MatmulCubeSpace {
            space: self.partitioning.box_of(ComputeScope::Cube),
            leaf: self.partitioning.leaf(),
            axes_kind: MatmulAxesKind::Plain,
            overhangs_output: false,
            batched: false,
        }
    }

    /// The launcher of the box's one cube, of planes `plane_width` wide.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot place it.
    pub(crate) fn launcher(&self, client: &Client, plane_width: u32) -> Result<Launcher, Refusal> {
        let partitioning = &self.partitioning;
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(plane_width),
        };
        Launcher::new(client, partitioning.clone(), partitioning.space(), grid)
    }
}

impl MatmulElements {
    /// `lhs`, `rhs` and `out` bound over `[m, k]`, `[k, n]` (or `[n, k]`, as `rhs_layout` says)
    /// and `[m, n]`, one element a line, the reads and writes past the problem masked.
    pub(crate) fn bound(
        &self,
        [lhs, rhs, out]: [TensorArg; 3],
        rhs_layout: MatmulLayout,
    ) -> [TileTensorLaunch; 3] {
        let rhs_axes: &[_] = match rhs_layout {
            MatmulLayout::Col => &[N, K],
            _ => &[K, N],
        };
        let bound = |arg, element, axes: &[_]| {
            TileTensorLaunch::new(
                arg,
                element,
                1,
                TileSpec::direct(axes).boundary(BoundaryPolicy::Every(Boundary::Zero)),
            )
        };
        [
            bound(lhs, self.input, &[M, K]),
            bound(rhs, self.input, rhs_axes),
            bound(out, self.out, &[M, N]),
        ]
    }
}
