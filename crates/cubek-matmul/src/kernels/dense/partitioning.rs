use cubecl::tile::launch::Decline;
use cubecl::tile::{
    Dimension, Partitioning, Space, TileSpec,
    space::{ComputeScope, Coverage, CubeOrder, PlaneOrder},
};
use cubek_microkernel::matmul::cube::MatmulCubeSpace;
use cubek_microkernel::matmul::{MatmulDimensionsKind, MatmulRowsKind, MatmulStage};
use cubek_std::matmul_dimensions::{M, MB, MI, N, NB, NI};

/// A tiled launch whose output is masked where its tiles do not overhang it.
const MASKED_WITHOUT_OVERHANG: Decline = Decline::Broken {
    condition: "the output's writes are masked only where its tiles overhang it",
    cause: None,
};

/// How a matmul product places its microkernel's work items over a problem: the tile DSL's levels,
/// which the kernel launches and walks, and the box each cube is handed. Each product builds one
/// from its problem; a cube microkernel is handed the space its cube covers
/// ([`cube_space`](Self::cube_space)). Stated along matmul's own `M`, `N` and `K`, or over a packed
/// product's block-split space ([`MatmulDimensionsKind`]).
///
/// Its outermost level is the cubes, and the one inside it the walk a cube takes through its box;
/// or, for a routine whose planes share nothing, the launch's planes, which a plane order groups
/// in cubes. Either way the box one cube covers and the one a step of its walk stages are
/// questions this answers for every routine alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulPartitioning {
    partitioning: Partitioning,
    /// The box one cube covers, its `k` a cube's run of the walk where several share a box.
    cube: Space,
    batch_dimensions: Vec<Dimension>,
    rows_kind: MatmulRowsKind,
    dimensions_kind: MatmulDimensionsKind,
}

impl MatmulPartitioning {
    /// `partitioning`, its cubes each covering `cube`, one batch along each of `batch_dimensions`,
    /// the rows held as `rows_kind` says.
    pub fn new(
        partitioning: Partitioning,
        cube: Space,
        batch_dimensions: Vec<Dimension>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let dimensions_kind = MatmulDimensionsKind::of(partitioning.space());
        Self {
            partitioning,
            cube,
            batch_dimensions,
            rows_kind,
            dimensions_kind,
        }
    }

    /// `partitioning`, its cubes each handed the box its cube level cuts.
    pub fn of_cubes(
        partitioning: Partitioning,
        batch_dimensions: Vec<Dimension>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let cube = partitioning.box_of(ComputeScope::Cube);
        Self::new(partitioning, cube, batch_dimensions, rows_kind)
    }

    /// `partitioning`, whose outermost level spans the launch's planes in `order`: a cube's box is
    /// one block of that order, a plane's box wider by the tiles the block holds along each
    /// dimension.
    pub fn of_launch_planes(
        partitioning: Partitioning,
        order: &PlaneOrder,
        batch_dimensions: Vec<Dimension>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let plane = partitioning.box_of(ComputeScope::Plane);
        let cube = Space::new(
            &plane
                .extents()
                .into_iter()
                .map(|(dimension, extent)| (dimension, extent * order.block_along(dimension)))
                .collect::<Vec<_>>(),
        );
        Self::new(partitioning, cube, batch_dimensions, rows_kind)
    }

    /// How the launch it places reads the rows.
    pub fn rows_kind(&self) -> MatmulRowsKind {
        self.rows_kind
    }

    /// The partitioning a launch compiles: this one with every extent its rows' kind leaves to the
    /// launch freed, so one kernel serves every product those extents tell apart. A block-split
    /// space compiles its blocks in, and frees only the rows inside the one block no scale cuts
    /// `M` into, where its rows' kind leaves them to the launch: a prompt's rows change with every
    /// call, and no factor is windowed along them. A launch holding every row whole compiles in the
    /// rows its cube's box holds rather than the product's, so every count a box holds alike runs
    /// one kernel, the rows past the product's masked against the lhs and the output as stored.
    pub fn kernel(&self) -> Partitioning {
        self.kernel_compiling(&[])
    }

    /// [`kernel`](Self::kernel) with the extents of `compiled` compiled in rather than freed: what
    /// a launch whose operands carry a factor over blocks of a dimension needs of that dimension,
    /// since the tile engine windows such a factor through extents it knows at expansion. One
    /// kernel a distinct extent of those dimensions.
    pub fn kernel_compiling(&self, compiled: &[Dimension]) -> Partitioning {
        let dynamic: Vec<Dimension> = match self.dimensions_kind {
            MatmulDimensionsKind::Plain => self
                .batch_dimensions
                .iter()
                .chain(self.rows_kind.dynamic_dimensions())
                .copied()
                .filter(|dimension| !compiled.contains(dimension))
                .collect(),
            MatmulDimensionsKind::BlockSplit => {
                let rows_free = self.rows_kind.dynamic_dimensions().contains(&M)
                    && self.partitioning.space().extent(MB) == 1
                    && !compiled.contains(&MI);
                rows_free.then_some(MI).into_iter().collect()
            }
        };
        Partitioning::new(
            self.compiled_space().with_dynamic(&dynamic),
            self.partitioning.levels().to_vec(),
        )
    }

    /// The space a kernel states before any extent is freed: the problem's, its rows the ones a
    /// cube's box holds where every row is held whole.
    fn compiled_space(&self) -> Space {
        let space = self.partitioning.space();
        match self.rows_kind {
            MatmulRowsKind::Tiled => space.clone(),
            MatmulRowsKind::Whole => {
                let rows = self.dimensions_kind.rows_dimension();
                let held = self.cube.extent(rows);
                Space::new(
                    &space
                        .extents()
                        .into_iter()
                        .map(|(dimension, extent)| {
                            (dimension, if dimension == rows { held } else { extent })
                        })
                        .collect::<Vec<_>>(),
                )
            }
        }
    }

    /// The batch dimensions a cube is handed one of: the problem's longer than one.
    pub fn batch_dimensions(&self) -> &[Dimension] {
        &self.batch_dimensions
    }

    /// The space one cube covers: one batch of the output's `m × n` box, and the whole of `k`, or
    /// its run of it where several cubes share the box; with what of this placement a cube
    /// microkernel's declines read.
    pub fn cube_space(&self) -> MatmulCubeSpace {
        MatmulCubeSpace {
            space: self.cube.clone(),
            leaf: self.leaf(),
            dimensions_kind: self.dimensions_kind,
            overhangs_output: self.overhangs_output(),
            batched: !self.batch_dimensions.is_empty(),
        }
    }

    /// The dimensions it is stated over.
    pub fn dimensions_kind(&self) -> MatmulDimensionsKind {
        self.dimensions_kind
    }

    /// The output's cells one cube's box covers: its [`rows`](Self::rows) by its
    /// [`columns`](Self::columns).
    pub fn cells(&self) -> usize {
        (self.rows() * self.columns()) as usize
    }

    /// The output's rows one cube's box covers, over both dimensions `M` is split into in a
    /// block-split space.
    pub fn rows(&self) -> u64 {
        let rows = match self.dimensions_kind {
            MatmulDimensionsKind::Plain => self.cube.extent(M),
            MatmulDimensionsKind::BlockSplit => self.cube.extent(MB) * self.cube.extent(MI),
        };
        rows as u64
    }

    /// The output's columns one cube's box covers, over both dimensions `N` is split into in a
    /// block-split space.
    pub fn columns(&self) -> u64 {
        let columns = match self.dimensions_kind {
            MatmulDimensionsKind::Plain => self.cube.extent(N),
            MatmulDimensionsKind::BlockSplit => self.cube.extent(NB) * self.cube.extent(NI),
        };
        columns as u64
    }

    /// Boxes of the output the cubes cover along `m` and along `n`, one batch of it.
    pub fn cubes(&self) -> (usize, usize) {
        let (space, cube) = (self.partitioning.space(), self.cube.clone());
        let boxes = |dimension| space.extent(dimension).div_ceil(cube.extent(dimension));
        (boxes(M), boxes(N))
    }

    /// The order the cubes take their boxes in: where the planes span the launch, the order its
    /// blocks are taken in.
    pub fn cube_order(&self) -> CubeOrder {
        match self.partitioning.levels()[0].plane_order() {
            Some(order) => order.block_order(),
            None => self
                .partitioning
                .level_distributed(ComputeScope::Cube)
                .expect("every matmul partitioning hands its cubes a box or its planes the launch")
                .order(),
        }
    }

    /// The box one step of the walk through `K` covers: what a staged routine or microkernel
    /// stages at once. The walk is the first level inside the cubes that steps `K` in sequence
    /// (its blocks, in a block-split space):
    /// the cube's own, where it stages for all its planes, or each plane's, where the cube's
    /// planes each stage their own.
    ///
    /// # Panics
    ///
    /// A partitioning that walks no `K` inside its cubes.
    pub fn stage(&self) -> MatmulStage {
        // Where the planes span the launch, the levels inside the outermost cut a plane's box.
        let mut enclosing = match self.partitioning.levels()[0].plane_order() {
            Some(_) => self.partitioning.box_of(ComputeScope::Plane),
            None => self.cube.clone(),
        };
        for level in &self.partitioning.levels()[1..] {
            let tile = level.child(&enclosing);
            if level.coverage() == Coverage::Walk
                && level.tile(self.dimensions_kind.walked()).is_some()
            {
                return MatmulStage::new(tile, self.dimensions_kind.stage_dimensions());
            }
            enclosing = tile;
        }
        panic!("the partitioning walks no `K` inside its cubes")
    }

    /// The innermost box: what one instruction, or one unit's block, covers.
    pub fn leaf(&self) -> Space {
        self.partitioning.leaf()
    }

    /// Whether the tiles overhang `dimension`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, dimension: Dimension) -> bool {
        self.partitioning.overhangs(dimension)
    }

    /// The problem's space: every dimension at its extent, or dynamic where a launch states it.
    pub fn space(&self) -> &Space {
        self.partitioning.space()
    }

    /// Whether the tiles overhang the output: a tile the output's edge cuts short, which a write
    /// that cannot be masked would reach past.
    pub fn overhangs_output(&self) -> bool {
        self.dimensions_kind
            .output_dimensions()
            .into_iter()
            .any(|dimension| self.overhangs(dimension))
    }

    /// Checks that the engine masks `out`'s writes exactly where the tiles overhang it, which is
    /// what a tiling routine decides from: a masked write is one a fragment's store intrinsic
    /// cannot make, so the plane routine bounces its fragments where the tiles overhang. A launch
    /// holding every row whole stores no fragment and frees its columns, so the claim is not its.
    ///
    /// # Errors
    ///
    /// [`MASKED_WITHOUT_OVERHANG`]: a tiled launch whose output is masked where its tiles do not
    /// overhang it.
    pub fn check_masked_where_overhanging(&self, out: &TileSpec) -> Result<(), Decline> {
        let as_claimed =
            self.rows_kind == MatmulRowsKind::Whole || !out.is_checked() || self.overhangs_output();
        match as_claimed {
            true => Ok(()),
            false => Err(MASKED_WITHOUT_OVERHANG),
        }
    }

    /// Tiles along `dimension` a level distributes its units in turns, where one does
    /// ([`Partitioning::distributed`]).
    pub fn distributed(&self, dimension: Dimension) -> Option<usize> {
        self.partitioning.distributed(dimension)
    }

    /// Whether its cubes share the boxes as one flat index, each walking a run of them
    /// ([`MatmulView::persistent_partitioning`](crate::kernels::MatmulView::persistent_partitioning)),
    /// rather than each taking one.
    pub fn shares_boxes(&self) -> bool {
        self.partitioning
            .levels()
            .iter()
            .any(|level| level.shared_by().is_some())
    }

    /// The tile DSL's partitioning, for what it reads of it: the launch and the binding.
    pub fn as_tile(&self) -> &Partitioning {
        &self.partitioning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::tile::Levels;
    use cubek_std::matmul_dimensions::{KB, KI};

    /// A packed product of `rows` in `row_blocks` blocks along `M`, its rows tiled.
    fn packed(row_blocks: usize, rows: usize) -> MatmulPartitioning {
        let space = Space::new(&[
            (MB, row_blocks),
            (MI, rows),
            (NB, 4),
            (NI, 16),
            (KB, 8),
            (KI, 16),
        ]);
        let levels = Levels::leaf(&[(MI, 16), (NI, 16), (KI, 16)])
            .cubes(&[MI, NI])
            .ordered(CubeOrder::RowMajor)
            .batches(&[])
            .build();
        MatmulPartitioning::of_cubes(
            Partitioning::new(space, levels),
            Vec::new(),
            MatmulRowsKind::Tiled,
        )
    }

    /// Tiled rows inside the one block along `M` are read at launch, so every prompt length runs
    /// one kernel; rows over several blocks, which a factor windows along, stay compiled in.
    #[test]
    fn a_packed_kernel_reads_the_rows_of_its_one_block_at_launch() {
        assert!(packed(1, 33).kernel().space().is_dynamic(MI));
        assert_eq!(packed(1, 33).kernel(), packed(1, 34).kernel());
        assert!(!packed(2, 33).kernel().space().is_dynamic(MI));
        assert!(!packed(1, 33).kernel_compiling(&[MI]).space().is_dynamic(MI));
    }
}
