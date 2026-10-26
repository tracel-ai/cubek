use crate::matmul::staged_walk::MatmulStage;
use crate::matmul::{MatmulAxes, MatmulRowsKind, MatmulWorkItem};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{
    Axis, Partitioning, Space, TileSpec,
    space::{ComputeScope, Count, Coverage, CubeOrder, PlaneOrder},
};
use cubek_std::matmul_axes::{K, KB, M, MB, MI, N, NB, NI};

/// How a matmul product places its microkernel's work items over a problem: the tile DSL's levels,
/// which the kernel launches and walks, and the box each cube is handed. Each product builds one
/// from its problem; a cube microkernel reads its box and stage off it. Stated along matmul's own
/// `M`, `N` and `K`, or over a packed product's block-split space ([`MatmulSpace`]).
///
/// Its outermost level is the cubes, and the one inside it the walk a cube takes through its box;
/// or, for a routine whose planes share nothing, the launch's planes, which a plane order groups
/// in cubes. Either way the box one cube covers and the one a step of its walk stages are
/// questions this answers for every routine alike.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulPartitioning {
    partitioning: Partitioning,
    /// The box one cube covers, its `k` a cube's run of the walk where several share a box.
    cube_box: Space,
    batch_axes: Vec<Axis>,
    rows_kind: MatmulRowsKind,
    space: MatmulSpace,
}

/// The space a partitioning is stated over, which its walk, its stage and its output's axes are
/// read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulSpace {
    /// Matmul's own `M`, `N` and `K`, and the batches beside them.
    Plain,
    /// A packed product's: each axis as the blocks its rhs's scales make of it and the position
    /// inside one ([`MatmulBlocks`](crate::matmul::MatmulBlocks)), every extent
    /// compiled in.
    BlockSplit,
}

impl MatmulSpace {
    /// The space `space` is: block-split where it carries `K`'s blocks.
    fn of(space: &Space) -> Self {
        match space.contains(KB) {
            true => Self::BlockSplit,
            false => Self::Plain,
        }
    }

    /// The axes a stage of a product over this space lies along: matmul's own, or those inside a
    /// block.
    pub fn stage_axes(self) -> MatmulAxes {
        match self {
            Self::Plain => MatmulAxes::MNK,
            Self::BlockSplit => MatmulAxes::INSIDE_BLOCKS,
        }
    }

    /// The axis a cube's walk steps through `K` along: `K`, or its blocks.
    pub fn walked(self) -> Axis {
        match self {
            Self::Plain => K,
            Self::BlockSplit => KB,
        }
    }

    /// The axis the product's rows lie along inside a block: `M`, or the rows inside its one
    /// block.
    pub fn rows_axis(self) -> Axis {
        match self {
            Self::Plain => M,
            Self::BlockSplit => MI,
        }
    }

    /// The axes the output's rows and columns are cut short along at the problem's edge.
    pub fn output_axes(self) -> [Axis; 2] {
        match self {
            Self::Plain => [M, N],
            Self::BlockSplit => [MI, NI],
        }
    }
}

impl MatmulPartitioning {
    /// What a cube microkernel declines an item on other axes with: it places and stages in
    /// matmul's own `M`, `N` and `K`.
    pub const OTHER_AXES: Refusal = Refusal::Declined {
        why: "a cube microkernel works in matmul's own M, N and K",
    };

    /// Whether `item` lies along matmul's own axes, which cube scope works in.
    ///
    /// # Errors
    ///
    /// [`OTHER_AXES`](Self::OTHER_AXES), for an item on other axes.
    pub fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.axes == MatmulAxes::MNK {
            true => Ok(()),
            false => Err(Self::OTHER_AXES),
        }
    }

    /// `partitioning`, its cubes each handed `cube_box`, one batch along each of `batch_axes`, the
    /// rows held as `rows_kind` says.
    pub fn new(
        partitioning: Partitioning,
        cube_box: Space,
        batch_axes: Vec<Axis>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let space = MatmulSpace::of(partitioning.space());
        Self {
            partitioning,
            cube_box,
            batch_axes,
            rows_kind,
            space,
        }
    }

    /// `partitioning`, its cubes each handed the box its cube level cuts.
    pub fn of_cubes(
        partitioning: Partitioning,
        batch_axes: Vec<Axis>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let cube_box = partitioning.box_of(ComputeScope::Cube);
        Self::new(partitioning, cube_box, batch_axes, rows_kind)
    }

    /// `partitioning`, whose outermost level spans the launch's planes in `order`: a cube's box is
    /// one block of that order, a plane's box wider by the tiles the block holds along each axis.
    pub fn of_launch_planes(
        partitioning: Partitioning,
        order: &PlaneOrder,
        batch_axes: Vec<Axis>,
        rows_kind: MatmulRowsKind,
    ) -> Self {
        let plane = partitioning.box_of(ComputeScope::Plane);
        let cube_box = Space::new(
            &plane
                .extents()
                .into_iter()
                .map(|(axis, extent)| (axis, extent * order.block_along(axis)))
                .collect::<Vec<_>>(),
        );
        Self::new(partitioning, cube_box, batch_axes, rows_kind)
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
    /// a launch whose operands carry a factor over blocks of an axis needs of that axis, since
    /// the tile engine windows such a factor through extents it knows at expansion. One kernel a
    /// distinct extent of those axes.
    pub fn kernel_compiling(&self, compiled: &[Axis]) -> Partitioning {
        let dynamic: Vec<Axis> = match self.space {
            MatmulSpace::Plain => self
                .batch_axes
                .iter()
                .chain(self.rows_kind.dynamic_axes())
                .copied()
                .filter(|axis| !compiled.contains(axis))
                .collect(),
            MatmulSpace::BlockSplit => {
                let rows_free = self.rows_kind.dynamic_axes().contains(&M)
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
                let rows = self.space.rows_axis();
                let held = self.cube_box.extent(rows);
                Space::new(
                    &space
                        .extents()
                        .into_iter()
                        .map(|(axis, extent)| (axis, if axis == rows { held } else { extent }))
                        .collect::<Vec<_>>(),
                )
            }
        }
    }

    /// The batch axes a cube is handed one of: the problem's longer than one.
    pub fn batch_axes(&self) -> &[Axis] {
        &self.batch_axes
    }

    /// The box one cube covers: one batch of the output's `m × n` box, and the whole of `k`, or
    /// its run of it where several cubes share the box.
    pub fn cube_box(&self) -> Space {
        self.cube_box.clone()
    }

    /// The space it is stated over.
    pub fn space(&self) -> MatmulSpace {
        self.space
    }

    /// The output's rows one cube's box covers, over both digits of `M` in a block-split space.
    pub fn rows(&self) -> u64 {
        let rows = match self.space {
            MatmulSpace::Plain => self.cube_box.extent(M),
            MatmulSpace::BlockSplit => self.cube_box.extent(MB) * self.cube_box.extent(MI),
        };
        rows as u64
    }

    /// The output's columns one cube's box covers, over both digits of `N` in a block-split space.
    pub fn columns(&self) -> u64 {
        let columns = match self.space {
            MatmulSpace::Plain => self.cube_box.extent(N),
            MatmulSpace::BlockSplit => self.cube_box.extent(NB) * self.cube_box.extent(NI),
        };
        columns as u64
    }

    /// Boxes of the output the cubes cover along `m` and along `n`, one batch of it.
    pub fn cubes(&self) -> (usize, usize) {
        let (space, cube) = (self.partitioning.space(), self.cube_box());
        let boxes = |axis| space.extent(axis).div_ceil(cube.extent(axis));
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
            None => self.cube_box(),
        };
        for level in &self.partitioning.levels()[1..] {
            let tile = level.child(&enclosing);
            if level.coverage() == Coverage::Walk && level.tile(self.space.walked()).is_some() {
                return MatmulStage::new(tile, self.space.stage_axes());
            }
            enclosing = tile;
        }
        panic!("the partitioning walks no `K` inside its cubes")
    }

    /// The innermost box: what one instruction, or one unit's block, covers.
    pub fn leaf(&self) -> Space {
        self.partitioning.leaf()
    }

    /// Whether the tiles overhang `axis`: a tile the problem's edge cuts short along it.
    pub fn overhangs(&self, axis: Axis) -> bool {
        self.partitioning.overhangs(axis)
    }

    /// Whether the tiles overhang the output: a tile the output's edge cuts short, which a write
    /// that cannot be masked would reach past.
    pub fn overhangs_output(&self) -> bool {
        self.space
            .output_axes()
            .into_iter()
            .any(|axis| self.overhangs(axis))
    }

    /// Checks that the engine masks `out`'s writes exactly where the tiles overhang it, which is
    /// what a tiling routine decides from: a masked write is one a fragment's store intrinsic
    /// cannot make, so the plane routine bounces its fragments where the tiles overhang. A launch
    /// holding every row whole stores no fragment and frees its columns, so the claim is not its.
    ///
    /// # Panics
    ///
    /// A tiled launch whose output is masked where its tiles do not overhang it.
    pub fn assert_masked_where_overhanging(&self, out: &TileSpec) {
        assert!(
            self.rows_kind == MatmulRowsKind::Whole || !out.is_checked() || self.overhangs_output(),
            "the output's writes are masked where its tiles do not overhang it"
        );
    }

    /// Tiles along `axis` a level distributes its units in turns ([`Count::Distributed`]), where
    /// one does.
    pub fn distributed(&self, axis: Axis) -> Option<usize> {
        self.partitioning
            .levels()
            .iter()
            .find_map(|level| match level.count(axis) {
                Some(Count::Distributed(tiles)) => Some(tiles),
                _ => None,
            })
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
    use cubek_std::matmul_axes::KI;

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
