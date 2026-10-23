use crate::kernels::{MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED, MatmulRoutineProblem};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{
    Axis, Levels, Partitioning, Space,
    space::{ComputeScope, CubeOrder, PlaneOrder},
};
use cubek_microkernel::matmul::MatmulPartitioning;
use cubek_microkernel::matmul::{
    MATMUL_PACKED_LHS_SPLIT, MatmulAxes, MatmulBlocks, MatmulLayout, MatmulMicrokernel,
    MatmulOperands, MatmulRows, MatmulRowsKind, MatmulStep, MatmulWorkItem,
};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N};

/// What a launch holding every row whole declines whatever the device: an activation or an output
/// that does not hold every batch, whose batches no merge makes rows of one matrix.
pub const MATMUL_WHOLE_ROWS_BATCHES_UNFOLDED: Refusal = Refusal::Declined {
    why: "the microkernel gemv folds an activation's batches into rows, so it reads an lhs and an output holding every batch",
};

/// What a launch holding every row whole declines whatever the device: a weight held per batch,
/// since such a launch reads one weight matrix against every row, and none has a batch axis to
/// step through several.
pub const MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH: Refusal = Refusal::Declined {
    why: "a launch holding every row whole reads one weight matrix against every row, so it reads no weight held per batch",
};

/// What a launch declines whatever the device: two packed operands whose scales group `K` in
/// blocks of two lengths, so no one block-split space has both their blocks whole.
pub const MATMUL_PACKED_BLOCKS_DISAGREE: Refusal = Refusal::Declined {
    why: "two packed operands are read in one block-split space, so their scales group k alike",
};

/// A [`MatmulRoutineProblem`] as a launch holding its rows as its [`MatmulRowsKind`] says reads
/// it: the rows tiled and the batches distributed over cubes, or every row whole, the batches
/// merged into the rows, since each of them reads the same weight.
///
/// Against a packed operand, the launch reads the problem in its block-split
/// [`space`](Self::space): each of `m`, `n` and `k` the tiles its scales are constant inside and
/// the position within one.
///
/// It borrows the problem and adds only the reading, derived wherever a routine needs it and never
/// carried in a blueprint. [`rows`](Self::rows) is what the work item holds, and the one reading of
/// the rows to take: a problem read off a tuning key states one batch's rows in `stated_rows`
/// beside its batches, while one read off tensors that hold every row whole states them all there,
/// its batches already merged.
#[derive(Debug, Clone, Copy)]
pub struct MatmulView<'a> {
    problem: &'a MatmulRoutineProblem,
    rows_kind: MatmulRowsKind,
}

impl<'a> MatmulView<'a> {
    /// One cube a box of the output: the count every routine but the hypercube's distributes.
    pub const ONE_CUBE_A_BOX: NonZeroUsize = NonZeroUsize::MIN;

    /// The element every microkernel sums in, whatever it reads and writes, and that an output its
    /// cubes add into is held in.
    pub const SUM: ElemType = ElemType::Float(FloatKind::F32);

    /// Why a problem a routine prepared has a view: `prepare` derives it first and declines what
    /// has none, so `estimate`, `time` and the launch read one that exists. Handing them a problem
    /// `prepare` declined panics with it.
    pub const PREPARED: &'static str = "a problem a routine prepared has its view";

    /// `problem` as a launch on `M` reads it, holding the rows as its
    /// [`ROWS_KIND`](MatmulMicrokernel::ROWS_KIND) says.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn of<M: MatmulMicrokernel>(problem: &'a MatmulRoutineProblem) -> Result<Self, Refusal> {
        Self::new(problem, M::ROWS_KIND)
    }

    /// `problem` as a launch holding its rows as `rows_kind` says reads it.
    ///
    /// # Errors
    ///
    /// [`MATMUL_PACKED_BLOCKS_DISAGREE`], for two packed operands whose scales cut `K` apart.
    /// Where every row is held whole: [`MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED`], for a
    /// weight lined along neither axis; [`MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH`], for one held per
    /// batch; [`MATMUL_WHOLE_ROWS_BATCHES_UNFOLDED`], for an activation or an output that does not
    /// hold every batch, unless a table is read per batch, which keeps the batches apart.
    pub fn new(
        problem: &'a MatmulRoutineProblem,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Refusal> {
        if let MatmulOperands {
            lhs: Some(lhs),
            rhs: Some(rhs),
            ..
        } = problem.packings
            && let (Some(lhs), Some(rhs)) = (
                lhs.operand.grouping(K, problem.k),
                rhs.operand.grouping(K, problem.k),
            )
            && lhs != rhs
        {
            return Err(MATMUL_PACKED_BLOCKS_DISAGREE);
        }
        if rows_kind == MatmulRowsKind::Whole {
            if problem.layouts.rhs == MatmulLayout::Strided {
                return Err(MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED);
            }
            if problem.batches_held.rhs > 1 {
                return Err(MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH);
            }
            let batches: usize = problem.batches.iter().product();
            let merges = problem.batches_held.lhs == batches && problem.batches_held.out == batches;
            if problem.index.is_none() && !merges {
                return Err(MATMUL_WHOLE_ROWS_BATCHES_UNFOLDED);
            }
        }
        Ok(Self { problem, rows_kind })
    }

    /// The rows a work item holds: every row, the batches merged in, or one batch's.
    pub fn rows(&self) -> usize {
        self.read().rows
    }

    /// The problem partitioned with `inside` a cube's box, the cubes cutting `axes` and taking
    /// their boxes in `order`, and `cubes_along_k` of them sharing every box
    /// ([`ONE_CUBE_A_BOX`](Self::ONE_CUBE_A_BOX) for a box to itself), each taking a run of its `K`
    /// walk (the tile DSL's [`Levels::across`]). A box's cubes ride the grid's third dimension
    /// beside the batches, so an order still strips the first two. Where every row is held whole,
    /// `inside` holds them all and there are no batches left to distribute, so the cubes cut only
    /// the columns.
    ///
    /// The count is the one asked for, whatever the walk: it is part of what a kernel compiles, and
    /// one fitted to each problem's steps would compile a kernel per `K`. The tile DSL distributes
    /// each cube the walk's steps over the count rounded up, so where the count does not divide the
    /// steps the trailing cubes' runs come up short or empty; an empty one sums nothing and skips
    /// its drain.
    pub fn partitioning(
        &self,
        inside: Levels,
        axes: &[Axis],
        order: CubeOrder,
        cubes_along_k: NonZeroUsize,
    ) -> MatmulPartitioning {
        let distributed_batch_axes = self.distributed_batch_axes();
        let cubes = match cubes_along_k.get() {
            1 => inside.cubes(axes),
            cubes => inside.cubes(&[axes, &[K]].concat()).across(K, cubes),
        };
        let levels = cubes
            .ordered(order)
            .batches(&distributed_batch_axes)
            .build();
        let partitioning = Partitioning::new(self.space(), levels);
        let tile = self.one_slab(partitioning.box_of(ComputeScope::Cube));
        let cube_box = match cubes_along_k.get() {
            1 => tile,
            // The cube level's own tile is one step of `K` where cubes share a box, distributed
            // in runs: the box states the run, as long as the longest cube's.
            cubes => {
                let step = tile.extent(K);
                let run = self.problem.k.div_ceil(step).max(1).div_ceil(cubes) * step;
                Space::new(
                    &tile
                        .extents()
                        .into_iter()
                        .map(|(axis, extent)| (axis, if axis == K { run } else { extent }))
                        .collect::<Vec<_>>(),
                )
            }
        };
        MatmulPartitioning::new(
            partitioning,
            cube_box,
            distributed_batch_axes,
            self.rows_kind,
        )
    }

    /// The problem partitioned with `inside` a plane's box, every box `axes` cut a plane of the
    /// launch's, the planes taking them in `order`: what a routine whose planes share nothing
    /// launches, which cube holds a plane being the launch's grouping alone. The batches ride
    /// outermost, each plane one batch's box. A cube's box is a block's of the order.
    pub fn plane_partitioning(
        &self,
        inside: Levels,
        axes: &[Axis],
        order: PlaneOrder,
    ) -> MatmulPartitioning {
        let distributed_batch_axes = self.distributed_batch_axes();
        let levels = inside
            .launch_planes(&[axes, &distributed_batch_axes].concat(), order.clone())
            .build();
        MatmulPartitioning::of_launch_planes(
            Partitioning::new(self.space(), levels),
            &order,
            distributed_batch_axes,
            self.rows_kind,
        )
    }

    /// The problem partitioned with `inside` a cube's box, the cubes cutting `axes`, every box's
    /// steps of `K` laid end to end and shared between `cubes` cubes, each taking a run of them
    /// that may start inside one box and end inside another (the tile DSL's [`Levels::shared_by`]).
    /// A cube's box holds the whole of `K`. The runs follow the boxes in the order of the line, so
    /// no cube order applies.
    pub fn shared_partitioning(
        &self,
        inside: Levels,
        axes: &[Axis],
        cubes: NonZeroUsize,
    ) -> MatmulPartitioning {
        let distributed_batch_axes = self.distributed_batch_axes();
        let levels = inside
            .cubes(axes)
            .shared_by(cubes.get())
            .batches(&distributed_batch_axes)
            .build();
        MatmulPartitioning::of_cubes(
            Partitioning::new(self.space(), levels),
            distributed_batch_axes,
            self.rows_kind,
        )
    }

    /// `tile`, one slab wide along the axis a table's entry picks along: no level cuts that axis,
    /// so a box spans every slab, but each cube's tile holds the one its entry names, and so does
    /// the stage it fills. Any other tile as it is.
    fn one_slab(&self, tile: Space) -> Space {
        match self.problem.index.and_then(|index| index.axis()) {
            Some((stacked, _)) => Space::new(
                &tile
                    .extents()
                    .into_iter()
                    .map(|(axis, extent)| (axis, if axis == stacked { 1 } else { extent }))
                    .collect::<Vec<_>>(),
            ),
            None => tile,
        }
    }

    /// The space a launch cuts: the batches its cubes are handed one of, then the rows, `n` and
    /// `k`. Against a packed operand, each of the three is the tiles its scales are constant
    /// inside and the position within one, so a microkernel's tiles land on whole scale tiles.
    /// Last, the axis a table's entry picks along, which no level cuts: a stacked rhs's slabs.
    pub fn space(&self) -> Space {
        let product = [(M, self.rows()), (N, self.problem.n), (K, self.problem.k)];
        let product = match self.blocks() {
            None => product.to_vec(),
            Some(blocks) => [M, N, K]
                .into_iter()
                .flat_map(|axis| blocks.axes(axis))
                .collect(),
        };
        let indexed = self.problem.index.and_then(|index| index.axis());
        Space::new(
            &[
                self.distributed_batches(),
                product,
                indexed.into_iter().collect(),
            ]
            .concat(),
        )
    }

    /// Whether a packed weight is a whole number of its [step](MatmulStep) on `hw`, each read of
    /// it inside one scale tile, and a packed activation whole words of itself in each of the
    /// weight's reads along `K`, the runs the two are contracted in. A float weight has no step to
    /// close.
    ///
    /// # Errors
    ///
    /// [`MATMUL_STEPS_UNCLOSED`](cubek_microkernel::matmul::MATMUL_STEPS_UNCLOSED), and
    /// [`MATMUL_PACKED_LHS_SPLIT`](cubek_microkernel::matmul::MATMUL_PACKED_LHS_SPLIT).
    pub fn check_steps(&self, hw: &Hardware) -> Result<(), Refusal> {
        let MatmulOperands { lhs, rhs, .. } = self.problem.packings;
        let Some(rhs) = rhs else {
            return Ok(());
        };
        MatmulStep::new(hw, &rhs.operand, self.problem.elements.lhs).check(
            &rhs.operand,
            self.problem.n,
            self.problem.k,
        )?;
        match lhs {
            Some(lhs) if !rhs.operand.read(K).is_multiple_of(lhs.operand.read(K)) => {
                Err(MATMUL_PACKED_LHS_SPLIT)
            }
            _ => Ok(()),
        }
    }

    /// What every microkernel is handed, summing in [`SUM`](Self::SUM): its rows held whole, every
    /// one, or tiled. Against a packed operand, the operands' blocks ([`MatmulBlocks`]), over the
    /// axes inside a block.
    pub fn work_item(&self) -> MatmulWorkItem {
        let blocks = self.blocks();
        MatmulWorkItem {
            elements: self.problem.elements,
            sum: Self::SUM,
            contraction: self.problem.contraction,
            layouts: self.problem.layouts,
            axes: match blocks {
                None => MatmulAxes::MNK,
                Some(_) => MatmulAxes::INSIDE_BLOCKS,
            },
            blocks,
            rows: self.read().work_item_rows,
            destination: self.problem.destination,
        }
    }

    /// The block-split space packed operands put the product in, the rows those a work item
    /// holds; none where every operand is float.
    pub fn blocks(&self) -> Option<MatmulBlocks> {
        let packed = self.problem.packings != MatmulOperands::all(None);
        packed.then(|| MatmulBlocks {
            packings: self.problem.packings,
            m: self.rows(),
            n: self.problem.n,
            k: self.problem.k,
        })
    }

    /// The problem's rows as the view's kind reads them: those a work item holds and those its
    /// microkernel is told.
    fn read(&self) -> RowsReading {
        match (self.rows_kind, self.problem.index) {
            (MatmulRowsKind::Whole, None) => self.read_folded(),
            (MatmulRowsKind::Whole, Some(_)) => self.read_per_batch(),
            (MatmulRowsKind::Tiled, _) => self.read_tiled(),
        }
    }

    /// Every row of one batch held whole, each batch handed to the cubes: a table read per batch
    /// tells the batches apart, so none merges into another's rows.
    fn read_per_batch(&self) -> RowsReading {
        RowsReading {
            rows: self.problem.stated_rows,
            work_item_rows: MatmulRows::Whole(self.problem.stated_rows),
        }
    }

    /// Every row held whole: the batches merged into the rows, and none left for the cubes.
    fn read_folded(&self) -> RowsReading {
        let rows = self.problem.stated_rows * self.problem.batches.iter().product::<usize>();
        RowsReading {
            rows,
            work_item_rows: MatmulRows::Whole(rows),
        }
    }

    /// The rows tiled: one batch's rows, and each batch longer than one handed to the cubes.
    fn read_tiled(&self) -> RowsReading {
        RowsReading {
            rows: self.problem.stated_rows,
            work_item_rows: MatmulRows::Tiled,
        }
    }

    /// The batch axes the cubes are handed one of.
    fn distributed_batch_axes(&self) -> Vec<Axis> {
        self.distributed_batches()
            .into_iter()
            .map(|(axis, _)| axis)
            .collect()
    }

    /// The batch axes the cubes are handed one of, with their extents: in a tiled launch those
    /// longer than one; a launch holding every row whole has merged them into its rows. Under a
    /// table, every one, of one batch too: the table's dims are the batches, all cut at the level
    /// that reads its entry.
    fn distributed_batches(&self) -> Vec<(Axis, usize)> {
        let batches = self
            .problem
            .batch_axes()
            .into_iter()
            .zip(self.problem.batches.iter().copied());
        match (self.rows_kind, self.problem.index) {
            (_, Some(_)) => batches.collect(),
            (MatmulRowsKind::Whole, None) => Vec::new(),
            (MatmulRowsKind::Tiled, None) => batches.filter(|&(_, extent)| extent > 1).collect(),
        }
    }
}

/// What a launch reads of a problem's rows: the rows a work item holds, every one with the batches
/// merged in or one batch's, and the rows the microkernel is told it holds.
struct RowsReading {
    rows: usize,
    work_item_rows: MatmulRows,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::MatmulIndex;
    use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::{MatmulLayout, MatmulOperands, MatmulRows};
    use cubek_std::matmul_axes::{EXPERT, batch_axis};

    fn shape(batch: usize, batching: MatmulBatching) -> MatmulShape {
        MatmulShape {
            m: 2,
            n: 48,
            k: 64,
            batch,
            elem: ElemType::Float(FloatKind::F16),
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(N), batching),
        }
    }

    /// Batches that share the weight merge into its rows where every row is held whole; batches
    /// that each hold one are no matrix such a launch reads, and a single batch is the same product
    /// either way. Tiled, the rows stay one batch's.
    #[test]
    fn only_a_shared_weight_folds_its_batches_into_rows() {
        let rows = |batch, batching, rows_kind| {
            let problem = MatmulRoutineProblem::from(&shape(batch, batching));
            MatmulView::new(&problem, rows_kind).map(|view| view.rows())
        };
        assert_eq!(
            rows(4, MatmulBatching::Shared, MatmulRowsKind::Whole),
            Ok(8)
        );
        assert_eq!(
            rows(4, MatmulBatching::Shared, MatmulRowsKind::Tiled),
            Ok(2)
        );
        assert_eq!(
            rows(4, MatmulBatching::Batched, MatmulRowsKind::Whole),
            Err(MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH)
        );
        assert_eq!(
            rows(1, MatmulBatching::Batched, MatmulRowsKind::Whole),
            rows(1, MatmulBatching::Shared, MatmulRowsKind::Whole)
        );
    }

    /// A launch holding every row whole compiles its rows in and reads the rest at launch; a tiled
    /// one reads every extent at launch, its batch included.
    #[test]
    fn a_kernel_frees_the_extents_its_rows_kind_leaves_to_the_launch() {
        let problem = MatmulRoutineProblem::from(&shape(4, MatmulBatching::Shared));
        let kernel = |rows_kind| {
            MatmulView::new(&problem, rows_kind)
                .expect("a shared weight folds")
                .partitioning(
                    Levels::leaf(&[(M, 2), (N, 16), (K, 16)]),
                    &[M, N],
                    CubeOrder::RowMajor,
                    MatmulView::ONE_CUBE_A_BOX,
                )
                .kernel()
        };
        let whole = kernel(MatmulRowsKind::Whole);
        assert!(!whole.space().is_dynamic(M));
        assert!(whole.space().is_dynamic(N) && whole.space().is_dynamic(K));
        let tiled = kernel(MatmulRowsKind::Tiled);
        let batch = problem.batch_axes()[0];
        assert!(
            [M, N, K, batch]
                .into_iter()
                .all(|axis| tiled.space().is_dynamic(axis))
        );
    }

    /// An activation broadcast along the batches holds fewer rows than the output, so no merge
    /// makes them one matrix of rows; tiled, the batch is a grid axis and nothing merges.
    #[test]
    fn a_broadcast_activation_does_not_fold() {
        let problem = MatmulRoutineProblem::from(&shape(4, MatmulBatching::Shared));
        let broadcast = MatmulRoutineProblem {
            batches_held: MatmulOperands {
                lhs: 1,
                ..problem.batches_held
            },
            ..problem
        };
        assert_eq!(
            MatmulView::new(&broadcast, MatmulRowsKind::Whole).map(|view| view.rows()),
            Err(MATMUL_WHOLE_ROWS_BATCHES_UNFOLDED)
        );
        assert!(MatmulView::new(&broadcast, MatmulRowsKind::Tiled).is_ok());
    }

    /// A routed product: 5 tokens, each routed to 3 of 4 slabs, batches `[5, 3]` of one row, the
    /// activation broadcast along the picks.
    fn routed() -> MatmulRoutineProblem {
        let problem = MatmulRoutineProblem::dense(
            1,
            48,
            64,
            ElemType::Float(FloatKind::F16),
            MatmulLayout::Row,
        );
        MatmulRoutineProblem {
            batches: vec![5, 3],
            batches_held: MatmulOperands {
                lhs: 5,
                rhs: 1,
                out: 15,
            },
            index: Some(MatmulIndex::Slab { slabs: 4 }),
            ..problem
        }
    }

    /// Under a table, a launch holding rows whole keeps each batch's row apart instead of merging
    /// the batches into rows, so a broadcast activation is no refusal, and the rows are one batch's.
    #[test]
    fn a_table_keeps_the_batches_apart_where_rows_are_held_whole() {
        let problem = routed();
        let view = MatmulView::new(&problem, MatmulRowsKind::Whole).expect("a table keeps batches");
        assert_eq!(view.rows(), 1);
        assert_eq!(view.work_item().rows, MatmulRows::Whole(1));
    }

    /// A cube's box spans one slab of the stack, the one its entry names, though no level cuts
    /// the slabs: what its stage is sized for.
    #[test]
    fn a_cube_holds_one_slab() {
        let problem = MatmulRoutineProblem {
            stated_rows: 64,
            ..routed()
        };
        let view = MatmulView::new(&problem, MatmulRowsKind::Tiled).unwrap();
        let partitioning = view.partitioning(
            Levels::leaf(&[(M, 16), (N, 16), (K, 16)]),
            &[M, N],
            CubeOrder::RowMajor,
            MatmulView::ONE_CUBE_A_BOX,
        );
        assert_eq!(view.space().extent(EXPERT), 4);
        assert_eq!(partitioning.cube_box().extent(EXPERT), 1);
    }

    /// Every batch is handed to the cubes, one of a single batch too, since the table's dims are
    /// the batches; the stack's slabs are in the space beside the product, for an entry to pick.
    #[test]
    fn a_table_puts_every_batch_and_the_stack_in_the_space() {
        let problem = MatmulRoutineProblem {
            batches: vec![5, 1],
            batches_held: MatmulOperands {
                lhs: 5,
                rhs: 1,
                out: 5,
            },
            ..routed()
        };
        for rows_kind in [MatmulRowsKind::Whole, MatmulRowsKind::Tiled] {
            let space = MatmulView::new(&problem, rows_kind).unwrap().space();
            let [first, second] = [batch_axis(0), batch_axis(1)];
            assert_eq!(
                [first, second, M, N, K, EXPERT].map(|axis| space.extent(axis)),
                [5, 1, 1, 48, 64, 4],
                "{rows_kind:?}"
            );
        }
    }
}
