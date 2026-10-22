use super::counters::ZeroedCounters;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::{
    MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH, MatmulBinding, MatmulBlueprint, MatmulInputs, MatmulOutput,
    MatmulOutputBuffer, MatmulRoutineProblem, MatmulView,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::{BufferArg, TensorMapSwizzle};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{
    Axis, Geometry, Launcher, Space, StageStorage, TileArgLaunch, launch::Bound, launch::TmaBox,
    launch::TmaOperand, launch::Unbound,
};
use cubek_microkernel::matmul::staged_walk::MatmulStage;
use cubek_microkernel::matmul::{
    MatmulBlocks, MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulOrder, MatmulRowsKind,
    MatmulWorkItem,
};
use cubek_std::MatrixLayout;
use cubek_std::launch::tma::{tma_operand, tma_store_operand};
use cubek_std::layout::contiguous;
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::routine::ADMITTED_PLAN_RUNS;
use cubek_std::routine::Grid;

/// What a launch writing into its caller's output declines: an output of another shape or element
/// than the product's, or one stored in tiles.
pub const MATMUL_OUTPUT_UNREAD: Refusal = Refusal::Declined {
    why: "the output handed in is not the product's shape and element, laid in rows",
};

/// What a launch beside a packed input declines before preparing anything: a float operand of an
/// element it is not verified at.
pub const MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the packed routines are verified at f32, f16 or bf16 float operands",
};

/// What a packed routine declines: a problem whose rhs is not packed.
pub const MATMUL_Q_RHS_NOT_PACKED: Refusal = Refusal::Declined {
    why: "the packed routines read a packed, block-scaled rhs, beside an lhs float or packed",
};

/// What a launch declines before preparing anything: operands of two elements, or of one the
/// routines do not take.
pub const MATMUL_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel matmul takes f32, f16 and bf16 operands of one element",
};

/// What a routed or grouped product declines before preparing anything: an operand stored in
/// tiles that its routines would read in rows, a slab at a time. That is either grouped operand,
/// and the routed product's stack; a routed activation is laid back in rows instead.
pub const MATMUL_SLABS_TAKE_NO_STORED_TILES: Refusal = Refusal::Declined {
    why: "the routed and grouped products read their operands in rows, slab by slab; store \
              them untiled",
};

/// An operand stored in tiles that each step down a column first
/// ([`PlainTensor::is_tiled_column_first`]), which a tile read by its columns cannot walk.
pub const MATMUL_COLUMN_FIRST_TILES: Refusal = Refusal::Declined {
    why: "the matmul kernels read storage tiles stored a row at a time; this operand's are \
              stored a column at a time",
};

/// What a launch tiling the rows declines before preparing anything: operands of two ranks, which
/// the routines do not broadcast.
pub const MATMUL_TILED_ROWS_RANKS_DIFFER: Refusal = Refusal::Declined {
    why: "the microkernel gemm's operands share a rank",
};

/// What a launch holding every row whole declines before preparing anything: a weight contiguous
/// along neither of its axes, or one stored in tiles, which no routine holding every row whole reads in vectors.
pub const MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED: Refusal = Refusal::Declined {
    why: "the microkernel gemv reads one weight matrix contiguous along `k` or `n`",
};

/// The tensors of one product, the output among them, from which the problem they state is read
/// ([`problem`](Self::problem)). `Out` is where
/// the product goes: a [`MatmulOutputBuffer`] the launch allocates, or a [`Geometry`] the caller
/// binds, and each binds and hands back its output its own way.
///
/// How a launch holds the rows ([`MatmulRowsKind`]) is applied here, once, as views over the
/// operands' own buffers, and everything after reads those views alike: the problem is stated off
/// them, and the binding binds them.
///
/// - **Rows tiled.** An operand is read where it lies whenever no two of its indices address one
///   element, a transposed or row-pitched one included; only a broadcast or a view aliasing itself
///   is copied row-major first. Copying the rest would move all of it on every launch.
/// - **Every row whole.** The activation's leading dims are merged into rows at the step it really
///   has, and copied first only where they do not merge; the weight is read as the one matrix it
///   holds, where it lies, since copying it would move the whole weight on every launch, which is
///   the whole cost of a product of few rows; the output is bound as its rows.
///
/// Each input is read as it is stored ([`StoredTensor`]): float ones as above, or, where either is
/// packed, each packed one bound as one tile of its words under its scales over the problem's
/// block-split space, the float ones restated over the same space. Against a packed input every
/// row is held whole, the activation's leading dims merged: the problem it states has no batches,
/// so a tiled launch reads it alike. Either way every input binds as one tile, so a kernel reads
/// it as the tile it is.
#[derive(Clone)]
pub struct MatmulTensors<Out = MatmulOutputBuffer> {
    client: Client,
    lhs: StoredTensor,
    rhs: StoredTensor,
    out: Out,
    rows_kind: MatmulRowsKind,
}

impl<Out: core::fmt::Debug> core::fmt::Debug for MatmulTensors<Out> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the product.
        f.debug_struct("MatmulTensors")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .field("out", &self.out)
            .field("rows_kind", &self.rows_kind)
            .finish()
    }
}

impl MatmulTensors {
    /// The product of `lhs` and `rhs`, both on `client`'s device, as a launch on `M` reads it, its
    /// output allocated in the activation's element: float inputs holding the rows as `M`'s
    /// [`ROWS_KIND`](MatmulMicrokernel::ROWS_KIND) says, a packed one every row whole.
    ///
    /// # Errors
    ///
    /// For float inputs, [`MATMUL_ELEMENT_UNVERIFIED`], and what the rows' kind declines:
    /// [`MATMUL_TILED_ROWS_RANKS_DIFFER`] and [`MATMUL_COLUMN_FIRST_TILES`] for tiled
    /// rows; [`MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH`] and
    /// [`MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED`] for every row whole. Beside a packed one,
    /// [`MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED`].
    ///
    /// # Panics
    ///
    /// Operands whose contraction or batch dims do not agree: no product at all.
    pub fn new<M: MatmulMicrokernel>(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
    ) -> Result<Self, Refusal> {
        match (lhs, rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                Self::float(client, lhs, rhs, M::ROWS_KIND)
            }
            (lhs, rhs) => {
                let out = Self::packed_output(&lhs);
                Self::packed(client, lhs, rhs, out)
            }
        }
    }

    /// [`new`](Self::new) into an output of `out`: what a product whose activation is packed, and
    /// so names no float element, is written in.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), for a packed input.
    pub fn written_in(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: ElemType,
    ) -> Result<Self, Refusal> {
        Self::packed(client, lhs, rhs, out)
    }

    /// The product of the float `lhs` and `rhs`, both on `client`'s device, as a launch on `M`
    /// reads it, written into `out`, which the caller allocated in the operands' element at the
    /// product's shape.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new) for float inputs, and [`MATMUL_OUTPUT_UNREAD`] for an output of
    /// another shape or element.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new).
    pub fn writing_into<M: MatmulMicrokernel>(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        out: PlainTensor,
    ) -> Result<Self, Refusal> {
        let operands = Operands::read(client, lhs, rhs, M::ROWS_KIND)?;
        if out.shape().as_slice() != operands.out_shape.as_slice()
            || out.elem != operands.lhs.elem
            || out.is_tiled()
        {
            return Err(MATMUL_OUTPUT_UNREAD);
        }
        let bound = operands.bound_out(&out);
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(operands.lhs),
            rhs: StoredTensor::Plain(operands.rhs),
            out: MatmulOutputBuffer {
                tensor: None,
                bound,
            },
            rows_kind: M::ROWS_KIND,
        })
    }

    /// `tensor`, a float operand over `axes`, as a product beside a packed input reads it: dense
    /// rows of an element the packed routines are verified at, a copy where it is not dense.
    ///
    /// # Errors
    ///
    /// [`MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED`].
    ///
    /// # Panics
    ///
    /// An operand whose contraction dim is not `k`.
    pub fn beside_packed(
        client: &Client,
        tensor: PlainTensor,
        axes: [Axis; 2],
        k: usize,
    ) -> Result<PlainTensor, Refusal> {
        if !MatmulInputs::VERIFIED.contains(&tensor.elem) {
            return Err(MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED);
        }
        let tensor = tensor.into_contiguous(client);
        let shape = tensor.shape();
        let contracted = axes
            .iter()
            .position(|&axis| axis == K)
            .expect("a product's input spans its contraction");
        assert_eq!(
            shape[shape.rank() - 2 + contracted],
            k,
            "the contraction dims must agree"
        );
        Ok(tensor)
    }

    /// `operand` as a product beside a packed input reads it, along `axes` and `k` deep: a float
    /// one [`beside_packed`](Self::beside_packed), a packed one as it is stored.
    ///
    /// # Errors
    ///
    /// As [`beside_packed`](Self::beside_packed).
    fn stored_beside_packed(
        client: &Client,
        operand: StoredTensor,
        axes: [Axis; 2],
        k: usize,
    ) -> Result<StoredTensor, Refusal> {
        match operand {
            StoredTensor::Plain(tensor) => {
                Self::beside_packed(client, tensor, axes, k).map(StoredTensor::Plain)
            }
            StoredTensor::Packed(_) => Ok(operand),
        }
    }

    /// The rows of `lhs`, `k` deep, its leading dims merged into them.
    fn rows_of(lhs: &StoredTensor, k: usize) -> usize {
        match lhs {
            StoredTensor::Plain(lhs) => lhs.shape().num_elements() / k.max(1),
            StoredTensor::Packed(lhs) => lhs.extent(M),
        }
    }

    /// The element a product beside a packed input is written in: the activation's, or, for a
    /// packed one, which names none, the element its words unpack into
    /// ([`PACKED`](MatmulRoutineProblem::PACKED)).
    pub(super) fn packed_output(lhs: &StoredTensor) -> ElemType {
        match lhs {
            StoredTensor::Plain(lhs) => lhs.elem,
            StoredTensor::Packed(_) => MatmulRoutineProblem::PACKED,
        }
    }

    /// `lhs` against the float `rhs`, holding the rows as `rows_kind` says, the output allocated
    /// in the operands' element.
    fn float(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Refusal> {
        let operands = Operands::read(client, lhs, rhs, rows_kind)?;
        let (tensor, out) =
            PlainTensor::empty(client, operands.out_shape.clone().into(), operands.lhs.elem);
        let bound = operands.bound_out(&out);
        let tensor = Some(tensor);
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(operands.lhs),
            rhs: StoredTensor::Plain(operands.rhs),
            out: MatmulOutputBuffer { tensor, bound },
            rows_kind,
        })
    }

    /// `lhs` against `rhs`, either packed, the output allocated as `[.., n]` in `out`.
    fn packed(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: ElemType,
    ) -> Result<Self, Refusal> {
        let k = rhs.extent([K, N], K);
        let lhs = Self::stored_beside_packed(client, lhs, [M, K], k)?;
        let rhs = Self::stored_beside_packed(client, rhs, [K, N], k)?;
        let mut shape = match &lhs {
            StoredTensor::Plain(lhs) => lhs.shape().to_vec(),
            StoredTensor::Packed(lhs) => vec![lhs.extent(M), k],
        };
        *shape.last_mut().expect("an activation has a rank") = rhs.extent([K, N], N);
        let (tensor, bound) = PlainTensor::empty(client, shape.into(), out);
        let tensor = Some(tensor);
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            out: MatmulOutputBuffer { tensor, bound },
            rows_kind: MatmulRowsKind::Whole,
        })
    }

    /// The three operands bound as `blueprint` states them, on `grid`: under its partitioning, at
    /// its widths, the output its buffer.
    pub fn bind<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Bound, TileArgLaunch, TileArgLaunch> {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                self.bind_float(lhs, rhs, blueprint, grid)
            }
            _ => self.bind_packed(blueprint, grid),
        }
    }

    /// The operands bound as float ones, each read in the order its microkernel reads it.
    fn bind_float<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Bound, TileArgLaunch, TileArgLaunch> {
        let (launcher, [lhs, rhs]) = self.bind_operands(lhs, rhs, blueprint, grid);
        let out = launcher
            .arg(self.out.bound.binding.clone(), self.out.bound.elem)
            .axes(&[M, N])
            .batches(&self.batch_axes())
            .vectorize(blueprint.widths.out)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        blueprint
            .partitioning
            .assert_masked_where_overhanging(&out.spec);
        MatmulBinding {
            launcher,
            lhs: lhs.arg(),
            rhs: rhs.arg(),
            out,
        }
    }

    /// The operands bound where either input is packed, the output restated over the block-split
    /// space.
    fn bind_packed<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Bound, TileArgLaunch, TileArgLaunch> {
        let launcher = self.packed_launcher(blueprint, grid);
        let (out, out_axes) = self.out.bound.restated(&self.blocks(), M, N);
        let out = launcher
            .arg(out.binding, out.elem)
            .axes(&out_axes)
            .vectorize(blueprint.widths.out)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        self.bind_packed_inputs(launcher, blueprint, out)
    }
}

impl MatmulTensors<Geometry> {
    /// The product of `lhs` and `rhs` as a launch on `M` reads it, handed to a sink the caller
    /// binds: nothing is allocated, and the output is laid out densely, row-major,
    /// in the activation's element, as rows where the launch reads every row whole.
    ///
    /// # Errors
    ///
    /// As [`MatmulTensors::new`].
    ///
    /// # Panics
    ///
    /// As [`MatmulTensors::new`].
    pub fn sink<M: MatmulMicrokernel>(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
    ) -> Result<Self, Refusal> {
        match (lhs, rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                Self::float(client, lhs, rhs, M::ROWS_KIND)
            }
            (lhs, rhs) => Self::packed(client, lhs, rhs),
        }
    }

    /// `lhs` against the float `rhs`, holding the rows as `rows_kind` says.
    fn float(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Refusal> {
        let operands = Operands::read(client, lhs, rhs, rows_kind)?;
        let out = Geometry::row_major(&operands.sink_shape());
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(operands.lhs),
            rhs: StoredTensor::Plain(operands.rhs),
            out,
            rows_kind,
        })
    }

    /// `lhs` against `rhs`, either packed, the output laid out as dense rows.
    fn packed(client: &Client, lhs: StoredTensor, rhs: StoredTensor) -> Result<Self, Refusal> {
        let k = rhs.extent([K, N], K);
        let lhs = MatmulTensors::stored_beside_packed(client, lhs, [M, K], k)?;
        let rhs = MatmulTensors::stored_beside_packed(client, rhs, [K, N], k)?;
        let out = Geometry::row_major(&[MatmulTensors::rows_of(&lhs, k), rhs.extent([K, N], N)]);
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            out,
            rows_kind: MatmulRowsKind::Whole,
        })
    }

    /// The three operands bound as `blueprint` states them, on `grid`, the output derived over the
    /// sink's layout rather than a buffer: its spec, and the geometry the derivation settled, which
    /// the caller's sink addresses through.
    pub fn bind<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Unbound, TileArgLaunch, TileArgLaunch> {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                self.bind_float(lhs, rhs, blueprint, grid)
            }
            _ => self.bind_packed(blueprint, grid),
        }
    }

    /// The operands bound as float ones, each read in the order its microkernel reads it.
    fn bind_float<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Unbound, TileArgLaunch, TileArgLaunch> {
        let (launcher, [lhs, rhs]) = self.bind_operands(lhs, rhs, blueprint, grid);
        let out = launcher
            .unbound(&self.out)
            .axes(&[M, N])
            .batches(&self.batch_axes())
            .vectorize(blueprint.widths.out)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        blueprint
            .partitioning
            .assert_masked_where_overhanging(&out.spec);
        MatmulBinding {
            launcher,
            lhs: lhs.arg(),
            rhs: rhs.arg(),
            out,
        }
    }

    /// The operands bound where either input is packed, the output stated as a buffer output is
    /// ([`MatmulBlocks::restated`]): a dim for each axis of the block-split space, each at its own
    /// stride over the sink's dense rows.
    fn bind_packed<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> MatmulBinding<Unbound, TileArgLaunch, TileArgLaunch> {
        let launcher = self.packed_launcher(blueprint, grid);
        let dims = self
            .blocks()
            .restated_dims(M, N, self.rhs.extent([K, N], N));
        let geometry: Vec<_> = dims
            .iter()
            .map(|&(_, extent, stride)| (extent, stride))
            .collect();
        let axes: Vec<_> = dims.iter().map(|&(axis, _, _)| axis).collect();
        let out = launcher
            .unbound(&Geometry::new(&geometry))
            .axes(&axes)
            .vectorize(blueprint.widths.out)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        self.bind_packed_inputs(launcher, blueprint, out)
    }
}

impl<Out: MatmulOutput> MatmulTensors<Out> {
    /// The output's tensor map, its descriptor moving `stage`'s box of the output, dense: what a
    /// routine binds where the TMA engine stores its cubes' boxes
    /// ([`TmaStored`](cubecl::tile::launch::TmaStored)).
    ///
    /// # Panics
    ///
    /// On tensors read for every row whole, and on a sink, which no descriptor addresses: the
    /// store that chose the engine refused both.
    pub fn tma_output(&self, stage: &MatmulStage) -> TmaOperand {
        self.assert_tiled();
        let out = self
            .out
            .addressed()
            .expect("the TMA engine stores a box only into a buffer the launch allocates");
        let (map, transposed) = tma_store_operand(
            out.binding.clone(),
            1,
            MatrixLayout::RowMajor,
            (stage.extent(M), stage.extent(N)),
            out.elem,
            TensorMapSwizzle::None,
        );
        TmaOperand {
            map,
            axes: vec![M, N],
            shape: TmaBox {
                rows: self.rows() as u32,
                cols: self.problem().n as u32,
                batch: None,
                transposed,
            },
        }
    }
}

impl MatmulTensors<MatmulOutputBuffer> {
    /// The output as a launch wrote it, laid out as the runtime pads it, where the launch
    /// allocated it; `None` where the caller handed its own ([`writing_into`](Self::writing_into)).
    pub fn into_out(self) -> Option<TensorHandle> {
        self.out.tensor
    }
}

impl<Out> MatmulTensors<Out> {
    /// The lhs, as the launch reads it.
    pub fn lhs(&self) -> &StoredTensor {
        &self.lhs
    }

    /// The rhs, as the launch reads it.
    pub fn rhs(&self) -> &StoredTensor {
        &self.rhs
    }

    pub fn out(&self) -> &Out {
        &self.out
    }

    /// The client the operands live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The launcher over the blueprint's partitioning, every extent compiled in.
    fn packed_launcher<Mk, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Launcher {
        grid.launcher(
            self.client(),
            blueprint.partitioning.kernel(),
            blueprint.partitioning.as_tile().space(),
        )
    }
}

impl<Out: MatmulOutput> MatmulTensors<Out> {
    /// The problem the three tensors state.
    pub fn problem(&self) -> MatmulRoutineProblem {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => self.out.problem(lhs, rhs),
            (lhs, rhs) => {
                let k = rhs.extent([K, N], K);
                let packing = |operand: &StoredTensor| match operand {
                    StoredTensor::Plain(_) => None,
                    StoredTensor::Packed(operand) => Some(operand.packing()),
                };
                let packings = MatmulOperands {
                    lhs: packing(lhs),
                    rhs: packing(rhs),
                    out: None,
                };
                MatmulRoutineProblem {
                    destination: Out::DESTINATION,
                    ..MatmulRoutineProblem::packed(
                        MatmulTensors::rows_of(lhs, k),
                        rhs.extent([K, N], N),
                        k,
                        self.out.element(lhs),
                        packings,
                    )
                }
            }
        }
    }

    /// The block-split space a packed input puts the product in.
    ///
    /// # Panics
    ///
    /// On float inputs, which put it in none.
    fn blocks(&self) -> MatmulBlocks {
        self.read(|view| view.blocks())
            .expect("a packed input puts the product in a block-split space")
    }

    /// Both inputs under `launcher`, beside `out`: each packed one as one tile of its words under
    /// its scales, each float one restated over the block-split space.
    fn bind_packed_inputs<Mk: MatmulMicrokernel, SM, O>(
        &self,
        launcher: Launcher,
        blueprint: &MatmulBlueprint<Mk, SM>,
        out: O,
    ) -> MatmulBinding<O, TileArgLaunch, TileArgLaunch> {
        let blocks = self.blocks();
        let bind = |operand: &StoredTensor, [rows, columns]: [Axis; 2], width: usize| match operand
        {
            StoredTensor::Plain(tensor) => {
                let (tensor, axes) = tensor.restated(&blocks, rows, columns);
                launcher
                    .arg(tensor.binding, tensor.elem)
                    .axes(&axes)
                    .vectorize(width)
                    .build()
                    .expect(ADMITTED_PLAN_RUNS)
                    .arg()
            }
            StoredTensor::Packed(tensor) => {
                tensor.bind(&launcher, width, blueprint.microkernel.block_scales())
            }
        };
        let lhs = bind(&self.lhs, [M, K], blueprint.widths.lhs);
        let rhs = bind(&self.rhs, [K, N], blueprint.widths.rhs);
        MatmulBinding {
            launcher,
            lhs,
            rhs,
            out,
        }
    }

    /// The float inputs, which only a launch tiling its rows stages or splits `K` over.
    ///
    /// # Panics
    ///
    /// On a packed input: its tensors hold every row whole, which
    /// [`assert_tiled`](Self::assert_tiled) refuses first.
    fn plain(&self) -> [&PlainTensor; 2] {
        [&self.lhs, &self.rhs].map(|operand| match operand {
            StoredTensor::Plain(tensor) => tensor,
            StoredTensor::Packed(_) => panic!("a packed input is read with every row whole"),
        })
    }

    /// The work item the launch they were read for hands its microkernel.
    pub fn work_item(&self) -> MatmulWorkItem {
        self.read(|view| view.work_item())
    }

    /// The axes the problem's batch dims are labelled with, read off the operands' rank without
    /// stating the whole problem ([`MatmulRoutineProblem::batch_axes`]).
    fn batch_axes(&self) -> Vec<Axis> {
        MatmulRoutineProblem::batch_axes_of(self.lhs.values().shape().rank() - 2)
    }

    /// The rows a work item of the launch they were read for holds.
    fn rows(&self) -> usize {
        self.read(|view| view.rows())
    }

    /// `f` of the problem as the launch they were read for reads it: the tensors already hold
    /// their rows as its kind says, and declined what it cannot hold.
    fn read<T>(&self, f: impl FnOnce(MatmulView<'_>) -> T) -> T {
        let problem = self.problem();
        let view = MatmulView::new(&problem, self.rows_kind)
            .expect("tensors read for a rows' kind hold their rows as it reads them");
        f(view)
    }

    /// The lhs's and the rhs's tensor maps, each descriptor moving `stage`'s box of its operand,
    /// one matrix of it read one element wide, so it lands as `storage` holds the stage: the rhs
    /// described in `rhs` order, a col weight as the `[n, k]` matrix its buffer is, and each box
    /// swizzled as the tile DSL says the stage keeps its rows (`StageStorage::tma_swizzle`). What a
    /// routine binds where the TMA engine fills its stages.
    ///
    /// # Panics
    ///
    /// On tensors read for every row whole: only a launch tiling its rows stages them. On a stage
    /// no box lands as `storage` holds it, which the fill that chose the engine refused.
    pub fn tma_operands(
        &self,
        stage: &MatmulStage,
        storage: &StageStorage,
        orders: MatmulOperands<MatmulOrder>,
    ) -> (TileArgLaunch, TileArgLaunch) {
        self.assert_tiled();
        let problem = self.problem();
        let [lhs, rhs] = self.plain();
        let landing = |tensor: &PlainTensor, operand: MatmulOperand| {
            let stage = stage.operand_space(operand, orders);
            let swizzle = storage
                .tma_swizzle(&stage, 1, tensor.elem.size_bits())
                .expect("the fill delivers by TMA only a stage a box lands");
            TmaLanding { stage, swizzle }
        };
        let landed = landing(rhs, MatmulOperand::Rhs);
        let lhs = Self::tma_matrix(
            lhs,
            [(M, self.rows()), (K, problem.k)],
            landing(lhs, MatmulOperand::Lhs),
        );
        let rhs = match orders.rhs {
            MatmulOrder::Stated => Self::tma_matrix(rhs, [(K, problem.k), (N, problem.n)], landed),
            MatmulOrder::InBufferOrder => Self::tma_matrix(
                &rhs.clone().in_buffer_order(),
                [(N, problem.n), (K, problem.k)],
                landed,
            ),
        };
        (lhs, rhs)
    }

    /// The carry the cubes splitting `K` take their turns at, in the sum's element, bound as the
    /// output is, beside one turn counter a cube, holding zero.
    ///
    /// # Panics
    ///
    /// On tensors read for every row whole: only a launch tiling its rows splits `K` over cubes.
    pub fn bind_relay<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        launcher: &Launcher,
    ) -> MatmulRelayBinding {
        self.assert_tiled();
        let [lhs, rhs] = self.plain();
        let shape = Operands::product_shape(lhs, rhs).expect("the product's shape was checked");
        let (_, carry) = PlainTensor::empty(&self.client, shape.into(), MatmulView::SUM);
        MatmulRelayBinding {
            carry: launcher
                .arg(carry.binding, MatmulView::SUM)
                .axes(&[M, N])
                .batches(&self.batch_axes())
                .vectorize(blueprint.widths.out)
                .build()
                .expect(ADMITTED_PLAN_RUNS),
            turns: ZeroedCounters::new(
                blueprint.partitioning.as_tile().relay_counters(),
                &self.client,
            )
            .into_arg(),
        }
    }

    /// The slots the cubes sharing a box park their parts in, over the output's axes in the sum's
    /// element, two cube boxes a cube
    /// ([`Partitioning::arrival_slots`](cubecl::tile::Partitioning::arrival_slots)), beside one
    /// arrival counter a box, holding zero.
    ///
    /// # Panics
    ///
    /// On tensors read for every row whole: only a launch tiling its rows shares its boxes' `K`.
    pub fn bind_last_arrival<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        launcher: &Launcher,
    ) -> MatmulLastArrivalBinding {
        self.assert_tiled();
        let partitioning = blueprint.partitioning.as_tile();
        // The output's axes in its order: the batches the cube level hands out, then its rows and
        // columns. Stated as labels, so a batch axis one box long stays a dim of the slots.
        let axes: Vec<Axis> = blueprint
            .partitioning
            .batch_axes()
            .iter()
            .copied()
            .chain([M, N])
            .collect();
        let (_, slots) = PlainTensor::empty(
            &self.client,
            partitioning.arrival_slots(&axes),
            MatmulView::SUM,
        );
        MatmulLastArrivalBinding {
            slots: launcher
                .arg(slots.binding, MatmulView::SUM)
                .axes(&axes)
                .vectorize(blueprint.widths.out)
                .build()
                .expect(ADMITTED_PLAN_RUNS),
            arrivals: ZeroedCounters::new(partitioning.arrival_counters(), &self.client).into_arg(),
        }
    }

    /// Checks the tensors were read for a launch tiling its rows, the only one a staged or relayed
    /// binding belongs to.
    fn assert_tiled(&self) {
        assert_eq!(
            self.rows_kind,
            MatmulRowsKind::Tiled,
            "a staged or relayed binding is a tiled launch's"
        );
    }

    /// The launcher `blueprint`'s partitioning launches on `grid`, and the lhs and the rhs bound
    /// under it.
    fn bind_operands<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> (Launcher, [Bound; 2]) {
        let partitioning = blueprint.partitioning.as_tile();
        let launcher = grid.launcher(
            self.client(),
            blueprint.partitioning.kernel(),
            partitioning.space(),
        );
        let (batch_axes, widths) = (self.batch_axes(), blueprint.widths);
        let delivery = blueprint.microkernel.delivery();
        let source = |tensor: &PlainTensor, axes: &'static [Axis]| {
            launcher
                .arg(tensor.binding.clone(), tensor.elem)
                .axes(axes)
                .batches(&batch_axes)
                .delivery(delivery)
        };
        let orders = blueprint.microkernel.orders();
        let rhs_source = orders.rhs.applied(source(rhs, &[K, N]));
        let lhs = orders
            .lhs
            .applied(source(lhs, &[M, K]))
            .vectorize(widths.lhs)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let rhs = rhs_source
            .vectorize(widths.rhs)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        (launcher, [lhs, rhs])
    }

    /// `tensor`'s tensor map, a row-major matrix of `rows × columns` (each an axis and its extent),
    /// its descriptor moving `landing`'s box of it.
    fn tma_matrix(
        tensor: &PlainTensor,
        [(rows, row_count), (columns, column_count)]: [(Axis, usize); 2],
        landing: TmaLanding,
    ) -> TileArgLaunch {
        let matrices = 1;
        let (map, transposed) = tma_operand(
            tensor.binding.clone(),
            matrices,
            MatrixLayout::RowMajor,
            (landing.stage.extent(rows), landing.stage.extent(columns)),
            tensor.elem,
            landing.swizzle,
        );
        TileArgLaunch::tensor_map(
            map,
            &[rows, columns],
            TmaBox {
                rows: row_count as u32,
                cols: column_count as u32,
                batch: None,
                transposed,
            },
        )
    }
}

/// The carry and the turn counters of a launch whose cubes take turns at the output.
pub struct MatmulRelayBinding {
    /// The partial sums the turns add up, the output's shape in the sum's element.
    pub carry: Bound,
    /// A turn counter a cube, holding zero.
    pub turns: BufferArg,
}

/// What a launch whose cubes share its boxes' `K` binds beside its operands
/// ([`bind_last_arrival`](MatmulTensors#method.bind_last_arrival)).
pub struct MatmulLastArrivalBinding {
    /// Two cube boxes a cube, in the sum's element, for the parts of the boxes it shares.
    pub slots: Bound,
    /// An arrival counter a box, holding zero.
    pub arrivals: BufferArg,
}

/// The operands as a launch holding rows one way binds them, and the output it allocates.
struct Operands {
    lhs: PlainTensor,
    rhs: PlainTensor,
    /// The shape of the output a launch allocates.
    out_shape: Vec<usize>,
    rows_kind: MatmulRowsKind,
}

impl Operands {
    /// `lhs` and `rhs` read as a launch holding rows as `rows_kind` says reads them.
    ///
    /// # Errors
    ///
    /// As [`MatmulTensors::new`].
    ///
    /// # Panics
    ///
    /// As [`MatmulTensors::new`].
    fn read(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Refusal> {
        match rows_kind {
            MatmulRowsKind::Tiled => Self::tiled(client, lhs, rhs),
            MatmulRowsKind::Whole => Self::whole(client, lhs, rhs),
        }
    }

    /// Both operands where they lie unless they alias themselves, the output their product's
    /// shape.
    fn tiled(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Refusal> {
        let out_shape = Self::product_shape(&lhs, &rhs)?;
        if lhs.is_tiled_column_first() || rhs.is_tiled_column_first() {
            return Err(MATMUL_COLUMN_FIRST_TILES);
        }
        let (lhs, rhs) = MatmulInputs::new(client, lhs, rhs)?
            .addressable(client)
            .into_parts();
        Ok(Self {
            lhs,
            rhs,
            out_shape,
            rows_kind: MatmulRowsKind::Tiled,
        })
    }

    /// The activation merged into `[m, k]` rows, the weight its one `[k, n]` matrix, the output
    /// allocated as `[.., n]` and bound as `[m, n]` rows.
    fn whole(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Refusal> {
        let (lhs, rhs) = MatmulInputs::new(client, lhs, rhs)?.into_parts();
        let rhs_shape = rhs.shape();
        if rhs_shape.as_slice()[..rhs_shape.rank().saturating_sub(2)]
            .iter()
            .any(|&dim| dim > 1)
        {
            return Err(MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH);
        }
        let weight =
            Self::one_weight_matrix(&rhs).ok_or(MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED)?;
        let lhs = Self::with_mergeable_rows(client, lhs);
        let lhs_shape = lhs.shape();
        let k = lhs_shape[lhs_shape.rank() - 1];
        let weight_shape = weight.shape();
        let [weight_k, n] = [weight_shape[0], weight_shape[1]];
        assert_eq!(
            k,
            weight_k,
            "the contraction dims must agree: {lhs_shape:?} and {:?}",
            rhs.shape()
        );
        let m = lhs_shape.num_elements() / k.max(1);
        let mut out_shape = lhs_shape.to_vec();
        *out_shape.last_mut().expect("an activation has a rank") = n;
        Ok(Self {
            lhs: lhs.as_rows(m, k),
            rhs: weight,
            out_shape,
            rows_kind: MatmulRowsKind::Whole,
        })
    }

    /// The rows and the line the output is bound as, where the launch reads every row whole: the
    /// activation's rows, the weight's columns.
    fn out_rows(&self) -> Option<[usize; 2]> {
        match self.rows_kind {
            MatmulRowsKind::Whole => Some([self.lhs.shape()[0], self.rhs.shape()[1]]),
            MatmulRowsKind::Tiled => None,
        }
    }

    /// The view of `out` the launch binds: its rows, where it reads every row whole.
    fn bound_out(&self, out: &PlainTensor) -> PlainTensor {
        match self.out_rows() {
            Some([rows, line]) => out.as_rows(rows, line),
            None => out.clone(),
        }
    }

    /// The shape a sink's dense layout is counted over: the rows, where the launch reads every row
    /// whole.
    fn sink_shape(&self) -> Vec<usize> {
        match self.out_rows() {
            Some(rows) => rows.to_vec(),
            None => self.out_shape.clone(),
        }
    }

    /// `rhs` as the one `[k, n]` matrix it holds, in rows or in tiles, contiguous along the axis
    /// the tuning key reads ([`contiguous`]), so a key and its tensors state one layout; `None`
    /// where it holds several matrices or is contiguous along neither axis.
    fn one_weight_matrix(rhs: &PlainTensor) -> Option<PlainTensor> {
        rhs.as_one_matrix()
            .filter(|matrix| contiguous(&matrix.meta(), [K, N]).is_some())
    }

    /// `lhs` where its leading dims merge into rows of its last at one step, a row-major copy of it
    /// where they do not: an activation the launch reads as rows either way.
    fn with_mergeable_rows(client: &Client, lhs: PlainTensor) -> PlainTensor {
        let (shape, strides) = (lhs.shape(), lhs.binding.strides.to_vec());
        let rank = shape.rank();
        let merges = rank >= 1
            && strides[rank - 1] == 1
            && (0..rank.saturating_sub(2))
                .all(|dim| strides[dim] == shape[dim + 1] * strides[dim + 1]);
        match merges && !lhs.is_tiled() {
            true => lhs,
            false => lhs.into_contiguous(client),
        }
    }

    /// The shape of `lhs · rhs`: the broadcast batch dims, then `m × n`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_TILED_ROWS_RANKS_DIFFER`].
    ///
    /// # Panics
    ///
    /// Operands of a rank under two, or whose contraction or batch dims do not agree.
    fn product_shape(lhs: &PlainTensor, rhs: &PlainTensor) -> Result<Vec<usize>, Refusal> {
        let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
        let rank = lhs_shape.num_dims();
        if rank != rhs_shape.num_dims() {
            return Err(MATMUL_TILED_ROWS_RANKS_DIFFER);
        }
        assert!(
            rank >= 2,
            "a matmul's operands have a rank of at least two: {lhs_shape:?}"
        );
        assert_eq!(
            lhs_shape[rank - 1],
            rhs_shape[rank - 2],
            "the contraction dims must agree: {lhs_shape:?} and {rhs_shape:?}"
        );
        let batches = (0..rank - 2).map(|dim| {
            let (l, r) = (lhs_shape[dim], rhs_shape[dim]);
            assert!(
                l == r || l == 1 || r == 1,
                "the batch dims do not broadcast: {lhs_shape:?} and {rhs_shape:?}"
            );
            l.max(r)
        });
        Ok(batches
            .chain([lhs_shape[rank - 2], rhs_shape[rank - 1]])
            .collect())
    }
}

/// Where a descriptor lands its box: the stage of the operand, its axes in the order the operand
/// is bound in, and the swizzle that lays the box's rows as the stage keeps them.
struct TmaLanding {
    stage: Space,
    swizzle: TensorMapSwizzle,
}

#[cfg(test)]
mod tests {
    use cubecl::tile::layout::ElemAlignment;
    use cubecl::zspace::metadata::Metadata;
    use cubek_microkernel::matmul::{MatmulLayout, MatmulOperand};

    /// A stack's alignment over the order its buffer steps, as a problem derives it from the
    /// stack's layout, is the one a binding stepping that order reads, for every slab view of a
    /// row or col stack: laid along the output and along the contraction, then pitched by an odd
    /// step, since every extent here is a multiple of sixteen and the pitch keeps a reading that
    /// drops a stride from hiding.
    #[test]
    fn a_stack_aligns_in_buffer_order_as_its_binding_steps() {
        let (k, n, slabs, pitch) = (64, 48, 4, 3);
        let stacks: [[usize; 3]; 6] = [
            [n, slabs * n, 1],
            [k * n, n, 1],
            [n * k, 1, k],
            [k, 1, slabs * k],
            [n, slabs * n + pitch, 1],
            [n * (k + pitch), 1, k + pitch],
        ];
        let shape = [slabs, k, n];
        for strides in stacks {
            let bound = ElemAlignment::in_stride_order(&shape, &strides);
            let derived = MatmulLayout::new(
                &Metadata::new(shape.to_vec(), strides.to_vec()),
                MatmulOperand::Rhs,
            )
            .buffer_alignment(&shape, &strides);
            assert_eq!(bound, derived, "slabs at {strides:?}");
        }
    }

    /// A row weight with a pitch along `K` has one alignment whichever order binds it: its buffer
    /// steps in the order the product names it. That holds for one matrix, a pitched one and every
    /// slab view of a stack, so the shuffle microkernels, which bind a row weight as stated, read it
    /// at the widths its buffer gives.
    #[test]
    fn a_row_weight_aligns_alike_in_either_order() {
        let (k, n, slabs, pitch) = (64, 48, 4, 3);
        for strides in [[n, 1], [n + pitch, 1]] {
            assert_eq!(
                ElemAlignment::new(&[k, n], &strides),
                ElemAlignment::in_stride_order(&[k, n], &strides),
                "{strides:?}"
            );
        }
        let shape = [slabs, k, n];
        for strides in [[n, slabs * n, 1], [k * n, n, 1], [n, slabs * n + pitch, 1]] {
            assert_eq!(
                ElemAlignment::new(&shape, &strides),
                ElemAlignment::in_stride_order(&shape, &strides),
                "slabs at {strides:?}"
            );
        }
    }
}
