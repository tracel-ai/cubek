use super::counters::ZeroedCounters;
use crate::kernels::tensor::{
    PACKED_STACK_UNREAD, PackedTensor, PlainTensor, ScaledTensor, StoredTensor,
};
use crate::kernels::{
    MATMUL_GROUPED_ROWS_OVER_TILED_LHS, MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH, MatmulBinding,
    MatmulBlueprint, MatmulIndex, MatmulIndexTable, MatmulInputs, MatmulOutput, MatmulOutputBuffer,
    MatmulRoutineProblem, MatmulView, RowGroups,
};
use core::marker::PhantomData;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::{BufferArg, TensorMapArg, TensorMapSwizzle};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Arg, Decline, Labelled, TmaCoordinates, TmaTerm};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{
    Dimension, Geometry, Launcher, Space, StageStorage, TileSpec, TileTensorLaunch, launch::Bound,
    launch::TmaBox, launch::Unbound,
};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::{Shape, Strides};
use cubecl_environment::sync::{Arc, SyncOnceCell};
use cubek_microkernel::matmul::MatmulStage;
use cubek_microkernel::matmul::{
    MatmulBlocks, MatmulLayout, MatmulMicrokernel, MatmulOperand, MatmulOperands, MatmulRowsKind,
    MatmulWorkItem,
};
use cubek_std::MatrixLayout;
use cubek_std::launch::tma::{swizzle_span_bytes, tma_meta_tiled, tma_operand, tma_store_operand};
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::{BlockScales, StoredInput};
use cubek_std::layout::{contiguous, tile_order};
use cubek_std::matmul_dimensions::{K, KB, KI, M, N};
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable, Grid};

/// What a launch writing into its caller's output declines: an output of another shape than the
/// product's, of an element the routines do not write, stored in tiles, or, beside a packed input,
/// other than row-major.
pub const MATMUL_OUTPUT_UNREAD: Decline = Decline::Unserved {
    why: "the output handed in is not the product's shape, in an element the routines write, laid in rows",
};

/// What a launch beside a packed input declines before preparing anything: a float operand of an
/// element it is not verified at.
pub const MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED: Decline = Decline::Unserved {
    why: "the packed routines are verified at f32, f16 or bf16 float operands",
};

/// What should hold of a stage the TMA engine fills, and its launch is [`Broken`](Decline::Broken)
/// without.
const TMA_FILLS_A_STAGE_A_BOX_LANDS: &str = "the fill delivers by TMA only a stage a box lands";

/// What should hold of a packed product the TMA engine fills.
const TMA_FILLS_A_PACKED_RHS: &str = "a packed product the TMA engine fills has a packed rhs";

/// What should hold of how a packed product states the output the TMA engine stores.
const TMA_STORES_ROWS_AND_COLUMNS: &str =
    "the TMA engine stores a box of an output stated as one dim of rows and one of columns";

/// What should hold of where the TMA engine stores.
const TMA_STORES_INTO_A_BUFFER: &str =
    "the TMA engine stores a box only into a buffer the launch allocates";

/// What a launch declines of tensors a caller read for another way of holding the rows than its
/// routine's ([`MatmulRoutine::ROWS_KIND`](crate::kernels::MatmulRoutine::ROWS_KIND)): float
/// operands laid out, copied or bound as another launch reads them.
pub const MATMUL_TENSORS_READ_FOR_OTHER_ROWS: Decline = Decline::Unserved {
    why: "the tensors were read for a launch holding its rows another way than this routine",
};

/// What a product of 8-bit floats under scales declines: another input that is not packed, or a
/// stack read slab by slab. Beside a packed input alone is it read in the block-split space.
pub const MATMUL_SCALED_BESIDE_FLOAT: Decline = Decline::Unserved {
    why: "8-bit floats under scales are multiplied with a packed input, as one matrix",
};

/// What a routine streaming the `[k, n]` input packed declines: a float one. A packed `[m, k]`
/// input against it is decoded first.
pub const MATMUL_STREAMED_INPUT_FLOAT: Decline = Decline::Unserved {
    why: "the packed placement walks the block-split space of the [k, n] input it streams, and this one is float",
};

/// What a launch declines before preparing anything: operands of two elements, or of one the
/// routines do not take.
pub const MATMUL_ELEMENT_UNVERIFIED: Decline = Decline::Unserved {
    why: "the microkernel matmul takes f32, f16, bf16 and i32 operands of one element",
};

/// What a routed or grouped product declines before preparing anything: a float operand stored
/// in tiles that its routines would read in rows, a slab at a time. That is either operand of a
/// K-grouped product, the rows or a float stack of an M-grouped one, and a routed product's float
/// stack; a routed lhs is laid back in rows instead. A packed stack may lie in tiles
/// ([`PackedTensor::stacked`]).
pub const MATMUL_SLABS_TAKE_NO_FLOAT_TILES: Decline = Decline::Unserved {
    why: "the routed and grouped products read float operands in rows, slab by slab; store \
              them untiled",
};

/// A table of entries other than `u32`.
pub const MATMUL_TABLE_NOT_U32: Decline = Decline::Unserved {
    why: "a product reading a table reads u32 entries",
};

/// A stack a table does not fit: under [`MatmulIndex::Slab`], other than `[slabs, k, n]` for the
/// table's slabs; under [`MatmulIndex::RunOfK`], other than one `[k, n]` matrix.
pub const MATMUL_STACK_NOT_THE_TABLES: Decline = Decline::Unserved {
    why: "a product reading a table reads the stack its index names: the table's slabs, or one \
          matrix it reads runs of",
};

/// Row group ends that are not one `u32` end per row group, as the entries are one slab.
pub const MATMUL_GROUP_ENDS_NOT_ONE_PER_GROUP: Decline = Decline::Unserved {
    why: "grouped rows read one `u32` end per row group, in a table shaped as the entries",
};

/// Inputs of a product that are no two matrices sharing their contraction.
pub const MATMUL_CONTRACTIONS_DISAGREE: Decline = Decline::Unserved {
    why: "a product's inputs are matrices sharing their contraction",
};

/// Inputs of a product tiling its rows whose batch dims do not broadcast against each other.
pub const MATMUL_BATCHES_DO_NOT_BROADCAST: Decline = Decline::Unserved {
    why: "a product's inputs have batch dims that agree, or where they differ, one of them is one",
};

/// An operand stored in tiles that each step down a column first
/// ([`PlainTensor::is_tiled_column_first`]), which a tile read by its columns cannot walk.
pub const MATMUL_COLUMN_FIRST_TILES: Decline = Decline::Unserved {
    why: "the matmul kernels read storage tiles stored a row at a time; this operand's are \
              stored a column at a time",
};

/// What a launch tiling the rows declines before preparing anything: operands of two ranks, which
/// the routines do not broadcast.
pub const MATMUL_TILED_ROWS_RANKS_DIFFER: Decline = Decline::Unserved {
    why: "a matmul tiling its rows reads operands of one rank",
};

/// What a launch holding every row whole declines before preparing anything: an rhs contiguous
/// along neither of its dimensions, or one stored in tiles, which no routine holding every row
/// whole reads in vectors.
pub const MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED: Decline = Decline::Unserved {
    why: "a matmul holding every row whole reads one rhs matrix contiguous along `k` or `n`",
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
/// - **Every row whole.** The lhs's leading dims are merged into rows at the step it really
///   has, and copied first only where they do not merge; the rhs is read as the one matrix it
///   holds, where it lies, since copying it would move the whole rhs on every launch, which is
///   the whole cost of a product of few rows; the output is bound as its rows.
///
/// Each input is read as it is stored ([`StoredTensor`]): float ones as above, or, where either is
/// packed, each packed one bound as one tile of its words under its scales over the problem's
/// block-split space, the float ones restated over the same space. Against a packed input every
/// row is held whole, the lhs's leading dims merged: the problem it states has no batches,
/// so a tiled launch reads it alike. Either way every input binds as one tile, so a kernel reads
/// it as the tile it is.
#[derive(Clone)]
pub struct MatmulTensors<Out = MatmulOutputBuffer> {
    client: Client,
    lhs: StoredTensor,
    rhs: StoredTensor,
    out: Out,
    rows_kind: MatmulRowsKind,
    table: Option<MatmulIndexTable>,
    /// The problem the tensors state, read the first time it is asked: a launch reads it to look
    /// up its prepared plan and again where it walks its work item.
    stated: StatedProblem,
}

/// A problem read once off the tensors it is stated by, shared by their clones: none of their
/// fields changes once they are read.
#[derive(Clone, Default)]
struct StatedProblem(Arc<SyncOnceCell<MatmulRoutineProblem>>);

impl<Out: core::fmt::Debug> core::fmt::Debug for MatmulTensors<Out> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The client says nothing about the product.
        f.debug_struct("MatmulTensors")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .field("out", &self.out)
            .field("rows_kind", &self.rows_kind)
            .field("table", &self.table)
            .finish()
    }
}

impl MatmulTensors {
    /// The product of `lhs` and `rhs`, both on `client`'s device, its output allocated in a float
    /// input's element: float inputs holding the rows as `rows_kind` says (the routine's
    /// [`ROWS_KIND`](crate::kernels::MatmulRoutine::ROWS_KIND)), a packed one every row whole.
    ///
    /// # Errors
    ///
    /// [`MATMUL_CONTRACTIONS_DISAGREE`]. For float inputs, [`MATMUL_ELEMENT_UNVERIFIED`], and what
    /// the rows' kind declines: [`MATMUL_TILED_ROWS_RANKS_DIFFER`],
    /// [`MATMUL_BATCHES_DO_NOT_BROADCAST`] and [`MATMUL_COLUMN_FIRST_TILES`] for tiled rows;
    /// [`MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH`] and [`MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED`] for
    /// every row whole. Beside a packed one, [`MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED`].
    pub fn new(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        match (lhs, rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                Self::float(client, lhs, rhs, rows_kind)
            }
            (lhs, rhs) => {
                let out = Self::packed_output(&lhs, &rhs);
                Self::packed(client, lhs, rhs, out)
            }
        }
    }

    /// [`new`](Self::new) into an output of `out`: what a product whose lhs is packed, and
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
    ) -> Result<Self, Decline> {
        Self::packed(client, lhs, rhs, out)
    }

    /// The product of `lhs` and `rhs`, both on `client`'s device, written into `out`, which the
    /// caller allocated at the product's shape, in any element the routines write: the sum is
    /// cast to it as it drains. Float inputs hold the rows as `rows_kind` says, a packed one every
    /// row whole, the output then its dense rows.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new), and [`MATMUL_OUTPUT_UNREAD`] for an output of another shape, of an
    /// element the routines do not write, stored in tiles, or, beside a packed input, other than
    /// row-major; read for every row whole, one whose leading dims do not merge into rows.
    pub fn writing_into(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        let (lhs, rhs) = match (lhs, rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => (lhs, rhs),
            (lhs, rhs) => return Self::packed_into(client, lhs, rhs, out),
        };
        // The output is checked off the shapes before a launch copies an operand.
        let out_shape = Operands::out_shape(&lhs, &rhs, rows_kind)?;
        // A leading axis of extent 1 steps nowhere, so its stride says nothing about the rows.
        let out = match rows_kind {
            MatmulRowsKind::Whole => Operands::with_lone_axes_restrided(&out),
            MatmulRowsKind::Tiled => out,
        };
        let rows_unmerged = rows_kind == MatmulRowsKind::Whole && !Operands::merges_into_rows(&out);
        if out.shape().as_slice() != out_shape.as_slice()
            || !MatmulInputs::VERIFIED.contains(&out.elem)
            || out.is_tiled()
            || rows_unmerged
        {
            return Err(MATMUL_OUTPUT_UNREAD);
        }
        let operands = Operands::read(client, lhs, rhs, rows_kind)?;
        let bound = operands.bound_out(&out);
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(operands.lhs),
            rhs: StoredTensor::Plain(operands.rhs),
            out: MatmulOutputBuffer {
                tensor: None,
                bound,
            },
            rows_kind: rows_kind,
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// The float `lhs` against `rhs`, every batch reading its entry of `table`, as a launch holding
    /// its rows as `rows_kind` says reads them, written into `out`, `[batches.., rows, n]` in a
    /// float input's element, which the caller allocated as its product lies: a grouped one's slabs
    /// side by side, say. The lhs's batch dims broadcast against the table's. Under
    /// [`MatmulIndex::Slab`] the rhs is the stack: a float one `[slabs, k, n]`, read where it lies;
    /// a packed one read as its slabs ([`PackedTensor::stacked`]), every row of a batch then held
    /// whole.
    ///
    /// # Errors
    ///
    /// As [`MatmulInputs::new`] for a float rhs; for a packed one, as
    /// [`beside_packed`](Self::beside_packed), and [`PACKED_STACK_UNREAD`] for one read along a
    /// run of `K`, which no packed product reads. [`MATMUL_OUTPUT_UNREAD`] for an output of
    /// another shape or element, or stored in tiles. [`MATMUL_TABLE_NOT_U32`],
    /// [`MATMUL_STACK_NOT_THE_TABLES`], [`MATMUL_CONTRACTIONS_DISAGREE`],
    /// [`MATMUL_GROUP_ENDS_NOT_ONE_PER_GROUP`] and, for grouped rows over an lhs stored in tiles,
    /// [`MATMUL_GROUPED_ROWS_OVER_TILED_LHS`].
    pub fn indexed(
        client: &Client,
        lhs: PlainTensor,
        rhs: StoredTensor,
        table: MatmulIndexTable,
        out: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        if table.entries().elem != ElemType::UInt(UIntKind::U32) {
            return Err(MATMUL_TABLE_NOT_U32);
        }
        if let MatmulIndexTable::GroupedRows {
            entries, groups, ..
        } = &table
        {
            Self::admit_group_ends(groups, entries)?;
        }
        match rhs {
            StoredTensor::Plain(rhs) => {
                Self::indexed_float(client, lhs, rhs, table, out, rows_kind)
            }
            StoredTensor::Scaled(_) => Err(MATMUL_SCALED_BESIDE_FLOAT),
            StoredTensor::Packed(rhs) => Self::indexed_packed(client, lhs, rhs, table, out),
        }
    }

    /// `tensor`, a float operand over `dimensions`, as a product beside a packed input reads it:
    /// dense rows of an element the packed routines are verified at, a copy where it is not dense.
    ///
    /// # Assumptions
    ///
    /// `dimensions` holds `K`: the operand is one of the product's inputs.
    ///
    /// # Errors
    ///
    /// [`MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED`], and [`MATMUL_CONTRACTIONS_DISAGREE`] for an
    /// operand whose contraction dim is not `k`.
    ///
    /// # Panics
    ///
    /// Where `dimensions` lacks `K`, against its Assumption.
    pub fn beside_packed(
        client: &Client,
        tensor: PlainTensor,
        dimensions: [Dimension; 2],
        k: usize,
    ) -> Result<PlainTensor, Decline> {
        if !MatmulInputs::FLOATS.contains(&tensor.elem) {
            return Err(MATMUL_PACKED_FLOAT_ELEMENT_UNVERIFIED);
        }
        if !Self::contracts(tensor.shape().as_slice(), dimensions, k) {
            return Err(MATMUL_CONTRACTIONS_DISAGREE);
        }
        Ok(tensor.into_contiguous(client))
    }

    /// The three operands bound as `blueprint` states them, on `grid`: under its partitioning, at
    /// its widths, the output its buffer.
    ///
    /// # Errors
    ///
    /// Why the tile engine refuses to bind a float input as it is stored under the launch: one
    /// stored in tiles that the launch's last cubes reach past, which the routine, reading the
    /// problem and not the tensors, could not see. [`Decline::Broken`] where it refuses anything
    /// else the routine prepared, and on an output masked where its tiles do not overhang it.
    pub fn bind<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Bound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                self.bind_float(lhs, rhs, blueprint, grid)
            }
            _ => self.bind_packed(blueprint, grid),
        }
    }
}

impl MatmulTensors<Geometry> {
    /// The product of `lhs` and `rhs` as a launch holding its rows as `rows_kind` says reads it,
    /// handed to a sink the caller binds: nothing is allocated, and the output is laid out densely,
    /// row-major, in a float input's element, as rows where the launch reads every row whole.
    ///
    /// # Errors
    ///
    /// As [`MatmulTensors::new`].
    pub fn sink(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        match (lhs, rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                Self::float(client, lhs, rhs, rows_kind)
            }
            (lhs, rhs) => Self::packed(client, lhs, rhs),
        }
    }

    /// The three operands bound as `blueprint` states them, on `grid`, the output derived over the
    /// sink's layout rather than a buffer: its spec, and the geometry the derivation settled, which
    /// the caller's sink addresses through.
    ///
    /// # Errors
    ///
    /// As the `bind` of the tensors writing into a buffer.
    pub fn bind<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Unbound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => {
                self.bind_float(lhs, rhs, blueprint, grid)
            }
            _ => self.bind_packed(blueprint, grid),
        }
    }
}

impl MatmulTensors<MatmulOutputBuffer> {
    /// The output, bound, as the argument of a launch, read through the table where one places it.
    ///
    /// # Errors
    ///
    /// Why the tile engine refuses to place the output by its table.
    pub fn output_through_table(&self, out: Bound) -> Result<TileTensorLaunch, Decline> {
        self.read_through_table(MatmulOperand::Out, out.arg())
    }

    /// The output as a launch wrote it, laid out as the runtime pads it, where the launch
    /// allocated it; `None` where the caller handed its own ([`writing_into`](Self::writing_into)).
    pub fn into_out(self) -> Option<TensorHandle> {
        self.out.tensor
    }
}

impl<Out> MatmulTensors<Out> {
    /// Whether the tensors were read for a launch holding its rows as `rows_kind` says: float
    /// ones read for that way of holding the rows, and packed ones, which every launch reads
    /// alike, always.
    pub fn is_read_for(&self, rows_kind: MatmulRowsKind) -> bool {
        self.holds_rows_as(rows_kind)
    }

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
}

impl<Out: MatmulOutput> MatmulTensors<Out> {
    /// The problem the three tensors state.
    pub fn problem(&self) -> MatmulRoutineProblem {
        self.stated.0.get_or_init(|| self.state_problem()).clone()
    }

    /// The work item the launch they were read for hands its microkernel.
    pub fn work_item(&self) -> MatmulWorkItem {
        self.read(|view| view.work_item())
    }
}

impl MatmulTensors {
    /// Whether `groups` ends each row group in one `u32`, as `entries` is one slab.
    ///
    /// # Errors
    ///
    /// [`MATMUL_TABLE_NOT_U32`] and [`MATMUL_GROUP_ENDS_NOT_ONE_PER_GROUP`].
    fn admit_group_ends(groups: &RowGroups, entries: &PlainTensor) -> Result<(), Decline> {
        let ends = &groups.ends;
        match (ends.elem, ends.shape() == entries.shape()) {
            (ElemType::UInt(UIntKind::U32), true) if entries.shape().rank() == 1 => Ok(()),
            (ElemType::UInt(UIntKind::U32), _) => Err(MATMUL_GROUP_ENDS_NOT_ONE_PER_GROUP),
            _ => Err(MATMUL_TABLE_NOT_U32),
        }
    }

    /// [`indexed`](Self::indexed) against a float rhs, holding the rows as `rows_kind` says.
    fn indexed_float(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        table: MatmulIndexTable,
        out: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        // Every check reads shapes alone, so they run before a self-aliasing operand is copied.
        if table.groups().is_some() && lhs.is_tiled() {
            return Err(MATMUL_GROUPED_ROWS_OVER_TILED_LHS);
        }
        Self::admit_stack(&rhs, table.index())?;
        let matrix_shape = Self::matrix_of(&rhs, table.index()).shape();
        let out = Self::indexed_output(&lhs, matrix_shape[0], matrix_shape[1], &table, out)?;
        let (lhs, rhs) = MatmulInputs::new(client, lhs, rhs)?
            .addressable(client)
            .into_parts();
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(lhs),
            rhs: StoredTensor::Plain(rhs),
            out,
            rows_kind,
            table: Some(table),
            stated: StatedProblem::default(),
        })
    }

    /// [`indexed`](Self::indexed) against a packed stack, every row of a batch held whole.
    fn indexed_packed(
        client: &Client,
        lhs: PlainTensor,
        rhs: PackedTensor,
        table: MatmulIndexTable,
        out: PlainTensor,
    ) -> Result<Self, Decline> {
        let (MatmulIndex::Slab { slabs } | MatmulIndex::GroupedRows { slabs }) = table.index() else {
            return Err(PACKED_STACK_UNREAD);
        };
        if rhs.slabs() != Some(slabs) {
            return Err(MATMUL_STACK_NOT_THE_TABLES);
        }
        let k = rhs.extent(K);
        let lhs = Self::beside_packed(client, lhs, [M, K], k)?;
        let out = Self::indexed_output(&lhs, k, rhs.extent(N), &table, out)?;
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(lhs),
            rhs: StoredTensor::Packed(rhs),
            out,
            rows_kind: MatmulRowsKind::Whole,
            table: Some(table),
            stated: StatedProblem::default(),
        })
    }

    /// `out`, the output of `lhs` against one `[k, n]` matrix per batch of `table`, checked to be
    /// `[batches.., rows, n]` in a float input's element, or the flat `[rows, n]` of grouped rows.
    ///
    /// # Errors
    ///
    /// [`MATMUL_OUTPUT_UNREAD`] for an output of another shape or element, or stored in tiles, and
    /// [`MATMUL_CONTRACTIONS_DISAGREE`] for an lhs that is no `[.., rows, k]`.
    fn indexed_output(
        lhs: &PlainTensor,
        k: usize,
        n: usize,
        table: &MatmulIndexTable,
        out: PlainTensor,
    ) -> Result<MatmulOutputBuffer, Decline> {
        let shape = lhs.shape();
        let rank = shape.rank();
        if rank < 2 || shape[rank - 1] != k {
            return Err(MATMUL_CONTRACTIONS_DISAGREE);
        }
        let out_shape = match table.groups() {
            Some(_) => Self::flat_output_shape(shape.as_slice(), n)?,
            None => [table.entries().shape().as_slice(), &[shape[rank - 2], n]].concat(),
        };
        if out.shape().as_slice() != out_shape.as_slice() || out.elem != lhs.elem || out.is_tiled()
        {
            return Err(MATMUL_OUTPUT_UNREAD);
        }
        Ok(MatmulOutputBuffer {
            tensor: None,
            bound: out,
        })
    }

    /// The `[rows, n]` output of a flat lhs of `shape`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_CONTRACTIONS_DISAGREE`] for an lhs that is no `[rows, k]`.
    fn flat_output_shape(shape: &[usize], n: usize) -> Result<Vec<usize>, Decline> {
        match shape {
            [rows, _] => Ok(vec![*rows, n]),
            _ => Err(MATMUL_CONTRACTIONS_DISAGREE),
        }
    }

    /// Whether `rhs` is the stack `index` reads: `[slabs, k, n]`, or one `[k, n]` every batch
    /// reads a run of.
    ///
    /// # Errors
    ///
    /// [`MATMUL_STACK_NOT_THE_TABLES`].
    fn admit_stack(rhs: &PlainTensor, index: MatmulIndex) -> Result<(), Decline> {
        match (index, rhs.shape().as_slice()) {
            (MatmulIndex::Slab { slabs } | MatmulIndex::GroupedRows { slabs }, [stacked, _, _])
                if *stacked == slabs =>
            {
                Ok(())
            }
            (MatmulIndex::RunOfK, [_, _]) => Ok(()),
            _ => Err(MATMUL_STACK_NOT_THE_TABLES),
        }
    }

    /// The one `[k, n]` matrix of `rhs` a batch contracts against: a stacked rhs's first slab,
    /// at the slab's strides, or the rhs itself.
    ///
    /// # Assumptions
    ///
    /// The rhs is one [`admit_stack`](Self::admit_stack) takes.
    fn matrix_of(rhs: &PlainTensor, index: MatmulIndex) -> PlainTensor {
        match index {
            MatmulIndex::Slab { .. } | MatmulIndex::GroupedRows { .. } => Self::first_slab(rhs),
            MatmulIndex::RunOfK => rhs.clone(),
        }
    }

    /// A stacked rhs's first `[k, n]` slab, at the slab's strides.
    fn first_slab(rhs: &PlainTensor) -> PlainTensor {
        let (shape, strides) = (&rhs.binding.shape, &rhs.binding.strides);
        rhs.viewed(Metadata::new(shape[1..].to_vec(), strides[1..].to_vec()))
    }

    /// `operand` as a product beside a packed input reads it, along `dimensions` and `k` deep: a
    /// float one [`beside_packed`](Self::beside_packed), a packed one as it is stored, and 8-bit
    /// floats under scales where they lie, in rows or in tiles that the packed input's `block` of
    /// `K` splits.
    ///
    /// # Errors
    ///
    /// As [`beside_packed`](Self::beside_packed).
    fn stored_beside_packed(
        client: &Client,
        operand: StoredTensor,
        dimensions: [Dimension; 2],
        k: usize,
        block: Option<usize>,
    ) -> Result<StoredTensor, Decline> {
        match operand {
            StoredTensor::Plain(tensor) => {
                Self::beside_packed(client, tensor, dimensions, k).map(StoredTensor::Plain)
            }
            StoredTensor::Scaled(tensor) => {
                Self::scaled_beside_packed(client, tensor, dimensions, k, block)
            }
            StoredTensor::Packed(_) => Ok(operand),
        }
    }

    /// `tensor`, along `dimensions` and `k` deep, read in rows or in the tiles it lies in where the
    /// packed input's `block` of `K` splits them ([`PlainTensor::readable_in_blocks`]).
    ///
    /// # Assumptions
    ///
    /// `dimensions` holds `K`: the operand is one of the product's inputs.
    ///
    /// # Errors
    ///
    /// [`MATMUL_CONTRACTIONS_DISAGREE`] for values whose contraction dim is not `k`.
    ///
    /// # Panics
    ///
    /// Where `dimensions` lacks `K`, against its Assumption.
    fn scaled_beside_packed(
        client: &Client,
        tensor: ScaledTensor,
        dimensions: [Dimension; 2],
        k: usize,
        block: Option<usize>,
    ) -> Result<StoredTensor, Decline> {
        if !Self::contracts(&tensor.values.logical_shape(), dimensions, k) {
            return Err(MATMUL_CONTRACTIONS_DISAGREE);
        }
        let values = tensor.values.readable_in_blocks(client, block);
        Ok(StoredTensor::Scaled(ScaledTensor { values, ..tensor }))
    }

    /// Whether a matrix of `shape`, its last two dims along `dimensions`, contracts `k` deep.
    ///
    /// # Panics
    ///
    /// Where `dimensions` lacks `K`.
    fn contracts(shape: &[usize], dimensions: [Dimension; 2], k: usize) -> bool {
        let contracted = dimensions
            .iter()
            .position(|&dimension| dimension == K)
            .expect("dimensions holds K");
        shape.len() >= 2 && shape[shape.len() - 2 + contracted] == k
    }

    /// The rows of `lhs`, `k` deep, its leading dims merged into them.
    fn rows_of(lhs: &StoredTensor, k: usize) -> usize {
        match lhs {
            StoredTensor::Plain(_) | StoredTensor::Scaled(_) => {
                lhs.values().shape().num_elements() / k.max(1)
            }
            StoredTensor::Packed(lhs) => lhs.extent(M),
        }
    }

    /// The element a product beside a packed input is written in: a float input's, or, where
    /// neither input names one, the element their words unpack into
    /// ([`PACKED`](MatmulRoutineProblem::PACKED)).
    pub(super) fn packed_output(lhs: &StoredTensor, rhs: &StoredTensor) -> ElemType {
        match (lhs, rhs) {
            (StoredTensor::Plain(input), _) | (_, StoredTensor::Plain(input)) => input.elem,
            _ => MatmulRoutineProblem::PACKED,
        }
    }

    /// `lhs` against the float `rhs`, holding the rows as `rows_kind` says, the output allocated
    /// in the operands' element.
    fn float(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
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
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// `lhs` against `rhs`, either packed, the output allocated as `[.., n]` in `out`.
    fn packed(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: ElemType,
    ) -> Result<Self, Decline> {
        let (lhs, rhs, shape) = Self::beside_packed_inputs(client, lhs, rhs)?;
        let (tensor, bound) = PlainTensor::empty(client, shape.into(), out);
        let tensor = Some(tensor);
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            out: MatmulOutputBuffer { tensor, bound },
            rows_kind: MatmulRowsKind::Whole,
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// `lhs` against `rhs`, either packed, written into `out`, which the caller allocated as the
    /// product's dense rows `[.., n]`.
    ///
    /// # Errors
    ///
    /// As [`packed`](Self::packed), and [`MATMUL_OUTPUT_UNREAD`] for an output of another shape,
    /// of an element the routines do not write, or other than row-major.
    fn packed_into(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: PlainTensor,
    ) -> Result<Self, Decline> {
        let (lhs, rhs, shape) = Self::beside_packed_inputs(client, lhs, rhs)?;
        if out.shape().as_slice() != shape.as_slice()
            || !MatmulInputs::VERIFIED.contains(&out.elem)
            || !out.is_contiguous()
        {
            return Err(MATMUL_OUTPUT_UNREAD);
        }
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            out: MatmulOutputBuffer {
                tensor: None,
                bound: out,
            },
            rows_kind: MatmulRowsKind::Whole,
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// `lhs` and `rhs`, either packed, as a product beside a packed input reads them
    /// ([`stored_beside_packed`](Self::stored_beside_packed)), and the shape of its output: the
    /// float lhs's, or a packed one's rows, then `n`.
    fn beside_packed_inputs(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
    ) -> Result<(StoredTensor, StoredTensor, Vec<usize>), Decline> {
        let (lhs, rhs, k) = Self::stored_inputs(client, lhs, rhs)?;
        let mut shape = match &lhs {
            StoredTensor::Plain(_) | StoredTensor::Scaled(_) => lhs.values().logical_shape(),
            StoredTensor::Packed(lhs) => vec![lhs.extent(M), k],
        };
        *shape.last_mut().expect("the lhs has a rank") = rhs.extent([K, N], N);
        Ok((lhs, rhs, shape))
    }

    /// `lhs` and `rhs`, either packed, as a product beside a packed input reads them
    /// ([`stored_beside_packed`](Self::stored_beside_packed)), and their `k`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_SCALED_BESIDE_FLOAT`] where neither is packed, and as
    /// [`stored_beside_packed`](Self::stored_beside_packed).
    fn stored_inputs(
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
    ) -> Result<(StoredTensor, StoredTensor, usize), Decline> {
        let k = rhs.extent([K, N], K);
        let block = match (&rhs, &lhs) {
            (StoredTensor::Packed(packed), _) | (_, StoredTensor::Packed(packed)) => {
                packed.operand().grouping(K, k)
            }
            _ => return Err(MATMUL_SCALED_BESIDE_FLOAT),
        };
        Ok((
            Self::stored_beside_packed(client, lhs, [M, K], k, block)?,
            Self::stored_beside_packed(client, rhs, [K, N], k, block)?,
            k,
        ))
    }

    /// The operands bound as float ones, each read in the order its microkernel reads it.
    fn bind_float<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Bound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        let (launcher, [lhs, rhs]) = self.bind_operands(lhs, rhs, blueprint, grid)?;
        let (batches, _) = self.frame(MatmulOperand::Out, &self.out.bound, self.batch_dimensions());
        let out = launcher
            .arg(self.out.bound.binding.clone(), self.out.bound.elem)
            .dimensions(&[M, N])
            .batches(&batches)
            .vectorize(blueprint.widths.out)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        blueprint
            .partitioning
            .check_masked_where_overhanging(&out.spec)?;
        Ok(MatmulBinding {
            launcher,
            lhs,
            rhs,
            out,
        })
    }

    /// The operands bound where either input is packed, the output restated over the block-split
    /// space.
    fn bind_packed<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Bound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        let launcher = self.packed_launcher(blueprint, grid)?;
        let blocks = self.blocks();
        let (batches, flat_rows) =
            self.frame(MatmulOperand::Out, &self.out.bound, self.table_batches());
        let blocks = MatmulBlocks {
            m: flat_rows.unwrap_or(blocks.m),
            ..blocks
        };
        let (out, out_dimensions) = self.out.bound.restated(&blocks, M, N, &batches);
        let out = launcher
            .arg(out.binding, out.elem)
            .dimensions(&out_dimensions)
            .batches(&batches)
            .vectorize(blueprint.widths.out)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        self.bind_packed_inputs(launcher, blueprint, out)
    }
}

impl MatmulTensors<Geometry> {
    /// `lhs` against the float `rhs`, holding the rows as `rows_kind` says.
    fn float(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        let operands = Operands::read(client, lhs, rhs, rows_kind)?;
        let out = Geometry::row_major(&operands.sink_shape());
        Ok(Self {
            client: client.clone(),
            lhs: StoredTensor::Plain(operands.lhs),
            rhs: StoredTensor::Plain(operands.rhs),
            out,
            rows_kind,
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// `lhs` against `rhs`, either packed, the output laid out as dense rows.
    fn packed(client: &Client, lhs: StoredTensor, rhs: StoredTensor) -> Result<Self, Decline> {
        let (lhs, rhs, k) = MatmulTensors::stored_inputs(client, lhs, rhs)?;
        let out = Geometry::row_major(&[MatmulTensors::rows_of(&lhs, k), rhs.extent([K, N], N)]);
        Ok(Self {
            client: client.clone(),
            lhs,
            rhs,
            out,
            rows_kind: MatmulRowsKind::Whole,
            table: None,
            stated: StatedProblem::default(),
        })
    }

    /// The operands bound as float ones, each read in the order its microkernel reads it.
    fn bind_float<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Unbound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        let (launcher, [lhs, rhs]) = self.bind_operands(lhs, rhs, blueprint, grid)?;
        let out = launcher
            .unbound(&self.out)
            .dimensions(&[M, N])
            .batches(&self.batch_dimensions())
            .vectorize(blueprint.widths.out)
            .build_spec()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        blueprint
            .partitioning
            .check_masked_where_overhanging(&out.spec)?;
        Ok(MatmulBinding {
            launcher,
            lhs,
            rhs,
            out,
        })
    }

    /// The operands bound where either input is packed, the output stated as a buffer output is
    /// ([`MatmulBlocks::restated`]): a dim for each dimension of the block-split space, each at its
    /// own stride over the sink's dense rows.
    fn bind_packed<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<MatmulBinding<Unbound, TileTensorLaunch, TileTensorLaunch>, Decline> {
        let launcher = self.packed_launcher(blueprint, grid)?;
        let dims = self
            .blocks()
            .restated_dims(M, N, self.rhs.extent([K, N], N));
        let geometry: Vec<_> = dims
            .iter()
            .map(|&(_, extent, stride)| (extent, stride))
            .collect();
        let dimensions: Vec<_> = dims.iter().map(|&(dimension, _, _)| dimension).collect();
        let out = launcher
            .unbound(&Geometry::new(&geometry))
            .dimensions(&dimensions)
            .vectorize(blueprint.widths.out)
            .build_spec()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        self.bind_packed_inputs(launcher, blueprint, out)
    }
}

impl<Out: MatmulOutput> MatmulTensors<Out> {
    /// The output's tensor map, its descriptor moving `stage`'s box of the output, dense: what a
    /// routine binds where the TMA engine stores its cubes' boxes
    /// ([`TmaStored`](cubecl::tile::launch::TmaStored)).
    ///
    /// # Assumptions
    ///
    /// Float tensors are read for a launch tiling its rows ([`is_read_for`](Self::is_read_for)).
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a sink, which no descriptor addresses, and on a packed product whose
    /// blocks cut its rows or columns: the store that chose the engine refused both.
    pub(crate) fn tma_output(&self, stage: &MatmulStage) -> Result<TileTensorLaunch, Decline> {
        // A packed product states its output over the block-split space, its rows and columns
        // one block each where the store took it (`CubeStagedTmaStoreDecline::BlockSplit`): the
        // positions inside them, which the descriptor's rows and columns are.
        let output: [Dimension; 2] = match self.read(|view| view.blocks()) {
            None => [M, N],
            Some(blocks) => match blocks.restated_dims(M, N, 1).as_slice() {
                [(rows, ..), (columns, ..)] => [*rows, *columns],
                _ => return Breakable::broken(TMA_STORES_ROWS_AND_COLUMNS),
            },
        };
        let dimensions = stage.dimensions();
        let Some(out) = self.out.addressed() else {
            return Breakable::broken(TMA_STORES_INTO_A_BUFFER);
        };
        let (map, transposed) = tma_store_operand(
            out.binding.clone(),
            1,
            MatrixLayout::RowMajor,
            (stage.extent(dimensions.m), stage.extent(dimensions.n)),
            out.elem,
            TensorMapSwizzle::None,
        );
        Ok(TileTensorLaunch::tensor_map(
            map,
            &output,
            TmaBox {
                rows: self.rows() as u32,
                cols: self.problem().n as u32,
                batch: None,
                transposed,
            },
        ))
    }
}

impl<Out: MatmulOutput> MatmulTensors<Out> {
    /// The element the packed inputs are decoded in: the other input's, the output's where both are
    /// packed.
    fn decoded_element(&self) -> ElemType {
        [&self.lhs, &self.rhs]
            .into_iter()
            .find_map(|input| match input {
                StoredTensor::Plain(tensor) => Some(tensor.elem),
                StoredTensor::Scaled(tensor) => Some(tensor.values.elem),
                StoredTensor::Packed(_) => None,
            })
            .unwrap_or_else(|| self.problem().elements.out)
    }

    /// The problem these tensors state, read off them.
    fn state_problem(&self) -> MatmulRoutineProblem {
        match (&self.lhs, &self.rhs) {
            (StoredTensor::Plain(lhs), StoredTensor::Plain(rhs)) => match &self.table {
                None => self.out.problem(lhs, rhs),
                Some(table) => self.indexed_problem(lhs, rhs, table),
            },
            (lhs, rhs) => {
                let k = rhs.extent([K, N], K);
                let packing = |operand: &StoredTensor| match operand {
                    StoredTensor::Plain(_) | StoredTensor::Scaled(_) => None,
                    StoredTensor::Packed(operand) => Some(operand.packing()),
                };
                let packings = MatmulOperands {
                    lhs: packing(lhs),
                    rhs: packing(rhs),
                    out: None,
                };
                let stored_tile = |operand: &StoredTensor| match operand {
                    StoredTensor::Plain(_) | StoredTensor::Scaled(_) => None,
                    StoredTensor::Packed(operand) => operand.stored_tile(),
                };
                // A float input is read at its own element, whatever the output is written in: an
                // f16 activation into an f32 output is still an f16 activation.
                let element = |operand: &StoredTensor| match operand {
                    StoredTensor::Plain(_) | StoredTensor::Scaled(_) => operand.values().elem,
                    StoredTensor::Packed(_) => MatmulRoutineProblem::PACKED,
                };
                let out = self.out.element(lhs, rhs);
                let packed = MatmulRoutineProblem::packed(
                    MatmulTensors::rows_of(lhs, k),
                    rhs.extent([K, N], N),
                    k,
                    out,
                    packings,
                );
                let input = |operand: &StoredTensor, stated: StoredInput| match operand {
                    StoredTensor::Plain(_) | StoredTensor::Scaled(_) => {
                        StoredInput::plain(operand.values().elem)
                    }
                    StoredTensor::Packed(_) => stated,
                };
                // A float input stored in tiles is read in the pieces it is stored in.
                let alignment =
                    |operand: &StoredTensor, dense: ElemAlignment| match packing(operand).is_none()
                        && operand.values().is_tiled()
                    {
                        true => operand.values().elem_alignment(),
                        false => dense,
                    };
                let alignments = MatmulOperands {
                    lhs: alignment(lhs, packed.elem_alignments.lhs),
                    rhs: alignment(rhs, packed.elem_alignments.rhs),
                    out: packed.elem_alignments.out,
                };
                let problem = MatmulRoutineProblem {
                    destination: Out::DESTINATION,
                    elem_alignments: alignments,
                    buffer_alignments: alignments,
                    stored_tiles: MatmulOperands {
                        lhs: stored_tile(lhs),
                        rhs: stored_tile(rhs),
                        out: None,
                    },
                    elements: MatmulOperands {
                        lhs: element(lhs),
                        rhs: element(rhs),
                        out,
                    },
                    inputs: MatmulOperands {
                        lhs: input(lhs, packed.inputs.lhs),
                        rhs: input(rhs, packed.inputs.rhs),
                        out: packed.inputs.out,
                    },
                    ..packed
                };
                match &self.table {
                    None => problem,
                    Some(table) => Self::indexed_packed_problem(problem, lhs, table),
                }
            }
        }
    }

    /// `problem`, a packed one over every row of `lhs`, restated over the batches of `table`, each
    /// holding its own rows and reading its entry.
    fn indexed_packed_problem(
        problem: MatmulRoutineProblem,
        lhs: &StoredTensor,
        table: &MatmulIndexTable,
    ) -> MatmulRoutineProblem {
        let shape = lhs.values().shape();
        let rank = shape.rank();
        let batches = table.entries().shape().to_vec();
        let count = |dims: &[usize]| dims.iter().product();
        let problem = match table.groups() {
            Some(groups) => problem.batched_by_row_groups(groups.max_rows, batches),
            None => MatmulRoutineProblem {
                stated_rows: shape[rank - 2],
                batches_held: MatmulOperands {
                    lhs: count(&shape[..rank - 2]),
                    rhs: 1,
                    out: count(&batches),
                },
                batches,
                ..problem
            },
        };
        MatmulRoutineProblem {
            index: Some(table.index()),
            ..problem
        }
    }

    /// The product of `lhs` and `rhs`, every batch reading its entry of `table`: each input stated
    /// at the output's rank, its missing batch dims broadcast, a stacked rhs as one slab of it and
    /// aligned as the stack is, so a line never reads across two slabs.
    fn indexed_problem(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        table: &MatmulIndexTable,
    ) -> MatmulRoutineProblem {
        let rank = match table.groups() {
            Some(_) => 2,
            None => table.entries().shape().rank() + 2,
        };
        let matrix = MatmulTensors::matrix_of(rhs, table.index());
        let problem = self
            .out
            .problem(&lhs.broadcast_to(rank), &matrix.broadcast_to(rank));
        let (shape, strides) = (&rhs.binding.shape, &rhs.binding.strides);
        let rhs_layout = match problem.layouts.rhs {
            MatmulLayout::Strided => Self::stacked_layout(&matrix),
            layout => layout,
        };
        let problem = match table.groups() {
            Some(groups) => {
                problem.batched_by_row_groups(groups.max_rows, table.entries().shape().to_vec())
            }
            None => problem,
        };
        MatmulRoutineProblem {
            elem_alignments: MatmulOperands {
                rhs: rhs.elem_alignment(),
                ..problem.elem_alignments
            },
            layouts: MatmulOperands {
                rhs: rhs_layout,
                ..problem.layouts
            },
            buffer_alignments: MatmulOperands {
                rhs: rhs_layout.buffer_alignment(shape, strides),
                ..problem.buffer_alignments
            },
            index: Some(table.index()),
            ..problem
        }
    }

    /// How `slab`, one `[k, n]` slab of a stack, lies where it is one wide along a dimension: along
    /// that dimension where its stride there is the unit one. A dim of extent one addresses nothing
    /// in a matrix of its own, but a slab's is cut from its stack, whose lines run along it.
    fn stacked_layout(slab: &PlainTensor) -> MatmulLayout {
        let (shape, strides) = (&slab.binding.shape, &slab.binding.strides);
        match ([shape[0], shape[1]], [strides[0], strides[1]]) {
            ([_, 1], [_, 1]) => MatmulLayout::Row,
            ([1, _], [1, _]) => MatmulLayout::Col,
            _ => MatmulLayout::Strided,
        }
    }

    /// The launcher over the blueprint's partitioning, as its kernel compiles it: the blocks in,
    /// and the rows inside the one block along `M` read at launch where they are tiled and no
    /// packed input's scales vary along them ([`MatmulBlocks::scaled_dimensions`]).
    fn packed_launcher<Mk, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<Launcher, Decline> {
        grid.launcher(self.client(), || {
            (
                blueprint
                    .partitioning
                    .kernel_compiling(&self.blocks().scaled_dimensions()),
                blueprint.partitioning.as_tile().space().clone(),
            )
        })
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

    /// Its inputs read as `planned`, the space a microkernel planned on: every packed input at the
    /// reads it takes ([`MatmulMicrokernel::planned_space`]), whichever engine binds it; as stored
    /// where `planned` is `None`.
    fn planned_inputs(&self, planned: Option<MatmulBlocks>) -> [StoredTensor; 2] {
        [
            (&self.lhs, MatmulOperand::Lhs),
            (&self.rhs, MatmulOperand::Rhs),
        ]
        .map(|(operand, input)| {
            match (operand, planned.and_then(|space| space.packings.get(input))) {
                (StoredTensor::Packed(tensor), Some(packing)) => {
                    StoredTensor::Packed(tensor.read_as(packing.operand))
                }
                _ => operand.clone(),
            }
        })
    }

    /// Both inputs under `launcher`, beside `out`: each packed one as one tile of its words under
    /// its scales, each float one restated over the block-split space, each moved into the stages
    /// as the microkernel says ([`delivery`](MatmulMicrokernel::delivery),
    /// [`words_delivery`](MatmulMicrokernel::words_delivery)).
    fn bind_packed_inputs<Mk: MatmulMicrokernel, SM, O>(
        &self,
        launcher: Launcher,
        blueprint: &MatmulBlueprint<Mk, SM>,
        out: O,
    ) -> Result<MatmulBinding<O, TileTensorLaunch, TileTensorLaunch>, Decline> {
        let blocks = self.blocks();
        let outer_scale = blueprint
            .microkernel
            .outer_scale()
            .read_in(self.decoded_element());
        let values = |operand: MatmulOperand,
                      tensor: &PlainTensor,
                      [rows, columns]: [Dimension; 2],
                      width: usize| {
            let (batches, flat_rows) = self.frame(operand, tensor, self.table_batches());
            let read = MatmulBlocks {
                m: flat_rows.unwrap_or(blocks.m),
                ..blocks
            };
            let (tensor, dimensions) = tensor.restated(&read, rows, columns, &batches);
            launcher
                .arg(tensor.binding, tensor.elem)
                .dimensions(&dimensions)
                .batches(&batches)
                .vectorize(width)
                .delivery(blueprint.microkernel.delivery())
                .build()
                .or_broken(ADMITTED_PLAN_RUNS)
                .map(Bound::arg)
        };
        let bind = |which: MatmulOperand,
                    operand: &StoredTensor,
                    dimensions: [Dimension; 2],
                    width: usize| match operand {
            StoredTensor::Plain(tensor) => values(which, tensor, dimensions, width),
            StoredTensor::Scaled(tensor) => tensor.under_scales(
                &launcher,
                &blocks,
                values(which, &tensor.values, dimensions, width)?,
            ),
            // A launch the TMA engine fills binds its packed inputs again through descriptors
            // (`tma_packed_operands`), scales and all, so the launcher states their scales as a
            // reader of fields takes them, which reads them where they lie, in tiles or in
            // rows.
            StoredTensor::Packed(tensor) => tensor.bind(
                &launcher,
                width,
                match blueprint.microkernel.fills_by_tma() {
                    true => BlockScales::Fields,
                    false => blueprint.microkernel.block_scales(),
                },
                outer_scale,
                blueprint.microkernel.words_delivery(),
            ),
        };
        let [lhs, rhs] = self.planned_inputs(blueprint.microkernel.planned_space());
        let lhs = bind(MatmulOperand::Lhs, &lhs, [M, K], blueprint.widths.lhs)?;
        let rhs = bind(MatmulOperand::Rhs, &rhs, [K, N], blueprint.widths.rhs)?;
        let lhs = self.read_through_table(MatmulOperand::Lhs, lhs)?;
        let rhs = self.read_through_table(MatmulOperand::Rhs, rhs)?;
        Ok(MatmulBinding {
            launcher,
            lhs,
            rhs,
            out,
        })
    }

    /// The float inputs, which only a launch tiling its rows stages or splits `K` over.
    ///
    /// # Panics
    ///
    /// On a packed input or one under scales: both inputs are float.
    fn plain(&self) -> [&PlainTensor; 2] {
        [&self.lhs, &self.rhs].map(|operand| match operand {
            StoredTensor::Plain(tensor) => tensor,
            StoredTensor::Scaled(_) | StoredTensor::Packed(_) => panic!("both inputs are float"),
        })
    }

    /// The dimensions the problem's batch dims are labelled with, read off the operands' rank
    /// without stating the whole problem ([`MatmulRoutineProblem::batch_dimensions`]): the lhs's,
    /// or a table's dims, which are the batches.
    fn batch_dimensions(&self) -> Vec<Dimension> {
        let dims = match &self.table {
            Some(table) => table.entries().shape().rank(),
            None => self.lhs.values().shape().rank() - 2,
        };
        MatmulRoutineProblem::batch_dimensions_of(dims)
    }

    /// The batches a table's dims are, which a packed product keeps apart; none without a table.
    fn table_batches(&self) -> Vec<Dimension> {
        self.table.as_ref().map_or_else(Vec::new, |table| {
            MatmulRoutineProblem::batch_dimensions_of(table.entries().shape().rank())
        })
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
    /// described in `rhs` order, a col rhs as the `[n, k]` matrix its buffer is, and each box
    /// swizzled as the tile DSL says the stage keeps its rows (`StageStorage::tma_swizzle`). What a
    /// routine binds where the TMA engine fills its stages.
    ///
    /// # Assumptions
    ///
    /// The tensors are float ones read for a launch tiling its rows: only such a launch stages
    /// them through these descriptors.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a stage no box lands as `storage` holds it, which the fill that chose
    /// the engine refused.
    ///
    /// # Panics
    ///
    /// On a packed input, against its Assumption.
    pub(crate) fn tma_operands(
        &self,
        stage: &MatmulStage,
        storage: &StageStorage,
        orders: MatmulOperands<BindingOrder>,
    ) -> Result<(TileTensorLaunch, TileTensorLaunch), Decline> {
        let problem = self.problem();
        let [lhs, rhs] = self.plain();
        let landing = |tensor: &PlainTensor, operand: MatmulOperand| {
            let stage = stage.operand_space(operand, orders);
            let swizzle = storage
                .tma_swizzle(&stage, 1, tensor.elem.size_bits())
                .or_broken(TMA_FILLS_A_STAGE_A_BOX_LANDS)?;
            Ok::<_, Decline>(TmaLanding { stage, swizzle })
        };
        let landed = landing(rhs, MatmulOperand::Rhs)?;
        let lhs = Self::tma_matrix(
            lhs,
            [(M, self.rows()), (K, problem.k)],
            landing(lhs, MatmulOperand::Lhs)?,
        );
        let rhs = match orders.rhs {
            BindingOrder::Stated => Self::tma_matrix(rhs, [(K, problem.k), (N, problem.n)], landed),
            BindingOrder::InBufferOrder => Self::tma_matrix(
                &rhs.clone().in_buffer_order(),
                [(N, problem.n), (K, problem.k)],
                landed,
            ),
        };
        Ok((lhs, rhs))
    }

    /// A packed product's inputs for stages the TMA engine fills, each landing as `storage`
    /// holds its stage of `stage` and served `widths` wide: a float lhs through a descriptor over
    /// the matrix its blocks split; each packed input's
    /// words through one over the rows or the tiles they are stored in
    /// ([`PackedTensor::bind_by_tma`]), its scales bound under `launcher` as `scales` says, for
    /// the copy engine to keep beside them. What a fill
    /// [`CubeStagedTmaDecline`](cubek_microkernel::matmul::cube::CubeStagedTmaDecline) admits.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] on a float rhs, since a packed product's rhs is packed; as
    /// [`PackedTensor::bind_by_tma`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tma_packed_operands(
        &self,
        launcher: &Launcher,
        stage: &MatmulStage,
        storage: &StageStorage,
        orders: MatmulOperands<BindingOrder>,
        widths: MatmulOperands<usize>,
        scales: BlockScales,
        planned: Option<MatmulBlocks>,
    ) -> Result<(TileTensorLaunch, TileTensorLaunch), Decline> {
        // Every packed input landed at the reads its microkernel planned on.
        let [planned_lhs, planned_rhs] = self.planned_inputs(planned);
        let StoredTensor::Packed(rhs) = &planned_rhs else {
            return Breakable::broken(TMA_FILLS_A_PACKED_RHS);
        };
        // A packed input `operand` across `across`, its words landing a word a cell.
        let packed = |tensor: &PackedTensor, operand: MatmulOperand, across, width| {
            let values_dimensions = tile_order::values_dimensions(across);
            let landed = stage.landed_space(operand, orders, &values_dimensions);
            let values_per_word = tensor.scheme().num_quants();
            let swizzle = storage
                .tma_swizzle(
                    &landed,
                    values_per_word,
                    u32::BITS as usize / values_per_word,
                )
                .or_broken(TMA_FILLS_A_STAGE_A_BOX_LANDS)?;
            tensor.bind_by_tma(
                launcher,
                width,
                landed.extent(values_dimensions[0]),
                landed.extent(KB) * landed.extent(KI),
                swizzle,
                scales,
            )
        };
        let lhs = match &planned_lhs {
            StoredTensor::Plain(lhs) => {
                self.tma_float_lhs(lhs, stage, storage, orders, widths.lhs)?
            }
            StoredTensor::Scaled(lhs) => lhs.under_scales(
                launcher,
                &self.blocks(),
                self.tma_float_lhs(&lhs.values, stage, storage, orders, widths.lhs)?,
            )?,
            StoredTensor::Packed(lhs) => packed(lhs, MatmulOperand::Lhs, M, widths.lhs)?,
        };
        let rhs = packed(rhs, MatmulOperand::Rhs, N, widths.rhs)?;
        Ok((lhs, rhs))
    }

    /// A float lhs beside a packed rhs, for stages the TMA engine fills: through a descriptor over
    /// the matrix its blocks split, `k` cut into rows of the swizzle's span so a box lands a
    /// stage's blocks of `K` side by side, served `width` wide.
    fn tma_float_lhs(
        &self,
        lhs: &PlainTensor,
        stage: &MatmulStage,
        storage: &StageStorage,
        orders: MatmulOperands<BindingOrder>,
        width: usize,
    ) -> Result<TileTensorLaunch, Decline> {
        let blocks = self.blocks();
        let batches = self.table_batches();
        let (lhs, dimensions) = lhs.restated(&blocks, M, K, &batches);
        let lhs_stage = stage.landed_space(MatmulOperand::Lhs, orders, &dimensions);
        let swizzle = storage
            .tma_swizzle(&lhs_stage, 1, lhs.elem.size_bits())
            .or_broken(TMA_FILLS_A_STAGE_A_BOX_LANDS)?;
        let (rows, k) = (
            lhs.binding.shape[0],
            lhs.binding.shape[1] * lhs.binding.shape[2],
        );
        let depth = lhs_stage.extent(KB) * lhs_stage.extent(KI);
        // A box row is the swizzle's span, or the stage's depth where it swizzles nothing.
        let row = swizzle_span_bytes(swizzle)
            .map_or(depth, |span| span / lhs.elem.size())
            .min(depth);
        let metadata = Metadata::new(Shape::new([rows, k / row, row]), Strides::new(&[k, row, 1]));
        let map = TensorMapArg {
            tensor: lhs.binding.clone().into_tensor_arg(),
            metadata: tma_meta_tiled(
                metadata,
                Shape::new([lhs_stage.extent(dimensions[0]), depth / row, row]),
                lhs.elem,
                swizzle,
            ),
            _kind: PhantomData,
        };
        // A box starts on a stage's first block of `K`, a whole number of rows in.
        let coordinates = TmaCoordinates {
            axes: vec![
                vec![TmaTerm::dimension(0)],
                vec![
                    TmaTerm::dimension(1)
                        .times(lhs_stage.extent(KI) as u32)
                        .over(row as u32),
                ],
                vec![],
            ],
        };
        let extents: Vec<u32> = lhs.binding.shape.iter().map(|&e| e as u32).collect();
        Ok(
            TileTensorLaunch::tensor_map_folded(map, &dimensions, &extents, coordinates)
                .served_wide(width),
        )
    }

    /// The slots the cubes sharing a box park their parts in, over the output's dimensions in the
    /// sum's element, two cube boxes a cube
    /// ([`Partitioning::arrival_slots`](cubecl::tile::Partitioning::arrival_slots)), beside one
    /// arrival counter a box, holding zero.
    ///
    /// It allocates the slots and the counters, and launches the zeroing of the counters.
    ///
    /// # Assumptions
    ///
    /// Float tensors are read for a launch tiling its rows: only such a launch shares its boxes'
    /// `K`. Packed ones merge their batches into the rows whatever the launch.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses to bind the slots the routine prepared.
    pub(crate) fn bind_last_arrival<Mk: MatmulMicrokernel, SM>(
        &self,
        blueprint: &MatmulBlueprint<Mk, SM>,
        launcher: &Launcher,
    ) -> Result<MatmulLastArrivalBinding, Decline> {
        let partitioning = blueprint.partitioning.as_tile();
        // The output's dimensions in its order: the batches the cube level hands out, then its rows
        // and columns, against packed inputs as the block-split space restates them
        // ([`bind_packed`](Self::bind_packed)). Stated as labels, so a batch dimension one box long
        // stays a dim of the slots.
        let output: Vec<Dimension> = match self.read(|view| view.blocks()) {
            None => vec![M, N],
            Some(blocks) => blocks
                .restated_dims(M, N, 1)
                .into_iter()
                .map(|(dimension, _, _)| dimension)
                .collect(),
        };
        let dimensions: Vec<Dimension> = blueprint
            .partitioning
            .batch_dimensions()
            .iter()
            .copied()
            .chain(output)
            .collect();
        let sum = self.problem().sum();
        let (_, slots) =
            PlainTensor::empty(&self.client, partitioning.arrival_slots(&dimensions), sum);
        Ok(MatmulLastArrivalBinding {
            slots: launcher
                .arg(slots.binding, sum)
                .dimensions(&dimensions)
                .vectorize(blueprint.widths.out)
                .build()
                .or_broken(ADMITTED_PLAN_RUNS)?,
            arrivals: ZeroedCounters::new(partitioning.arrival_counters(), &self.client).into_arg(),
        })
    }

    /// The launcher `blueprint`'s partitioning launches on `grid`, and the lhs and the rhs bound
    /// under it, each reading the table where it is one of its readers.
    fn bind_operands<Mk: MatmulMicrokernel, SM>(
        &self,
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        blueprint: &MatmulBlueprint<Mk, SM>,
        grid: &Grid,
    ) -> Result<(Launcher, [TileTensorLaunch; 2]), Decline> {
        let partitioning = blueprint.partitioning.as_tile();
        let launcher = grid.launcher(self.client(), || {
            (
                blueprint.partitioning.kernel(),
                partitioning.space().clone(),
            )
        })?;
        let (batch_dimensions, widths) = (self.batch_dimensions(), blueprint.widths);
        let (lhs_batches, _) = self.frame(MatmulOperand::Lhs, lhs, self.batch_dimensions());
        let delivery = blueprint.microkernel.delivery();
        let decoded = blueprint.microkernel.decoded_stage();
        let source = |tensor: &PlainTensor, dimensions: &'static [Dimension]| {
            Self::float_arg(&launcher, tensor, dimensions, decoded).delivery(delivery)
        };
        let orders = blueprint.microkernel.orders();
        // A stack is labelled, not batched: its slabs are the dimension an entry picks along.
        let stack_dimensions = self
            .table
            .as_ref()
            .and_then(|table| table.index().dimension())
            .map(|(stacked, _)| [stacked, K, N]);
        let rhs_source = match &stack_dimensions {
            Some(dimensions) => {
                Self::float_arg(&launcher, rhs, dimensions, decoded).delivery(delivery)
            }
            None => source(rhs, &[K, N]).batches(&batch_dimensions),
        };
        let lhs = orders
            .lhs
            .applied(source(lhs, &[M, K]).batches(&lhs_batches))
            .vectorize(widths.lhs)
            .build()?;
        let rhs = orders
            .rhs
            .applied(rhs_source)
            .vectorize(widths.rhs)
            .build()?;
        let lhs = self.read_through_table(MatmulOperand::Lhs, lhs.arg())?;
        let rhs = self.read_through_table(MatmulOperand::Rhs, rhs.arg())?;
        Ok((launcher, [lhs, rhs]))
    }

    /// A float input's argument under `launcher`: at its own element, or, where the stages it is
    /// read through hold `decoded`, another element than it is stored in, as its `u32` words, each
    /// one value of its float, which the fill decodes.
    fn float_arg<'a>(
        launcher: &'a Launcher,
        tensor: &PlainTensor,
        dimensions: &'a [Dimension],
        decoded: Option<ElemType>,
    ) -> Arg<'a, Labelled> {
        match (tensor.elem, decoded) {
            (ElemType::Float(kind), Some(decoded)) if decoded != tensor.elem => launcher
                .arg(tensor.binding.clone(), ElemType::UInt(UIntKind::U32))
                .dimensions(dimensions)
                .packed(kind),
            _ => launcher
                .arg(tensor.binding.clone(), tensor.elem)
                .dimensions(dimensions),
        }
    }

    /// `operand`, bound, reading its entry of the table where the index has it read one, the
    /// table's dims the batches; as bound otherwise.
    ///
    /// # Errors
    ///
    /// Why the tile engine refuses to place `operand` by its table.
    fn read_through_table(
        &self,
        operand: MatmulOperand,
        bound: TileTensorLaunch,
    ) -> Result<TileTensorLaunch, Decline> {
        let (Some(table), Some(indexing)) = (
            &self.table,
            self.problem().placements().get_ref(operand).clone(),
        ) else {
            return Ok(bound);
        };
        let buffer = table.read_by(&indexing);
        let entries = TileTensorLaunch::new(
            buffer.binding.clone().into_tensor_arg(),
            buffer.elem,
            1,
            TileSpec::direct(&self.table_batches()),
        );
        bound.indexed(entries, indexing)
    }

    /// The batch dimensions `operand`, bound from `tensor`, is over, and the rows its blocks hold
    /// where they are not the product's. The lhs and output of grouped rows are flat, one row group a
    /// batch of the table, so they have no batch axis (`batches` for any other operand) and their
    /// blocks hold every row of `tensor`, since a row group may start in any of them.
    fn frame(
        &self,
        operand: MatmulOperand,
        tensor: &PlainTensor,
        batches: Vec<Dimension>,
    ) -> (Vec<Dimension>, Option<usize>) {
        let flat = self
            .table
            .as_ref()
            .is_some_and(|table| table.groups().is_some())
            && operand != MatmulOperand::Rhs;
        match flat {
            true => (Vec::new(), Some(tensor.shape()[0])),
            false => (batches, None),
        }
    }

    /// `tensor`'s tensor map, a row-major matrix of `rows × columns` (each a dimension and its
    /// extent), its descriptor moving `landing`'s box of it.
    fn tma_matrix(
        tensor: &PlainTensor,
        [(rows, row_count), (columns, column_count)]: [(Dimension, usize); 2],
        landing: TmaLanding,
    ) -> TileTensorLaunch {
        let matrices = 1;
        let (map, transposed) = tma_operand(
            tensor.binding.clone(),
            matrices,
            MatrixLayout::RowMajor,
            (landing.stage.extent(rows), landing.stage.extent(columns)),
            tensor.elem,
            landing.swizzle,
        );
        TileTensorLaunch::tensor_map(
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

/// What a launch whose cubes share its boxes' `K` binds beside its operands: the extras of the
/// last-arrival routine ([`HypercubeLastArrivalMatmulRoutine`](crate::kernels::HypercubeLastArrivalMatmulRoutine)).
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

impl<Out> MatmulTensors<Out> {
    /// [`is_read_for`](Self::is_read_for).
    fn holds_rows_as(&self, rows_kind: MatmulRowsKind) -> bool {
        let packed = !matches!(
            (&self.lhs, &self.rhs),
            (StoredTensor::Plain(_), StoredTensor::Plain(_))
        );
        packed || self.rows_kind == rows_kind
    }
}

impl Operands {
    /// `lhs` and `rhs` read as a launch holding rows as `rows_kind` says reads them.
    ///
    /// # Errors
    ///
    /// As [`MatmulTensors::new`].
    fn read(
        client: &Client,
        lhs: PlainTensor,
        rhs: PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Self, Decline> {
        match rows_kind {
            MatmulRowsKind::Tiled => Self::tiled(client, lhs, rhs),
            MatmulRowsKind::Whole => Self::whole(client, lhs, rhs),
        }
    }

    /// Both operands where they lie unless they alias themselves, the output their product's
    /// shape.
    fn tiled(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Decline> {
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

    /// The lhs merged into `[m, k]` rows, the rhs its one `[k, n]` matrix, the output
    /// allocated as `[.., n]` and bound as `[m, n]` rows.
    fn whole(client: &Client, lhs: PlainTensor, rhs: PlainTensor) -> Result<Self, Decline> {
        let (lhs, rhs) = MatmulInputs::new(client, lhs, rhs)?.into_parts();
        let (weight, out_shape) = Self::whole_weight(&lhs, &rhs)?;
        let lhs = Self::with_mergeable_rows(client, lhs);
        let k = weight.shape()[0];
        let m = lhs.shape().num_elements() / k.max(1);
        Ok(Self {
            lhs: lhs.as_rows(m, k),
            rhs: weight,
            out_shape,
            rows_kind: MatmulRowsKind::Whole,
        })
    }

    /// The one `[k, n]` matrix `rhs` holds, and the shape of `lhs` against it, `[.., n]`, read off
    /// their shapes and strides, before any copy.
    ///
    /// # Errors
    ///
    /// [`MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH`], [`MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED`] and
    /// [`MATMUL_CONTRACTIONS_DISAGREE`].
    fn whole_weight(
        lhs: &PlainTensor,
        rhs: &PlainTensor,
    ) -> Result<(PlainTensor, Vec<usize>), Decline> {
        let rhs_shape = rhs.shape();
        if rhs_shape.as_slice()[..rhs_shape.rank().saturating_sub(2)]
            .iter()
            .any(|&dim| dim > 1)
        {
            return Err(MATMUL_WHOLE_ROWS_WEIGHT_PER_BATCH);
        }
        let weight =
            Self::one_weight_matrix(rhs).ok_or(MATMUL_WHOLE_ROWS_RHS_LAYOUT_UNSUPPORTED)?;
        let weight_shape = weight.shape();
        let [weight_k, n] = [weight_shape[0], weight_shape[1]];
        let mut out_shape = lhs.shape().to_vec();
        match out_shape.last_mut() {
            Some(k) if *k == weight_k => *k = n,
            _ => return Err(MATMUL_CONTRACTIONS_DISAGREE),
        }
        Ok((weight, out_shape))
    }

    /// The shape of the output of `lhs` against `rhs`, as a launch holding its rows as `rows_kind`
    /// says reads them, read off their shapes before any copy.
    ///
    /// # Errors
    ///
    /// As [`read`](Self::read), of their shapes.
    fn out_shape(
        lhs: &PlainTensor,
        rhs: &PlainTensor,
        rows_kind: MatmulRowsKind,
    ) -> Result<Vec<usize>, Decline> {
        match rows_kind {
            MatmulRowsKind::Tiled => Self::product_shape(lhs, rhs),
            MatmulRowsKind::Whole => Self::whole_weight(lhs, rhs).map(|(_, shape)| shape),
        }
    }

    /// The rows and the line the output is bound as, where the launch reads every row whole: the
    /// lhs's rows, the rhs's columns.
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

    /// `rhs` as the one `[k, n]` matrix it holds, in rows or in tiles, contiguous along the
    /// dimension the tuning key reads ([`contiguous`]), so a key and its tensors state one layout;
    /// `None` where it holds several matrices or is contiguous along neither dimension.
    fn one_weight_matrix(rhs: &PlainTensor) -> Option<PlainTensor> {
        rhs.as_one_matrix()
            .filter(|matrix| contiguous(&matrix.meta(), [K, N]).is_some())
    }

    /// `lhs` where its leading dims merge into rows of its last at one step, a row-major copy of it
    /// where they do not: an lhs the launch reads as rows either way.
    fn with_mergeable_rows(client: &Client, lhs: PlainTensor) -> PlainTensor {
        match Self::merges_into_rows(&lhs) {
            true => lhs,
            false => lhs.into_contiguous(client),
        }
    }

    /// `tensor` with every leading axis of extent 1 stepping by the axes inside it, outward from
    /// its rows: the same elements, its leading dims then merging wherever the others do.
    fn with_lone_axes_restrided(tensor: &PlainTensor) -> PlainTensor {
        let shape = tensor.binding.shape.to_vec();
        let mut strides = tensor.binding.strides.to_vec();
        for dim in (0..shape.len().saturating_sub(2)).rev() {
            if shape[dim] == 1 {
                strides[dim] = shape[dim + 1] * strides[dim + 1];
            }
        }
        tensor.viewed(Metadata::new(shape, strides))
    }

    /// Whether `tensor`'s leading dims merge into rows of its last at one step, its last one
    /// element a step, untiled: what a launch reading every row whole binds as rows.
    fn merges_into_rows(tensor: &PlainTensor) -> bool {
        let (shape, strides) = (tensor.shape(), &tensor.binding.strides);
        let rank = shape.rank();
        rank >= 1
            && strides[rank - 1] == 1
            && (0..rank.saturating_sub(2))
                .all(|dim| strides[dim] == shape[dim + 1] * strides[dim + 1])
            && !tensor.is_tiled()
    }

    /// The shape of `lhs · rhs`: the broadcast batch dims, then `m × n`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_TILED_ROWS_RANKS_DIFFER`], [`MATMUL_CONTRACTIONS_DISAGREE`] for operands of a rank
    /// under two or whose contraction dims differ, and [`MATMUL_BATCHES_DO_NOT_BROADCAST`].
    fn product_shape(lhs: &PlainTensor, rhs: &PlainTensor) -> Result<Vec<usize>, Decline> {
        let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
        let rank = lhs_shape.num_dims();
        if rank != rhs_shape.num_dims() {
            return Err(MATMUL_TILED_ROWS_RANKS_DIFFER);
        }
        if rank < 2 || lhs_shape[rank - 1] != rhs_shape[rank - 2] {
            return Err(MATMUL_CONTRACTIONS_DISAGREE);
        }
        let broadcast = |dim: usize| {
            let (l, r) = (lhs_shape[dim], rhs_shape[dim]);
            (l == r || l == 1 || r == 1).then_some(l.max(r))
        };
        let batches: Option<Vec<usize>> = (0..rank - 2).map(broadcast).collect();
        let batches = batches.ok_or(MATMUL_BATCHES_DO_NOT_BROADCAST)?;
        Ok(batches
            .into_iter()
            .chain([lhs_shape[rank - 2], rhs_shape[rank - 1]])
            .collect())
    }
}

/// Where a descriptor lands its box: the stage of the operand, its dimensions in the order the
/// operand is bound in, and the swizzle that lays the box's rows as the stage keeps them.
struct TmaLanding {
    stage: Space,
    swizzle: TensorMapSwizzle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubecl::tile::layout::ElemAlignment;
    use cubek_std::layout::stored::Operand;
    use cubek_test_utils::client;

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

    /// A row rhs with a pitch along `K` has one alignment whichever order binds it: its buffer
    /// steps in the order the product names it. That holds for one matrix, a pitched one and every
    /// slab view of a stack, so the shuffle microkernels, which bind a row rhs as stated, read it
    /// at the widths its buffer gives.
    #[test]
    fn a_row_rhs_aligns_alike_in_either_order() {
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

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const U32: ElemType = ElemType::UInt(UIntKind::U32);

    /// Zeros of `shape` and `element`.
    fn zeros(shape: &[usize], element: ElemType) -> PlainTensor {
        PlainTensor::of_handle(TensorHandle::zeros(&client(), shape.to_vec(), element))
    }

    /// What two batches of `lhs` against `rhs` decline, each batch reading its entry of a table of
    /// `entries` that `table` makes, into an output of two batches of one row of sixteen.
    fn declined_indexed(
        lhs: &[usize],
        rhs: &[usize],
        table: fn(PlainTensor) -> MatmulIndexTable,
        entries: ElemType,
    ) -> Option<Decline> {
        let table = table(zeros(&[2], entries));
        MatmulTensors::indexed(
            &client(),
            zeros(lhs, F32),
            StoredTensor::Plain(zeros(rhs, F32)),
            table,
            zeros(&[2, 1, 16], F32),
            MatmulRowsKind::Whole,
        )
        .err()
    }

    /// A table naming slabs of a stack of four.
    fn four_slabs(entries: PlainTensor) -> MatmulIndexTable {
        MatmulIndexTable::Slab { slabs: 4, entries }
    }

    #[test]
    fn a_float_table_declines() {
        let declined = declined_indexed(&[2, 1, 8], &[4, 8, 16], four_slabs, F32);

        assert_eq!(declined, Some(MATMUL_TABLE_NOT_U32));
    }

    #[test]
    fn a_stack_of_other_slabs_than_its_table_declines() {
        let declined = declined_indexed(&[2, 1, 8], &[3, 8, 16], four_slabs, U32);

        assert_eq!(declined, Some(MATMUL_STACK_NOT_THE_TABLES));
    }

    #[test]
    fn runs_of_k_over_a_stack_decline() {
        let declined = declined_indexed(&[2, 1, 8], &[4, 8, 16], |ends| MatmulIndexTable::RunOfK { ends }, U32);

        assert_eq!(declined, Some(MATMUL_STACK_NOT_THE_TABLES));
    }

    #[test]
    fn an_indexed_lhs_of_another_contraction_declines() {
        let declined = declined_indexed(&[2, 1, 6], &[4, 8, 16], four_slabs, U32);

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    #[test]
    fn an_indexed_lhs_of_no_rows_declines() {
        let declined = declined_indexed(&[8], &[4, 8, 16], four_slabs, U32);

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    /// A packed `[k, n]` rhs of `k = 64` and `n = 16`, in rows of NVFP4 words along `k`, read as a
    /// stack of two slabs along `n`.
    fn packed_stack_of_two() -> PackedTensor {
        let scheme = crate::kernels::select::packed::nvfp4_scheme();
        let (n, k) = (16, 64);
        let words = zeros(&[n, k / scheme.num_quants()], U32);
        // A scale byte a block of sixteen values, four a word.
        let block_scales = zeros(&[n, k / 16 / 4], U32);
        let operand = Operand::of_scheme(&scheme, K, [N, K]).expect("one block along k");
        PackedTensor::new(
            words,
            block_scales,
            Some(zeros(&[1, 1], U32)),
            scheme,
            operand,
            [(N, n), (K, k)],
        )
        .stacked(2, N)
        .expect("two slabs split no read or block")
    }

    #[test]
    fn a_packed_stack_of_other_slabs_than_its_table_declines() {
        let table = four_slabs(zeros(&[2], U32));

        let declined = MatmulTensors::indexed(
            &client(),
            zeros(&[2, 1, 64], F32),
            StoredTensor::Packed(packed_stack_of_two()),
            table,
            zeros(&[2, 1, 8], F32),
            MatmulRowsKind::Whole,
        )
        .err();

        assert_eq!(declined, Some(MATMUL_STACK_NOT_THE_TABLES));
    }

    #[test]
    fn an_operand_of_another_contraction_beside_a_packed_one_declines() {
        let declined =
            MatmulTensors::beside_packed(&client(), zeros(&[2, 6], F32), [M, K], 8).err();

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    #[test]
    fn an_operand_of_no_rows_beside_a_packed_one_declines() {
        let declined = MatmulTensors::beside_packed(&client(), zeros(&[8], F32), [M, K], 8).err();

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    /// What a product of `lhs` and `rhs`, both of `f32`, declines, holding its rows as
    /// `rows_kind` says.
    fn declined_product(
        lhs: &[usize],
        rhs: &[usize],
        rows_kind: MatmulRowsKind,
    ) -> Option<Decline> {
        MatmulTensors::new(
            &client(),
            StoredTensor::Plain(zeros(lhs, F32)),
            StoredTensor::Plain(zeros(rhs, F32)),
            rows_kind,
        )
        .err()
    }

    #[test]
    fn tiled_operands_of_another_contraction_decline() {
        let declined = declined_product(&[4, 8], &[6, 16], MatmulRowsKind::Tiled);

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    #[test]
    fn tiled_operands_of_no_rows_decline() {
        let declined = declined_product(&[8], &[8], MatmulRowsKind::Tiled);

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }

    #[test]
    fn tiled_operands_whose_batches_do_not_broadcast_decline() {
        let declined = declined_product(&[2, 4, 8], &[3, 8, 16], MatmulRowsKind::Tiled);

        assert_eq!(declined, Some(MATMUL_BATCHES_DO_NOT_BROADCAST));
    }

    #[test]
    fn an_output_of_rows_that_do_not_merge_declines_for_every_row_whole() {
        let mut out = zeros(&[2, 4, 16], F32);
        out = out.viewed(Metadata::new(vec![2, 4, 16], vec![128, 16, 1]));

        let declined = MatmulTensors::writing_into(
            &client(),
            StoredTensor::Plain(zeros(&[2, 4, 8], F32)),
            StoredTensor::Plain(zeros(&[8, 16], F32)),
            out,
            MatmulRowsKind::Whole,
        )
        .err();

        assert_eq!(declined, Some(MATMUL_OUTPUT_UNREAD));
    }

    /// What a product of `[.., 4, 8]` rows against an `[8, 16]` weight, holding every row whole,
    /// declines of an `out` of `shape` at `strides`.
    fn declined_output(shape: &[usize], strides: &[usize]) -> Option<Decline> {
        let lhs = [&shape[..shape.len() - 1], &[8]].concat();
        let out = zeros(shape, F32).viewed(Metadata::new(shape.to_vec(), strides.to_vec()));
        MatmulTensors::writing_into(
            &client(),
            StoredTensor::Plain(zeros(&lhs, F32)),
            StoredTensor::Plain(zeros(&[8, 16], F32)),
            out,
            MatmulRowsKind::Whole,
        )
        .err()
    }

    #[test]
    fn a_dense_output_whose_lone_batch_carries_any_stride_is_served() {
        assert_eq!(declined_output(&[1, 4, 16], &[16, 16, 1]), None);
    }

    #[test]
    fn an_output_whose_batches_overlap_declines() {
        assert_eq!(
            declined_output(&[2, 1, 16], &[16, 1, 1]),
            Some(MATMUL_OUTPUT_UNREAD)
        );
    }

    #[test]
    fn whole_rows_of_another_contraction_decline() {
        let declined = declined_product(&[4, 8], &[6, 16], MatmulRowsKind::Whole);

        assert_eq!(declined, Some(MATMUL_CONTRACTIONS_DISAGREE));
    }
}
