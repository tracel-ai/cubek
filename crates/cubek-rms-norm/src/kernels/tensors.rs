//! How an RMS-norm launch reads its operands and binds them.

use crate::kernels::{RmsNormBlueprint, RmsNormRoutineProblem};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{
    Axis, Geometry, Launcher,
    launch::{Bound, Unbound},
};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{
    ELEMENT_NOT_ON_DEVICE, PackedRows, PackedTensor, Placement, PlainTensor, TiledRows,
};
use cubek_microkernel::rms_norm::{B, D, R, RmsNormOperands};
use cubek_microkernel::row_store::RowStore;
use cubek_std::Hardware;
use cubek_std::routine::ADMITTED_PLAN_RUNS;
use cubek_std::routine::Grid;

/// What a launch of this kind declines before preparing anything: rows that are not floats.
pub const RMS_NORM_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel RMS norm normalizes float rows",
};

/// The tensors of one norm, its output among them, and the problem they state.
///
/// Every operand is read as dense rows starting on a whole read of the device: the rows as
/// `[b, r, d]` over the input, read where they lie wherever they come in batch entries
/// ([`PlainTensor::as_batched_rows`]), and copied into dense ones ([`PlainTensor::into_dense_at`])
/// otherwise; the weight as `[d]`, whatever leading dims of one it carries, or as `[r, d]`, a row
/// for each row of an entry; the output allocated dense in the input's shape and written through
/// the same `[b, r, d]` view. So every operand
/// binds at the one width a dense row does ([`RmsNormWorkItem::width`]), which is the width a
/// tuning key prepares its candidates at.
///
/// The output is `Out`: a [`RmsNormOutputBuffer`] the launch allocates, a [`Geometry`] the
/// caller binds — a fused tail's block, laid out as the dense rows the buffer would hold — or a
/// [`PackedRows`] or a [`TiledRows`] the caller allocates, which the launch encodes its rows into
/// or writes them into the tiles of.
///
/// [`RmsNormWorkItem::width`]: cubek_microkernel::rms_norm::RmsNormWorkItem::width
#[derive(Clone)]
pub struct RmsNormTensors<Out = RmsNormOutputBuffer> {
    client: Client,
    input: PlainTensor,
    weight: RmsNormWeight,
    out: Out,
    problem: RmsNormRoutineProblem,
}

impl<Out: core::fmt::Debug> core::fmt::Debug for RmsNormTensors<Out> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RmsNormTensors")
            .field("input", &self.input)
            .field("weight", &self.weight)
            .field("out", &self.out)
            .field("problem", &self.problem)
            .finish_non_exhaustive()
    }
}

/// The output a launch allocates: dense in the input's shape, and the `[b, r, d]` view of it the
/// kernel writes through.
#[derive(Debug, Clone)]
pub struct RmsNormOutputBuffer {
    handle: TensorHandle,
    rows: PlainTensor,
}

/// What a packed or tiled launch declines before preparing anything: rows in batch entries,
/// which a matrix of rows does not lie in.
pub const RMS_NORM_ROWS_IN_ENTRIES: Refusal = Refusal::Declined {
    why: "a packed or tiled RMS norm writes dense rows, one batch entry of them",
};

/// The rows and the weight as every launch reads them, before the output is decided.
struct RmsNormInputs {
    client: Client,
    input: PlainTensor,
    weight: RmsNormWeight,
    /// The input's shape, which the output takes.
    shape: Vec<usize>,
    batch: usize,
    rows: usize,
    dim: usize,
}

impl RmsNormInputs {
    /// The rows of `input`, in batch entries where they lie, and `weight`, read as dense rows
    /// starting on a whole read. Under a weight a row ([`RmsNormWeight::PerRow`]) each batch
    /// entry holds the weight's rows: rows that lie in other entries are made dense and taken as
    /// many at a time.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`].
    fn new(client: &Client, input: PlainTensor, weight: PlainTensor) -> Result<Self, Refusal> {
        assert_eq!(input.elem, weight.elem, "the weight is the rows' element");
        if !matches!(input.elem, ElemType::Float(_)) {
            return Err(RMS_NORM_ELEMENT_UNVERIFIED);
        }
        if !input.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let read_bytes = Hardware::from_client(client).properties.io_width() as u64 / 8;
        let shape = input.binding.shape.to_vec();
        let dim = *shape.last().expect("a tensor has a last dimension");
        let starts_on_a_read = input.start_bytes().is_multiple_of(read_bytes.max(1));
        let input = match input.as_batched_rows(dim) {
            Some(rows) if starts_on_a_read => rows,
            _ => input
                .into_dense_at(client, read_bytes)
                .as_batched_rows(dim)
                .expect("dense rows are one batch entry"),
        };
        let weight = RmsNormWeight::new(weight.into_dense_at(client, read_bytes), dim);
        let input = match weight.rows() {
            Some(each) if input.binding.shape[1] != each => {
                let dense = input.into_dense_at(client, read_bytes);
                let rows = dense.binding.shape.iter().product::<usize>() / dim;
                assert!(
                    rows.is_multiple_of(each),
                    "a weight of {each} rows scales whole groups of rows, got {rows} rows"
                );
                dense.viewed(Metadata::new(
                    [rows / each, each, dim],
                    [each * dim, dim, 1],
                ))
            }
            _ => input,
        };
        let [batch, rows, _] = input.binding.shape[..] else {
            unreachable!("batched rows have three dims")
        };
        Ok(Self {
            client: client.clone(),
            input,
            weight,
            shape,
            batch,
            rows,
            dim,
        })
    }

    /// These operands, the rows in one batch entry, as a matrix of rows lies.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_ROWS_IN_ENTRIES`].
    fn one_entry(self) -> Result<Self, Refusal> {
        match self.batch {
            1 => Ok(self),
            _ => Err(RMS_NORM_ROWS_IN_ENTRIES),
        }
    }

    /// The tensors of the norm of these operands into `out`, which lies at `placement` and stores
    /// its rows as `store` says.
    fn with_out<Out>(self, out: Out, placement: Placement, store: RowStore) -> RmsNormTensors<Out> {
        let problem = self.problem(placement, store);
        RmsNormTensors {
            client: self.client,
            input: self.input,
            weight: self.weight,
            out,
            problem,
        }
    }

    /// The problem these operands state, the output placed at `out` and its rows stored as
    /// `store` says.
    fn problem(&self, out: Placement, store: RowStore) -> RmsNormRoutineProblem {
        let (input, weight) = (&self.input, self.weight.tensor());
        RmsNormRoutineProblem {
            batch: self.batch,
            rows: self.rows,
            dim: self.dim,
            element: self.input.elem,
            elem_alignments: RmsNormOperands {
                input: input.elem_alignment(),
                weight: weight.elem_alignment(),
                out: out.elem_alignment,
            },
            start_bytes: RmsNormOperands {
                input: input.start_bytes(),
                weight: weight.start_bytes(),
                out: out.start_bytes,
            },
            store,
        }
    }
}

impl RmsNormTensors {
    /// The norm of `input`'s last dim, scaled by `weight`, its output allocated.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_ELEMENT_UNVERIFIED`], what the routines decline whatever the device, and
    /// [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does not run.
    ///
    /// # Panics
    ///
    /// Operands on different devices or of different elements, a weight that is not whole rows of
    /// the normalized dim, or a weight a row whose rows do not divide the input's: no norm at all.
    pub fn new(client: &Client, input: PlainTensor, weight: PlainTensor) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(client, input, weight)?;
        let (handle, tensor) = PlainTensor::empty_contiguous(
            client,
            operands.shape.clone().into(),
            operands.input.elem,
        );
        let (batch, rows, dim) = (operands.batch, operands.rows, operands.dim);
        let rows = tensor.viewed(Metadata::new([batch, rows, dim], [rows * dim, dim, 1]));
        let placement = Placement::from(&rows);
        Ok(operands.with_out(
            RmsNormOutputBuffer { handle, rows },
            placement,
            RowStore::Plain,
        ))
    }

    /// The output, in the input's shape.
    pub fn into_out(self) -> TensorHandle {
        self.out.handle
    }

    /// The three operands bound as `blueprint` states them, on `grid`: under its partitioning, at
    /// its width.
    pub fn bind<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<Bound> {
        let (launcher, [input, weight]) = self.bind_operands(blueprint, grid);
        let out = launcher
            .arg(self.out.rows.binding.clone(), self.problem.element)
            .axes(&[B, R, D])
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        RmsNormBinding {
            launcher,
            input,
            weight,
            out,
        }
    }
}

impl RmsNormTensors<Geometry> {
    /// The norm of `input`'s last dim, scaled by `weight`, handed to a sink the caller binds:
    /// nothing is allocated, and the output is laid out as the dense `[b, r, d]` rows a buffer
    /// would hold.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`].
    ///
    /// # Panics
    ///
    /// As [`RmsNormTensors::new`].
    pub fn sink(client: &Client, input: PlainTensor, weight: PlainTensor) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(client, input, weight)?;
        let out = Geometry::row_major(&[operands.batch, operands.rows, operands.dim]);
        let placement = Placement::dense_rows(operands.dim);
        Ok(operands.with_out(out, placement, RowStore::Plain))
    }

    /// The three operands bound as `blueprint` states them, on `grid`, the output derived over
    /// the sink's layout rather than a buffer: its spec, and the geometry the derivation settled,
    /// which the caller's sink addresses through.
    pub fn bind<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<Unbound> {
        let (launcher, [input, weight]) = self.bind_operands(blueprint, grid);
        let out = launcher
            .unbound(&self.out)
            .axes(&[B, R, D])
            .vectorize(blueprint.width)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        RmsNormBinding {
            launcher,
            input,
            weight,
            out,
        }
    }

    /// The sink the output is laid out as.
    pub fn out(&self) -> &Geometry {
        &self.out
    }
}

impl RmsNormTensors<PackedRows> {
    /// The norm of `input`'s last dim, scaled by `weight`, encoded into `out`, a packed
    /// `[rows, dim]` matrix over a product's lhs axes the caller allocated under one level of block
    /// scales along each row. The rows are staged in each cube's shared memory, dense, and encoded
    /// from there.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`]; [`RMS_NORM_ROWS_IN_ENTRIES`]; and what
    /// [`PackedRows::new`] declines of `out`.
    ///
    /// # Panics
    ///
    /// As [`RmsNormTensors::new`], and on an `out` not shaped `[rows, dim]`.
    pub fn packed(
        client: &Client,
        input: PlainTensor,
        weight: PlainTensor,
        out: PackedTensor,
    ) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(client, input, weight)?.one_entry()?;
        let out = PackedRows::new(out, operands.rows, operands.dim)?;
        let (placement, store) = (Placement::dense_rows(operands.dim), out.store());
        Ok(operands.with_out(out, placement, store))
    }

    /// The rows and the weight bound as `blueprint` states them, on `grid`, and the output as one
    /// tile of its words under their scales ([`PackedRows::bind`]).
    pub fn bind<M>(
        &self,
        blueprint: &RmsNormBlueprint<M>,
        grid: &Grid,
    ) -> RmsNormBinding<TileTensorLaunch> {
        let (launcher, [input, weight]) = self.bind_operands(blueprint, grid);
        let out = self.out.bind(&launcher, blueprint.width);
        RmsNormBinding {
            launcher,
            input,
            weight,
            out,
        }
    }
}

impl RmsNormTensors<TiledRows> {
    /// The norm of `input`'s last dim, scaled by `weight`, written into `out`, a float
    /// `[rows, dim]` matrix stored in tiles the caller allocated: its rows land in its tiles as
    /// they are written, through the binding its tiling states.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`]; [`RMS_NORM_ROWS_IN_ENTRIES`]; and what [`TiledRows::new`]
    /// declines of `out`.
    ///
    /// # Panics
    ///
    /// As [`RmsNormTensors::new`].
    pub fn tiled(
        client: &Client,
        input: PlainTensor,
        weight: PlainTensor,
        out: PlainTensor,
    ) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(client, input, weight)?.one_entry()?;
        let element = operands.input.elem;
        let out = TiledRows::new(out, operands.rows, operands.dim, element)?;
        let placement = Placement::from(&out);
        Ok(operands.with_out(out, placement, RowStore::Plain))
    }

    /// The rows and the weight bound as `blueprint` states them, on `grid`, and the output over
    /// the rows and the dim, through the tiles its binding states.
    pub fn bind<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<Bound> {
        let (launcher, [input, weight]) = self.bind_operands(blueprint, grid);
        let out = self.out.bind(&launcher, blueprint.width);
        RmsNormBinding {
            launcher,
            input,
            weight,
            out,
        }
    }
}

impl<Out> RmsNormTensors<Out> {
    /// The client the operands live on, which a launch submits to.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &RmsNormRoutineProblem {
        &self.problem
    }

    /// The launcher of `blueprint`'s partitioning on `grid`, and the rows and the weight bound
    /// under it at its width: what every output binds beside.
    fn bind_operands<M>(
        &self,
        blueprint: &RmsNormBlueprint<M>,
        grid: &Grid,
    ) -> (Launcher, [Bound; 2]) {
        let partitioning = &blueprint.partitioning;
        // The grid's launcher, settled over the extents on its first launch and cloned after.
        let launcher = grid.launcher(self.client(), || {
            (
                partitioning.kernel(),
                partitioning.as_tile().space().clone(),
            )
        });
        // Each operand as the store states the rows: as it lies, or its row split into blocks.
        let bind = |tensor: &PlainTensor, leading: &[Axis]| {
            let (meta, axes) = self.problem.store.restated(&tensor.meta(), leading);
            launcher
                .arg(tensor.bound_as(meta), self.problem.element)
                .axes(&axes)
                .vectorize(blueprint.width)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        };
        let operands = [
            bind(&self.input, &[B, R]),
            bind(self.weight.tensor(), self.weight.leading_axes()),
        ];
        (launcher, operands)
    }
}

/// What the normalized rows are scaled by: one row of `d` every row shares, or a row of `d` for
/// each row of a batch entry, `[r, d]`, every entry sharing them, as the query and key heads of a
/// decode step share their norms' weights.
#[derive(Debug, Clone)]
enum RmsNormWeight {
    Shared(PlainTensor),
    PerRow(PlainTensor),
}

impl RmsNormWeight {
    /// `weight`, dense, over the same buffer: `d` elements whatever leading dims of one they
    /// carry, or whole rows of `d`, one for each row of a batch entry.
    ///
    /// # Panics
    ///
    /// A weight that is not whole rows of `dim`.
    fn new(weight: PlainTensor, dim: usize) -> Self {
        let elements = weight.binding.shape.iter().product::<usize>();
        assert!(
            elements > 0 && elements.is_multiple_of(dim),
            "the weight spans the normalized dim, once or once a row"
        );
        let rows = elements / dim;
        match rows {
            1 => Self::Shared(weight.viewed(Metadata::new([dim], [1]))),
            rows => Self::PerRow(weight.viewed(Metadata::new([rows, dim], [dim, 1]))),
        }
    }

    /// The rows a batch entry holds under a weight a row.
    fn rows(&self) -> Option<usize> {
        match self {
            Self::Shared(_) => None,
            Self::PerRow(weight) => Some(weight.binding.shape[0]),
        }
    }

    fn tensor(&self) -> &PlainTensor {
        match self {
            Self::Shared(weight) | Self::PerRow(weight) => weight,
        }
    }

    /// The axes the weight runs along before the dim: none, or each row's.
    fn leading_axes(&self) -> &'static [Axis] {
        match self {
            Self::Shared(_) => &[],
            Self::PerRow(_) => &[R],
        }
    }
}

/// The operands of a launch bound under one launcher, the output as `Out`: bound to a buffer, or
/// derived for a sink. Not `Debug`: cubecl's `Launcher` is not.
pub struct RmsNormBinding<Out> {
    pub launcher: Launcher,
    pub input: Bound,
    pub weight: Bound,
    pub out: Out,
}

impl<Out> RmsNormBinding<Out> {
    /// The same binding, its output turned into `T` — what a destination's launch takes.
    pub fn map_out<T>(self, f: impl FnOnce(Out) -> T) -> RmsNormBinding<T> {
        RmsNormBinding {
            launcher: self.launcher,
            input: self.input,
            weight: self.weight,
            out: f(self.out),
        }
    }
}
