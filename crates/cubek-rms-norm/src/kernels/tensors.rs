//! How an RMS-norm launch reads its operands and binds them.

use crate::kernels::{
    RMS_NORM_OUTPUT_UNSUPPORTED, RmsNormBlueprint, RmsNormCall, RmsNormRoutineProblem,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{
    Axis, Geometry, Launcher,
    launch::{Bound, Unbound},
};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{PackedRows, PackedTensor, Placement, PlainTensor, TiledRows};
use cubek_microkernel::rms_norm::{B, D, GateOrder, R, RmsNormOperands};
use cubek_microkernel::row_store::RowStore;
use cubek_std::Hardware;
use cubek_std::routine::ADMITTED_PLAN_RUNS;
use cubek_std::routine::Grid;

/// What a launch of this kind declines before preparing anything: rows that are not floats.
pub const RMS_NORM_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel RMS norm normalizes float rows",
};

/// The gate a norm multiplies its rows by, `silu(gate)`, and where it enters: a tensor of the
/// input's shape and element.
#[derive(Debug, Clone)]
pub struct RmsNormGate {
    pub tensor: PlainTensor,
    pub order: GateOrder,
}

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
    gate: Option<RmsNormGate>,
    out: Out,
    problem: RmsNormRoutineProblem,
}

impl<Out: core::fmt::Debug> core::fmt::Debug for RmsNormTensors<Out> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RmsNormTensors")
            .field("input", &self.input)
            .field("weight", &self.weight)
            .field("gate", &self.gate)
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

/// The rows, the weight and the gate as every launch reads them, before the output is decided.
struct RmsNormInputs {
    client: Client,
    input: PlainTensor,
    weight: RmsNormWeight,
    gate: Option<RmsNormGate>,
    /// The input's shape, which the output takes.
    shape: Vec<usize>,
    output_element: ElemType,
    batch: usize,
    rows: usize,
    dim: usize,
}

impl RmsNormInputs {
    /// The rows of `call`'s input, in batch entries where they lie, its gate cut into the same
    /// entries, and its weight, read as dense rows starting on a whole read. Under a weight a row
    /// ([`RmsNormWeight::PerRow`]) each batch entry holds the weight's rows: rows that lie in
    /// other entries are made dense and taken as many at a time.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`].
    fn new(call: &RmsNormCall) -> Result<Self, Refusal> {
        let (client, input) = (&call.client, call.input.clone());
        assert_eq!(
            input.elem, call.weight.elem,
            "the weight is the rows' element"
        );
        if let Some(gate) = &call.gate {
            assert_eq!(
                gate.tensor.elem, input.elem,
                "the gate is the rows' element"
            );
            assert_eq!(
                gate.tensor.binding.shape, input.binding.shape,
                "the gate has the rows' shape"
            );
        }
        let output_element = call.output_element;
        let hw = Hardware::from_client(client);
        RmsNormRoutineProblem::admit_elements(&hw, input.elem, output_element)?;
        let read_bytes = hw.properties.io_width() as u64 / 8;
        let shape = input.binding.shape.to_vec();
        let dim = *shape.last().expect("a tensor has a last dimension");
        let weight = RmsNormWeight::new(call.weight.clone().into_dense_at(client, read_bytes), dim);
        let cut = |tensor: PlainTensor| match tensor.as_batched_rows(dim) {
            Some(rows) if rows.start_bytes().is_multiple_of(read_bytes.max(1)) => rows,
            _ => dense_entries(client, tensor, dim, read_bytes),
        };
        let input = cut(input);
        let gate = call.gate.as_ref().map(|gate| RmsNormGate {
            tensor: cut(gate.tensor.clone()),
            order: gate.order,
        });
        // The rows an entry holds: the weight's under a weight a row, else the multi-entry
        // operand's, and every operand is cut to them.
        let each = weight.rows().or_else(|| {
            let shapes = [Some(&input), gate.as_ref().map(|gate| &gate.tensor)];
            shapes
                .into_iter()
                .flatten()
                .find(|tensor| tensor.binding.shape[0] > 1)
                .map(|tensor| tensor.binding.shape[1])
        });
        let to_each = |tensor: PlainTensor| match each {
            Some(each) if tensor.binding.shape[1] != each => {
                let rows = match tensor.binding.shape[0] {
                    1 => tensor,
                    _ => dense_entries(client, tensor, dim, read_bytes),
                };
                let rows_in_all = rows.binding.shape[1];
                assert!(
                    rows_in_all.is_multiple_of(each),
                    "a weight of {each} rows scales whole groups of rows, got {rows_in_all} rows"
                );
                rows.viewed(Metadata::new(
                    [rows_in_all / each, each, dim],
                    [each * dim, dim, 1],
                ))
            }
            _ => tensor,
        };
        let input = to_each(input);
        let gate = gate.map(|gate| RmsNormGate {
            tensor: to_each(gate.tensor),
            ..gate
        });
        let [batch, rows, _] = input.binding.shape[..] else {
            unreachable!("batched rows have three dims")
        };
        Ok(Self {
            client: client.clone(),
            input,
            weight,
            gate,
            shape,
            output_element,
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

    /// These operands, the output written at the input's element, as every output but a buffer
    /// holds it.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_OUTPUT_UNSUPPORTED`].
    fn writing_the_input_element(self) -> Result<Self, Refusal> {
        match self.output_element == self.input.elem {
            true => Ok(self),
            false => Err(RMS_NORM_OUTPUT_UNSUPPORTED),
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
            gate: self.gate,
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
            output_element: self.output_element,
            elem_alignments: RmsNormOperands {
                input: input.elem_alignment(),
                weight: weight.elem_alignment(),
                gate: self
                    .gate
                    .as_ref()
                    .map_or(ElemAlignment::dense_rows(self.dim), |gate| {
                        gate.tensor.elem_alignment()
                    }),
                out: out.elem_alignment,
            },
            start_bytes: RmsNormOperands {
                input: input.start_bytes(),
                weight: weight.start_bytes(),
                gate: self
                    .gate
                    .as_ref()
                    .map_or(0, |gate| gate.tensor.start_bytes()),
                out: out.start_bytes,
            },
            store,
            gate: self.gate.as_ref().map(|gate| gate.order),
        }
    }
}

impl RmsNormTensors {
    /// The norm `call` states, its output allocated at the call's output element.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_ELEMENT_UNVERIFIED`], what the routines decline whatever the device,
    /// [`RMS_NORM_OUTPUT_UNSUPPORTED`], and `ELEMENT_NOT_ON_DEVICE`: an element this device's
    /// runtime does not run.
    ///
    /// # Panics
    ///
    /// Operands on different devices or of different elements, a gate not shaped as the input, a
    /// weight that is not whole rows of
    /// the normalized dim, or a weight a row whose rows do not divide the input's: no norm at all.
    pub fn new(call: &RmsNormCall) -> Result<Self, Refusal> {
        let (client, operands) = (&call.client, RmsNormInputs::new(call)?);
        let (handle, tensor) = PlainTensor::empty_contiguous(
            client,
            operands.shape.clone().into(),
            operands.output_element,
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
        let binding = self.bind_operands(blueprint, grid);
        let out = binding
            .launcher
            .arg(self.out.rows.binding.clone(), self.problem.output_element)
            .axes(&[B, R, D])
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        binding.map_out(|()| out)
    }
}

impl RmsNormTensors<Geometry> {
    /// The norm `call` states, handed to a sink the caller binds:
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
    pub fn sink(call: &RmsNormCall) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(call)?.writing_the_input_element()?;
        let out = Geometry::row_major(&[operands.batch, operands.rows, operands.dim]);
        let placement = Placement::dense_rows(operands.dim);
        Ok(operands.with_out(out, placement, RowStore::Plain))
    }

    /// The three operands bound as `blueprint` states them, on `grid`, the output derived over
    /// the sink's layout rather than a buffer: its spec, and the geometry the derivation settled,
    /// which the caller's sink addresses through.
    pub fn bind<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<Unbound> {
        let binding = self.bind_operands(blueprint, grid);
        let out = binding
            .launcher
            .unbound(&self.out)
            .axes(&[B, R, D])
            .vectorize(blueprint.width)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        binding.map_out(|()| out)
    }

    /// The sink the output is laid out as.
    pub fn out(&self) -> &Geometry {
        &self.out
    }
}

impl RmsNormTensors<PackedRows> {
    /// The norm `call` states, encoded into `out`, a packed
    /// `[rows, dim]` matrix over a product's lhs axes the caller allocated under one level of block
    /// scales along each row. The rows are staged in each cube's shared memory, dense, and encoded
    /// from there.
    ///
    /// # Errors
    ///
    /// As [`RmsNormTensors::new`], with [`RMS_NORM_OUTPUT_UNSUPPORTED`] for an `output_element` other
    /// than the input's; [`RMS_NORM_ROWS_IN_ENTRIES`]; and what
    /// [`PackedRows::new`] declines of `out`.
    ///
    /// # Panics
    ///
    /// As [`RmsNormTensors::new`], and on an `out` not shaped `[rows, dim]`.
    pub fn packed(call: &RmsNormCall, out: PackedTensor) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(call)?
            .writing_the_input_element()?
            .one_entry()?;
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
        let binding = self.bind_operands(blueprint, grid);
        let out = self.out.bind(&binding.launcher, blueprint.width);
        binding.map_out(|()| out)
    }
}

impl RmsNormTensors<TiledRows> {
    /// The norm `call` states, written into `out`, a float
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
    pub fn tiled(call: &RmsNormCall, out: PlainTensor) -> Result<Self, Refusal> {
        let operands = RmsNormInputs::new(call)?
            .writing_the_input_element()?
            .one_entry()?;
        let element = operands.input.elem;
        let out = TiledRows::new(out, operands.rows, operands.dim, element)?;
        let placement = Placement::from(&out);
        Ok(operands.with_out(out, placement, RowStore::Plain))
    }

    /// The rows and the weight bound as `blueprint` states them, on `grid`, and the output over
    /// the rows and the dim, through the tiles its binding states.
    pub fn bind<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<Bound> {
        let binding = self.bind_operands(blueprint, grid);
        let out = self.out.bind(&binding.launcher, blueprint.width);
        binding.map_out(|()| out)
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

    /// The launcher of `blueprint`'s partitioning on `grid`, and the rows, the weight and the gate
    /// bound under it at its width: what every output binds beside.
    fn bind_operands<M>(&self, blueprint: &RmsNormBlueprint<M>, grid: &Grid) -> RmsNormBinding<()> {
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
        let input = bind(&self.input, &[B, R]);
        let weight = bind(self.weight.tensor(), self.weight.leading_axes());
        let gate = self.gate.as_ref().map(|gate| bind(&gate.tensor, &[B, R]));
        RmsNormBinding {
            launcher,
            input,
            weight,
            gate,
            out: (),
        }
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
    /// The gate, where the norm is gated.
    pub gate: Option<Bound>,
    pub out: Out,
}

impl<Out> RmsNormBinding<Out> {
    /// The same binding, its output turned into `T` — what a destination's launch takes.
    pub fn map_out<T>(self, f: impl FnOnce(Out) -> T) -> RmsNormBinding<T> {
        RmsNormBinding {
            launcher: self.launcher,
            input: self.input,
            weight: self.weight,
            gate: self.gate,
            out: f(self.out),
        }
    }
}

/// `tensor` copied into dense rows starting on a read, as `[1, rows, dim]`.
fn dense_entries(client: &Client, tensor: PlainTensor, dim: usize, read_bytes: u64) -> PlainTensor {
    tensor
        .into_dense_at(client, read_bytes)
        .as_batched_rows(dim)
        .expect("dense rows are one batch entry")
}
