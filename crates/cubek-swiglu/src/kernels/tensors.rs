//! How a SwiGLU launch reads its operands and binds them.

use crate::kernels::{SwigluBlueprint, SwigluRoutineProblem};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Launcher, launch::Bound};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{
    ELEMENT_NOT_ON_DEVICE, PackedRows, PackedTensor, Placement, PlainTensor, TiledRows,
};
use cubek_microkernel::row_store::RowStore;
use cubek_microkernel::swiglu::{D, H, R, SwigluOperands};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// What a launch of this kind declines before preparing anything: rows that are not floats.
pub const SWIGLU_ELEMENT_UNVERIFIED: Refusal = Refusal::Declined {
    why: "the microkernel SwiGLU gates float rows",
};

/// The tensors of one gate, its output among them, and the problem they state.
///
/// The fused rows are read as dense rows starting on a whole read of the device, as a
/// projection's fresh output is, and copied into dense ones ([`PlainTensor::into_dense_at`])
/// otherwise, and bound as `[r, h, d]`: the gate at `h = 0` and the up at `h = 1`, a half `d`
/// elements apart. The up half is a stride of the one binding, never a view
/// starting inside the buffer, which a runtime binds only at its own offset alignment. So the
/// input binds at the one width a dense run of `d` does ([`SwigluWorkItem::width`]), which is the
/// width a tuning key prepares its candidates at.
///
/// The output is `Out`: a [`SwigluOutputBuffer`] the launch allocates, dense in the input's shape
/// at half its width, or a [`PackedRows`] or a [`TiledRows`] the caller allocates, which the
/// launch encodes the gated rows into or writes them into the tiles of.
///
/// [`SwigluWorkItem::width`]: cubek_microkernel::swiglu::SwigluWorkItem::width
#[derive(Clone)]
pub struct SwigluTensors<Out = SwigluOutputBuffer> {
    client: Client,
    input: PlainTensor,
    out: Out,
    problem: SwigluRoutineProblem,
}

impl<Out: core::fmt::Debug> core::fmt::Debug for SwigluTensors<Out> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SwigluTensors")
            .field("input", &self.input)
            .field("out", &self.out)
            .field("problem", &self.problem)
            .finish_non_exhaustive()
    }
}

/// The output a launch allocates: dense in the input's shape at half its width, and the `[r, d]`
/// view of it the kernel writes through.
#[derive(Debug, Clone)]
pub struct SwigluOutputBuffer {
    handle: TensorHandle,
    rows: PlainTensor,
}

/// The operands of a launch bound under one launcher, the output as `Out`. Not `Debug`: cubecl's
/// `Launcher` is not.
pub struct SwigluBinding<Out> {
    pub launcher: Launcher,
    pub input: Bound,
    pub out: Out,
}

impl SwigluTensors {
    /// The gate of `both`, `(..., 2 · d)` laid out `[gate | up]`, its output allocated.
    ///
    /// # Errors
    ///
    /// [`SWIGLU_ELEMENT_UNVERIFIED`], what the routines decline whatever the device, and
    /// [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does not run.
    ///
    /// # Panics
    ///
    /// On an odd last dim: no fused row of two halves.
    pub fn new(client: &Client, both: PlainTensor) -> Result<Self, Refusal> {
        let input = SwigluInput::new(client, both)?;
        let (handle, tensor) =
            PlainTensor::empty_contiguous(client, input.shape.clone().into(), input.fused.elem);
        let rows = tensor.viewed(Metadata::new([input.rows, input.dim], [input.dim, 1]));
        let placement = Placement::from(&rows);
        Ok(input.with_out(
            SwigluOutputBuffer { handle, rows },
            placement,
            RowStore::Plain,
        ))
    }

    /// The output, in the input's shape at half its width.
    pub fn into_out(self) -> TensorHandle {
        self.out.handle
    }

    /// The two operands bound as `blueprint` states them, on `grid`: under its partitioning, at
    /// its width.
    pub fn bind<M>(&self, blueprint: &SwigluBlueprint<M>, grid: &Grid) -> SwigluBinding<Bound> {
        let (launcher, input) = self.bind_input(blueprint, grid);
        let out = launcher
            .arg(self.out.rows.binding.clone(), self.problem.element)
            .axes(&[R, D])
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        SwigluBinding {
            launcher,
            input,
            out,
        }
    }
}

impl SwigluTensors<PackedRows> {
    /// The gate of `both`, encoded into `out`, a packed `[rows, d]` matrix over a product's lhs
    /// axes the caller allocated under one level of block scales along each row. The gated rows are
    /// staged in each cube's shared memory, dense, and encoded from there.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`], and what [`PackedRows::new`] declines of `out`.
    ///
    /// # Panics
    ///
    /// As [`SwigluTensors::new`], and on an `out` not shaped `[rows, d]`.
    pub fn packed(client: &Client, both: PlainTensor, out: PackedTensor) -> Result<Self, Refusal> {
        let input = SwigluInput::new(client, both)?;
        let out = PackedRows::new(out, input.rows, input.dim)?;
        let (placement, store) = (Placement::dense_rows(input.dim), out.store());
        Ok(input.with_out(out, placement, store))
    }

    /// The input bound as `blueprint` states it, on `grid`, and the output as one tile of its
    /// words under their scales ([`PackedRows::bind`]).
    pub fn bind<M>(
        &self,
        blueprint: &SwigluBlueprint<M>,
        grid: &Grid,
    ) -> SwigluBinding<TileTensorLaunch> {
        let (launcher, input) = self.bind_input(blueprint, grid);
        let out = self.out.bind(&launcher, blueprint.width);
        SwigluBinding {
            launcher,
            input,
            out,
        }
    }
}

impl SwigluTensors<TiledRows> {
    /// The gate of `both`, written into `out`, a float `[rows, d]` matrix stored in tiles the
    /// caller allocated: the gated rows land in its tiles as they are written, through the binding
    /// its tiling states.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`], and what [`TiledRows::new`] declines of `out`.
    ///
    /// # Panics
    ///
    /// As [`SwigluTensors::new`].
    pub fn tiled(client: &Client, both: PlainTensor, out: PlainTensor) -> Result<Self, Refusal> {
        let input = SwigluInput::new(client, both)?;
        let element = input.fused.elem;
        let out = TiledRows::new(out, input.rows, input.dim, element)?;
        let placement = Placement::from(&out);
        Ok(input.with_out(out, placement, RowStore::Plain))
    }

    /// The input bound as `blueprint` states it, on `grid`, and the output over the rows and the
    /// width, through the tiles its binding states.
    pub fn bind<M>(&self, blueprint: &SwigluBlueprint<M>, grid: &Grid) -> SwigluBinding<Bound> {
        let (launcher, input) = self.bind_input(blueprint, grid);
        let out = self.out.bind(&launcher, blueprint.width);
        SwigluBinding {
            launcher,
            input,
            out,
        }
    }
}

impl<Out> SwigluTensors<Out> {
    /// The client the operands live on, which a launch submits to.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The problem the tensors state.
    pub fn problem(&self) -> &SwigluRoutineProblem {
        &self.problem
    }

    /// The launcher of `blueprint`'s partitioning on `grid`, and the fused rows bound under it at
    /// its width, as the store states the rows: what every output binds beside.
    fn bind_input<M>(&self, blueprint: &SwigluBlueprint<M>, grid: &Grid) -> (Launcher, Bound) {
        let partitioning = &blueprint.partitioning;
        // The grid's launcher, settled over the extents on its first launch and cloned after.
        let launcher = grid.launcher(self.client(), || {
            (
                partitioning.kernel(),
                partitioning.as_tile().space().clone(),
            )
        });
        let (meta, axes) = self.problem.store.restated(&self.input.meta(), &[R, H]);
        let input = launcher
            .arg(self.input.bound_as(meta), self.problem.element)
            .axes(&axes)
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        (launcher, input)
    }
}

impl<Out> SwigluBinding<Out> {
    /// The same binding, its output turned into `T` — what a destination's launch takes.
    pub fn map_out<T>(self, f: impl FnOnce(Out) -> T) -> SwigluBinding<T> {
        SwigluBinding {
            launcher: self.launcher,
            input: self.input,
            out: f(self.out),
        }
    }
}

/// The fused rows as every launch reads them, before the output is decided.
struct SwigluInput {
    client: Client,
    /// `[r, h, d]` over the dense fused rows.
    fused: PlainTensor,
    /// The input's shape at half its width, which the output takes.
    shape: Vec<usize>,
    rows: usize,
    dim: usize,
}

impl SwigluInput {
    /// The fused rows of `both`, `(..., 2 · d)` laid out `[gate | up]`, as dense rows starting on
    /// a whole read, bound as `[r, h, d]`.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`].
    fn new(client: &Client, both: PlainTensor) -> Result<Self, Refusal> {
        if !matches!(both.elem, ElemType::Float(_)) {
            return Err(SWIGLU_ELEMENT_UNVERIFIED);
        }
        if !both.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        let mut shape = both.binding.shape.to_vec();
        let last = shape.last_mut().expect("a tensor has a last dimension");
        assert!(
            last.is_multiple_of(2),
            "a fused gate/up row has an even width, got {last}"
        );
        *last /= 2;
        let dim = *last;
        let rows = shape[..shape.len() - 1].iter().product();
        let read_bytes = Hardware::from_client(client).properties.io_width() as u64 / 8;
        let both = both.into_dense_at(client, read_bytes);
        let fused = both.viewed(Metadata::new([rows, 2, dim], [2 * dim, dim, 1]));
        Ok(Self {
            client: client.clone(),
            fused,
            shape,
            rows,
            dim,
        })
    }

    /// The tensors of the gate of these rows into `out`, which lies at `placement` and stores its
    /// rows as `store` says.
    fn with_out<Out>(self, out: Out, placement: Placement, store: RowStore) -> SwigluTensors<Out> {
        let problem = self.problem(placement, store);
        SwigluTensors {
            client: self.client,
            input: self.fused,
            out,
            problem,
        }
    }

    /// The problem the fused rows state, the output placed at `out` and its rows stored as
    /// `store` says.
    fn problem(&self, out: Placement, store: RowStore) -> SwigluRoutineProblem {
        let input = Placement::from(&self.fused);
        SwigluRoutineProblem {
            rows: self.rows,
            dim: self.dim,
            element: self.fused.elem,
            elem_alignments: SwigluOperands {
                input: input.elem_alignment,
                out: out.elem_alignment,
            },
            start_bytes: SwigluOperands {
                input: input.start_bytes,
                out: out.start_bytes,
            },
            store,
        }
    }
}
