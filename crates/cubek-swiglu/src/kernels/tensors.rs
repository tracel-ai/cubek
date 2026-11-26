//! How a SwiGLU launch reads its operands and binds them.

use crate::kernels::{SwigluBlueprint, SwigluRoutineProblem};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Launcher, launch::Bound};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::{
    ELEMENT_NOT_ON_DEVICE, PackedRows, PackedTensor, Placement, PlainTensor, TiledRows,
};
use cubek_microkernel::row_store::RowStore;
use cubek_microkernel::swiglu::{D, H, R, SwigluOperands};
use cubek_std::Hardware;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable, Grid};

/// What a launch of this kind declines before preparing anything: rows that are not floats.
pub const SWIGLU_ELEMENT_UNVERIFIED: Decline = Decline::Unserved {
    why: "the microkernel SwiGLU gates float rows",
};

/// What a launch of this kind declines before preparing anything: an input of no dim, or whose
/// last dim is odd, which holds no fused row of two halves.
pub const SWIGLU_ROW_NOT_TWO_HALVES: Decline = Decline::Unserved {
    why: "a fused gate/up row is the last dim of the input, of an even width",
};

/// What a launch into a buffer its caller allocated declines: an output of another shape or
/// element than the gate's, or not dense.
pub const SWIGLU_OUTPUT_UNREAD: Decline = Decline::Unserved {
    why: "the output takes the input's element and shape at half its width, laid densely",
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
    /// [`SWIGLU_ELEMENT_UNVERIFIED`] and [`SWIGLU_ROW_NOT_TWO_HALVES`], what the routines decline
    /// whatever the device, and [`ELEMENT_NOT_ON_DEVICE`]: an element this device's runtime does
    /// not run.
    pub fn new(client: &Client, both: PlainTensor) -> Result<Self, Decline> {
        let shape = SwigluInput::admit(client, &both)?;
        let input = SwigluInput::new(client, both, shape);
        let (out, _) =
            PlainTensor::empty_contiguous(client, input.shape.clone().into(), input.fused.elem);
        Ok(input.into_contiguous(out))
    }

    /// The output, in the input's shape at half its width.
    pub fn into_out(self) -> TensorHandle {
        self.out.handle
    }

    /// The two operands bound as `blueprint` states them, on `grid`: under its partitioning, at
    /// its width.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    pub fn bind<M>(
        &self,
        blueprint: &SwigluBlueprint<M>,
        grid: &Grid,
    ) -> Result<SwigluBinding<Bound>, Decline> {
        let (launcher, input) = self.bind_input(blueprint, grid)?;
        let out = launcher
            .arg(self.out.rows.binding.clone(), self.problem.element)
            .dimensions(&[R, D])
            .vectorize(blueprint.width)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok(SwigluBinding {
            launcher,
            input,
            out,
        })
    }
}

impl SwigluTensors {
    /// A contiguous output for the gate of `both`: the input's shape at half its width.
    ///
    /// # Errors
    ///
    /// [`SWIGLU_ROW_NOT_TWO_HALVES`].
    pub fn allocate_out(client: &Client, both: &PlainTensor) -> Result<TensorHandle, Decline> {
        let shape = gated_shape(both)?;
        let (handle, _) = PlainTensor::empty_contiguous(client, shape.into(), both.elem);
        Ok(handle)
    }

    /// [`SwigluTensors::new`] into `out`, a contiguous buffer the caller allocated
    /// ([`allocate_out`](Self::allocate_out)): what every candidate of a race writes into.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`], and [`SWIGLU_OUTPUT_UNREAD`].
    pub(crate) fn contiguous(
        client: &Client,
        both: PlainTensor,
        out: TensorHandle,
    ) -> Result<Self, Decline> {
        let shape = SwigluInput::admit(client, &both)?;
        let dense = PlainTensor::of_handle(out.clone()).is_contiguous();
        if !out.shape().iter().eq(shape.iter()) || out.dtype != both.elem || !dense {
            return Err(SWIGLU_OUTPUT_UNREAD);
        }
        Ok(SwigluInput::new(client, both, shape).into_contiguous(out))
    }
}

impl SwigluTensors<PackedRows> {
    /// The gate of `both`, encoded into `out`, a packed `[rows, d]` matrix over a product's lhs
    /// dimensions the caller allocated under one level of block scales along each row. The gated
    /// rows are staged in each cube's shared memory, dense, and encoded from there.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`], and what [`PackedRows::new`] declines of `out`.
    pub fn packed(client: &Client, both: PlainTensor, out: PackedTensor) -> Result<Self, Decline> {
        let shape = SwigluInput::admit(client, &both)?;
        let [rows, dim] = rows_and_dim(&shape);
        let out = PackedRows::new(out, rows, dim)?;
        let input = SwigluInput::new(client, both, shape);
        let (placement, store) = (Placement::dense_rows(input.dim), out.store());
        Ok(input.with_out(out, placement, store))
    }

    /// The input bound as `blueprint` states it, on `grid`, and the output as one tile of its
    /// words under their scales ([`PackedRows::bind`]).
    ///
    /// # Errors
    ///
    /// As the `bind` of the tensors writing into a buffer.
    pub fn bind<M>(
        &self,
        blueprint: &SwigluBlueprint<M>,
        grid: &Grid,
    ) -> Result<SwigluBinding<TileTensorLaunch>, Decline> {
        let (launcher, input) = self.bind_input(blueprint, grid)?;
        let out = self.out.bind(&launcher, blueprint.width)?;
        Ok(SwigluBinding {
            launcher,
            input,
            out,
        })
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
    pub fn tiled(client: &Client, both: PlainTensor, out: PlainTensor) -> Result<Self, Decline> {
        let shape = SwigluInput::admit(client, &both)?;
        let [rows, dim] = rows_and_dim(&shape);
        let out = TiledRows::new(out, rows, dim, both.elem)?;
        let input = SwigluInput::new(client, both, shape);
        let placement = Placement::from(&out);
        Ok(input.with_out(out, placement, RowStore::Plain))
    }

    /// The input bound as `blueprint` states it, on `grid`, and the output over the rows and the
    /// width, through the tiles its binding states.
    ///
    /// # Errors
    ///
    /// As the `bind` of the tensors writing into a buffer.
    pub fn bind<M>(
        &self,
        blueprint: &SwigluBlueprint<M>,
        grid: &Grid,
    ) -> Result<SwigluBinding<Bound>, Decline> {
        let (launcher, input) = self.bind_input(blueprint, grid)?;
        let out = self.out.bind(&launcher, blueprint.width)?;
        Ok(SwigluBinding {
            launcher,
            input,
            out,
        })
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
    fn bind_input<M>(
        &self,
        blueprint: &SwigluBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, Bound), Decline> {
        let partitioning = &blueprint.partitioning;
        // The grid's launcher, settled over the extents on its first launch and cloned after.
        let launcher = grid.launcher(self.client(), || {
            (
                partitioning.kernel(),
                partitioning.as_tile().space().clone(),
            )
        })?;
        let (meta, dimensions) = self.problem.store.restated(&self.input.meta(), &[R, H]);
        let input = launcher
            .arg(self.input.bound_as(meta), self.problem.element)
            .dimensions(&dimensions)
            .vectorize(blueprint.width)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        Ok((launcher, input))
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
/// The shape of the gate of `both`: its shape at half its width.
///
/// # Errors
///
/// [`SWIGLU_ROW_NOT_TWO_HALVES`].
fn gated_shape(both: &PlainTensor) -> Result<Vec<usize>, Decline> {
    let mut shape = both.binding.shape.to_vec();
    match shape.last_mut() {
        Some(last) if last.is_multiple_of(2) => *last /= 2,
        _ => return Err(SWIGLU_ROW_NOT_TWO_HALVES),
    }
    Ok(shape)
}

/// The rows and the dim of a gate of `shape`, `[rows, dim]`.
///
/// # Assumptions
///
/// `shape` has a last dim, as a gated shape does.
fn rows_and_dim(shape: &[usize]) -> [usize; 2] {
    let (dim, rows) = shape.split_last().expect("a gated shape has a last dim");
    [rows.iter().product(), *dim]
}

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
    /// # Assumptions
    ///
    /// `shape` is what [`admit`](Self::admit) returned for `both`.
    fn new(client: &Client, both: PlainTensor, shape: Vec<usize>) -> Self {
        let [rows, dim] = rows_and_dim(&shape);
        let read_bytes = Hardware::from_client(client).properties.io_width() as u64 / 8;
        let both = both.into_dense_at(client, read_bytes);
        let fused = both.viewed(Metadata::new([rows, 2, dim], [2 * dim, dim, 1]));
        Self {
            client: client.clone(),
            fused,
            shape,
            rows,
            dim,
        }
    }

    /// The gate's shape of `both`, checked from its shape and element alone, before any copy.
    ///
    /// # Errors
    ///
    /// As [`SwigluTensors::new`].
    fn admit(client: &Client, both: &PlainTensor) -> Result<Vec<usize>, Decline> {
        if !matches!(both.elem, ElemType::Float(_)) {
            return Err(SWIGLU_ELEMENT_UNVERIFIED);
        }
        if !both.runs_on(client) {
            return Err(ELEMENT_NOT_ON_DEVICE);
        }
        gated_shape(both)
    }

    /// The tensors of the gate of these rows into `out`, a contiguous buffer in the input's shape
    /// at half its width.
    ///
    /// # Assumptions
    ///
    /// `out` takes the gate's shape and the input's element.
    fn into_contiguous(self, out: TensorHandle) -> SwigluTensors {
        let rows = PlainTensor::of_handle(out.clone())
            .viewed(Metadata::new([self.rows, self.dim], [self.dim, 1]));
        let placement = Placement::from(&rows);
        self.with_out(
            SwigluOutputBuffer { handle: out, rows },
            placement,
            RowStore::Plain,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;
    use cubek_test_utils::client;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Zeros of `shape` in `f32`.
    fn zeros(shape: &[usize]) -> PlainTensor {
        PlainTensor::of_handle(TensorHandle::zeros(&client(), shape.to_vec(), F32))
    }

    #[test]
    fn an_odd_fused_row_declines() {
        let declined = SwigluTensors::new(&client(), zeros(&[4, 7])).err();

        assert_eq!(declined, Some(SWIGLU_ROW_NOT_TWO_HALVES));
    }

    #[test]
    fn a_strided_output_declines() {
        let mut out = TensorHandle::zeros(&client(), vec![4, 4], F32);
        out.metadata = Box::new(Metadata::new(vec![4, 4], vec![1, 4]));

        let declined = SwigluTensors::contiguous(&client(), zeros(&[4, 8]), out).err();

        assert_eq!(declined, Some(SWIGLU_OUTPUT_UNREAD));
    }

    #[test]
    fn an_output_of_another_shape_declines() {
        let out = TensorHandle::zeros(&client(), vec![4, 8], F32);

        let declined = SwigluTensors::contiguous(&client(), zeros(&[4, 8]), out).err();

        assert_eq!(declined, Some(SWIGLU_OUTPUT_UNREAD));
    }
}
