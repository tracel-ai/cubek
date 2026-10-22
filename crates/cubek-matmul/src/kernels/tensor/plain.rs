use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::std::tensor::is_contiguous;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Geometry, Projection};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubecl::zspace::{Shape, Tiling};
use cubek_std::layout::{is_dense, is_nonoverlapping, one_matrix};

use cubek_microkernel::matmul::MatmulBlocks;

/// A float operand as the routines read it: cubecl's binding of its buffer, and the element it
/// holds. Every view below reads the same buffer under other dims, so a launch states its problem
/// off exactly what it binds.
#[derive(Debug, Clone)]
pub struct PlainTensor {
    pub binding: TensorBinding,
    pub elem: ElemType,
}

impl PlainTensor {
    pub fn new(binding: TensorBinding, elem: ElemType) -> Self {
        Self { binding, elem }
    }

    /// `handle`, bound.
    pub fn of_handle(handle: TensorHandle) -> Self {
        let elem = handle.dtype;
        Self::new(handle.binding(), elem)
    }

    /// An uninitialized tensor shaped `shape` in `elem`, at whatever layout the runtime's
    /// allocator prefers: on CUDA and ROCm its rows come back pitched, and the launch reads them by
    /// their real strides. The handle is the caller's to hand back; the view binds it.
    pub fn empty(client: &Client, shape: Shape, elem: ElemType) -> (TensorHandle, Self) {
        let layout = client.empty_tensor(shape.clone(), elem.size());
        let handle = TensorHandle::new(layout.memory, shape, layout.strides, elem);
        let plain = Self::of_handle(handle.clone());
        (handle, plain)
    }

    /// An uninitialized row-major tensor shaped `shape` in `elem`, gapless whatever the runtime: a
    /// view that factors its strides, as a reshape does, stays valid. The handle is the caller's
    /// to hand back; the view binds it.
    pub fn empty_contiguous(client: &Client, shape: Shape, elem: ElemType) -> (TensorHandle, Self) {
        let memory = client.empty(shape.num_elements() * elem.size());
        let handle = TensorHandle::new_contiguous(shape, memory, elem);
        let plain = Self::of_handle(handle.clone());
        (handle, plain)
    }

    /// Whether `client`'s runtime computes in this tensor's element and stores it in a buffer:
    /// what a kernel at that element needs to compile there.
    pub fn runs_on(&self, client: &Client) -> bool {
        client
            .properties()
            .features
            .type_usage(self.elem)
            .is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer)
    }

    /// The buffer's shape, strides and tiling.
    pub fn meta(&self) -> Metadata {
        Metadata::new(self.binding.shape.clone(), self.binding.strides.clone())
            .with_tiling(self.binding.tiling)
            .expect("a binding's tiling describes its own rank")
    }

    /// The *logical* shape: a storage-tiled buffer stores each matrix dim as two physical dims,
    /// and its metadata folds them back. The physical shape for a buffer in rows.
    pub fn shape(&self) -> Shape {
        self.meta()
            .logical_shape()
            .expect("a binding's tiling describes its own rank")
    }

    /// Bytes this view starts past its buffer's own start.
    pub fn start_bytes(&self) -> u64 {
        self.binding.handle.offset_start.unwrap_or(0)
    }

    /// Whether the buffer is stored one tile at a time rather than in rows.
    pub fn is_tiled(&self) -> bool {
        self.binding.tiling.is_tiled()
    }

    /// Whether this tensor is stored in tiles that each step down a column first: a tile read by
    /// its columns, as the matmul kernels read one, steps along its rows.
    pub fn is_tiled_column_first(&self) -> bool {
        self.is_tiled() && self.binding.strides.last() != Some(&1)
    }

    /// The alignment of this view's layout, as a launch binding it reads it: a storage-tiled
    /// one's is its stored piece, which its strides, being the tiles', do not state.
    pub fn elem_alignment(&self) -> ElemAlignment {
        ElemAlignment::from(&self.binding)
    }

    /// This buffer read as `meta` states.
    pub fn viewed(&self, meta: Metadata) -> Self {
        Self::new(self.bound_as(meta), self.elem)
    }

    /// This view's last two dims swapped, over the same buffer.
    pub fn transposed(&self) -> Self {
        let (mut shape, mut strides) = (self.binding.shape.clone(), self.binding.strides.clone());
        let rank = shape.len();
        shape.swap(rank - 2, rank - 1);
        strides.swap(rank - 2, rank - 1);
        self.viewed(Metadata::new(shape, strides))
    }

    /// This buffer bound as `meta` states.
    pub fn bound_as(&self, meta: Metadata) -> TensorBinding {
        TensorBinding {
            handle: self.binding.handle.clone(),
            strides: meta.strides,
            shape: meta.shape,
            tiling: meta.tiling,
        }
    }

    /// This buffer as `rows` rows of `line` values, stepping by the row stride it really has: a
    /// pitched allocator pads a row, so the step is read, never derived from the shape.
    ///
    /// # Panics
    ///
    /// On an innermost stride other than 1: a row is a contiguous line.
    pub fn as_rows(&self, rows: usize, line: usize) -> Self {
        let step = self.row_step().unwrap_or(line);
        self.viewed(Metadata::new([rows, line], [step, 1]))
    }

    /// [`as_rows`](Self::as_rows), bound.
    pub fn rows(&self, rows: usize, line: usize) -> TensorBinding {
        self.as_rows(rows, line).binding
    }

    /// [`rows`](Self::rows) over a buffer whose every stored element holds `values_per_element`
    /// values: the binding is in values, the strides in stored elements.
    pub fn rows_unpacked(
        &self,
        rows: usize,
        line: usize,
        values_per_element: usize,
    ) -> TensorBinding {
        let step = self
            .row_step()
            .map_or(line, |step| step * values_per_element);
        TensorBinding {
            handle: self.binding.handle.clone(),
            strides: [step, 1].into(),
            shape: [rows, line].into(),
            tiling: Tiling::UNTILED,
        }
    }

    /// This buffer as the one matrix its last two logical dims hold, stored as they are, in rows
    /// or in tiles; `None` where a leading dim holds more than one.
    pub fn as_one_matrix(&self) -> Option<Self> {
        one_matrix(&self.meta()).map(|meta| self.viewed(meta))
    }

    /// This tensor, dense rows along `rows` by `columns`, restated over the axes each is in the
    /// block-split space of `blocks`, and those axes: the same bytes, a dim an axis, which the
    /// tile engine stages and masks as it lies ([`MatmulBlocks::restated_dims`]).
    pub fn restated(&self, blocks: &MatmulBlocks, rows: Axis, columns: Axis) -> (Self, Vec<Axis>) {
        let dense = self.as_rows(blocks.extent(rows), blocks.extent(columns));
        let dims = blocks.restated_dims(rows, columns, dense.binding.strides[0]);
        let meta = Metadata::new(
            dims.iter()
                .map(|&(_, extent, _)| extent)
                .collect::<Vec<_>>(),
            dims.iter()
                .map(|&(_, _, stride)| stride)
                .collect::<Vec<_>>(),
        );
        (
            dense.viewed(meta),
            dims.iter().map(|&(axis, _, _)| axis).collect(),
        )
    }

    /// A rank-2 view as its buffer lies: the same buffer, its two dims named in the order they
    /// step. A transposed view over a row-major buffer comes back as that buffer; an ordered one
    /// comes back unchanged.
    ///
    /// # Panics
    ///
    /// On a view the two-dim projection cannot address.
    pub fn in_buffer_order(self) -> Self {
        let (shape, strides) = (&self.binding.shape, &self.binding.strides);
        let projection = Projection::direct(&[Axis(0), Axis(1)]);
        let dims = Geometry::new(&[(shape[0], strides[0]), (shape[1], strides[1])]);
        assert!(
            shape.rank() == 2 && projection.is_addressable(&dims),
            "a rank-2 view this projection can describe, got shape {shape:?} strides {strides:?}"
        );
        // Coarsest stride first; two dims either already step that way or swap.
        let (outer, inner) = match strides[0] >= strides[1] {
            true => (0, 1),
            false => (1, 0),
        };
        let meta = Metadata::new(
            [shape[outer], shape[inner]],
            [strides[outer], strides[inner]],
        );
        self.viewed(meta)
    }

    /// Whether the strides are *exactly* the row-major ones for this shape.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.binding.shape, &self.binding.strides)
    }

    /// Whether the elements are laid out row-major with no gaps, the axes of extent 1 ignored
    /// ([`is_dense`]).
    pub fn is_dense(&self) -> bool {
        is_dense(&self.binding.shape, &self.binding.strides)
    }

    /// Whether no two indices address the same element ([`is_nonoverlapping`]).
    pub fn is_nonoverlapping(&self) -> bool {
        is_nonoverlapping(&self.binding.shape, &self.binding.strides)
    }

    /// This tensor row-major and gapless: as it is where it already is, restrided where only an
    /// axis of extent 1 is out of place, a copy otherwise. A storage-tiled one is laid back in
    /// rows first.
    pub fn into_contiguous(self, client: &Client) -> Self {
        let tensor = match self.is_tiled() {
            true => Self::of_handle(
                crate::tiled::storage::untile(client, self.binding, self.elem)
                    .expect("a storage-tiled binding describes its own tiles"),
            ),
            false => self,
        };
        if tensor.is_contiguous() {
            return tensor;
        }
        // Dense already, with only an unobservable stride out of place: restride rather than
        // copy, the buffer is what a copy would have produced.
        if tensor.is_dense() {
            let shape = tensor.binding.shape.clone();
            let strides = row_major_contiguous_strides(&shape);
            return tensor.viewed(Metadata::new(shape, strides));
        }
        Self::of_handle(cubecl::std::tensor::into_contiguous(
            client,
            tensor.binding,
            tensor.elem,
        ))
    }

    /// The step between the rows of a row-bound operand, in stored elements: the stride it really
    /// has; `None` for one of a single dim, which has no step.
    ///
    /// # Panics
    ///
    /// On an innermost stride other than 1: a row is a contiguous line.
    fn row_step(&self) -> Option<usize> {
        let strides = &self.binding.strides;
        let rank = strides.len();
        assert_eq!(
            strides[rank - 1],
            1,
            "a row-bound operand's innermost stride must be 1, got {strides:?}"
        );
        (rank >= 2).then(|| strides[rank - 2])
    }
}

/// What a launch declines on a device whose runtime does not compute in the operands' element and
/// store it in a buffer, as WGSL does not bf16: the kernel would fail to compile there.
pub const ELEMENT_NOT_ON_DEVICE: Refusal = Refusal::Declined {
    why: "the device's runtime does not compute in this element and store it in a buffer",
};
