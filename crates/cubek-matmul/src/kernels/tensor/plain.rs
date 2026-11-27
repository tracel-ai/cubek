use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::TensorBinding;
use cubecl::std::tensor::TensorHandle;
use cubecl::std::tensor::is_contiguous;
use cubecl::tile::launch::Decline;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Dimension, Geometry, Projection};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubecl::zspace::{Shape, Strides, Tiling};
use cubek_std::layout::{
    Fragments, fragments, fragments_of, is_dense, is_nonoverlapping, one_matrix,
};

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

    /// The bytes of the buffer it binds.
    pub fn bytes(&self) -> usize {
        self.binding.handle.size as usize
    }

    /// A zero-filled tensor shaped `shape` in `elem` at `strides`, uploaded rather than filled: an
    /// upload is not a launch, so a process mode that discards launches keeps it. An axis at stride
    /// zero holds one element.
    pub fn zeros(
        client: &Client,
        shape: &[usize],
        strides: impl Into<Strides>,
        elem: ElemType,
    ) -> Self {
        let strides: Strides = strides.into();
        let elements = shape
            .iter()
            .zip(strides.iter())
            .map(|(extent, stride)| (extent - 1) * stride)
            .sum::<usize>()
            + 1;
        let handle = client.create_from_slice(&vec![0u8; elements * elem.size()]);
        let meta = Metadata::new(Shape::from(shape.to_vec()), strides);
        Self::of_handle(TensorHandle::from_metadata(handle, meta, elem))
    }

    /// `values` as a row-major `u32` tensor shaped `shape`, uploaded.
    pub fn of_u32s(client: &Client, shape: &[usize], values: &[u32]) -> Self {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let handle = client.create_from_slice(&bytes);
        let meta = Metadata::new(
            Shape::from(shape.to_vec()),
            row_major_contiguous_strides(shape),
        );
        Self::of_handle(TensorHandle::from_metadata(
            handle,
            meta,
            ElemType::UInt(UIntKind::U32),
        ))
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
    /// and its metadata merges them back. The physical shape for a buffer in rows.
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
        let steps_along_a_row = |meta: &Metadata| {
            let dims = fragments(meta);
            let finest = dims.last().and_then(|dim| dim.last());
            finest.is_some_and(|&(_, stride)| stride == 1)
        };
        self.is_tiled() && !steps_along_a_row(&self.meta())
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

    /// This view at `rank` logical dims, the ones it lacks leading ones of one, which broadcast:
    /// the same buffer, in the tiles it is stored in. A view already as long is unchanged.
    pub fn broadcast_to(&self, rank: usize) -> Self {
        let meta = self.meta();
        match meta.is_tiled() {
            false => self.viewed(Self::led(&meta, rank)),
            true => self.viewed(Self::led_in_tiles(&meta, rank)),
        }
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
    /// tile engine stages and masks as it lies ([`MatmulBlocks::restated_dims`]). Its leading
    /// dims merge into the rows, unless they are `batches`, which stay dims of their own ahead of
    /// the block-split ones, for the binding to read as batches (`Arg::batches`), broadcast where
    /// one is a single entry.
    pub fn restated(
        &self,
        blocks: &MatmulBlocks,
        rows: Dimension,
        columns: Dimension,
        batches: &[Dimension],
    ) -> (Self, Vec<Dimension>) {
        if self.is_tiled() {
            return self.restated_in_pieces(blocks, [rows, columns]);
        }
        let (shape, strides) = (&self.binding.shape, &self.binding.strides);
        let kept = batches.len();
        let dense = match kept {
            0 => self.as_rows(blocks.extent(rows), blocks.extent(columns)),
            _ => self.clone(),
        };
        let step = dense.row_step().unwrap_or(blocks.extent(columns));
        let dims = blocks.restated_dims(rows, columns, step);
        let meta = Metadata::new(
            shape[..kept]
                .iter()
                .copied()
                .chain(dims.iter().map(|&(_, extent, _)| extent))
                .collect::<Vec<_>>(),
            strides[..kept]
                .iter()
                .copied()
                .chain(dims.iter().map(|&(_, _, stride)| stride))
                .collect::<Vec<_>>(),
        );
        (
            dense.viewed(meta),
            dims.iter().map(|&(axis, _, _)| axis).collect(),
        )
    }

    /// The logical shape, whatever pieces the buffer is stored in.
    pub fn logical_shape(&self) -> Vec<usize> {
        self.meta()
            .logical_shape()
            .expect("a tensor's tiling describes its own rank")
            .to_vec()
    }

    /// A dim's `pieces`, coarsest first, as those across blocks of `block` values and those within
    /// one, a piece holding the boundary split in two. `None` where they close no block.
    fn split_pieces(pieces: &Fragments, block: usize) -> Option<(Fragments, Fragments)> {
        let mut across = pieces.clone();
        let mut within = Fragments::new();
        let mut left = block;
        while left > 1 {
            let (extent, stride) = across.pop()?;
            let taken = extent.min(left);
            if !extent.is_multiple_of(taken) || !left.is_multiple_of(taken) {
                return None;
            }
            within.insert(0, (taken, stride));
            if extent > taken {
                across.push((extent / taken, stride * taken));
            }
            left /= taken;
        }
        Some((across, within))
    }

    /// [`restated`](Self::restated) of one matrix stored in tiles, its pieces each a piece of a
    /// dimension of the space: a piece that holds a block's boundary is two.
    ///
    /// # Panics
    ///
    /// On a matrix [`readable_in_blocks`](Self::readable_in_blocks) does not return.
    fn restated_in_pieces(
        &self,
        blocks: &MatmulBlocks,
        dimensions: [Dimension; 2],
    ) -> (Self, Vec<Dimension>) {
        let matrix = fragments_of::<2>(&self.meta()).expect("one matrix, stored in tiles");
        let mut pieces: Vec<(Dimension, Fragments)> = Vec::new();
        for (dimension, stored) in dimensions.into_iter().zip(matrix) {
            let [(block, count), (inside, extent)] = blocks.dimensions(dimension);
            let (across, within) = match count {
                1 => (Fragments::new(), stored),
                _ => Self::split_pieces(&stored, extent).expect("tiles a block of K splits"),
            };
            if count > 1 {
                pieces.push((block, across));
            }
            pieces.push((inside, within));
        }
        let mut axes: Vec<(usize, usize)> = pieces
            .iter()
            .flat_map(|(_, pieces)| pieces.iter().copied())
            .collect();
        axes.sort_by_key(|&(extent, stride)| core::cmp::Reverse((stride, extent)));
        let mut listed = vec![false; axes.len()];
        let mut place = |piece: &(usize, usize)| {
            let axis = (0..axes.len())
                .find(|&axis| !listed[axis] && axes[axis] == *piece)
                .expect("every piece is an axis");
            listed[axis] = true;
            axis
        };
        let placed: Vec<Vec<usize>> = pieces
            .iter()
            .map(|(_, pieces)| pieces.iter().map(&mut place).collect())
            .collect();
        let (shape, strides): (Vec<usize>, Vec<usize>) = axes.iter().copied().unzip();
        let meta = Metadata::new(shape, strides)
            .with_tiling(Tiling::from_physical_dims(&placed).expect("each axis is one piece"))
            .expect("the tiling lists the axes");
        (
            self.viewed(meta),
            pieces.iter().map(|&(dimension, _)| dimension).collect(),
        )
    }

    /// This operand where it lies, or a copy of it in rows: a matrix stored in tiles is read in
    /// them where its contraction, its innermost dim, splits into blocks of `block` at its pieces.
    pub fn readable_in_blocks(self, client: &Client, block: Option<usize>) -> Self {
        let splits = match (fragments_of::<2>(&self.meta()), block) {
            (Ok([_, contracted]), Some(block)) => Self::split_pieces(&contracted, block).is_some(),
            (Ok(_), None) => true,
            (Err(_), _) => false,
        };
        match self.is_tiled() && splits && !self.is_tiled_column_first() {
            true => self,
            false => self.into_contiguous(client),
        }
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
        let projection = Projection::direct(&[Dimension(0), Dimension(1)]);
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

    /// This tensor where no two of its indices address one element, a row-major copy of it on
    /// `client` where they do.
    ///
    /// Neither a pitch nor gaps between rows stop a read by strides; the runtime pitches rows whose
    /// bytes are not aligned, so a check on the buffer's size would copy those. A storage-tiled
    /// operand is read through its tiling, stored as a launch reads it, so it stands: a copy would
    /// lay it back in rows on every launch.
    pub fn addressable(self, client: &Client) -> Self {
        match self.is_tiled() || self.is_contiguous() || self.is_nonoverlapping() {
            true => self,
            false => self.into_contiguous(client),
        }
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

    /// This view as `[entries, rows, line]` rows of `line` values in batch entries, read where they
    /// lie: the rows run outward from the line while each dim starts where the ones inside it end,
    /// dims of extent 1 skipped, and the entries take the rest at one stride across. `None` where
    /// they do not lie so: a storage-tiled buffer, a last dim other than `line` or not contiguous,
    /// or entries that do not step evenly.
    pub fn as_batched_rows(&self, line: usize) -> Option<Self> {
        if self.is_tiled() {
            return None;
        }
        let (shape, strides) = (&self.binding.shape, &self.binding.strides);
        let rank = shape.len();
        if shape[rank - 1] != line || (line != 1 && strides[rank - 1] != 1) {
            return None;
        }
        let (mut rows, mut row_end) = (1, line);
        let mut batch: Option<(usize, usize, usize)> = None;
        for (&extent, &stride) in shape[..rank - 1].iter().zip(&strides[..rank - 1]).rev() {
            match (extent, batch.as_mut()) {
                (1, _) => {}
                (_, None) if stride == row_end => {
                    rows *= extent;
                    row_end *= extent;
                }
                (_, None) => batch = Some((extent, stride, stride * extent)),
                (_, Some((entries, _, end))) if stride == *end => {
                    *entries *= extent;
                    *end *= extent;
                }
                (_, Some(_)) => return None,
            }
        }
        let (entries, entry_stride, _) = batch.unwrap_or((1, row_end, row_end));
        Some(self.viewed(Metadata::new(
            [entries, rows, line],
            [entry_stride, line, 1],
        )))
    }

    /// This tensor dense and starting on a whole read of `read_bytes`: as it is where it already
    /// is ([`into_contiguous`](Self::into_contiguous)), a copy into a fresh buffer otherwise.
    ///
    /// What a kernel binding every operand at the width a dense row admits wants: a view starting
    /// partway into a read, one into a larger buffer, binds narrower, a width a tuning key over the
    /// shape cannot see.
    ///
    /// The copy reads one element at a time. cubecl's copy picks its vector width from the shape
    /// and strides alone, not from where the view starts, so a vectorized read of a misaligned
    /// start would issue a misaligned load, which CUDA faults on; the view is handed to it as
    /// `[n, 1]`, whose last dim admits no wider read, and the copy is viewed back in this shape.
    pub fn into_dense_at(self, client: &Client, read_bytes: u64) -> Self {
        let tensor = self.into_contiguous(client);
        if tensor.start_bytes().is_multiple_of(read_bytes.max(1)) {
            return tensor;
        }
        let shape = tensor.binding.shape.clone();
        let elements = shape.iter().product::<usize>();
        let one_a_read = tensor.bound_as(Metadata::new([elements, 1], [1, 1]));
        let copy = cubecl::std::tensor::into_contiguous(client, one_a_read, tensor.elem);
        Self::of_handle(copy).viewed(Metadata::new(
            shape.clone(),
            row_major_contiguous_strides(&shape),
        ))
    }

    /// The step between the rows of a row-bound operand, in stored elements: the stride it really
    /// has; `None` for one of a single dim, which has no step.
    ///
    /// # Panics
    ///
    /// On an innermost stride other than 1: a row is a contiguous line.
    pub(crate) fn row_step(&self) -> Option<usize> {
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

/// Where a tensor lies, as a problem reads it: the alignment its layout gives a line, and the bytes
/// it starts past its buffer's own start, which together bound the width a launch binds it at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub elem_alignment: ElemAlignment,
    pub start_bytes: u64,
}

impl Placement {
    /// Dense rows of `width` elements starting where their buffer does: an output a launch lays
    /// out itself, as a sink's or a cube's stage.
    pub fn dense_rows(width: usize) -> Self {
        Self {
            elem_alignment: ElemAlignment::dense_rows(width),
            start_bytes: 0,
        }
    }
}

impl From<&PlainTensor> for Placement {
    fn from(tensor: &PlainTensor) -> Self {
        Self {
            elem_alignment: tensor.elem_alignment(),
            start_bytes: tensor.start_bytes(),
        }
    }
}

/// What a launch declines on a device whose runtime does not compute in the operands' element and
/// store it in a buffer, as WGSL does not bf16: the kernel would fail to compile there.
pub const ELEMENT_NOT_ON_DEVICE: Decline = Decline::Unserved {
    why: "the device's runtime does not compute in this element and store it in a buffer",
};

impl PlainTensor {
    /// `meta`, a view in rows, at `rank` dims, leading ones of one added.
    fn led(meta: &Metadata, rank: usize) -> Metadata {
        let missing = rank.saturating_sub(meta.shape.len());
        let lead = |dims: &[usize]| [&vec![1; missing][..], dims].concat();
        Metadata::new(lead(&meta.shape), lead(&meta.strides))
    }

    /// `meta`, a view in tiles, at `rank` logical dims, leading ones of one added ahead of its
    /// pieces.
    fn led_in_tiles(meta: &Metadata, rank: usize) -> Metadata {
        let logical = meta
            .logical_rank()
            .expect("a binding's tiling describes its own rank");
        let missing = rank.saturating_sub(logical);
        let lead = |dims: &[usize]| [&vec![1; missing][..], dims].concat();
        let pieces = meta.tiling.physical_dims(logical);
        let dims: Vec<Vec<usize>> = (0..missing)
            .map(|dim| vec![dim])
            .chain(
                pieces
                    .iter()
                    .map(|dim| dim.iter().map(|&piece| piece + missing).collect()),
            )
            .collect();
        let tiling = Tiling::from_physical_dims(&dims).expect("leading dims ahead of its pieces");
        Metadata::new(lead(&meta.shape), lead(&meta.strides))
            .with_tiling(tiling)
            .expect("the tiling lists every dim")
    }
}

#[cfg(test)]
mod tests {
    use super::PlainTensor;
    use cubek_std::layout::Fragments;

    /// `k` of 64 in tiles of 8: eight tiles 128 apart, then eight values.
    fn tiles_of_8() -> Fragments {
        Fragments::from_slice(&[(8, 128), (8, 1)])
    }

    /// Blocks of 32 take the values of a tile and four tiles: the eight tiles split in two.
    #[test]
    fn a_piece_holding_a_block_boundary_is_split_in_two() {
        let (across, within) = PlainTensor::split_pieces(&tiles_of_8(), 32).expect("whole blocks");
        assert_eq!(
            (across.to_vec(), within.to_vec()),
            (vec![(2, 512)], vec![(4, 128), (8, 1)])
        );
    }

    #[test]
    fn blocks_that_cut_a_piece_unevenly_are_none() {
        assert!(PlainTensor::split_pieces(&tiles_of_8(), 12).is_none());
    }
}
