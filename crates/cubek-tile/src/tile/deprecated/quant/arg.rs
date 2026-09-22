//! A quantized operand as a single launch argument: values, scales, spec and scheme together.

use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantScheme, QuantStore, ScaleDtype};
use cubecl::std::quant::view::KnownScale;
use cubecl::std::tensor::layout::linear::{LinearView, linear_view};

use crate::*;

/// One quantized operand as a single launch argument: the storage-typed values tensor, its scales,
/// and the comptime spec + scheme. A quantized tensor is one thing, so its pieces travel together;
/// [`TileArg`] is its plain twin; [`tile`](QuantTileArg::tile) projects the kernel's [`Space`].
#[derive(CubeType, CubeLaunch)]
pub struct QuantTileArg<'a, E: Numeric, V: Size> {
    pub values: &'a Tensor<Vector<E, V>>,
    pub scales: &'a Tensor<f32>,
    /// The global level's whole-tensor scale in its first element, bound exactly when the scheme
    /// has a second level.
    pub global: ComptimeOption<LinearView<'static, f32>>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode): reads reconstruct
    /// `table[field] * scale` instead of casting the field.
    pub table: ComptimeOption<Box<[f32]>>,
    #[cube(comptime)]
    pub spec: TileSpec,
    #[cube(comptime)]
    pub scheme: QuantScheme,
    /// How far this operand stays quantized; stated at launch, where what can decode is known.
    #[cube(comptime)]
    pub dequant_at: DequantAt,
}

#[cube]
impl<'a, E: Numeric, V: Size> QuantTileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`] of the served type `O`: the kernel's one `space`
    /// projected onto this operand's `spec` axes, reads dequantizing per the scheme.
    pub fn tile<O: Numeric>(&self, #[comptime] space: Partitioning) -> Tile<O> {
        // The engine's own backstop, like `validate_dequant_at`: the builder checks the binding
        // contract too, but a hand-built `QuantTileArgLaunch` reaches here without it.
        comptime!(cubecl::std::quant::check_scale_bindings(
            &self.scheme,
            1 + self.global.is_some() as usize
        ));
        // One read for the whole kernel; every window below shares the register.
        let known = if comptime!(self.global.is_some()) {
            let buffer = self.global.unwrap();
            KnownScale::new_Global(buffer.read(0))
        } else {
            KnownScale::new_None()
        };
        dequant_operand::<O, Vector<E, V>>(
            self.values,
            self.scales,
            known,
            self.table.clone(),
            comptime!(self.scheme),
            comptime!(self.dequant_at),
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .tile(comptime!(space.levels().to_vec()))
    }
}

/// Reject a [`QuantScheme`] this operand cannot serve, at launch and on the caller's thread: a
/// kernel-side assert would fire on a device thread and read as zeroed output rather than a
/// rejection, so this is the one gate.
///
/// A line is one read, so it may not straddle two scale blocks: the innermost block must be a
/// multiple of the served width. Whether a tile straddles a block is the blueprint's to refuse,
/// where the tiles are decided.
pub(crate) fn validate_scheme(space: &Space, vector_size: usize, scheme: QuantScheme) {
    // `Native` holds one element per value; `PackedU32` carries `num_quants` of them per `u32`,
    // which the view unpacks on read. A packed store must pack along the innermost (contiguous,
    // vectorized) axis, whose lanes the view lays down contiguously. Sub-byte native isn't wired.
    match scheme.store {
        QuantStore::Native => {}
        QuantStore::PackedU32(dim) => {
            assert!(
                dim == 0,
                "a packed-u32 quantized operand must pack along the innermost axis (dim 0), got {dim}"
            );
            assert!(
                vector_size.is_multiple_of(scheme.num_quants()),
                "a quantized operand's innermost axis is served in {vector_size}-wide lines, which must \
                 be a multiple of the {}-value packing factor, else a line splits a u32",
                scheme.num_quants()
            );
        }
        other => panic!("quantization storage {other:?} is not supported (native or packed-u32)"),
    }
    // The scales ride a plain `f32` tensor read straight through, so a narrower param
    // would reinterpret its bytes.
    assert!(
        scheme.scale_dtype() == ScaleDtype::F32,
        "a quantized operand's scales are read as f32, got {:?}",
        scheme.scale_dtype()
    );

    // A per-tensor scale has no edge to straddle, whatever the cuts and the served width.
    if scheme.block_size().is_none() {
        return;
    }

    let rank = space.rank();
    let block = block_edges(scheme, rank);
    let inner = block[rank - 1];
    assert!(
        inner.is_multiple_of(vector_size),
        "a quantized operand's innermost axis is served in {vector_size}-wide lines, which its \
         {inner}-element scale blocks must be a multiple of, else one line straddles two scales"
    );
}

/// How an operand is quantized: the scales beside its values, the scheme saying how to fold them
/// back in, and how far the quantized form travels before something decodes it. One thing, because
/// none says anything alone: scales need a scheme, a [`DequantAt`] without one bounds nothing.
pub struct Quantization {
    /// The innermost level's scales, the only ones addressed per position.
    pub scales: TensorArg,
    /// The global level's scale, one for the whole tensor, present exactly when the scheme has a
    /// second level ([`validate`](Self::validate) holds the two together).
    pub global: Option<TensorBinding>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode);
    /// [`validate`](Self::validate) holds the two together.
    pub table: Option<BufferArg>,
    pub scheme: QuantScheme,
    pub dequant_at: DequantAt,
}

impl Quantization {
    /// `inner` holds the innermost level's scales, addressed per position; `global` the whole
    /// tensor's, read once from its first element, present exactly when the scheme has a second
    /// level ([`validate`](Self::validate) holds the two together).
    pub fn new(
        inner: TensorBinding,
        global: Option<TensorBinding>,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> Self {
        Quantization {
            scales: inner.into_tensor_arg(),
            global,
            table: None,
            scheme,
            dequant_at,
        }
    }

    /// [`new`](Self::new) for a lookup scheme, where a read reconstructs `table[field] * scale`
    /// ([`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode)). `table` holds `2^bits` f32
    /// entries, unchecked here: the unpack's mask bounds every index.
    pub fn lookup(
        scales: TensorArg,
        table: BufferArg,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> Self {
        Quantization {
            scales,
            global: None,
            table: Some(table),
            scheme,
            dequant_at,
        }
    }

    /// Values per stored element: `1` unless the scheme packs several into each.
    pub fn num_quants(&self) -> usize {
        self.scheme.num_quants()
    }

    /// Refuse what this quantization cannot serve, on the caller's thread: the scheme against the
    /// operand's cuts and served width. Where the [`DequantAt`] can be honoured is the fragment
    /// load's to say, at the kernel's own call.
    /// This quantization against the operand it rides: a gathered operand cannot be quantized
    /// (its scale grid is shaped over its logical axes, which its buffer's dims no longer match),
    /// and the scheme must fit the served width and the operand's space.
    pub(crate) fn check(
        &self,
        spec: &TileSpec,
        space: &Space,
        width: usize,
    ) -> Result<(), Refusal> {
        if !spec.projection.untiled().is_direct() {
            return Err(Refusal::QuantizedGather);
        }
        self.validate(&space.subspace(spec.axes()), width);
        Ok(())
    }

    fn validate(&self, space: &Space, vector_size: usize) {
        cubecl::std::quant::check_scale_bindings(&self.scheme, 1 + self.global.is_some() as usize);
        validate_scheme(space, vector_size, self.scheme);
        cubecl::std::quant::check_table_bindings(&self.scheme, self.table.is_some());
    }

    /// The operand as the kernel's [`QuantTileArg`] launch argument: values, scales, spec and
    /// scheme as one thing.
    pub(crate) fn arg<E: Numeric, V: Size>(
        self,
        tensor: TensorArg,
        spec: TileSpec,
    ) -> QuantTileArgLaunch<'static, E, V> {
        QuantTileArgLaunch::new(
            tensor,
            self.scales,
            self.global.map(linear_view).into(),
            self.table.into(),
            spec,
            self.scheme,
            self.dequant_at,
        )
    }
}

/// A quantized operand: the values tensor is storage-typed (`u32` words
/// for a packed scheme, `i8` native), the scales ride as a plain second tensor, and the scheme
/// says how reads fold them back in. The served width is the binding's × the packing factor.
#[cube]
#[allow(clippy::too_many_arguments)]
fn dequant_operand<T: Numeric, E: CubePrimitive>(
    values: &Tensor<E>,
    scales: &Tensor<f32>,
    known: KnownScale,
    table: ComptimeOption<Box<[f32]>>,
    #[comptime] scheme: QuantScheme,
    #[comptime] dequant_at: DequantAt,
    #[comptime] space: Space,
    #[comptime] spec: TileSpec,
) -> GlobalOperand<T> {
    comptime!(cubecl::std::quant::check_table_bindings(
        &scheme,
        table.is_some()
    ));
    let rank = comptime!(spec.axes().len());
    let block = comptime!(block_edges(scheme, rank));
    let mut strides = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(scheme.block_size().is_none()) {
            strides.push(0u32);
        } else {
            strides.push(scales.stride(p) as u32);
        }
    }
    let info = QuantInfo {
        buffer: unsafe { scales.as_slice().as_boxed_unchecked() },
        known,
        strides,
        window_start: 0u32,
        block: comptime!(block),
        extent: comptime!(window_extents(&space.subspace(spec.axes()), rank)),
        dequant_at: comptime!(dequant_at),
        // A gmem operand reads the tensor's scales in place; only a staged stage grids them.
        scale_shape: comptime!(Vec::new()),
        table,
        scheme: comptime!(scheme),
    };
    GlobalOperand::<T>::of_tensor::<E>(
        values,
        space,
        spec,
        ComptimeOption::new_Some(info),
        Coords::<u32>::new(),
        Coords::<i32>::new(),
    )
}
