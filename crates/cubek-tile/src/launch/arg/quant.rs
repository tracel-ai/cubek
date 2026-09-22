//! A quantized operand as a single launch argument: values, scales, spec and scheme together.

use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantScheme, QuantStore, ScaleDtype};
use cubecl::std::quant::view::KnownScale;
use cubecl::std::tensor::layout::linear::LinearView;

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
        Tile::<O>::of_dequant(
            self.values,
            self.scales,
            known,
            self.table.clone(),
            comptime!(self.scheme),
            comptime!(self.dequant_at),
            comptime!(space.space().clone()),
            comptime!(self.spec.clone()),
        )
        .under(comptime!(space.levels().to_vec()))
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
