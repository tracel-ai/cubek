//! The strided matmul routine: `out = lhs @ rhs` through the tile engine's register leaf,
//! either operand optionally quantized (native or packed-u32, per-tensor or block scales) —
//! dequantization happens at the staged fill, so the microkernel only ever sees the served
//! float type. The memory-bound decode GEMV is the degenerate `m = 1` case of the same
//! launch.

use cubecl::{
    ir::{ElemType, StorageType},
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore},
};

use crate::{Axis, Leaf, Partitioner, Space};

/// The routine's axis labels. A caller's [`Partitioner`] must cut these — they are what
/// [`matmul`] labels the space with.
pub const M: Axis = Axis(0);
/// See [`M`].
pub const N: Axis = Axis(1);
/// See [`M`].
pub const K: Axis = Axis(2);

/// One matmul operand: its binding plus the optional quantization side-channel. A quantized
/// binding's shape/strides are declared **in values** (a packed store's buffer is narrower
/// than its shape by the scheme's packing factor).
pub struct Operand<R: Runtime> {
    binding: TensorBinding<R>,
    quant: Option<(TensorBinding<R>, QuantScheme)>,
}

impl<R: Runtime> Operand<R> {
    pub fn plain(binding: TensorBinding<R>) -> Self {
        Operand {
            binding,
            quant: None,
        }
    }

    pub fn quantized(
        binding: TensorBinding<R>,
        scales: TensorBinding<R>,
        scheme: QuantScheme,
    ) -> Self {
        Operand {
            binding,
            quant: Some((scales, scheme)),
        }
    }

    /// The scheme's packing factor: served values per stored element, `1` when plain or
    /// native.
    fn pack(&self) -> usize {
        self.quant
            .as_ref()
            .map(|(_, scheme)| scheme.num_quants())
            .unwrap_or(1)
    }

    /// Reject a dtype this operand cannot be bound at. A plain operand is served
    /// unconverted, so its dtype must *be* the served (out) type — the binding is downcast,
    /// not converted, and a mismatch is a silent bit-reinterpretation. A quantized operand's
    /// dtype is its storage element, which the scheme dictates.
    fn validate_dtype(&self, name: &str, dtype: StorageType, served: StorageType) {
        match &self.quant {
            None => assert!(
                dtype == served,
                "routine::matmul: a plain {name} is served unconverted, so its dtype \
                 ({dtype:?}) must be the out dtype ({served:?})"
            ),
            Some((_, scheme)) => {
                let storage = match scheme.store {
                    QuantStore::PackedU32(_) => {
                        StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32))
                    }
                    _ => StorageType::Scalar(ElemType::from_quant_value(scheme.value)),
                };
                assert!(
                    dtype == storage,
                    "routine::matmul: a quantized {name}'s dtype names its storage element — \
                     {storage:?} for {:?}/{:?}, got {dtype:?}",
                    scheme.value,
                    scheme.store
                );
            }
        }
    }
}

/// `out = lhs @ rhs`: `lhs` is `(m, k)`, `rhs` is `(k, n)`, `out` is `(m, n)`, all
/// row-major-contiguous innermost. `partitioner` is the caller's plan over [`M`]/[`N`]/[`K`]
/// (the blueprint — a selector derives it from the problem and device); the routine derives
/// the vector widths and cube geometry and launches the engine kernel.
///
/// `dtypes` is the operands' storage types `[lhs, rhs, out]`: a quantized operand names its
/// *storage* element (`u32` for packed), a plain one must name `out`'s type (it is served
/// unconverted).
///
/// # Panics
///
/// Every invariant the engine can only refuse at expand time — where a panic is swallowed by
/// the compile thread and reads as zeroed output — is validated here on the caller's thread:
///
/// * a quantized operand with a plan that never stages (dequantization happens at a stage's
///   fill; the leaf's direct matrix reads refuse a storage-typed tile);
/// * a non-[`Register`](Leaf::Register) leaf (this routine does not promote an accumulator);
/// * a packed operand whose packing factor exceeds the device's vector width (a served line
///   covers whole words, and its stage allocates that wide);
/// * a dtype that isn't what the operand is truly bound at (plain ⇒ the out type, quantized
///   ⇒ the scheme's storage element) — a mismatch is a silent bit-reinterpretation;
/// * plus the launch-layer checks (scheme support, divisibility, scale straddling).
pub fn matmul<R: Runtime>(
    client: &ComputeClient<R>,
    (m, n, k): (usize, usize, usize),
    partitioner: Partitioner,
    lhs: Operand<R>,
    rhs: Operand<R>,
    out: TensorBinding<R>,
    dtypes: [StorageType; 3],
) {
    let quantized = lhs.quant.is_some() || rhs.quant.is_some();
    assert!(
        !quantized || partitioner.stages(),
        "routine::matmul: a quantized operand requires a staged level — dequantization happens \
         at the stage's fill, and an all-direct plan reaches the leaf's matrix reads still \
         storage-typed (an expand-time refusal, which a launch swallows into zeroed output)"
    );
    assert!(
        partitioner.leaf() == Leaf::Register,
        "routine::matmul: this routine runs the register (plain-ALU) leaf and never promotes \
         an accumulator; a cmma-leaf plan needs the tensor-core routine"
    );
    lhs.validate_dtype("lhs", dtypes[0], dtypes[2]);
    rhs.validate_dtype("rhs", dtypes[1], dtypes[2]);

    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);
    let launcher = space.launcher(client);

    // Served widths. A packed operand serves whole words — `pack` values per line — which is
    // also what its smem stage allocates, so the device's vector width must reach it. A plain
    // or native operand takes the widest line the bindings sharing its innermost axis admit.
    // The register microkernel lines the accumulator at the RHS's width, so `rhs` and `out`
    // share one.
    let max_width = client.properties().hardware.max_vector_size;
    let type_size = |ty: StorageType| ty.size();
    let wl = match lhs.pack() {
        1 => launcher.vector_size(K, &[(&lhs.binding, &[M, K])], type_size(dtypes[0])),
        pack => {
            assert!(
                pack <= max_width,
                "routine::matmul: lhs packs {pack} values per word, but the device's vectors \
                 cap at {max_width} — a served line covers whole words"
            );
            pack
        }
    };
    let wr = match rhs.pack() {
        1 => launcher
            .vector_size(N, &[(&rhs.binding, &[K, N])], type_size(dtypes[1]))
            .min(launcher.vector_size(N, &[(&out, &[M, N])], type_size(dtypes[2]))),
        pack => {
            assert!(
                pack <= max_width,
                "routine::matmul: rhs packs {pack} values per word, but the device's vectors \
                 cap at {max_width} — a served line covers whole words"
            );
            pack
        }
    };

    let cube_count = launcher.cube_count();
    let cube_dim = launcher.cube_dim();

    let mut lhs_arg = launcher
        .arg(lhs.binding)
        .subspace(&[M, K])
        .vectorize(wl)
        .build();
    if let Some((scales, scheme)) = lhs.quant {
        lhs_arg = lhs_arg.quantized(scales.into_tensor_arg(), scheme);
    }
    let mut rhs_arg = launcher
        .arg(rhs.binding)
        .subspace(&[K, N])
        .vectorize(wr)
        .build();
    if let Some((scales, scheme)) = rhs.quant {
        rhs_arg = rhs_arg.quantized(scales.into_tensor_arg(), scheme);
    }
    let out_arg = launcher.arg(out).subspace(&[M, N]).vectorize(wr).build();

    matmul_kernel::launch::<R>(
        client, cube_count, cube_dim, lhs_arg, rhs_arg, out_arg, dtypes,
    );
}

/// The engine kernel behind [`matmul`]: both operands served as `EO` through
/// [`tile_dequant`](crate::StridedTileArg::tile_dequant), so the same kernel runs plain
/// (`EL == EO`, no side-channel) or quantized — the trace only ever sees floats past the
/// fill. The gmem accumulator keeps [`mma`](crate::Tile) on the register leaf.
#[cube(launch)]
fn matmul_kernel<EL: Numeric, ER: Numeric, EO: Numeric>(
    lhs: &crate::StridedTileArg<'_, EL>,
    rhs: &crate::StridedTileArg<'_, ER>,
    out: &crate::StridedTileArg<'_, EO>,
    #[define(EL, ER, EO)] _dtypes: [StorageType; 3],
) {
    let a = lhs.tile_dequant::<EO>();
    let b = rhs.tile_dequant::<EO>();
    let mut c = out.tile();
    c.mma(&a, &b);
}
