use cubek_test_utils::CatalogEntry;

use crate::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

/// One dequantize launch: an `[m, n]` q8s tensor restored to f32 under `scheme`.
pub struct DequantizeProblem {
    pub m: usize,
    pub n: usize,
    pub scheme: QuantScheme,
}

/// One decoder-weight-sized matrix for every case: the op is memory-bound, so the sweep
/// varies the scale layout and the store, not the shape.
const M: usize = 4096;
const N: usize = 4096;

fn q8s(store: QuantStore) -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_store(store)
}

fn problem(scheme: QuantScheme) -> DequantizeProblem {
    DequantizeProblem { m: M, n: N, scheme }
}

pub fn problems() -> Vec<CatalogEntry<DequantizeProblem>> {
    vec![
        CatalogEntry::new(
            "native_tensor",
            "native q8s per-tensor 4096x4096",
            problem(q8s(QuantStore::Native).per_tensor(ScaleDtype::F32)),
        ),
        CatalogEntry::new(
            "native_block32",
            "native q8s [32]-block 4096x4096",
            problem(q8s(QuantStore::Native).per_block([32], ScaleDtype::F32)),
        ),
        CatalogEntry::new(
            "packed_block32",
            "packed-u32 q8s [32]-block 4096x4096",
            problem(q8s(QuantStore::PackedU32(0)).per_block([32], ScaleDtype::F32)),
        ),
        CatalogEntry::new(
            "two_level_block16",
            "native q8s [16]-block + tensor 4096x4096",
            problem(
                q8s(QuantStore::Native)
                    .per_block([16], ScaleDtype::F32)
                    .per_tensor(ScaleDtype::F32),
            ),
        ),
    ]
}
