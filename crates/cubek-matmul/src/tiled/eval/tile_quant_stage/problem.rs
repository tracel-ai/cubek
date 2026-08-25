use cubek_test_utils::CatalogEntry;

/// One `C = A · dequant(B)` shape, `B` the packed weight. `m` is the reuse factor: a cube stages one
/// B tile and reuses it across all `m` rows, so `m = 1` is the decode path the packed stage targets.
pub struct TileQuantStageProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Scale block along the packed axis. Staged scales cost `pack / bn` of the staged values, so
    /// `32`/`128` bracket production group sizes.
    pub bn: usize,
}

pub fn problems() -> Vec<CatalogEntry<TileQuantStageProblem>> {
    vec![
        CatalogEntry::new(
            "gemv_4096_bn32",
            "gemv m=1 n=k=4096 bn=32",
            TileQuantStageProblem {
                m: 1,
                n: 4096,
                k: 4096,
                bn: 32,
            },
        ),
        CatalogEntry::new(
            "gemv_4096_bn128",
            "gemv m=1 n=k=4096 bn=128",
            TileQuantStageProblem {
                m: 1,
                n: 4096,
                k: 4096,
                bn: 128,
            },
        ),
        CatalogEntry::new(
            "gemm_m8_4096_bn128",
            "gemm m=8 n=k=4096 bn=128",
            TileQuantStageProblem {
                m: 8,
                n: 4096,
                k: 4096,
                bn: 128,
            },
        ),
    ]
}
