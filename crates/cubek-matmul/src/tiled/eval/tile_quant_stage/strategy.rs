use cubek_test_utils::CatalogEntry;

/// Stage depth: `tk` rows of the weight staged per region. The packed stage costs `tk · tn` bytes
/// against `4 · tk · tn` dequantized, so depth is what the packing buys — a dequantized stage hits
/// the shared-memory ceiling ~4x sooner. Deep enough and even the packed stage overruns it, which is
/// why the sweep runs past the knee.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct StageDepth(pub usize);

pub fn strategies() -> Vec<CatalogEntry<StageDepth>> {
    [32usize, 64, 128, 256]
        .into_iter()
        .map(|tk| {
            CatalogEntry::new(
                format!("tk{tk}"),
                format!("stage depth tk={tk}"),
                StageDepth(tk),
            )
        })
        .collect()
}
