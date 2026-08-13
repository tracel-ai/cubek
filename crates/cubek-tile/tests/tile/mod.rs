mod attention;
mod conv;
mod instruction;
mod launcher;
mod matmul;
mod quant;
mod recursive;
mod reduce;
mod references;
mod softmax;
mod space;

use cubecl::{TestRuntime, ir::ElemType, prelude::*};
use cubek_test_utils::{TestOutcome, ValidationResult};

/// Skip guard for the manual-mma (`Leaf::Mma`) tests.
///
/// The tile mma leaf builds `MmaDefinition::<T, T, T>`, so A, B and the accumulator all share
/// one element type. Checking only that *some* mma config exists is not enough — a device
/// advertises specific `(a, b, cd)` triples, and asking for one it doesn't have is rejected at
/// compile time, the same hazard `require_cmma_8x8x8_f32` documents for cmma.
///
/// Be aware of what this currently skips. No backend registers a uniform triple today: CUDA's
/// f32 accumulator takes `tf32` operands and its f16/bf16 shapes accumulate into f32, HIP
/// registers only f16/bf16 into f32, and Metal registers no manual mma at all (it exposes
/// `cmma` instead, which is why the cmma twins of these tests do run there). So these tests
/// skip everywhere until the leaf can carry a register type distinct from its storage type,
/// the way `MatmulElems` does for cubek-matmul. Before that, they were failing at ptxas on
/// CUDA rather than skipping, which is what this guard fixes.
pub(crate) fn require_uniform_mma(client: &ComputeClient<TestRuntime>, dtype: ElemType) -> bool {
    let supported = client
        .properties()
        .features
        .matmul
        .mma
        .iter()
        .any(|cfg| cfg.a_type == dtype && cfg.b_type == dtype && cfg.cd_type == dtype);

    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device has no uniform-type {dtype} mma (a == b == cd)"
        )))
        .enforce();
    }
    supported
}
