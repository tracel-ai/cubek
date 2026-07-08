//! CPU GEMM on *tiled* (packed) storage vs the plain row-major baseline.
//!
//! The generic `gemm` bench only ever feeds plain row/col-major buffers, so the tiled path
//! (`levels > 0`) is unmeasured. This category drives `cpu_gemm::launch_ref` directly with tiled
//! operands. Every strategy fixes the register-fit leaf and plane grid and varies only the storage
//! packing, so `strided_pN` vs `tiled_pN` isolates the storage layout.
//!
//! Finding: with tiled reads vectorized (`launch.rs::vectorizes_n` no longer gates on `levels == 0`)
//! the tiled path is ~10× faster than when it was scalar, but still ~1.5× *slower* than strided and
//! **insensitive to block size** (a 16²→256² sweep was flat) — it is addressing-bound on the
//! `TiledViewLayout` coord math, never reaching the memory wall. Storage packing buys no locality
//! here; cache locality is a compute-tiling (schedule) matter, not a storage-reblocking one. Kept as
//! the evidence for that + a regression guard on the vectorized tiled path.

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_std::InputBinding;
use cubek_test_utils::{CatalogEntry, RunSamples, TestInput};

use crate::definition::MatmulElems;
use crate::routines::BlueprintStrategy;
use crate::routines::cpu_gemm::{CpuGemmBlueprint, Instruction, PlaneGrid, WithLayout, launch_ref};

/// The register-fit leaf shared by every strategy: the optimized `2 × 32 × 64` microkernel (no
/// spill), so the only variable across a `strided_pN`/`tiled_pN` pair is the storage packing.
const LEAF: Instruction = Instruction { m: 2, n: 32, k: 64 };

/// Storage-tile edge for the packed variants: square `64 × 64` blocks (16 KiB in f32, L1-resident)
/// that divide every benchmarked shape. A sweep of 16²→256² was flat, so one representative edge is
/// enough (see the module doc).
const EDGE: usize = 64;

/// How an operand's matrix axes are physically stored.
#[derive(Clone, Copy)]
enum Packing {
    /// Plain row-major (`levels = 0`) — the vectorized reference path.
    RowMajor,
    /// One level of square `edge × edge` storage tiles (`levels = 1`) — the packed path under test.
    Tiled { edge: usize },
}

impl Packing {
    /// Storage-tiling depth: `0` strided, `1` for a single level of tiles.
    fn levels(self) -> usize {
        match self {
            Packing::RowMajor => 0,
            Packing::Tiled { .. } => 1,
        }
    }

    /// Physical buffer dims for a logical `(batch, rows, cols)` operand. Row-major keeps the logical
    /// shape; tiled expands each matrix axis into its `[grid, tile]` pair, level-major, so the load
    /// builder reads it back as `levels = 1`.
    fn physical_dims(self, batch: usize, rows: usize, cols: usize) -> Vec<usize> {
        match self {
            Packing::RowMajor => vec![batch, rows, cols],
            Packing::Tiled { edge } => vec![batch, rows / edge, cols / edge, edge, edge],
        }
    }
}

#[derive(Clone, Copy)]
pub struct TiledProblem {
    b: usize,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy)]
pub struct TiledStrategy {
    packing: Packing,
    planes: PlaneGrid,
}

/// A fresh uniform-random operand of the given physical `packing`; a contiguous buffer whose
/// row-major strides already realize the layout (tiled dims are just higher rank).
fn make(
    client: &ComputeClient<TestRuntime>,
    packing: Packing,
    batch: usize,
    rows: usize,
    cols: usize,
    dtype: StorageType,
    seed: u64,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(
        client.clone(),
        Shape::from(packing.physical_dims(batch, rows, cols)),
    )
    .dtype(dtype)
    .uniform(seed, 0.0, 1.0)
    .generate_without_host_data()
}

struct TiledBench {
    problem: TiledProblem,
    strategy: TiledStrategy,
    client: ComputeClient<TestRuntime>,
    dtypes: MatmulElems,
    samples: usize,
}

impl Benchmark for TiledBench {
    type Input = (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let TiledProblem { b, m, n, k } = self.problem;
        let packing = self.strategy.packing;
        let lhs = make(&self.client, packing, b, m, k, self.dtypes.lhs_global, 0);
        let rhs = make(&self.client, packing, b, k, n, self.dtypes.rhs_global, 1);
        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let TiledProblem { b, m, n, k: _ } = self.problem;
        let packing = self.strategy.packing;
        let levels = packing.levels();
        let out = TensorHandle::empty(
            &self.client,
            packing.physical_dims(b, m, n),
            self.dtypes.acc_global,
        );

        launch_ref::<TestRuntime>(
            &self.client,
            WithLayout {
                binding: InputBinding::Normal(lhs.binding(), self.dtypes.lhs_global),
                levels,
            },
            WithLayout {
                binding: InputBinding::Normal(rhs.binding(), self.dtypes.rhs_global),
                levels,
            },
            WithLayout {
                binding: out.binding(),
                levels,
            },
            &BlueprintStrategy::Forced(CpuGemmBlueprint {
                instruction: LEAF,
                planes: self.strategy.planes,
            }),
            &self.dtypes,
        )
        .map_err(|err| format!("{err:?}"))?;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let packing = match self.strategy.packing {
            Packing::RowMajor => "strided".to_string(),
            Packing::Tiled { edge } => format!("tiled{edge}"),
        };
        let planes = self.strategy.planes;
        format!(
            "{}-cpu-gemm-tiled-{}-p{}x{}",
            <TestRuntime as Runtime>::name(&self.client),
            packing,
            planes.m,
            planes.n,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "cpu-gemm-tiled-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

pub fn bench(
    strategy: &TiledStrategy,
    problem: &TiledProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let flops = 2.0 * problem.b as f64 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let bench = TiledBench {
        problem: *problem,
        strategy: *strategy,
        client,
        dtypes: MatmulElems::from_single_dtype(f32::as_type_native_unchecked()),
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_flops(flops))
}

/// Square-ish shapes whose dims all divide `EDGE`, so both packings run maskless. `1536³` carries a
/// factor of 3 so its plane grid can split evenly across a 12-core machine.
const SHAPES: &[(&str, &str, usize, usize, usize, usize)] = &[
    ("rect_1x512", "512³", 1, 512, 512, 512),
    ("square_2x1024", "b=2 1024³", 2, 1024, 1024, 1024),
    ("square_1x1536", "1536³", 1, 1536, 1536, 1536),
];

/// Worker-thread grids, 1 → 16, mirroring the `gemm_cpu` fast-core ladder.
const PLANE_LADDER: &[(&str, PlaneGrid)] = &[
    ("p1", PlaneGrid { m: 1, n: 1 }),
    ("p4", PlaneGrid { m: 2, n: 2 }),
    ("p8", PlaneGrid { m: 4, n: 2 }),
    ("p16", PlaneGrid { m: 8, n: 2 }),
];

pub fn problems() -> Vec<CatalogEntry<TiledProblem>> {
    SHAPES
        .iter()
        .map(|&(tag, label, b, m, n, k)| CatalogEntry::new(tag, label, TiledProblem { b, m, n, k }))
        .collect()
}

/// For each thread count: the strided (vectorized row-major) baseline and the packed variant at the
/// same leaf/planes, so `strided_pN` vs `tiled_pN` isolates the storage packing.
pub fn strategies() -> Vec<CatalogEntry<TiledStrategy>> {
    let mut out = Vec::new();
    for &(tag, planes) in PLANE_LADDER {
        out.push(CatalogEntry::new(
            format!("strided_{tag}"),
            format!("Strided (row-major, {tag})"),
            TiledStrategy {
                packing: Packing::RowMajor,
                planes,
            },
        ));
        out.push(CatalogEntry::new(
            format!("tiled_{tag}"),
            format!("Tiled ({EDGE}² blocks, {tag})"),
            TiledStrategy {
                packing: Packing::Tiled { edge: EDGE },
                planes,
            },
        ));
    }
    out
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = TiledProblem;
    type Strategy = TiledStrategy;

    fn id(&self) -> &'static str {
        "gemm_cpu_tiled"
    }

    fn label(&self) -> &'static str {
        "GEMM (CPU, tiled storage)"
    }

    fn problems(&self) -> Vec<CatalogEntry<TiledProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<TiledStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &TiledStrategy,
        problem: &TiledProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
