//! The Cmma routine on a block-stored weight vs the same plan on a row-major one.
//!
//! The hypothesis under test: a `[stage_k x stage_n]` stage read from a row-major weight touches
//! `stage_k` short runs, half of every cache line wasted at a narrow stage; packed into blocks of
//! exactly the plan's stage, the stage is one contiguous run. Prefill shapes only (a tall `M`
//! against a square-ish weight): decode is already at the bandwidth slope.
//!
//! Every strategy runs the plan the selector picks for the problem, so the storage of the weight
//! is the only variable. The row-major plan is listed twice, first and last: the machine drifts
//! thermally, and two readings of the same row bracket the drift the middle row sits in.

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    ir::FloatKind,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::{Shape, Tiling, shape},
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{CatalogEntry, CategoryWork, ComputeWork, RunSamples, TestInput, client};

use crate::{
    definition::{AvailableVectorSizes, MatmulCost, MatmulElems, MatmulGlobalElems, MatmulProblem},
    routine::{BlueprintStrategy, DeviceSettings},
    tiled::cmma::{CmmaBlueprint, CmmaRoutine, CmmaStrategy, launch_ref},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    fn dtype(self) -> ElemType {
        match self {
            Precision::F32 => f32::elem_type_native(),
            Precision::F16 => half::f16::elem_type_native(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct BlockProblem {
    m: usize,
    n: usize,
    k: usize,
    precision: Precision,
}

/// How the weight is stored, which is the only thing a strategy varies.
#[derive(Clone, Copy)]
pub enum Storage {
    /// Plain `[k, n]`, the strided delivery.
    RowMajor,
    /// `[k / stage_k, n / stage_n, stage_k, stage_n]`, blocks of exactly the plan's stage, the
    /// Block delivery.
    Block,
}

#[derive(Clone, Copy)]
pub struct BlockStrategy {
    storage: Storage,
}

/// The plan the selector picks for `problem` under `strategy`'s delivery: the same geometry
/// either way, so the two storages are compared on one plan.
fn plan(
    client: &Client,
    problem: &MatmulProblem,
    dtype: ElemType,
    strategy: CmmaStrategy,
) -> Result<CmmaBlueprint, String> {
    let acc = match dtype {
        ElemType::Float(FloatKind::F16 | FloatKind::BF16) => f32::elem_type_native(),
        other => other,
    };
    let sz = dtype.size();
    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: client.properties().hardware.plane_size_max,
        vector_sizes: AvailableVectorSizes::from_type_sizes(client, sz, sz, sz)
            .pick_max()
            .map_err(|e| format!("{e:?}"))?,
        max_cube_count: client.properties().hardware.max_cube_count,
    };
    CmmaRoutine::blueprint(
        &BlueprintStrategy::Inferred(strategy),
        problem,
        &device_settings,
        acc,
    )
    .map_err(|e| format!("{e:?}"))
}

struct BlockBench {
    problem: BlockProblem,
    storage: Storage,
    blueprint: CmmaBlueprint,
    client: Client,
    dtypes: MatmulElems,
    samples: usize,
}

impl Benchmark for BlockBench {
    type Input = (TensorHandle, TensorHandle);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let BlockProblem { m, n, k, precision } = self.problem;
        let dtype = precision.dtype();
        let lhs = TestInput::builder(self.client.clone(), shape![m, k])
            .dtype(dtype)
            .uniform(0, 0.0, 1.0)
            .generate_without_host_data();
        // The weight, packed at load: its contents do not matter to the timing, only its layout.
        let (stage_m, stage_n) = self.blueprint.stage();
        let _ = stage_m;
        let stage_k = self.blueprint.stage_k;
        let rhs_dims = match self.storage {
            Storage::RowMajor => vec![k, n],
            Storage::Block => vec![k / stage_k, n / stage_n, stage_k, stage_n],
        };
        let rhs = TestInput::builder(self.client.clone(), Shape::from(rhs_dims))
            .dtype(dtype)
            .uniform(1, 0.0, 1.0)
            .generate_without_host_data();
        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let BlockProblem { m, n, .. } = self.problem;
        let dtype = self.problem.precision.dtype();
        let out = TensorHandle::empty(&self.client, shape![m, n], dtype);
        let mut rhs = rhs.binding();
        if let Storage::Block = self.storage {
            rhs.tiling = Tiling::new(&[2, 2]).expect("two matrix dims, two fragments each");
        }
        launch_ref(
            &self.client,
            InputBinding::Normal(lhs.binding(), dtype),
            InputBinding::Normal(rhs, dtype),
            out.binding(),
            &BlueprintStrategy::Forced(self.blueprint.clone()),
            &self.dtypes,
        )
        .map_err(|err| format!("{err:?}"))?;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        let storage = match self.storage {
            Storage::RowMajor => "rowmajor",
            Storage::Block => "block",
        };
        let BlockProblem { m, n, k, precision } = self.problem;
        let precision = match precision {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
        };
        format!(
            "{}-gemm-block-{storage}-{m}x{n}x{k}-{precision}",
            self.client.name()
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let (launched, duration) = self
            .client
            .profile(|| self.execute(args), "gemm-block-bench")
            .map_err(|err| format!("{err:?}"))?;
        launched.map(|_| duration)
    }
}

pub fn bench(
    strategy: &BlockStrategy,
    problem: &BlockProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();
    let dtype = problem.precision.dtype();
    let elems = MatmulElems::from_single_dtype(dtype);
    let matmul = MatmulProblem::from_parameters(
        problem.m,
        problem.n,
        problem.k,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        elems.as_global_elems(),
        cubecl::ir::AddressType::U32,
    );
    let delivery = match strategy.storage {
        Storage::RowMajor => CmmaStrategy::default(),
        Storage::Block => CmmaStrategy::block(),
    };
    let blueprint = plan(&client, &matmul, dtype, delivery)?;

    let bench = BlockBench {
        problem: *problem,
        storage: strategy.storage,
        blueprint,
        client: client.clone(),
        dtypes: elems,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::System)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations))
}

/// Prefill shapes: a batch of tokens against a square-ish weight, at the widths a decoder's
/// projections have.
const SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    (
        "prefill_512x4096x4096",
        "512 tokens, 4096x4096",
        512,
        4096,
        4096,
    ),
    (
        "prefill_2048x4096x4096",
        "2048 tokens, 4096x4096",
        2048,
        4096,
        4096,
    ),
    (
        "prefill_512x12288x4096",
        "512 tokens, 12288x4096 (qkv)",
        512,
        12288,
        4096,
    ),
    (
        "prefill_2048x4096x11008",
        "2048 tokens, 4096x11008 (down)",
        2048,
        4096,
        11008,
    ),
];

const PRECISIONS: &[(&str, Precision)] = &[("f16", Precision::F16), ("f32", Precision::F32)];

pub fn problems() -> Vec<CatalogEntry<BlockProblem>> {
    SHAPES
        .iter()
        .flat_map(|&(tag, label, m, n, k)| {
            PRECISIONS.iter().map(move |&(suffix, precision)| {
                CatalogEntry::new(
                    format!("{tag}_{suffix}"),
                    format!("{label} [{suffix}]"),
                    BlockProblem { m, n, k, precision },
                )
            })
        })
        .collect()
}

/// Row-major, block-stored, row-major again: the two readings of the row-major plan bracket the
/// machine's drift over the run, which is what the block-stored row is read against.
pub fn strategies() -> Vec<CatalogEntry<BlockStrategy>> {
    vec![
        CatalogEntry::new(
            "rowmajor",
            "Cmma, row-major weight",
            BlockStrategy {
                storage: Storage::RowMajor,
            },
        ),
        CatalogEntry::new(
            "block",
            "Cmma, weight packed to the stage",
            BlockStrategy {
                storage: Storage::Block,
            },
        ),
        CatalogEntry::new(
            "rowmajor_again",
            "Cmma, row-major weight (control, second reading)",
            BlockStrategy {
                storage: Storage::RowMajor,
            },
        ),
    ]
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = BlockProblem;
    type Strategy = BlockStrategy;

    fn id(&self) -> &'static str {
        "gemm_block"
    }

    fn label(&self) -> &'static str {
        "GEMM (cmma, block-stored weight)"
    }

    fn problems(&self) -> Vec<CatalogEntry<BlockProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<BlockStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &BlockStrategy,
        problem: &BlockProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    fn work(&self, problem: &BlockProblem) -> Option<CategoryWork> {
        let dtype = problem.precision.dtype();
        let cost = MatmulCost {
            batches: 1,
            m: problem.m,
            n: problem.n,
            k: problem.k,
            elems: MatmulGlobalElems {
                lhs: dtype,
                rhs: dtype,
                out: dtype,
            },
        };
        let (bytes_read, bytes_written) = cost.traffic();
        Some(CategoryWork {
            compute: Some(ComputeWork {
                ops: cost.compute_ops(),
                key: cost.compute_key(&client()),
            }),
            bytes_read,
            bytes_written,
        })
    }
}
