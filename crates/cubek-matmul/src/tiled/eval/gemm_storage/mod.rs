//! The Cmma routine on a storage-tiled weight vs the same plan on a row-major one.
//!
//! The hypothesis under test: a `[stage_k x stage_n]` stage read from a row-major weight touches
//! `stage_k` short runs, half of every cache line wasted at a narrow stage; packed into storage tiles of
//! exactly the plan's stage, the stage is one contiguous run. Prefill shapes only (a tall `M`
//! against a square-ish weight): decode is already at the bandwidth slope.
//!
//! Every strategy runs the plan the selector picks for the problem, so the storage of the weight
//! is the only variable. The row-major plan is listed twice, first and last: the machine drifts
//! thermally, and two readings of the same row bracket the drift the middle row sits in.
//!
//! Each strategy is proved on a small shape against the CPU reference before it is timed, so a
//! wrong read of the storage tiles cannot win by being fast.

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    ir::FloatKind,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{
    CatalogEntry, CategoryWork, ComputeWork, RunSamples, TestInput, ValidationResult,
    assert_equals_approx, client,
};

use crate::{
    definition::{AvailableVectorSizes, MatmulCost, MatmulElems, MatmulGlobalElems, MatmulProblem},
    eval::cpu_reference::{cpu_reference_result, matmul_epsilon, produce_with},
    routine::{BlueprintStrategy, DeviceSettings},
    tiled::{
        cmma::{CmmaBlueprint, CmmaRoutine, CmmaStrategy, StoredTiles, launch_ref},
        pack::pack,
    },
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
pub struct StorageProblem {
    m: usize,
    n: usize,
    k: usize,
    precision: Precision,
}

/// How the weight is stored, which is the only thing a strategy varies.
#[derive(Clone, Copy)]
pub enum Weight {
    /// Plain `[k, n]`, the strided delivery.
    RowMajor,
    /// `[k / stage_k, n / stage_n, stage_k, stage_n]`, storage tiles of exactly the plan's stage, the
    /// Tiled delivery.
    Tiled,
}

impl Weight {
    fn name(self) -> &'static str {
        match self {
            Weight::RowMajor => "rowmajor",
            Weight::Tiled => "tiled",
        }
    }

    /// The weight as this storage holds it: plain, or packed to the plan's stage. Packing is a
    /// real relayout of the data, so the packed weight computes the same product as the plain one.
    fn store(
        self,
        client: &Client,
        rhs: TensorBinding,
        dtype: ElemType,
        blueprint: &CmmaBlueprint,
    ) -> Result<TensorBinding, String> {
        match self {
            Weight::RowMajor => Ok(rhs),
            Weight::Tiled => {
                let (_, stage_n) = blueprint.stage();
                pack(client, rhs, dtype, (blueprint.stage_k, stage_n))
                    .map(TensorHandle::binding)
                    .map_err(|e| format!("{e:?}"))
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct StorageStrategy {
    weight: Weight,
}

/// The `m x n x k` problem and the plan the selector picks for it: one delivery, the cube's
/// units, either way, so the weight's storage is the only variable. Nothing is stored at plan
/// time, since the bench plans first and packs the weight to the stage the plan picked.
fn plan(
    client: &Client,
    (m, n, k): (usize, usize, usize),
    dtype: ElemType,
) -> Result<(MatmulProblem, CmmaBlueprint), String> {
    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        None,
        None,
        MatmulElems::from_single_dtype(dtype).as_global_elems(),
        cubecl::ir::AddressType::U32,
    );
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
    let blueprint = CmmaRoutine::blueprint(
        &BlueprintStrategy::Inferred(CmmaStrategy::default()),
        &problem,
        &device_settings,
        acc,
        StoredTiles::default(),
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok((problem, blueprint))
}

struct StorageBench {
    problem: StorageProblem,
    weight: Weight,
    blueprint: CmmaBlueprint,
    client: Client,
    dtypes: MatmulElems,
    samples: usize,
}

impl Benchmark for StorageBench {
    type Input = (TensorHandle, TensorBinding);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let StorageProblem { m, n, k, precision } = self.problem;
        let dtype = precision.dtype();
        let lhs = TestInput::builder(self.client.clone(), shape![m, k])
            .dtype(dtype)
            .uniform(0, 0.0, 1.0)
            .generate_without_host_data();
        // The weight, packed at load where the strategy stores it in tiles.
        let rhs = TestInput::builder(self.client.clone(), shape![k, n])
            .dtype(dtype)
            .uniform(1, 0.0, 1.0)
            .generate_without_host_data();
        let rhs = self
            .weight
            .store(&self.client, rhs.binding(), dtype, &self.blueprint)
            .expect("the proof packed this weight before the timing");
        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let StorageProblem { m, n, .. } = self.problem;
        let dtype = self.problem.precision.dtype();
        let out = TensorHandle::empty(&self.client, shape![m, n], dtype);
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
        let storage = self.weight.name();
        let StorageProblem { m, n, k, precision } = self.problem;
        let precision = match precision {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
        };
        format!(
            "{}-gemm-storage-{storage}-{m}x{n}x{k}-{precision}",
            self.client.name()
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "gemm-storage-bench", || self.execute(args))
    }
}

pub fn bench(
    strategy: &StorageStrategy,
    problem: &StorageProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();
    let dtype = problem.precision.dtype();
    let elems = MatmulElems::from_single_dtype(dtype);
    let weight = strategy.weight;

    // Prove the storage correct on a small shape before timing it: the same plan selection and
    // the same weight storage as the timed run, against the CPU reference.
    let (proof, proof_plan) = plan(&client, (64, 256, 256), dtype)?;
    let (seed_lhs, seed_rhs) = (7, 11);
    let actual = produce_with(
        client.clone(),
        proof.clone(),
        seed_lhs,
        seed_rhs,
        |c, lhs, rhs, out, dtypes| {
            let rhs = weight
                .store(c, rhs.into_data(), dtype, &proof_plan)
                .map_err(|e| crate::definition::MatmulSetupError::InvalidConfig(Box::new(e)))?;
            launch_ref(
                c,
                lhs,
                InputBinding::Normal(rhs, dtype),
                out,
                &BlueprintStrategy::Forced(proof_plan.clone()),
                dtypes,
            )
        },
    )?;
    let expected = cpu_reference_result(client.clone(), proof, seed_lhs, seed_rhs, None)?;
    match assert_equals_approx(&actual, &expected, matmul_epsilon(&elems, 500.)) {
        ValidationResult::Pass | ValidationResult::Skipped(_) => {}
        ValidationResult::Fail(reason) | ValidationResult::Error(reason) => {
            return Err(format!(
                "the {} weight computes the wrong product, so its timing would be meaningless: \
                 {reason}",
                weight.name()
            ));
        }
    }

    let (_, blueprint) = plan(&client, (problem.m, problem.n, problem.k), dtype)?;
    let bench = StorageBench {
        problem: *problem,
        weight,
        blueprint,
        client: client.clone(),
        dtypes: elems,
        samples: num_samples,
    };

    let durations = bench
        .run(cubek_test_utils::timing_method(TimingMethod::System))
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

pub fn problems() -> Vec<CatalogEntry<StorageProblem>> {
    SHAPES
        .iter()
        .flat_map(|&(tag, label, m, n, k)| {
            PRECISIONS.iter().map(move |&(suffix, precision)| {
                CatalogEntry::new(
                    format!("{tag}_{suffix}"),
                    format!("{label} [{suffix}]"),
                    StorageProblem { m, n, k, precision },
                )
            })
        })
        .collect()
}

/// Row-major, storage-tiled, row-major again: the two readings of the row-major plan bracket the
/// machine's drift over the run, which is what the storage-tiled row is read against.
pub fn strategies() -> Vec<CatalogEntry<StorageStrategy>> {
    vec![
        CatalogEntry::new(
            "rowmajor",
            "Cmma, row-major weight",
            StorageStrategy {
                weight: Weight::RowMajor,
            },
        ),
        CatalogEntry::new(
            "tiled",
            "Cmma, weight packed to the stage",
            StorageStrategy {
                weight: Weight::Tiled,
            },
        ),
        CatalogEntry::new(
            "rowmajor_again",
            "Cmma, row-major weight (control, second reading)",
            StorageStrategy {
                weight: Weight::RowMajor,
            },
        ),
    ]
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = StorageProblem;
    type Strategy = StorageStrategy;

    fn id(&self) -> &'static str {
        "gemm_storage"
    }

    fn label(&self) -> &'static str {
        "GEMM (cmma, storage-tiled weight)"
    }

    fn problems(&self) -> Vec<CatalogEntry<StorageProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<StorageStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &StorageStrategy,
        problem: &StorageProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }

    fn work(&self, problem: &StorageProblem) -> Option<CategoryWork> {
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
