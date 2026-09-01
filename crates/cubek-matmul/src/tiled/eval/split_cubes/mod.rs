//! Splitting the contraction across *cubes*, and the two ways of putting the pieces back
//! together.
//!
//! The sibling category [`split_k`](super::split_k) spends a plane's lanes on `K`. This one
//! spends cubes on it, which is the lever that matters once a shape has too few output tiles to
//! fill the device: `m = 1, n = 64` is 64 cubes however deep `K` is, and a GPU with more cores
//! than that idles through the whole contraction no matter how well each cube is written.
//!
//! Three mappings of one problem, differing only in how the pieces are added up:
//!
//! - `data_parallel`: `K` walked whole by each cube. The baseline, and the mapping that needs no
//!   combine because there is nothing to combine.
//! - `workspace`: `K` spelled as two axes, `(KB, KI)`, with the output bound over `[KB, M, N]` so
//!   it *spans* the split. Nothing is partial, at the price of a `splits × m × n` buffer and a
//!   second pass to fold it away. Both launches are timed, because both are the cost.
//! - `atomic`: `K` cut at cube scope and the drain folding each cube's slice into the output
//!   atomically. One launch, one buffer, and a result that is not bit-identical run to run.
//!
//! What the numbers are for: `workspace` and `atomic` compute the same thing and differ only in
//! where the pieces meet, so the gap between them is the price of the second pass against the
//! price of the atomics; the gap from `data_parallel` is what either of them buys. The split that
//! wins at `splits = 1` is not a split at all, which is the control.

use crate::definition::{MatmulElems, compute_peak_ops_per_s};
use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    features::AtomicUsage,
    future,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
};
use cubek_test_utils::{CatalogEntry, HostData, HostDataType, RunSamples, TileInput};
use cubek_tile::{
    AccumulateArg, AccumulateArgLaunch, Axis, Buffering, CubeAxis, Cut, Instruction, Monoid,
    PhysicalAxisMap, Projection, RegisterBlock, Residence, Semiring, Space, TileArg, TileArgLaunch,
    TileSpec, Tiling, WalkOrder,
};

/// Held fixed across mappings so the numbers compare the partitioning and not the instruction.
const INSTRUCTION: Instruction = Instruction::Registers {
    config: RegisterBlock::new(64),
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// `K` as two axes, for the mapping that binds the split into the output's own shape.
const KB: Axis = Axis(3);
const KI: Axis = Axis(4);

/// Columns each cube owns. Held at one so the mappings differ only in what they do with `K`.
const COLS: usize = 1;
/// The reduce pass's output tile: one row by this many columns, inside the leaf's unroll budget.
const FOLD_COLS: usize = 32;

#[cube(launch)]
fn plain_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm(&a, &b, Semiring::SUM_PROD);
}

#[cube(launch)]
fn atomic_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = out.tile(space);
    let mut acc = c.accumulate::<E, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

#[cube(launch)]
fn fold_splits<E: Numeric>(
    partials: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let partials = partials.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.reduce_axis(&partials, Monoid::Sum);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mapping {
    /// Each cube walks the whole `K`.
    DataParallel,
    /// `K` as `(KB, KI)`, the output spanning `KB`, then a second pass folds it away.
    Workspace { splits: usize },
    /// `K` cut at cube scope, the drain folding atomically.
    Atomic { splits: usize },
    /// [`Atomic`](Mapping::Atomic) with the lanes given work of their own: each cube's slice of
    /// `K` is cut again across the plane, so the lanes contract disjoint slices and combine in
    /// registers before one fold per cube reaches memory.
    ///
    /// The other mappings put nothing on the lanes, and a cube launches a full plane whatever the
    /// space says, so their 32 lanes all run the same code over the same numbers and 31 of them
    /// are waste. That is not what a split should look like, and this is the comparison that says
    /// what it costs.
    AtomicLanes { splits: usize },
}

impl Mapping {
    fn splits(self) -> usize {
        match self {
            Mapping::DataParallel => 1,
            Mapping::Workspace { splits }
            | Mapping::Atomic { splits }
            | Mapping::AtomicLanes { splits } => splits,
        }
    }

    pub fn tag(self) -> String {
        match self {
            Mapping::DataParallel => "data_parallel".to_string(),
            Mapping::Workspace { splits } => format!("workspace_s{splits}"),
            Mapping::Atomic { splits } => format!("atomic_s{splits}"),
            Mapping::AtomicLanes { splits } => format!("atomic_lanes_s{splits}"),
        }
    }

    pub fn label(self) -> String {
        match self {
            Mapping::DataParallel => "whole K per cube (no split)".to_string(),
            Mapping::Workspace { splits } => {
                format!("K split {splits} ways, partials buffer + fold pass")
            }
            Mapping::Atomic { splits } => format!("K split {splits} ways, atomic drain"),
            Mapping::AtomicLanes { splits } => {
                format!("K split {splits} ways over cubes then again over lanes, atomic drain")
            }
        }
    }

    /// The contraction's space. `N` rides the cubes in every mapping, so only the treatment of
    /// `K` differs.
    fn space(self, problem: Problem, lanes: usize) -> Space {
        let Problem { m, n, k } = problem;
        let splits = self.splits();
        match self {
            Mapping::DataParallel | Mapping::Atomic { .. } => Tiling::new()
                .extents(&[(M, m), (N, n), (K, k)])
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                    l.axis(M, Cut::sequential(m))
                        .axis(N, Cut::cube(CubeAxis::X, COLS))
                        .axis(K, Cut::cube(CubeAxis::Z, k / splits))
                })
                .build(),
            Mapping::Workspace { .. } => Tiling::new()
                .extents(&[(M, m), (N, n), (KB, splits), (KI, k / splits)])
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                    l.axis(M, Cut::sequential(m))
                        .axis(N, Cut::cube(CubeAxis::X, COLS))
                        .axis(KB, Cut::cube(CubeAxis::Z, 1))
                        .axis(KI, Cut::sequential(k / splits))
                })
                .build(),
            // The cube's slice of K cut again across the plane: each lane contracts its own
            // sixteenth (or whatever the lane count makes it), the plane combines in registers,
            // and one fold per cube reaches memory.
            Mapping::AtomicLanes { .. } => Tiling::new()
                .extents(&[(M, m), (N, n), (K, k)])
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                    l.axis(M, Cut::sequential(m))
                        .axis(N, Cut::cube(CubeAxis::X, COLS))
                        .axis(K, Cut::cube(CubeAxis::Z, k / splits))
                })
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                    l.axis(M, Cut::sequential(m))
                        .axis(N, Cut::sequential(COLS))
                        .axis(K, Cut::unit(k / splits / lanes))
                })
                .build()
                .resolve_lanes(lanes),
        }
        .with_instruction(INSTRUCTION)
    }

    /// The fold pass's space, for the mapping that has one.
    fn fold_space(self, problem: Problem) -> Space {
        let Problem { m, n, .. } = problem;
        Tiling::new()
            .extents(&[(M, m), (N, n), (KB, self.splits())])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::cube(CubeAxis::X, 1))
                    .axis(N, Cut::cube(CubeAxis::Y, FOLD_COLS))
                    .axis(KB, Cut::sequential(self.splits()))
            })
            .build()
            .with_instruction(INSTRUCTION)
    }

    /// The lhs spec: `[M, K]` in memory either way, addressed by one logical axis or two.
    fn lhs_spec(self, inside: usize) -> TileSpec {
        match self {
            Mapping::DataParallel | Mapping::Atomic { .. } | Mapping::AtomicLanes { .. } => {
                TileSpec::direct(&[M, K])
            }
            Mapping::Workspace { .. } => TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                ],
            )),
        }
    }

    fn rhs_spec(self, inside: usize) -> TileSpec {
        match self {
            Mapping::DataParallel | Mapping::Atomic { .. } | Mapping::AtomicLanes { .. } => {
                TileSpec::direct(&[K, N])
            }
            Mapping::Workspace { .. } => TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Problem {
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy)]
pub struct Strategy {
    mapping: Mapping,
}

/// Everything one mapping needs to launch, built once so only the launches are timed.
struct Bound {
    client: ComputeClient<TestRuntime>,
    mapping: Mapping,
    samples: usize,
    space: Space,
    fold_space: Space,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    fold_cube_count: CubeCount,
    fold_cube_dim: CubeDim,
    a: TileInput,
    b: TileInput,
    /// The destination the contraction writes: the output itself, or the partials buffer.
    c: TileInput,
    /// The final output, for the mapping whose contraction wrote partials.
    folded: TileInput,
    lhs_spec: TileSpec,
    rhs_spec: TileSpec,
    out_spec: TileSpec,
}

/// The operand seeds. Fixed rather than passed: only the shape varies between the run that
/// verifies a mapping and the run that times it, and two `u64`s at a call site say nothing.
const LHS_SEED: u64 = 0;
const RHS_SEED: u64 = 1;

impl Bound {
    fn new(client: &ComputeClient<TestRuntime>, mapping: Mapping, problem: Problem) -> Self {
        let lanes = client.properties().hardware.plane_size_max as usize;
        let Problem { m, n, k } = problem;
        let splits = mapping.splits();
        let inside = k / splits;
        let space = mapping.space(problem, lanes);
        let fold_space = mapping.fold_space(problem);

        let a = TileInput::builder(client, Space::new(&[(M, m), (K, k)]))
            .untiled()
            .uniform(LHS_SEED, 0.0, 1.0);
        let b = TileInput::builder(client, Space::new(&[(K, k), (N, n)]))
            .untiled()
            .uniform(RHS_SEED, 0.0, 1.0);
        // The contraction's destination: `[KB, M, N]` where the split is an axis of it, the plain
        // output otherwise. Zeroed either way, which the atomic drain needs and the others do not
        // mind.
        let c = match mapping {
            Mapping::Workspace { .. } => {
                TileInput::builder(client, Space::new(&[(KB, splits), (M, m), (N, n)]))
                    .untiled()
                    .zeros()
            }
            _ => TileInput::builder(client, Space::new(&[(M, m), (N, n)]))
                .untiled()
                .zeros(),
        };
        let folded = TileInput::builder(client, Space::new(&[(M, m), (N, n)]))
            .untiled()
            .zeros();

        Bound {
            client: client.clone(),
            mapping,
            // The verifying run never times anything; `bench` states the count it wants.
            samples: 1,
            cube_count: space.cube_count(),
            cube_dim: space.cube_dim(client),
            fold_cube_count: fold_space.cube_count(),
            fold_cube_dim: fold_space.cube_dim(client),
            space,
            fold_space,
            a,
            b,
            c,
            folded,
            lhs_spec: mapping.lhs_spec(inside),
            rhs_spec: mapping.rhs_spec(inside),
            out_spec: TileSpec::direct(&[M, N]).residence(&[Residence::Register]),
        }
    }

    /// How many timed samples this takes. Named rather than passed to
    /// [`new`](Bound::new), where it would be one of two bare numbers at every call site.
    fn samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// One full run of the mapping: the contraction, and the fold pass where there is one. Both
    /// launches, because the second pass is part of what the workspace mapping costs.
    fn launch(&self) {
        let dtype = f32::elem_type_native();
        match self.mapping {
            Mapping::Atomic { .. } | Mapping::AtomicLanes { .. } => {
                atomic_matmul::launch::<TestRuntime>(
                    &self.client,
                    self.cube_count.clone(),
                    self.cube_dim,
                    TileArgLaunch::new(self.a.tensor_arg(1), self.lhs_spec.clone()),
                    TileArgLaunch::new(self.b.tensor_arg(1), self.rhs_spec.clone()),
                    AccumulateArgLaunch::new(self.c.tensor_arg(1), self.out_spec.clone()),
                    self.space.clone(),
                    dtype,
                );
            }
            Mapping::DataParallel => {
                plain_matmul::launch::<TestRuntime>(
                    &self.client,
                    self.cube_count.clone(),
                    self.cube_dim,
                    TileArgLaunch::new(self.a.tensor_arg(1), self.lhs_spec.clone()),
                    TileArgLaunch::new(self.b.tensor_arg(1), self.rhs_spec.clone()),
                    TileArgLaunch::new(self.c.tensor_arg(1), TileSpec::direct(&[M, N])),
                    self.space.clone(),
                    dtype,
                );
            }
            Mapping::Workspace { .. } => {
                plain_matmul::launch::<TestRuntime>(
                    &self.client,
                    self.cube_count.clone(),
                    self.cube_dim,
                    TileArgLaunch::new(self.a.tensor_arg(1), self.lhs_spec.clone()),
                    TileArgLaunch::new(self.b.tensor_arg(1), self.rhs_spec.clone()),
                    TileArgLaunch::new(self.c.tensor_arg(1), TileSpec::direct(&[KB, M, N])),
                    self.space.clone(),
                    dtype,
                );
                fold_splits::launch::<TestRuntime>(
                    &self.client,
                    self.fold_cube_count.clone(),
                    self.fold_cube_dim,
                    TileArgLaunch::new(self.c.tensor_arg(1), TileSpec::direct(&[KB, M, N])),
                    TileArgLaunch::new(self.folded.tensor_arg(1), TileSpec::direct(&[M, N])),
                    self.fold_space.clone(),
                    dtype,
                );
            }
        }
    }

    /// Where this mapping's answer ends up.
    fn result(&self) -> &TileInput {
        match self.mapping {
            Mapping::Workspace { .. } => &self.folded,
            _ => &self.c,
        }
    }
}

impl Benchmark for Bound {
    type Input = ();
    type Output = ();

    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _: Self::Input) -> Result<Self::Output, String> {
        self.launch();
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!("split-cubes-{}", self.mapping.tag())
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "split-cubes-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

/// A mapping that computes the wrong answer still times fast, so each proves itself on a small
/// shape first. The trap this one is really guarding is the atomic drain onto a buffer that was
/// not zeroed, which reads as a plausible number rather than as garbage.
fn verify(client: &ComputeClient<TestRuntime>, mapping: Mapping) -> Result<(), String> {
    let lanes = client.properties().hardware.plane_size_max as usize;
    // Big enough that every mapping's cuts divide it: each cube's slice of `K` has to survive
    // being cut again across the plane.
    let (m, n, k) = (2usize, FOLD_COLS, mapping.splits() * lanes * 2);
    let problem = Problem { m, n, k };
    let bound = Bound::new(client, mapping, problem);
    bound.launch();

    let a = HostData::from_tensor_handle(client, bound.a.handle(), HostDataType::F32);
    let b = HostData::from_tensor_handle(client, bound.b.handle(), HostDataType::F32);
    let got = HostData::from_tensor_handle(client, bound.result().handle(), HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k)
                .map(|p| a.get_f32(&[i, p]) * b.get_f32(&[p, j]))
                .sum();
            let have = got.get_f32(&[i, j]);
            if (have - want).abs() > want.abs() * 1e-3 + 1e-3 {
                return Err(format!(
                    "{} computes the wrong result at ({i}, {j}): got {have}, want {want}: the \
                     mapping is misconfigured, so its timing would be meaningless",
                    mapping.tag()
                ));
            }
        }
    }
    Ok(())
}

pub fn bench(
    strategy: &Strategy,
    problem: &Problem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let mapping = strategy.mapping;
    let splits = mapping.splits();

    if !problem.k.is_multiple_of(splits) {
        return Err(format!(
            "k ({}) must divide into {splits} slices",
            problem.k
        ));
    }
    let lanes = client.properties().hardware.plane_size_max as usize;
    if let Mapping::AtomicLanes { .. } = mapping
        && !(problem.k / splits).is_multiple_of(lanes)
    {
        return Err(format!(
            "each cube's K slice ({}) must divide across the plane's {lanes} lanes",
            problem.k / splits
        ));
    }
    if !problem.n.is_multiple_of(FOLD_COLS) {
        return Err(format!(
            "n ({}) must be a multiple of the fold pass's {FOLD_COLS}-column tile",
            problem.n
        ));
    }
    if matches!(
        mapping,
        Mapping::Atomic { .. } | Mapping::AtomicLanes { .. }
    ) && !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        return Err("device has no f32 atomic add".to_string());
    }
    verify(&client, mapping)?;

    let bound = Bound::new(&client, mapping, *problem).samples(num_samples);
    let flops = 2.0 * problem.m as f64 * problem.n as f64 * problem.k as f64;
    let elems = MatmulElems::from_single_dtype(f32::elem_type_native());
    let durations = bound
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;
    Ok(RunSamples::new(durations).with_flops(flops, compute_peak_ops_per_s(&client, &elems)))
}

/// Shapes with too few output tiles to fill a device, which is the whole reason to spend cubes on
/// the contraction. `n` is the cube count without a split (`COLS = 1`), so `n = 32` starts well
/// under any GPU's core count and `n = 512` starts above it.
const SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    (
        "m1_n32_k8192",
        "m=1 n=32 k=8192 (32 cubes unsplit)",
        1,
        32,
        8192,
    ),
    ("m1_n128_k8192", "m=1 n=128 k=8192", 1, 128, 8192),
    (
        "m1_n512_k8192",
        "m=1 n=512 k=8192 (already wide)",
        1,
        512,
        8192,
    ),
    ("m8_n128_k4096", "m=8 n=128 k=4096", 8, 128, 4096),
];

const MAPPINGS: &[Mapping] = &[
    Mapping::DataParallel,
    Mapping::AtomicLanes { splits: 1 },
    Mapping::AtomicLanes { splits: 4 },
    Mapping::AtomicLanes { splits: 16 },
    Mapping::Workspace { splits: 1 },
    Mapping::Workspace { splits: 4 },
    Mapping::Workspace { splits: 16 },
    Mapping::Atomic { splits: 1 },
    Mapping::Atomic { splits: 4 },
    Mapping::Atomic { splits: 16 },
];

pub fn problems() -> Vec<CatalogEntry<Problem>> {
    SHAPES
        .iter()
        .map(|&(tag, label, m, n, k)| CatalogEntry::new(tag, label, Problem { m, n, k }))
        .collect()
}

pub fn strategies() -> Vec<CatalogEntry<Strategy>> {
    MAPPINGS
        .iter()
        .map(|&mapping| {
            CatalogEntry::new(
                Box::leak(mapping.tag().into_boxed_str()),
                Box::leak(mapping.label().into_boxed_str()),
                Strategy { mapping },
            )
        })
        .collect()
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = Problem;
    type Strategy = Strategy;

    fn id(&self) -> &'static str {
        "split_cubes"
    }

    fn label(&self) -> &'static str {
        "Split-K across cubes: workspace pass vs atomic drain"
    }

    /// Latency-bound shapes, so the launch is timed on the device rather than around a submit.
    fn timing_method(&self) -> TimingMethod {
        TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<Problem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<Strategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &Strategy,
        problem: &Problem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
