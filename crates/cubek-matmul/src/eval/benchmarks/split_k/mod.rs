//! Split-K on the tile-DSL register leaf: what should a plane's lanes be spent on?
//!
//! The register leaf contracts K serially per lane, so a thin K-heavy shape (`m = 1`, `k = 8192`,
//! the down-proj) leaves a plane's lanes with nothing to do unless something is spread across
//! them. Three mappings of the same problem, differing only in that:
//!
//! - `seq_k`    — nothing on the lanes: one lane per cube walks the whole K. The literal "no
//!   split-K" baseline, launched at `CubeDim::new_single()` so it really is one lane.
//! - `n_spread` — `Cut::unit` on N: each lane owns disjoint output columns and still walks the
//!   whole K. No combine (the columns are disjoint) — today's strategy, the
//!   `GemvUnitPerpendicular` mapping. Needs `n ≥ plane_size · cols` *per cube* to fill the lanes.
//! - `split_k`  — `Cut::unit` on K: each lane contracts a disjoint K-slice and the plane
//!   `plane_sum`-combines, one lane writing. Works however small N gets.
//!
//! All three solve the same `(m, n, k)`; only the mapping differs, so the comparison is total time
//! per problem. `seq_k` vs the other two is "does spending the lanes pay at all"; `n_spread` vs
//! `split_k` is "spend them on N or on K".
//!
//! Rhs traffic, not the combine, dominates these shapes, so the catalogue also sweeps the two
//! levers that shape it. `cols` (both spreads): on a row-major rhs a 1-column `split_k` lane
//! reads one scalar per row while the neighboring columns belong to other cubes, wasting the rest
//! of every cache line; `cols` per cube restores full lines and amortizes the lhs broadcast, at
//! the price of `cols`× fewer cubes. `split_kt` (rhs layout): the same split on a K-contiguous
//! rhs — a `[N, K]` buffer presented as `[K, N]` by stride swap ([`rhs_arg`]), sound at
//! `vector_size == 1` where the tile carries strides verbatim — making each lane's walk down its
//! K-slice sequential in memory (a pre-transposed weight, the layout the legacy `execute_dot`
//! demands). There `cols` flips sign: extra columns sit a whole K apart, so `split_kt` wants
//! `cols = 1`.
//!
//! Measured (Metal): `split_kt_c1` dominates small/mid N; large N stays with `n_spread`. The
//! residual gap is the traversal, not the layout: the legacy kernel interleaves K across the
//! lanes per step (adjacent lanes touch adjacent addresses every instant), and the tile DSL can
//! only hand a lane one *dense* K-window — an interleaved Unit-K cut would put a `plane_sum`
//! after every K element. Per-lane-sequential is as close as a dense window gets.
//!
//! Only meaningful on a GPU: `plane_size == 1` on CPU collapses every strategy to `seq_k`.

use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_test_utils::{
    CatalogEntry, HostData, HostDataType, RunSamples, TileInput, TileInputBuilder,
};
use cubek_tile::{
    Axis, CubeAxis, Cut, Leaf, Schedule, Space, StridedTileArg, StridedTileArgLaunch, Tiling,
    WalkOrder,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The kernel under test: the whole contraction is the space's, so the mapping is the only
/// variable. Mirrors the tile suite's `launch_staged_matmul`.
#[cube(launch)]
fn launch_split_k_matmul<E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, E>,
    c: &StridedTileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}

/// How a problem is mapped onto the plane's lanes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mapping {
    /// Nothing on the lanes: one lane walks the whole K.
    SeqK,
    /// `Cut::unit` on N: `cols` disjoint columns per lane, whole K each, no combine.
    NSpread { cols: usize },
    /// `Cut::unit` on K: a K-slice per lane over `cols` shared columns per cube,
    /// `plane_sum` combine, one lane writes.
    SplitK { cols: usize },
    /// [`SplitK`](Mapping::SplitK) on its home layout: the rhs buffer stored K-contiguous
    /// (a pre-transposed weight), so a lane's walk down its K-slice is sequential in memory.
    SplitKT { cols: usize },
}

/// Which rhs axis is contiguous in the buffer. The *space* is `[K, N]` either way — layout
/// lives in the tensor's strides, never in the space.
#[derive(Clone, Copy)]
enum RhsLayout {
    /// Row-major `[K, N]`: adjacent columns adjacent in memory — `n_spread`'s home.
    NContiguous,
    /// Transposed buffer `[N, K]` presented as `[K, N]` by stride swap — `split_k`'s home.
    KContiguous,
}

#[derive(Clone, Copy)]
pub struct SplitKProblem {
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy)]
pub struct SplitKStrategy {
    mapping: Mapping,
}

impl Mapping {
    /// The space for this mapping. N always rides the cubes, so every mapping loads the grid the
    /// same way and only the intra-plane split differs; each spread takes the columns per cube its
    /// `cols` implies (`plane_size · cols` for `n_spread`, `cols` for `split_k`), `seq_k` takes one.
    fn space(self, problem: SplitKProblem, lanes: usize) -> Space {
        let SplitKProblem { m, n, k } = problem;
        let seq = Cut::sequential;
        match self {
            // One column per cube, one lane, whole K walked serially.
            Mapping::SeqK => Tiling::new()
                .extents(&[(M, m), (N, n), (K, k)])
                .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                    l.axis(M, seq(m))
                        .axis(N, Cut::cube(CubeAxis::X, 1))
                        .axis(K, seq(k))
                })
                .leaf(Leaf::Register),
            // `plane_size · cols` columns per cube, then `cols` per lane, whole K each.
            Mapping::NSpread { cols } => Tiling::new()
                .extents(&[(M, m), (N, n), (K, k)])
                .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                    l.axis(M, seq(m))
                        .axis(N, Cut::cube(CubeAxis::X, lanes * cols))
                        .axis(K, seq(k))
                })
                .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                    l.axis(M, seq(m)).axis(N, Cut::unit(cols)).axis(K, seq(k))
                })
                .leaf(Leaf::Register),
            // `cols` columns per cube shared by the whole plane, K cut into one slice per lane.
            // The transposed variant is the same *space* — only the rhs strides differ.
            Mapping::SplitK { cols } | Mapping::SplitKT { cols } => Tiling::new()
                .extents(&[(M, m), (N, n), (K, k)])
                .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                    l.axis(M, seq(m))
                        .axis(N, Cut::cube(CubeAxis::X, cols))
                        .axis(K, Cut::unit(k / lanes))
                })
                .leaf(Leaf::Register),
        }
        .resolve_lanes(lanes)
    }

    fn rhs_layout(self) -> RhsLayout {
        match self {
            Mapping::SeqK | Mapping::NSpread { .. } | Mapping::SplitK { .. } => {
                RhsLayout::NContiguous
            }
            Mapping::SplitKT { .. } => RhsLayout::KContiguous,
        }
    }

    /// `seq_k` is the one-lane baseline, so it launches a single unit; the spread mappings take
    /// the space's own geometry (`plane_size` lanes on X).
    fn cube_dim(self, space: &Space, client: &ComputeClient<TestRuntime>) -> CubeDim {
        match self {
            Mapping::SeqK => CubeDim::new_single(),
            Mapping::NSpread { .. } | Mapping::SplitK { .. } | Mapping::SplitKT { .. } => {
                space.cube_dim(client)
            }
        }
    }

    fn tag(self) -> String {
        match self {
            Mapping::SeqK => "seq_k".to_string(),
            Mapping::NSpread { cols } => format!("n_spread_c{cols}"),
            Mapping::SplitK { cols } => format!("split_k_c{cols}"),
            Mapping::SplitKT { cols } => format!("split_kt_c{cols}"),
        }
    }

    fn label(self) -> String {
        match self {
            Mapping::SeqK => "No split (1 lane walks K)".to_string(),
            Mapping::NSpread { cols } => format!("N on lanes ({cols} col/lane, no combine)"),
            Mapping::SplitK { cols } => format!("K on lanes ({cols} col/cube, plane_sum)"),
            Mapping::SplitKT { cols } => {
                format!("K on lanes ({cols} col/cube, K-contig rhs)")
            }
        }
    }
}

/// The rhs operand for `mapping`: row-major `[K, N]`, or (`KContiguous`) a `[N, K]` row-major
/// buffer to be presented as `[K, N]` by [`rhs_arg`]. `fill` is the data finalizer.
fn rhs_input(
    client: &ComputeClient<TestRuntime>,
    mapping: Mapping,
    space: &Space,
    fill: impl FnOnce(TileInputBuilder) -> TileInput,
) -> TileInput {
    let axes: &[Axis] = match mapping.rhs_layout() {
        RhsLayout::NContiguous => &[K, N],
        RhsLayout::KContiguous => &[N, K],
    };
    fill(TileInput::builder(client, space.project(axes)).untiled())
}

/// The launch arg for [`rhs_input`]'s tensor: as-is, or the `[N, K]` buffer presented as shape
/// `[K, N]` with swapped strides `[1, k]` — a metadata-only transpose, exactly how
/// [`TileInput::tensor_arg`] re-presents for vectorization. Sound at `vector_size == 1`, where
/// [`MemData::from_tensor`](cubek_tile::MemData) carries strides verbatim.
fn rhs_arg(b: &TileInput, mapping: Mapping) -> TensorArg<TestRuntime> {
    match mapping.rhs_layout() {
        RhsLayout::NContiguous => b.tensor_arg(1),
        RhsLayout::KContiguous => {
            let handle = b.handle();
            let (n, k) = (handle.shape()[0], handle.shape()[1]);
            TensorHandle::<TestRuntime>::new(
                handle.handle.clone(),
                vec![k, n],
                vec![1, k],
                handle.dtype,
            )
            .binding()
            .into_tensor_arg()
        }
    }
}

/// One launch of `mapping` over `problem`, into a freshly zeroed accumulator.
fn run(
    client: &ComputeClient<TestRuntime>,
    mapping: Mapping,
    problem: SplitKProblem,
    lanes: usize,
) -> TileInput {
    let space = mapping.space(problem, lanes);
    let dtype = f32::as_type_native_unchecked().storage_type();
    let a = TileInput::builder(client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = rhs_input(client, mapping, &space, TileInputBuilder::arange);
    let c = TileInput::builder(client, space.project(&[M, N]))
        .untiled()
        .zeros();

    launch_split_k_matmul::launch::<TestRuntime>(
        client,
        space.cube_count(),
        mapping.cube_dim(&space, client),
        StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
        StridedTileArgLaunch::strided(rhs_arg(&b, mapping), 1, space.project(&[K, N]), b.storage()),
        StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
        dtype,
    );
    c
}

struct SplitKBench {
    problem: SplitKProblem,
    mapping: Mapping,
    client: ComputeClient<TestRuntime>,
    samples: usize,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    a: TileInput,
    b: TileInput,
    /// The kernel-side rhs space, always `[K, N]`; `b.space()` is `[N, K]` for the
    /// transposed layout, so it cannot serve here.
    rhs_space: Space,
    c: TileInput,
}

impl Benchmark for SplitKBench {
    type Input = ();
    type Output = ();

    /// Operands are built once in `SplitKBench::new` (`TileInput` is not `Clone`), so only the
    /// launch lands in the timed region.
    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _: Self::Input) -> Result<Self::Output, String> {
        let (a, b, c) = (&self.a, &self.b, &self.c);
        let dtype = f32::as_type_native_unchecked().storage_type();
        launch_split_k_matmul::launch::<TestRuntime>(
            &self.client,
            self.cube_count.clone(),
            self.cube_dim,
            StridedTileArgLaunch::strided(a.tensor_arg(1), 1, a.space(), a.storage()),
            StridedTileArgLaunch::strided(
                rhs_arg(b, self.mapping),
                1,
                self.rhs_space.clone(),
                b.storage(),
            ),
            StridedTileArgLaunch::strided(c.tensor_arg(1), 1, c.space(), c.storage()),
            dtype,
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "{}-split-k-{}-m{}-n{}-k{}",
            <TestRuntime as Runtime>::name(&self.client),
            self.mapping.tag(),
            self.problem.m,
            self.problem.n,
            self.problem.k,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "split-k-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

/// A mapping that computes the wrong answer would still time fast, so every strategy proves itself
/// on a small shape before it is measured. Guards the whole family of silent-zero traps: a
/// wrong-sized `Cut::unit`, an unresolved lane count, a combine that never fires.
fn verify(
    client: &ComputeClient<TestRuntime>,
    mapping: Mapping,
    lanes: usize,
) -> Result<(), String> {
    // `n = lanes · 4` divides evenly for every catalogued width (n_spread cols ≤ 4 fills its
    // cube exactly; split_k cols ∈ {1, 8, 32} all divide 128), so no mapping needs masking here.
    let (m, n, k) = (1usize, lanes * 4, lanes * 4);
    let problem = SplitKProblem { m, n, k };
    let c = run(client, mapping, problem, lanes);
    let out = HostData::from_tensor_handle(client, c.handle(), HostDataType::F32);

    // Arange lands on the *physical* buffer: lhs(i, p) = i·k + p either way, but the logical
    // rhs value depends on the layout — row-major [K, N] gives rhs(p, j) = p·n + j, the
    // transposed [N, K] buffer gives rhs(p, j) = j·k + p.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| {
                    let rhs = match mapping.rhs_layout() {
                        RhsLayout::NContiguous => p * n + j,
                        RhsLayout::KContiguous => j * k + p,
                    };
                    ((i * k + p) * rhs) as f32
                })
                .sum()
        })
        .collect();

    for (i, e) in expected.iter().enumerate() {
        let g = out.data.get_f32(i);
        if (g - e).abs() > e.abs() * 1e-3 + 1e-3 {
            return Err(format!(
                "{} computes the wrong result at {i}: got {g}, expected {e} — the mapping is \
                 misconfigured, so its timing would be meaningless",
                mapping.tag()
            ));
        }
    }
    Ok(())
}

pub fn bench(
    strategy: &SplitKStrategy,
    problem: &SplitKProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);
    let lanes = client.properties().hardware.plane_size_max as usize;
    let mapping = strategy.mapping;

    if lanes == 1 {
        return Err("plane_size == 1: every mapping collapses to seq_k (run on a GPU)".to_string());
    }
    match mapping {
        Mapping::SeqK => {}
        Mapping::NSpread { cols } => {
            if !problem.n.is_multiple_of(lanes * cols) {
                return Err(format!(
                    "n_spread needs n ({}) divisible by plane_size·cols ({lanes}·{cols})",
                    problem.n
                ));
            }
        }
        Mapping::SplitK { cols } | Mapping::SplitKT { cols } => {
            if !problem.k.is_multiple_of(lanes) {
                return Err(format!(
                    "split_k needs k ({}) divisible by plane_size ({lanes})",
                    problem.k
                ));
            }
            if !problem.n.is_multiple_of(cols) {
                return Err(format!(
                    "split_k needs n ({}) divisible by cols ({cols})",
                    problem.n
                ));
            }
        }
    }
    verify(&client, mapping, lanes)?;

    let space = mapping.space(*problem, lanes);
    let cube_count = space.cube_count();
    let cube_dim = mapping.cube_dim(&space, &client);
    let flops = 2.0 * problem.m as f64 * problem.n as f64 * problem.k as f64;

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .uniform(0, 0.0, 1.0);
    let b = rhs_input(&client, mapping, &space, |bld| bld.uniform(1, 0.0, 1.0));
    let rhs_space = space.project(&[K, N]);
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    let bench = SplitKBench {
        problem: *problem,
        mapping,
        client,
        samples: num_samples,
        cube_count,
        cube_dim,
        a,
        b,
        rhs_space,
        c,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    Ok(RunSamples::new(durations).with_flops(flops))
}

/// The down-proj family: `m = 1`, `k = 8192`, N sweeping the regime where it stops being able to
/// fill the lanes on its own. `n = 32` is one cube per lane-group; `n = 2048` has N parallelism to
/// spare, which is where `n_spread` should stop needing K.
const SHAPES: &[(&str, &str, usize, usize, usize)] = &[
    ("m1_n32_k8192", "m=1 n=32 k=8192", 1, 32, 8192),
    ("m1_n256_k8192", "m=1 n=256 k=8192", 1, 256, 8192),
    (
        "m1_n2048_k8192",
        "m=1 n=2048 k=8192 (down-proj)",
        1,
        2048,
        8192,
    ),
];

/// The width ladder: `c1` is each mapping's naive form, the wider entries buy cache-line
/// utilization with register-block width (all stay within the leaf's 64-cell unroll budget).
/// `n_spread` also gets a widened form so the comparison stays honest if widening helps everyone.
const MAPPINGS: &[Mapping] = &[
    Mapping::SeqK,
    Mapping::NSpread { cols: 1 },
    Mapping::NSpread { cols: 4 },
    Mapping::SplitK { cols: 1 },
    Mapping::SplitK { cols: 8 },
    Mapping::SplitK { cols: 32 },
    Mapping::SplitKT { cols: 1 },
    Mapping::SplitKT { cols: 8 },
    Mapping::SplitKT { cols: 32 },
];

pub fn problems() -> Vec<CatalogEntry<SplitKProblem>> {
    SHAPES
        .iter()
        .map(|&(tag, label, m, n, k)| CatalogEntry::new(tag, label, SplitKProblem { m, n, k }))
        .collect()
}

pub fn strategies() -> Vec<CatalogEntry<SplitKStrategy>> {
    MAPPINGS
        .iter()
        .map(|&mapping| {
            CatalogEntry::new(mapping.tag(), mapping.label(), SplitKStrategy { mapping })
        })
        .collect()
}

pub struct Category;

impl cubek_test_utils::Category for Category {
    type Problem = SplitKProblem;
    type Strategy = SplitKStrategy;

    fn id(&self) -> &'static str {
        "split_k"
    }

    fn label(&self) -> &'static str {
        "Split-K (tile register leaf)"
    }

    /// These shapes are latency-bound, so the launch must be timed on the device rather than
    /// around an async submit; matches the `TimingMethod` [`bench`] actually runs with.
    fn timing_method(&self) -> TimingMethod {
        TimingMethod::Device
    }

    fn problems(&self) -> Vec<CatalogEntry<SplitKProblem>> {
        problems()
    }

    fn strategies(&self) -> Vec<CatalogEntry<SplitKStrategy>> {
        strategies()
    }

    fn bench(
        &self,
        strategy: &SplitKStrategy,
        problem: &SplitKProblem,
        num_samples: usize,
    ) -> Result<RunSamples, String> {
        bench(strategy, problem, num_samples)
    }
}
