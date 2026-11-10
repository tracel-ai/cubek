//! `matmul -> RMS norm -> matmul` as three kernels against the norm deferred into the two
//! products: the first folds each row's squares into the norm's factor as it stores, and the
//! second scales each row by that factor as it stores. The norm's weight `gamma` is folded into
//! the second weight once, outside the timed region.

use std::time::{Duration, Instant};

use cubecl::client::Client;
use cubecl::future;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{
    MergeLaunch, Merged, MergedOperand, ReducedRows, RowReductionScratch, RowScale, SinkLaunch,
    VectorStageLaunch,
};
use cubecl::zspace::metadata::Metadata;
use cubecl::zspace::shape;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine,
    MatmulRoutine, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine, UnitWholeKMatmulRoutine,
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::rms_norm::RmsNormScaleLaunch;
use cubek_rms_norm::deferred::RmsNormFactor;
use cubek_rms_norm::kernels::{RmsNormCall, launch_tuned};
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_std::routine::Routine;

const DIM: usize = 4096;
const EPS: f32 = 1e-6;
const TOLERANCE: f32 = 2e-2;
const ELEM: ElemType = ElemType::Float(FloatKind::F16);
const ROWS: [usize; 4] = [1, 16, 512, 4096];
const WARM_UPS: usize = 10;
const FEW_ROWS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Row,
    Col,
}

/// A `[DIM, DIM]` weight stored both ways, so a routine picks the layout it reads best.
struct Weight {
    row: TensorHandle,
    col: TensorHandle,
}

impl Weight {
    fn new(client: &Client, host: &[half::f16]) -> Self {
        let upload = |values: &[half::f16]| client.create_from_slice(half::f16::as_bytes(values));
        let row = TensorHandle::new_contiguous(shape![DIM, DIM], upload(host), ELEM);
        let transposed: Vec<half::f16> = (0..DIM * DIM)
            .map(|i| host[(i % DIM) * DIM + i / DIM])
            .collect();
        let meta = Metadata::new(shape![DIM, DIM], [1usize, DIM]);
        let col = TensorHandle::from_metadata(upload(&transposed), meta, ELEM);
        Self { row, col }
    }

    fn at(&self, layout: Layout) -> &TensorHandle {
        match layout {
            Layout::Row => &self.row,
            Layout::Col => &self.col,
        }
    }
}

/// One matmul routine at one setting, launching the three kinds of product the arms need.
trait Product {
    fn name(&self) -> String;
    fn buffered(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle;
    fn row_factors(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle;
    fn row_scaled(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle;
    fn passed(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle;
}

struct Planned<R: Routine, M, SM> {
    routine: R,
    settings: R::Settings,
    rows: usize,
    hw: Hardware,
    client: Client,
    /// Where the first product folds each row's factor, made once for the launch it picked.
    factors: RowReductionScratch,
    _kernel: core::marker::PhantomData<fn() -> (M, SM)>,
}

fn stored(handle: &TensorHandle) -> StoredTensor {
    StoredTensor::Plain(PlainTensor::of_handle(handle.clone()))
}

impl<R, M, SM> Planned<R, M, SM>
where
    R: MatmulRoutine<M, SM> + Copy,
    M: MatmulMicrokernel,
{
    /// The product stored through `Merged<G, S>`.
    fn sunk<G: MergeLaunch, S: VectorStageLaunch>(
        &self,
        lhs: &TensorHandle,
        rhs: &TensorHandle,
        merge: G::Operand,
        stage: S::Operand,
    ) -> TensorHandle {
        let launch = self
            .routine
            .sink_launch_on(
                &self.hw,
                &self.client,
                stored(lhs),
                stored(rhs),
                &self.settings,
            )
            .expect("the routine served this product when it was picked");
        let (out, _) = PlainTensor::empty_contiguous(&self.client, shape![self.rows, DIM], ELEM);
        launch.launch::<Merged<G, S>>(|sink| MergedOperand {
            out: out.clone().into_arg(),
            merge,
            stage,
            sink,
        });
        out
    }
}

impl<R, M, SM> Product for Planned<R, M, SM>
where
    R: MatmulRoutine<M, SM> + Copy,
    R::Settings: 'static,
    M: MatmulMicrokernel + 'static,
    SM: 'static,
{
    fn name(&self) -> String {
        let routine = core::any::type_name::<R>()
            .rsplit("::")
            .next()
            .unwrap_or_default();
        format!("{routine} {}", self.settings)
    }

    fn buffered(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle {
        let (out, plain) =
            PlainTensor::empty_contiguous(&self.client, shape![self.rows, DIM], ELEM);
        self.routine
            .launch_into(
                &self.hw,
                &self.client,
                stored(lhs),
                stored(rhs),
                plain,
                &self.settings,
            )
            .expect("the routine served this product when it was picked");
        out
    }

    fn row_factors(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle {
        self.sunk::<ReducedRows<RmsNormFactor>, ()>(
            lhs,
            rhs,
            (self.factors.clone(), RmsNormScaleLaunch::new(EPS, DIM)),
            (),
        )
    }

    fn row_scaled(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle {
        self.sunk::<(), RowScale>(lhs, rhs, (), self.factors.values().clone())
    }

    fn passed(&self, lhs: &TensorHandle, rhs: &TensorHandle) -> TensorHandle {
        self.sunk::<(), ()>(lhs, rhs, (), ())
    }
}

struct Plan {
    product: Box<dyn Product>,
    layout: Layout,
}

/// `routine` at its first setting that serves `[rows, DIM] x [DIM, DIM]` as a sink in `layout`
/// whose rows can be folded across the cubes sharing them.
fn plan<R, M, SM>(routine: R, layout: Layout, bench: &Bench, x: &TensorHandle) -> Option<Plan>
where
    R: MatmulRoutine<M, SM> + Copy + 'static,
    R::Settings: 'static,
    M: MatmulMicrokernel + 'static,
    SM: 'static,
{
    let rows = x.metadata.shape()[0];
    let rhs = bench.w1.at(layout);
    let (settings, factors) = routine
        .settings(&bench.hw)
        .into_iter()
        .find_map(|settings| {
            let launch = routine
                .sink_launch_on(&bench.hw, &bench.client, stored(x), stored(rhs), &settings)
                .ok()?;
            let factors = RowReductionScratch::new(launch.launcher(), launch.out()).ok()?;
            Some((settings, factors))
        })?;
    let product = Planned {
        routine,
        settings,
        rows,
        hw: bench.hw.clone(),
        client: bench.client.clone(),
        factors,
        _kernel: core::marker::PhantomData,
    };
    Some(Plan {
        product: Box::new(product),
        layout,
    })
}

struct Bench {
    client: Client,
    device: cubecl::Device,
    hw: Hardware,
    gamma: PlainTensor,
    w1: Weight,
    w2: Weight,
    w2_gamma: Weight,
}

impl Bench {
    fn new() -> Self {
        let device = cubecl::test_device();
        let client = device.client();
        let gamma_host = uniform(DIM, 1, 0.5, 1.5);
        let w2_host = uniform(DIM * DIM, 3, -1.0 / 64.0, 1.0 / 64.0);
        let w2_gamma_host: Vec<_> = w2_host
            .iter()
            .enumerate()
            .map(|(i, w)| half::f16::from_f32(w.to_f32() * gamma_host[i / DIM].to_f32()))
            .collect();
        let gamma = TensorHandle::new_contiguous(
            shape![DIM],
            client.create_from_slice(half::f16::as_bytes(&gamma_host)),
            ELEM,
        );
        Self {
            hw: Hardware::from_client(&client),
            gamma: PlainTensor::of_handle(gamma),
            w1: Weight::new(&client, &uniform(DIM * DIM, 2, -1.0 / 64.0, 1.0 / 64.0)),
            w2: Weight::new(&client, &w2_host),
            w2_gamma: Weight::new(&client, &w2_gamma_host),
            client,
            device,
        }
    }

    fn input(&self, rows: usize) -> TensorHandle {
        let host = uniform(rows * DIM, 4, -1.0, 1.0);
        TensorHandle::new_contiguous(
            shape![rows, DIM],
            self.client.create_from_slice(half::f16::as_bytes(&host)),
            ELEM,
        )
    }

    /// The first routine of the list for `rows` that serves the product.
    fn pick(&self, x: &TensorHandle) -> Plan {
        type Try<'a> = Box<dyn Fn() -> Option<Plan> + 'a>;
        let tries: Vec<Try> = if x.metadata.shape()[0] <= FEW_ROWS {
            vec![
                Box::new(|| plan(PlaneShuffleMatmulRoutine, Layout::Col, self, x)),
                Box::new(|| plan(CubeShuffleMatmulRoutine, Layout::Col, self, x)),
                Box::new(|| plan(UnitWholeKMatmulRoutine, Layout::Col, self, x)),
                Box::new(|| plan(CubeStagedMatmulRoutine, Layout::Row, self, x)),
                Box::new(|| plan(UnitRegistersMatmulRoutine, Layout::Row, self, x)),
            ]
        } else {
            vec![
                Box::new(|| plan(CubeStagedMatmulRoutine, Layout::Row, self, x)),
                Box::new(|| plan(CubeStagedAsyncCopyMatmulRoutine, Layout::Row, self, x)),
                Box::new(|| plan(UnitRegistersMatmulRoutine, Layout::Row, self, x)),
            ]
        };
        tries
            .iter()
            .find_map(|attempt| attempt())
            .expect("a routine serves the product and its row factors")
    }

    fn three_kernels(&self, plan: &Plan, x: &TensorHandle) -> TensorHandle {
        let h = plan.product.buffered(x, self.w1.at(plan.layout));
        let norm = RmsNormCall {
            client: self.client.clone(),
            input: PlainTensor::of_handle(h),
            weight: self.gamma.clone(),
            epsilon: EPS,
        };
        let y = launch_tuned(norm, &self.device, GroupPriority::DEFAULT);
        plan.product.buffered(&y, self.w2.at(plan.layout))
    }

    fn deferred(&self, plan: &Plan, x: &TensorHandle) -> TensorHandle {
        let h = plan.product.row_factors(x, self.w1.at(plan.layout));
        plan.product.row_scaled(&h, self.w2_gamma.at(plan.layout))
    }

    fn two_products(&self, plan: &Plan, x: &TensorHandle) -> TensorHandle {
        let h = plan.product.buffered(x, self.w1.at(plan.layout));
        plan.product.buffered(&h, self.w2.at(plan.layout))
    }

    fn two_passed(&self, plan: &Plan, x: &TensorHandle) -> TensorHandle {
        let h = plan.product.passed(x, self.w1.at(plan.layout));
        plan.product.passed(&h, self.w2.at(plan.layout))
    }

    fn time(&self, arm: impl Fn() -> TensorHandle) -> Duration {
        future::block_on(self.client.sync()).unwrap();
        let start = Instant::now();
        let out = arm();
        future::block_on(self.client.sync()).unwrap();
        drop(out);
        start.elapsed()
    }

    fn read(&self, out: TensorHandle) -> Vec<f32> {
        let bytes = self.client.read_one_unchecked(out.handle);
        half::f16::from_bytes(&bytes)
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }
}

fn uniform(len: usize, seed: u64, lo: f32, hi: f32) -> Vec<half::f16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            half::f16::from_f32(lo + (hi - lo) * unit)
        })
        .collect()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn check(rows: usize, reference: &[f32], deferred: &[f32]) {
    let rms = (reference.iter().map(|v| v * v).sum::<f32>() / reference.len() as f32).sqrt();
    let worst = reference
        .iter()
        .zip(deferred)
        .map(|(expected, got)| (expected - got).abs() / expected.abs().max(rms))
        .fold(0.0f32, f32::max);
    assert!(
        worst <= TOLERANCE,
        "deferred norm differs from the three kernels at {rows} rows: worst relative error {worst} > {TOLERANCE}"
    );
}

fn main() {
    let samples: usize = std::env::var("CUBEK_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let bench = Bench::new();
    for rows in ROWS {
        let x = bench.input(rows);
        let plan = bench.pick(&x);
        let three_kernels = || bench.three_kernels(&plan, &x);
        let deferred = || bench.deferred(&plan, &x);
        let two_products = || bench.two_products(&plan, &x);
        let two_passed = || bench.two_passed(&plan, &x);
        check(rows, &bench.read(three_kernels()), &bench.read(deferred()));
        for _ in 0..WARM_UPS {
            bench.time(three_kernels);
            bench.time(deferred);
            bench.time(two_products);
            bench.time(two_passed);
        }
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..samples {
            times[0].push(bench.time(three_kernels));
            times[1].push(bench.time(deferred));
            times[2].push(bench.time(two_products));
            times[3].push(bench.time(two_passed));
        }
        let [
            three_median,
            deferred_median,
            products_median,
            passed_median,
        ] = times.map(median);
        let ratio = |time: Duration| time.as_secs_f64() / three_median.as_secs_f64();
        println!(
            "M={rows:<5} {}: three kernels {three_median:?}, deferred {deferred_median:?} ({:.3}), two products {products_median:?} ({:.3}), two through a pass sink {passed_median:?} ({:.3})",
            plan.product.name(),
            ratio(deferred_median),
            ratio(products_median),
            ratio(passed_median)
        );
    }
}
