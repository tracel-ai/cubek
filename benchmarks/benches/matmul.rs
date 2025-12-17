use crate::matmul::launch::launch;
use cubecl::{
    benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations, TimingMethod},
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::{
    matmul::{
        self as matmul,
        components::{batch::HypercubeSelection, stage::PartitionBuffering},
        definition::{
            LoadingPrecomputeStrategy, MatmulElems, MatmulPrecision, MatmulSelection, StageSize,
            TilingScheme,
        },
        launch::{
            AcceleratedTileKind, AsyncPartialReadingStrategy, MatmulInputHandle,
            PartialReadingStrategy, ReadingStrategy, Strategy,
        },
        routines::{
            Selection, TileSizeSelection, double_buffering::DoubleBufferingArgs,
            double_unit::DoubleUnitSelectionArgs, ordered_double_buffering::OrderedSelectionArgs,
            simple::SimpleArgs, simple_unit::SimpleUnitSelectionArgs,
        },
    },
    random::random_uniform,
};
use std::collections::BTreeMap;

impl<R: Runtime> Benchmark for MatmulBench<R> {
    type Input = (MatmulInputHandle<R>, MatmulInputHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);

        let mut lhs = TensorHandle::empty(
            &client,
            vec![self.b, self.m, self.k],
            *self.dtypes.lhs_global,
        );
        if self.tl {
            let len = lhs.shape.len();
            lhs.strides.swap(len - 2, len - 1);
        }
        random_uniform(&client, 0.0, 1.0, lhs.as_ref(), *self.dtypes.lhs_global).unwrap();

        let mut rhs = TensorHandle::empty(
            &client,
            vec![self.b, self.k, self.n],
            *self.dtypes.rhs_global,
        );

        if self.tr {
            let len = rhs.shape.len();
            rhs.strides.swap(len - 2, len - 1);
        }

        random_uniform(&client, 0.0, 1.1, rhs.as_ref(), *self.dtypes.rhs_global).unwrap();

        (
            MatmulInputHandle::Normal(lhs),
            MatmulInputHandle::Normal(rhs),
        )
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Result<Self::Output, String> {
        let client = R::client(&self.device);
        let out = TensorHandle::empty(
            &client,
            vec![self.b, self.m, self.n],
            *self.dtypes.acc_global,
        );

        match launch(
            &self.strategy,
            &self.client,
            lhs,
            rhs,
            out,
            self.dtypes.clone(),
        ) {
            Ok(_) => Ok(()),
            Err(err) => Err(format!("{err:?}")),
        }
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "{}-matmul-Lhs<{}-{}-{}>-Rhs<{}-{}-{}>-{}-{}-{:?}",
            R::name(&client),
            self.dtypes.lhs_global,
            self.dtypes.lhs_stage,
            self.dtypes.lhs_register,
            self.dtypes.rhs_global,
            self.dtypes.rhs_stage,
            self.dtypes.rhs_register,
            self.dtypes.acc_register,
            self.dtypes.acc_global,
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<cubecl::benchmark::ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "matmul-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

#[allow(dead_code)]
struct MatmulBench<R: Runtime> {
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    tl: bool,
    tr: bool,
    strategy: Strategy,
    device: R::Device,
    client: ComputeClient<R>,
    dtypes: MatmulElems,
}

#[allow(unused)]
fn entry(m: usize, n: usize, k: usize) -> (usize, usize, usize, usize) {
    let expected = 2 * 6144 * 6144 * 6144;
    let num_ops = 2 * m * n * k;

    let b = usize::max(expected / num_ops, 1);
    let b = 2usize.pow((b as f64).log(2.0).floor() as u32);
    let b = usize::min(4096, b);

    (b, m, n, k)
}

#[allow(dead_code, clippy::single_element_loop)]
fn run<R: Runtime, MP: MatmulPrecision>(device: R::Device, strategy: Strategy) {
    for tl in [true, false] {
        for tr in [true, false] {
            for (b, m, n, k) in [
                // entry(8192, 8192, 8192),
                // entry(6144, 6144, 6144),
                // entry(4096, 4096, 4096),
                // entry(2048, 2048, 2048),
                // (2, 1024, 1024, 1024),
                // entry(512, 512, 512),
                // entry(64, 1024, 64),
                // entry(32, 1024, 32),
                // entry(10, 1024, 10),
                // entry(64, 64, 1024),
                // entry(32, 32, 1024),
                // entry(10, 10, 1024),
                // entry(1024, 64, 64),
                // entry(1024, 32, 32),
                // entry(1024, 10, 10),
                // (16, 1, 2048, 8192),
                // (16, 1, 4096, 4096),
                (1, 512, 512, 512),
                // (2, 8192, 8192, 1), // Outer
                // (2, 8192, 1, 8192), // MatVec
                //(2, 1, 8192, 8192), // VecMat
            ] {
                println!("-------------------");
                let _ = run_one::<R, MP>(device.clone(), strategy.clone(), (b, m, n, k), (tl, tr));
            }
        }
    }
}

#[allow(dead_code)]
fn run_one<R: Runtime, MP: MatmulPrecision>(
    device: R::Device,
    strategy: Strategy,
    shapes: (usize, usize, usize, usize),
    transposed: (bool, bool),
) -> Result<(BenchmarkDurations, f64), String> {
    let client = R::client(&device);
    let (b, m, n, k) = shapes;
    let (tl, tr) = transposed;

    let bench = MatmulBench {
        b,
        m,
        k,
        n,
        tl,
        tr,
        client: client.clone(),
        device: device.clone(),
        strategy: strategy.clone(),
        dtypes: MatmulElems::new::<MP>(),
    };
    println!("b: {b} m: {m} n: {n} k: {k}, tl {tl}, tr {tr}");
    println!("{}", bench.name());

    match bench.run(TimingMethod::System) {
        Ok(val) => {
            let flops = 2 * b * m * n * k;
            let computed = BenchmarkComputations::new(&val);
            let tflops = flops as f64 / (computed.median.as_secs_f64() * 1e12);
            println!("TFLOPS: {tflops}");
            println!("Times: {val}");
            Ok((val, tflops))
        }
        Err(err) => {
            println!("{err:?}");
            Err(err)
        }
    }
}

#[allow(unused, clippy::single_element_loop)]
// This function should be customized to help build a proper selector that reduces the number of
// possibilities.
fn run_grid_search<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    let mut algos = BTreeMap::<u64, (BenchmarkDurations, MatmulSelection, f64)>::new();

    for t in [(16, 16, 16)] {
        for p in [(1, 1, 1)] {
            for s in [(1, 1, 1)] {
                let plane_dim = client.properties().hardware.plane_size_min;
                let tiling = TilingScheme::builder()
                    .with_tile_size(t.into())
                    .with_partition_size(p.into())
                    .with_stage_size(StageSize {
                        m: s.0,
                        n: s.1,
                        k: s.2,
                    })
                    .build()
                    .unwrap();
                let hypercube = HypercubeSelection::builder(&tiling)
                    .global_order(cubek::matmul::components::batch::GlobalOrderSelection::Default)
                    .cube_count_plan(
                        cubek::matmul::components::batch::CubeCountPlanSelection::Flattened,
                    )
                    .build();
                let selection = MatmulSelection::builder(tiling, plane_dim)
                    .plane_dim(plane_dim)
                    .partition_buffering(PartitionBuffering::Single)
                    .hypercube_config(hypercube)
                    .loading_precompute_strategy(LoadingPrecomputeStrategy::Always)
                    .build();
                let result = run_one::<R, MP>(
                    Default::default(),
                    Strategy::Simple {
                        read_strategy: ReadingStrategy::Cyclic,
                        selection: Selection::Forced(selection.clone()),
                        tile_kind: AcceleratedTileKind::Cmma,
                    },
                    (4096, 10, 64, 10),
                    (false, false),
                );

                if let Ok((duration, tflops)) = result {
                    let key = tflops * 1000000.0;
                    algos.insert(key as u64, (duration, selection, tflops));
                }
            }
        }
    }

    for (_, (duration, selection, tflops)) in algos.iter() {
        println!("==== TFLOPS: {tflops:?} ====");
        println!("Selection: {selection:?}");
        println!("Times: {duration}");
        println!("====================");
    }
}

#[allow(unused)]
fn run_algos_vecmat<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple VecMat");
    run::<R, MP>(
        Default::default(),
        Strategy::SimpleVecMat(Selection::Inferred(())),
    );

    println!("Double VecMat");
    run::<R, MP>(
        Default::default(),
        Strategy::DoubleVecMat(Selection::Inferred(())),
    );

    println!("Simple Unit Min");
    run::<R, MP>(
        Default::default(),
        Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );

    println!("Simple Unit Max");
    run::<R, MP>(
        Default::default(),
        Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );
}

#[allow(unused)]
fn run_algos_unit<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple Unit Min");
    run::<R, MP>(
        Default::default(),
        Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );

    println!("Simple Unit Max");
    run::<R, MP>(
        Default::default(),
        Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );

    println!("Double Unit Min");
    run::<R, MP>(
        Default::default(),
        Strategy::DoubleUnit(Selection::Inferred(DoubleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
    );

    println!("Double Unit Max");
    run::<R, MP>(
        Default::default(),
        Strategy::DoubleUnit(Selection::Inferred(DoubleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
    );
}

#[allow(unused)]
fn run_algos_wmma<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    println!("Simple");
    run::<R, MP>(
        Default::default(),
        Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Simple multi rows");
    run::<R, MP>(
        Default::default(),
        Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: true }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering");
    run::<R, MP>(
        Default::default(),
        Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: false }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering Specialized");
    run::<R, MP>(
        Default::default(),
        Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: true }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );

    println!("Double Buffering Ordered");
    run::<R, MP>(
        Default::default(),
        Strategy::OrderedDoubleBuffering {
            selection: Selection::Inferred(OrderedSelectionArgs {
                row_count: Some(8),
                rows_per_plane: Some(2),
                partition_k: Some(2),
            }),
            tile_kind: AcceleratedTileKind::Cmma,
        },
    );
}

#[allow(unused)]
fn run_algos_mma<R: Runtime, MP: MatmulPrecision>() {
    let client = R::client(&Default::default());

    // println!("Specialized TMA");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::Specialized {
    //         read_strategy: AsyncPartialReadingStrategy::Tma,
    //         selection: Selection::Inferred(()),
    //         tile_kind: AcceleratedTileKind::Mma,
    //     },
    // );

    // println!("Specialized Cyclic");
    // run::<R, MP>(
    //     Default::default(),
    //     matmul::Strategy::Specialized {
    //         read_strategy: AsyncPartialReadingStrategy::Cyclic,
    //         selection: Selection::Inferred(()),
    //         tile_kind: AcceleratedTileKind::Mma,
    //     },
    // );

    println!("Specialized Strided");
    run::<R, MP>(
        Default::default(),
        Strategy::Specialized {
            read_strategy: AsyncPartialReadingStrategy::Strided,
            selection: Selection::Inferred(()),
            tile_kind: AcceleratedTileKind::Mma,
        },
    );
}

#[allow(unused)]
fn run_benches<R: Runtime, MP: MatmulPrecision>() {
    // run_grid_search::<R, MP>();
    run_algos_unit::<R, MP>();
    // run_algos_wmma::<R, MP>();
    // run_algos_vecmat::<R, MP>();
    // run_algos_mma::<R, MP>();
}

fn main() {
    run_benches::<cubecl::TestRuntime, half::f16>();
}
