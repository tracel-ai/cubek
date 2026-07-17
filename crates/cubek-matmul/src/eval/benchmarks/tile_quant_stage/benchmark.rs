use std::sync::Arc;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubek_test_utils::{QuantizedTileInput, RunSamples, TileInput};
use cubek_tile::*;

use super::problem::TileQuantStageProblem;
use super::strategy::StageDepth;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// `C = A · dequant(B)`, `B` the packed weight — the staged lowering picks the stage form.
#[cube(launch)]
fn staged_matmul_quant_rhs<I: Numeric, E: Numeric>(
    a: &StridedTileArg<'_, E>,
    b: &StridedTileArg<'_, I>,
    c: &StridedTileArg<'_, E>,
    #[define(I)] _b_dtype: StorageType,
    #[define(E)] _e_dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile_dequant::<E>();
    let mut c = c.tile();
    c.mma(&a, &b);
}

pub fn bench(
    strategy: &StageDepth,
    problem: &TileQuantStageProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = <TestRuntime as Runtime>::Device::default();
    let client = <TestRuntime as Runtime>::client(&device);

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([1, problem.bn as u8]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);
    let pack = scheme.num_quants();
    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        return Err(format!(
            "device vectors cap at {max_width}, below the packing factor {pack}"
        ));
    }
    if !problem.k.is_multiple_of(strategy.0) {
        return Err(format!(
            "k={} is not a multiple of the stage depth {}",
            problem.k, strategy.0
        ));
    }

    let bench = TileQuantStageBench {
        m: problem.m,
        n: problem.n,
        k: problem.k,
        tk: strategy.0,
        scheme,
        pack,
        client: client.clone(),
        device,
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;

    let flops = 2.0 * problem.m as f64 * problem.n as f64 * problem.k as f64;
    Ok(RunSamples::new(durations).with_flops(flops))
}

struct TileQuantStageBench {
    m: usize,
    n: usize,
    k: usize,
    tk: usize,
    scheme: QuantScheme,
    pack: usize,
    device: <TestRuntime as Runtime>::Device,
    client: ComputeClient<TestRuntime>,
    samples: usize,
}

impl TileQuantStageBench {
    /// L0 stages one `m × tn × tk` cube tile; L1 spreads that tile's `N` across the plane's lanes,
    /// one served line each, so the leaf is `mr = m`, `nr = 1` — unrolled while `m <= 64` (the
    /// `mr·nr` cliff), keeping the unroll state constant as depth varies.
    fn space(&self) -> Space {
        let lanes = self.client.properties().hardware.plane_size_max as usize;
        let un = self.pack;
        let tn = lanes * un;
        Tiling::new()
            .extents(&[(M, self.m), (N, self.n), (K, self.k)])
            .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
                l.axis(M, Cut::sequential(self.m))
                    .axis(N, Cut::cube(CubeAxis::X, tn))
                    .axis(K, Cut::sequential(self.tk))
            })
            .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(M, Cut::sequential(self.m))
                    .axis(N, Cut::unit(un))
                    .axis(K, Cut::sequential(self.tk))
            })
            .leaf(Leaf::Register)
    }
}

impl Benchmark for TileQuantStageBench {
    // `Benchmark::Input` must be `Clone`; the tile inputs own device handles, so share them.
    type Input = Arc<(TileInput, QuantizedTileInput, TileInput)>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let space = self.space();
        let a = TileInput::builder(&self.client, space.project(&[M, K]))
            .untiled()
            .arange();
        let b = TileInput::builder(&self.client, space.project(&[K, N]))
            .untiled()
            .packed(&self.scheme)
            .arange();
        let c = TileInput::builder(&self.client, space.project(&[M, N]))
            .untiled()
            .zeros();
        Arc::new((a, b, c))
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let (a, b, c) = &*args;
        let space = self.space();
        let launcher = space.launcher(&self.client);
        staged_matmul_quant_rhs::launch::<TestRuntime>(
            &self.client,
            launcher.cube_count(),
            launcher.cube_dim(),
            launcher.arg(a.handle().binding()).subspace(&[M, K]).build(),
            launcher
                .arg(b.tile.handle().binding())
                .subspace(&[K, N])
                .vectorize(self.pack)
                .quantized(b.scales_arg(), self.scheme)
                .build(),
            // The register microkernel lines the accumulator at the RHS's served width.
            launcher
                .arg(c.handle().binding())
                .subspace(&[M, N])
                .vectorize(self.pack)
                .build(),
            u32::as_type_native_unchecked().storage_type(),
            f32::as_type_native_unchecked().storage_type(),
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "tile-quant-stage-{}-m{}-n{}-k{}-tk{}",
            <TestRuntime as Runtime>::name(&self.client),
            self.m,
            self.n,
            self.k,
            self.tk,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "tile-quant-stage-bench")
            .map(|it| it.1)
            .map_err(|it| format!("{it:?}"))
    }
}
