use std::sync::Arc;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};
use cubek_test_utils::{QuantizedTileInput, RunSamples, TileInput};
use cubek_tile::*;

use super::problem::TileQuantStageProblem;

use super::strategy::StageDepth;

/// What this bench contracts through: a 64-cell unroll budget, no edge specialization, no lane
/// fan-out, so the numbers measure the staging, not the instruction.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(64);

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// `C = A · dequant(B)`, `B` the packed weight staged in its stored form: both inputs stage
/// into shared memory per cube region, the plane's lanes read windows of the stage.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn staged_matmul_quant_rhs<I: Numeric, E: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &TileArg<'_, E, VA>,
    b: &QuantTileArg<'_, I, VB>,
    c: &TileArg<'_, E, VC>,
    space: Space,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] lanes: Level,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space.cubes(comptime!(cubes.clone())) {
        let a = a.at(&cube);
        let b = b.at(&cube);
        let c = c.at(&cube);
        let steps = cube.walk(comptime!(steps.clone()));
        let mut ring = Ring::smem(&steps, &a, &b, StageStorage::Strided, 1usize);
        pipelined(steps, &mut ring, |slot, step| {
            let c_step = c.at(step);
            slot.consume(|a_s, b_s| {
                for lane in step.lanes(comptime!(lanes.clone())) {
                    let mut c_lane = c_step.at(&lane);
                    c_lane.mma_with(
                        &a_s.at(&lane),
                        &b_s.at(&lane),
                        REGISTER_BLOCK,
                        Semiring::SUM_PROD,
                    );
                }
            });
        });
    }
}

/// The packed-weight scheme this bench quantizes `B` under: `Q8S`, block size `1 × bn`
/// (no blocking along `k`, `bn` along `n`), packed into `u32` words.
pub(super) fn quant_scheme(bn: usize) -> QuantScheme {
    QuantScheme::default()
        .per_block([1, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S)
}

pub fn bench(
    strategy: &StageDepth,
    problem: &TileQuantStageProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let device = cubecl::test_device();
    let client = device.client();

    let scheme = quant_scheme(problem.bn);
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
        samples: num_samples,
    };

    let durations = bench
        .run(TimingMethod::Device)
        .map_err(|e| format!("benchmark failed: {e}"))?
        .durations;
    // The mma contracts in f32; the packed u32 is how the RHS is stored, not what the
    // arithmetic runs at.

    Ok(RunSamples::new(durations))
}

struct TileQuantStageBench {
    m: usize,
    n: usize,
    k: usize,
    tk: usize,
    scheme: QuantScheme,
    pack: usize,
    client: Client,
    samples: usize,
}

impl TileQuantStageBench {
    /// L0 stages one `m × tn × tk` cube tile; L1 spreads that tile's `N` across the plane's lanes,
    /// one served line each, so the leaf is `mr = m`, `nr = 1`: unrolled while `m <= 64` (the
    /// `mr·nr` cliff), keeping the unroll state constant as depth varies. The kernel stages both
    /// inputs at L0 and reads windows of the stage at L1, which is the staging this bench
    /// measures. The output stages nothing.
    fn extents(&self) -> Vec<(Axis, usize)> {
        vec![(M, self.m), (N, self.n), (K, self.k)]
    }

    /// Three levels: a strip of `tn` columns per cube, `K` in `tk` steps, then `un` columns per
    /// lane.
    fn levels(&self) -> Vec<Level> {
        let plane_size = self.client.properties().hardware.plane_size_max as usize;
        let un = self.pack;
        let tn = plane_size * un;
        vec![
            Level::cubes(&[(N, tn)]),
            Level::walk(&[(K, self.tk)]),
            Level::lanes(&[Cut::new(N, un).across(plane_size)]),
        ]
    }

    fn space(&self) -> Space {
        Space::new(&self.extents())
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
            .packed(&self.scheme, DequantAt::Read)
            .arange();
        let c = TileInput::builder(&self.client, space.project(&[M, N]))
            .untiled()
            .zeros();
        Arc::new((a, b, c))
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let (a, b, c) = &*args;
        let launcher = Launcher::implied(
            &self.client,
            self.space(),
            self.levels(),
            KernelForm::Static,
        );
        let a = launcher.arg(a.handle().binding()).subspace(&[M, K]).build();
        let b = launcher
            .arg(b.tile.handle().binding())
            .subspace(&[K, N])
            .vectorize(self.pack)
            .quantized(&[b.scales_binding()], self.scheme, DequantAt::Read)
            .build();
        // The register instruction lines the accumulator at the RHS's served width.
        let c = launcher
            .arg(c.handle().binding())
            .subspace(&[M, N])
            .vectorize(self.pack)
            .build();
        let vb = b.bound_width();
        staged_matmul_quant_rhs::launch(
            &self.client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            vb,
            c.vector_size,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.space_arg(),
            launcher.level(0),
            launcher.level(1),
            launcher.level(2),
            u32::elem_type_native(),
            f32::elem_type_native(),
        );
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "tile-quant-stage-{}-m{}-n{}-k{}-tk{}",
            self.client.name(),
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
        let (launched, duration) = self
            .client
            .profile(|| self.execute(args), "tile-quant-stage-bench")
            .map_err(|it| format!("{it:?}"))?;
        launched.map(|_| duration)
    }
}
