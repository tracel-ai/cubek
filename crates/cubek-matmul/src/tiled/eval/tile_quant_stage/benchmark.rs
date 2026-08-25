use std::sync::Arc;

use cubecl::{
    Runtime, TestRuntime,
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::ComputeClient,
    future,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};
use cubek_test_utils::{QuantizedTileInput, RunSamples, TileInput};
use cubek_tile::*;

use super::problem::TileQuantStageProblem;

/// What this bench contracts through: a 64-cell unroll budget, no edge specialization, no lane
/// fan-out. Bound on the accumulator at the kernel top so the numbers measure the staging, not
/// the instruction.
const INSTRUCTION: Instruction = Instruction::Registers {
    config: RegisterBlock::new(64, false, false),
};
use super::strategy::StageDepth;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// `C = A · dequant(B)`, `B` the packed weight: the staged lowering picks the stage form.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn staged_matmul_quant_rhs<I: Numeric, E: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &TileArg<'_, E, VA>,
    b: &QuantTileArg<'_, I, VB>,
    c: &TileArg<'_, E, VC>,
    #[comptime] space: Space,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let mut c = c.tile(space);
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
        .per_block([1, problem.bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
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
    client: ComputeClient<TestRuntime>,
    samples: usize,
}

impl TileQuantStageBench {
    /// L0 stages one `m × tn × tk` cube tile; L1 spreads that tile's `N` across the plane's lanes,
    /// one served line each, so the leaf is `mr = m`, `nr = 1`: unrolled while `m <= 64` (the
    /// `mr·nr` cliff), keeping the unroll state constant as depth varies. Both inputs state
    /// their shared stage where L0 is declared: L0 fills it, L1 reads windows of it, which is
    /// the staging this bench measures. The output stages nothing.
    fn space(&self) -> (Space, (Operand, Operand, Operand)) {
        let lanes = self.client.properties().hardware.plane_size_max as usize;
        let un = self.pack;
        let tn = lanes * un;
        let f32t = f32::elem_type_native();
        let mut operands = (
            Operand::new(&[M, K], f32t),
            Operand::new(&[K, N], f32t),
            Operand::new(&[M, N], f32t),
        );
        let space = Tiling::over(&mut operands, &[(M, self.m), (N, self.n), (K, self.k)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
                l.axis(M, Cut::sequential(self.m))
                    .axis(N, Cut::cube(CubeAxis::X, tn))
                    .axis(K, Cut::sequential(self.tk));
                o.0.stage(Residence::Smem);
                o.1.stage(Residence::Smem);
            })
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.axis(M, Cut::sequential(self.m))
                    .axis(N, Cut::unit(un))
                    .axis(K, Cut::sequential(self.tk));
            })
            .build();
        (space, operands)
    }
}

impl Benchmark for TileQuantStageBench {
    // `Benchmark::Input` must be `Clone`; the tile inputs own device handles, so share them.
    type Input = Arc<(TileInput, QuantizedTileInput, TileInput)>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let (space, _) = self.space();
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
        let (space, ops) = self.space();
        let launcher = space.launcher(&self.client);
        let a = launcher.bind(&ops.0, a.handle().binding()).build();
        let b = launcher
            .bind(&ops.1, b.tile.handle().binding())
            .vectorize(self.pack)
            .quantized(&[b.scales_binding()], self.scheme, DequantAt::Read)
            .build();
        // The register instruction lines the accumulator at the RHS's served width.
        let c = launcher
            .bind(&ops.2, c.handle().binding())
            .vectorize(self.pack)
            .build();
        let vb = b.bound_width();
        staged_matmul_quant_rhs::launch::<TestRuntime>(
            &self.client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            vb,
            c.vector_size,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.space().clone().with_instruction(INSTRUCTION),
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
