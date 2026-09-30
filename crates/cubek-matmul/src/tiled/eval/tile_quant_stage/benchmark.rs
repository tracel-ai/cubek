use std::sync::Arc;

use cubecl::{
    benchmark::{Benchmark, ProfileDuration, TimingMethod},
    client::Client,
    future,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};
use cubek_test_utils::{QuantizedTileInput, RunSamples, TileInput};
use cubek_tile::launch::Grid;
use cubek_tile::layout::PhysicalAxisMap;
use cubek_tile::*;

use super::problem::TileQuantStageProblem;

use super::strategy::StageDepth;

/// What this bench contracts through: a 64-cell unroll budget, no edge specialization, no unit
/// fan-out, so the numbers measure the staging, not the instruction.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(64);

const M: Axis = Axis(0);
const K: Axis = Axis(2);
/// `N` as the scale blocks make it: which block, and where inside it.
const NB: Axis = Axis(3);
const NI: Axis = Axis(4);

/// `C = A · (B ⊗ S)`, `B` the packed weight staged in its stored form: both inputs stage into
/// shared memory per cube region, and the plane's units read windows of the stage, each scaled
/// where it is read (`b_s.at(&unit).mul(..)`), which is the decode this bench measures.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn staged_matmul_quant_rhs<E: Numeric, VA: Size, VB: Size, VC: Size>(
    a: &TileArg<'_, E, VA>,
    b: &TileArg<'_, u32, VB>,
    scales: &TileArg<'_, f32, Const<1>>,
    c: &TileArg<'_, E, VC>,
    space: Partitioning,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile_as::<E>(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let a = a.at(&cube);
        let b = b.at(&cube);
        let scales = scales.at(&cube);
        let c = c.at(&cube);
        let steps = cube.walk();
        let mut stages = Stages::smem(&steps, &a, &b, StageStorage::Strided, 1usize);
        stages.pipelined(steps, |slot, step| {
            let c_step = c.at(step);
            let scales_step = scales.at(step);
            slot.consume(|a_s, b_s| {
                for unit in step {
                    let mut c_unit = c_step.at(&unit);
                    c_unit.mma_with(
                        &a_s.at(&unit),
                        &b_s.at(&unit).mul(&scales_step.at(&unit)),
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
        .run(cubek_test_utils::timing_method(TimingMethod::Device))
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
    /// L0 stages one `m × tn × tk` cube tile; L1 spreads that tile's `N` across the plane's units,
    /// one served line each, so the leaf is `mr = m`, `nr = 1`: unrolled while `m <= 64` (the
    /// `mr·nr` cliff), keeping the unroll state constant as depth varies. The kernel stages both
    /// inputs at L0 and reads windows of the stage at L1, which is the staging this bench
    /// measures. The output stages nothing.
    ///
    /// `N` is split into its scale blocks (`NB` of `bn` columns, `NI` inside one), so a block's
    /// scale is looked up at the value's own coordinate.
    fn extents(&self) -> Vec<(Axis, usize)> {
        let bn = self.bn();
        vec![(M, self.m), (NB, self.n / bn), (NI, bn), (K, self.k)]
    }

    /// Three levels: a strip of `plane_size · un` columns per cube, `K` in `tk` steps, then `un`
    /// columns per unit, the units running across a block's columns first, then across blocks.
    fn levels(&self) -> Vec<Level> {
        let plane_size = self.client.properties().hardware.plane_size_max as usize;
        let (un, bn) = (self.pack, self.bn());
        Levels::leaf(&[(NI, un), (NB, 1), (K, self.tk)])
            .units(&[(NI, bn / un), (NB, plane_size * un / bn)])
            .walk_every(&[K])
            .cubes(&[NB])
            .build()
    }

    /// The scale block along `N`, which the scheme states.
    fn bn(&self) -> usize {
        // The block is `1 × bn`; the scheme keeps its trailing dims, the `N` one last.
        let block = self
            .scheme
            .block_size()
            .expect("the bench's scheme is block-scaled");
        block[block.len() - 1] as usize
    }

    /// `N`'s one physical dim, in blocks and inside one.
    fn n_dim(&self) -> PhysicalAxisMap {
        PhysicalAxisMap::disjoint(&[(NB, self.bn()), (NI, 1)])
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
        // The buffers are the plain matrices; only the kernel's view splits `N`.
        const N: Axis = Axis(1);
        let space = Space::new(&[(M, self.m), (N, self.n), (K, self.k)]);
        let a = TileInput::builder(&self.client, space.subspace(&[M, K]))
            .untiled()
            .arange();
        let b = TileInput::builder(&self.client, space.subspace(&[K, N]))
            .untiled()
            .packed(&self.scheme)
            .arange();
        let c = TileInput::builder(&self.client, space.subspace(&[M, N]))
            .untiled()
            .zeros();
        Arc::new((a, b, c))
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let (a, b, c) = &*args;
        let launcher = {
            let partitioning = Partitioning::new(self.space(), self.levels());
            let concrete = partitioning.space().clone();
            Launcher::new(&self.client, partitioning, &concrete, Grid::FromLevels)
                .map_err(|refusal| refusal.to_string())?
        };
        let a = launcher
            .arg(a.handle().binding())
            .axes(&[M, K])
            .build()
            .map_err(|refusal| refusal.to_string())?;
        let b_projection = Projection::new(&[K, NB, NI], &[PhysicalAxisMap::of(K), self.n_dim()]);
        // The register instruction lines the accumulator at the RHS's served width.
        let c_spec = TileSpec::new(Projection::new(
            &[M, NB, NI],
            &[PhysicalAxisMap::of(M), self.n_dim()],
        ));
        staged_matmul_quant_rhs::launch(
            &self.client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a.vector_size,
            1,
            self.pack,
            a.arg(),
            b.values_arg(TileSpec::new(b_projection.clone())),
            b.scales_arg(TileSpec::new(b_projection.scales_per(NB))),
            TileArgLaunch::new(c.tensor_arg(self.pack), c_spec),
            launcher.partitioning_arg(),
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
        cubek_test_utils::profile_launch(&self.client, "tile-quant-stage-bench", || {
            self.execute(args)
        })
    }
}
