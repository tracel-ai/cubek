//! What every random microkernel computes alone: the lines of its run drawn from one generator per
//! lane and component, seeded for its position in the launch, and turned into values of a
//! distribution. The host replays every generator, so each value is checked, not a statistic.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::random::plane::PlaneRandomMicrokernelSettings;
use cubek_microkernel::random::{Bernoulli, DistributionArgs, Normal, SeedsLaunch, Uniform};
use cubek_microkernel::testing::random::{RandomLaunch, RandomMicrokernel};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// The seeds of every launch: the three Tausworthe generators', then the LCG's.
const SEEDS: [u32; 4] = [12_345, 67_890, 13_579, 24_680];

/// The distributions a draw turns the generator's values into.
#[derive(Debug, Clone, Copy)]
pub(crate) enum HostDistribution {
    Uniform { lower: f32, upper: f32 },
    Bernoulli { probability: f32 },
    Normal { mean: f32, std: f32 },
}

/// Every distribution, at parameters away from their defaults.
pub(crate) const DISTRIBUTIONS: [HostDistribution; 3] = [
    HostDistribution::Uniform {
        lower: -3.0,
        upper: 5.0,
    },
    HostDistribution::Bernoulli { probability: 0.3 },
    HostDistribution::Normal {
        mean: 1.5,
        std: 2.0,
    },
];

impl HostDistribution {
    /// Output lines one draw writes.
    fn lines_per_draw(&self) -> usize {
        match self {
            Self::Normal { .. } => Normal::LINES_PER_DRAW,
            Self::Uniform { .. } => Uniform::LINES_PER_DRAW,
            Self::Bernoulli { .. } => Bernoulli::LINES_PER_DRAW,
        }
    }

    /// The values of one draw off `generator`, one per line it writes.
    fn draw(&self, generator: &mut HostGenerator) -> Vec<f32> {
        match *self {
            Self::Uniform { lower, upper } => {
                vec![HostGenerator::closed_open(generator.advance()).mul_add(upper - lower, lower)]
            }
            Self::Bernoulli { probability } => {
                let uniform = HostGenerator::closed_open(generator.advance());
                vec![if uniform < probability { 1.0 } else { 0.0 }]
            }
            Self::Normal { mean, std } => {
                let unit_0 = HostGenerator::open(generator.advance());
                let unit_1 = HostGenerator::open(generator.advance());
                let angle = 2.0 * core::f32::consts::PI * unit_1;
                let coeff = (unit_0.ln() * -2.0).sqrt() * std;
                vec![
                    angle.cos().mul_add(coeff, mean),
                    angle.sin().mul_add(coeff, mean),
                ]
            }
        }
    }
}

/// One component's generator: three Tausworthe generators and an LCG combined by xor, as a unit
/// advances it.
struct HostGenerator {
    taus: [u32; 3],
    lcg: u32,
}

impl HostGenerator {
    /// The generator of stream `stream`: a unit's position in the launch times its line's width,
    /// plus its component.
    fn seeded(stream: usize) -> Self {
        let spread = 1_000_000_007u32.wrapping_mul(stream as u32);
        Self {
            taus: [0, 1, 2].map(|i| spread.wrapping_add(SEEDS[i])),
            lcg: spread.wrapping_add(SEEDS[3]),
        }
    }

    /// Advance every generator and return their combined output.
    fn advance(&mut self) -> u32 {
        let taus = |z: u32, s1: u32, s2: u32, s3: u32, m: u32| {
            let b = ((z << s1) ^ z) >> s2;
            ((z & m) << s3) ^ b
        };
        self.taus[0] = taus(self.taus[0], 13, 19, 12, 4_294_967_294);
        self.taus[1] = taus(self.taus[1], 2, 25, 4, 4_294_967_288);
        self.taus[2] = taus(self.taus[2], 3, 11, 17, 4_294_967_280);
        self.lcg = self.lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.taus[0] ^ self.taus[1] ^ self.taus[2] ^ self.lcg
    }

    /// `value` in `[0, 1)`, from its upper 24 bits.
    fn closed_open(value: u32) -> f32 {
        (value >> 8) as f32 / 16_777_216.0
    }

    /// `value` in `(0, 1)`, from its upper 23 bits.
    fn open(value: u32) -> f32 {
        ((value >> 9) as f32 + 1.0) / 8_388_609.0
    }
}

/// The random microkernel `hw` picks for each case.
pub(crate) struct Random {
    pub hw: Hardware,
}

/// Which microkernel draws the run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RandomScope {
    /// One unit, over `lines` lines.
    Unit { lines: usize },
    /// One plane at its settings.
    Plane(PlaneRandomMicrokernelSettings),
}

/// A distribution drawn by a scope, `width` values a line.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RandomCase {
    pub distribution: HostDistribution,
    pub scope: RandomScope,
    pub width: usize,
}

impl RandomCase {
    /// Every distribution drawn by `scope`, at each of `widths`.
    pub(crate) fn every_distribution(scope: RandomScope, widths: &[usize]) -> Vec<Self> {
        widths
            .iter()
            .flat_map(|&width| {
                DISTRIBUTIONS.map(|distribution| Self {
                    distribution,
                    scope,
                    width,
                })
            })
            .collect()
    }
}

impl Random {
    /// # Errors
    ///
    /// Why the device runs no such microkernel.
    fn launch_at(&self, case: &RandomCase) -> Result<RandomLaunch, Refusal> {
        match case.scope {
            RandomScope::Unit { lines } => RandomLaunch::unit(&self.hw, lines, case.width),
            RandomScope::Plane(settings) => RandomLaunch::plane(&self.hw, &settings, case.width),
        }
    }

    /// Every value `launch` draws of `distribution`: each lane's generators replayed, a lane's
    /// lines landing every `stride` lines from its first.
    fn values(launch: &RandomLaunch, distribution: HostDistribution) -> Vec<f32> {
        let width = launch.width;
        let lines = launch.values / width;
        let (lanes, stride) = match launch.microkernel {
            RandomMicrokernel::Unit(_) => (1, 1),
            RandomMicrokernel::Plane(_) => (launch.plane_width, launch.plane_width),
        };
        let per_draw = distribution.lines_per_draw();
        let draws = lines / lanes / per_draw;
        let mut out = vec![f32::NAN; launch.values];
        for lane in 0..lanes {
            for component in 0..width {
                let mut generator = HostGenerator::seeded(lane * width + component);
                for nth in 0..draws {
                    for (i, value) in distribution.draw(&mut generator).into_iter().enumerate() {
                        let line = lane + (per_draw * nth + i) * stride;
                        out[line * width + component] = value;
                    }
                }
            }
        }
        out
    }
}

impl ScopeUnderTest for Random {
    type Case = RandomCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let out = TestInput::builder(client.clone(), shape![launch.values])
            .dtype(F32)
            .custom(vec![-7.0; launch.values])
            .generate_without_host_data();
        let arg = out.clone().binding().into_tensor_arg();
        let seeds = SeedsLaunch::new(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);
        match case.distribution {
            HostDistribution::Uniform { lower, upper } => launch.launch::<Uniform>(
                client,
                arg,
                F32,
                Uniform {
                    lower_bound: lower,
                    upper_bound: upper,
                }
                .args(),
                seeds,
            ),
            HostDistribution::Bernoulli { probability } => launch.launch::<Bernoulli>(
                client,
                arg,
                F32,
                Bernoulli { probability }.args(),
                seeds,
            ),
            HostDistribution::Normal { mean, std } => {
                launch.launch::<Normal>(client, arg, F32, Normal { mean, std }.args(), seeds)
            }
        }?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        let values = Self::values(&launch, case.distribution);
        vec![HostData::row_major(
            HostDataVec::F32(values),
            shape![launch.values],
        )]
    }

    /// Exact for a uniform or a Bernoulli draw; a normal one goes through the device's (or the
    /// polynomials') `ln`, `cos` and `sin`.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.distribution {
            HostDistribution::Normal { .. } => 1e-3,
            _ => 1e-6,
        }
    }
}
