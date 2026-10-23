//! The device as a routine sees it: cubecl's properties, read as they are, and the questions a
//! routine asks of them.

use core::cmp::Reverse;
use std::sync::Arc;

use cubecl::tile::launch::Refusal;
use cubecl::{
    CubeDim,
    client::Client,
    features::{AtomicUsage, MmaConfig, Plane, ScaledMmaConfig, Tma},
    ir::{DeviceProperties, ElemType, FloatKind, OpaqueType, TargetProperties, Type},
};

/// The device, as the routines see it.
///
/// A value rather than a client, so a routine's geometry is unit-tested against a device that is
/// not in the room. Cloning one costs a count increment: the properties are the client's own.
#[derive(Clone, Debug)]
pub struct Hardware {
    /// What the runtime reports about the device. Every capability and limit a routine adapts to
    /// is read here, as cubecl states it.
    pub properties: Arc<DeviceProperties>,
    /// Whether an `mma` accumulator holds its cells where the `A` operand of a contraction as
    /// deep as it is wide holds them, so a product contracted again reads it as that operand in
    /// the same registers (the tile DSL's `accumulator_as_a`).
    ///
    /// True on CUDA, and on Metal, whose simdgroup matrix gives every lane two cells of one row
    /// in each operand. False on RDNA, whose `A` runs along its rows against an accumulator down
    /// its columns, and whose RDNA3 `A` is held twice across the wave.
    pub accumulator_as_a: bool,
}

/// Bytes one coalesced memory transaction moves on the GPUs cubek has met: 128 on CUDA, HIP,
/// Metal and Vulkan hardware alike, at wave32 and wave64. A property of the memory system, not of
/// the plane width that fetches through it.
pub const GPU_TRANSACTION_BYTES: usize = 128;

/// A CPU's transaction is its cache line: 64 bytes on every x86 and Apple core a CPU runtime
/// runs on.
pub const CPU_CACHE_LINE_BYTES: usize = 64;

impl Hardware {
    /// The device `properties` describe, compiling for `target`.
    pub fn new(properties: Arc<DeviceProperties>, target: &TargetProperties) -> Self {
        let mma = &target.mma;
        Self {
            properties,
            accumulator_as_a: mma.register_layout_a == mma.register_layout_acc
                && mma.register_duplication_a == mma.register_duplication_acc,
        }
    }

    /// The device `client` runs on, compiling for its target.
    pub fn from_client(client: &Client) -> Self {
        Self::new(client.properties_shared(), client.target_properties())
    }

    /// The exact plane width, for a kernel whose correctness depends on one.
    ///
    /// The only way a routine reaches a plane width, and fallible, because both shortcuts are
    /// wrong. `plane_size_max > 1` does not mean planes work: the wgpu runtime reports
    /// `8..=128` for an adapter with no subgroups at all, and a `plane_sum` there is undefined.
    /// And `plane_size_max` is not the width: wgpu reports a range on AMD RDNA (32 to 64) and
    /// Intel (8 to 32), and a cube sized to the max sums over a fraction of each row.
    ///
    /// # Errors
    ///
    /// [`Refusal::PlanesUnavailable`] without plane operations, [`Refusal::ImprecisePlaneDim`]
    /// where the runtime reports a range.
    pub fn plane_dim(&self) -> Result<u32, Refusal> {
        let hardware = &self.properties.hardware;
        if !self.properties.features.plane.contains(Plane::Ops) {
            return Err(Refusal::PlanesUnavailable);
        }
        if hardware.plane_size_min != hardware.plane_size_max {
            return Err(Refusal::ImprecisePlaneDim {
                min: hardware.plane_size_min,
                max: hardware.plane_size_max,
            });
        }
        Ok(hardware.plane_size_max)
    }

    /// Units that cooperate as one plane: the device's width where it commits to one, and `1`
    /// where a plane is a single unit.
    ///
    /// The width a derivation asks for. [`plane_dim`](Self::plane_dim) is the width a kernel asks
    /// for when it issues plane instructions; a geometry that only divides a microkernel between
    /// cooperating units gets `1` on a runtime with none, the width the tile engine builds its
    /// cube from there. A single-unit plane is read off what makes it one, a committed width of
    /// one and no plane operations, as cubecl's CPU runtime reports.
    ///
    /// # Errors
    ///
    /// [`plane_dim`](Self::plane_dim)'s, for a fabricated width or a range.
    pub fn plane_size(&self) -> Result<u32, Refusal> {
        let hardware = &self.properties.hardware;
        let single_unit = !self.properties.features.plane.contains(Plane::Ops)
            && hardware.plane_size_min == 1
            && hardware.plane_size_max == 1;
        match single_unit {
            true => Ok(1),
            false => self.plane_dim(),
        }
    }

    /// The width the engine builds a cube from, whatever the device committed to: sound only
    /// where no axis of the space rides `Unit` scope.
    ///
    /// cubek builds `cube_dim.x` from `plane_size_max` and asserts only that the axes riding
    /// `Unit` scope multiply to it. A cut that hands disjoint work to whole planes has no `Unit`
    /// axis, so this is the width the launch is going to use rather than a guess about the
    /// device. Units that share an item and sum across each other need
    /// [`plane_size`](Self::plane_size) instead.
    pub fn dispatch_plane_size(&self) -> u32 {
        self.properties.hardware.plane_size_max.max(1)
    }

    /// The most planes of the width the device dispatches that one cube holds: a cube as full as
    /// the device allows. On a CPU runtime, where a plane is one unit, a unit on every core.
    pub fn full_cube_planes(&self) -> usize {
        (self.properties.hardware.max_units_per_cube / self.dispatch_plane_size()).max(1) as usize
    }

    /// A flat cube of `units`, clamped to what the device spawns: for the element-wise kernels
    /// that want a reasonable block, where the count is a throughput preference.
    pub fn cube_dim_1d(&self, units: u32) -> CubeDim {
        CubeDim::new_1d(
            units
                .min(self.properties.hardware.max_units_per_cube)
                .max(1),
        )
    }

    /// Shared memory a cube may claim: the reported maximum, less what the driver reserves for
    /// cube-scoped matrices. Every budget goes through here: reading the maximum directly
    /// over-subscribes on a runtime that reserves, as Vulkan does, and fails the launch.
    pub fn shared_memory_budget(&self) -> usize {
        let hardware = &self.properties.hardware;
        hardware
            .max_shared_memory_size
            .saturating_sub(hardware.cube_mma_reserved_shared_memory)
    }

    /// Independent workers to spread across: a CPU runtime's cores, or a GPU's compute units.
    /// `None` where nothing reports one, and a caller keeps its behaviour there rather than guess.
    pub fn workers(&self) -> Option<u32> {
        let hardware = &self.properties.hardware;
        hardware
            .num_cpu_cores
            .or(hardware.num_streaming_multiprocessors)
    }

    /// Units a launch wants in flight to keep this device busy, or `None` when the device does
    /// not say how wide it is.
    ///
    /// Units, not cubes: a worker hides memory latency behind other units' requests, so one cube
    /// per worker starves it however large the cube, and counting cubes would make the answer
    /// depend on a cube size decided after it. A CPU runtime's units are threads, one per core.
    ///
    /// The per-worker figure is a floor no runtime reports. Two measurements agree on it: a
    /// shared-memory gemv's best target was 8192 units, and the decode attention that measured
    /// fastest on an M2 Pro ran 256 single-plane cubes, also 8192 units, about 431 per worker.
    pub fn units_to_fill(&self) -> Option<u32> {
        const UNITS_PER_GPU_WORKER: u32 = 512;
        let hardware = &self.properties.hardware;
        match hardware.num_cpu_cores {
            Some(cores) => Some(cores.max(1)),
            None => Some(
                hardware
                    .num_streaming_multiprocessors?
                    .saturating_mul(UNITS_PER_GPU_WORKER)
                    .max(1),
            ),
        }
    }

    /// Cubes of `units_per_cube` units that fill this device, or `None` when it does not say how
    /// wide it is. A preference: the grid may exceed it.
    pub fn cubes_to_fill(&self, units_per_cube: u32) -> Option<u32> {
        Some((self.units_to_fill()? / units_per_cube.max(1)).max(1))
    }

    /// How many values of `elem` one load instruction holds: what a step counts its loads in.
    pub fn load_values(&self, elem: ElemType) -> usize {
        (self.properties.hardware.load_width as usize / elem.size_bits().max(1)).max(1)
    }

    /// The widest vector this device reads `elem` in: its IO width over the element's, capped by
    /// the most components a vector takes. The device's half of a ceiling: a launch reads at the
    /// microkernel's, narrowed until the operands admit it.
    pub fn vector_ceiling(&self, elem: ElemType) -> usize {
        (self.properties.io_width() as usize / elem.size_bits().max(1))
            .max(1)
            .min(self.properties.hardware.max_vector_size.max(1))
    }

    /// Bytes the memory system moves in one transaction, whatever share of them a read wants:
    /// [`GPU_TRANSACTION_BYTES`] on a GPU, [`CPU_CACHE_LINE_BYTES`] on a CPU runtime. No runtime
    /// reports one, so it is stated per kind of runtime.
    ///
    /// What it bounds is waste: a walk whose units read consecutive addresses is coalesced
    /// however many they are, and this decides whether the run they cover per step fills a
    /// transaction.
    pub fn transaction_bytes(&self) -> usize {
        match self.properties.hardware.num_cpu_cores {
            Some(_) => CPU_CACHE_LINE_BYTES,
            None => GPU_TRANSACTION_BYTES,
        }
    }

    /// The distance, in elements, between the starts of consecutive runs of `run` elements of
    /// `elem` laid one after another: a whole and odd number of transactions.
    ///
    /// Odd because a memory system that picks a channel from low power-of-two address bits sends
    /// runs a power of two bytes apart to one channel; an odd step reaches every slot of such an
    /// interleave. It costs at most one transaction of padding a run.
    pub fn staggered_pitch(&self, run: usize, elem: ElemType) -> usize {
        let elem_bits = elem.size_bits().max(1);
        let transaction_bits = self.transaction_bytes() * 8;
        let transactions = (run * elem_bits).div_ceil(transaction_bits).max(1);
        let odd = match transactions.is_multiple_of(2) {
            true => transactions + 1,
            false => transactions,
        };
        odd * transaction_bits / elem_bits
    }

    /// Planes per cube for a plane-parallel launch of `working_units` units: on a CPU runtime
    /// capped at the core count, on a GPU a power of two up to eight, and never a cube of no
    /// unit on a device whose unit budget is narrower than its plane.
    pub fn plane_count_per_cube(&self, working_units: usize, plane_dim: u32) -> u32 {
        const NUM_PLANE_MAX: u32 = 8;
        let hardware = &self.properties.hardware;
        let plane_count = match hardware.num_cpu_cores {
            Some(num_cores) => num_cores.min(working_units as u32),
            None => {
                let plane_count_max = (working_units / plane_dim as usize).max(1);
                2u32.pow(NUM_PLANE_MAX.ilog2().min(plane_count_max.ilog2()))
            }
        };
        let max_plane_per_cube = (hardware.max_units_per_cube / plane_dim).max(1);
        plane_count.min(max_plane_per_cube).max(1)
    }

    /// The tallest cube, in planes of `plane_size`, the device spawns once the plane width has
    /// taken `cube_dim.x`: the cap on `cube_dim.y`.
    ///
    /// A budget, not a preference. A bound that lowers what the caller asked for would also erase
    /// the differences between the values autotune measures, so what a taller cube is worth is
    /// measured and only what the device can spawn is stated here.
    pub fn strip_budget(&self, plane_size: usize) -> usize {
        let hardware = &self.properties.hardware;
        let cube = hardware.max_units_per_cube as usize / plane_size.max(1);
        cube.min(hardware.max_cube_dim.1 as usize).max(1)
    }

    /// Whether a cube's shared memory takes an atomic add on an `f32`: what a cube-scope
    /// microkernel sums its planes' partials through.
    pub fn shared_f32_atomic_add(&self) -> bool {
        self.properties
            .shared_atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
            .contains(AtomicUsage::Add)
    }

    /// Whether a stage may be filled by the TMA engine: the bulk tensor copy, and the barriers its
    /// copies land on.
    pub fn bulk_tensor_copy(&self) -> bool {
        self.properties.features.tma.contains(Tma::Base) && self.barrier_ops()
    }

    /// Whether a stage may be filled by the units' async copies: the copy engine, and the
    /// barriers its copies land on.
    pub fn unit_async_copy(&self) -> bool {
        self.properties.features.copy_async && self.barrier_ops()
    }

    /// Whether the runtime carries a barrier type, which is how a stage's producer and consumer
    /// meet when they are different planes.
    pub fn barrier_ops(&self) -> bool {
        self.properties
            .features
            .types
            .opaque
            .contains(&OpaqueType::Barrier)
    }
}

impl From<&Client> for Hardware {
    fn from(client: &Client) -> Self {
        Self::new(client.properties_shared(), client.target_properties())
    }
}

/// The ranking of a device's fragment contractions: an iterator over one of cubecl's
/// `MatmulFeatures` sets (`cmma`, `mma`), narrowed and ordered the way a routine picks from it.
pub trait Instructions: Iterator<Item = MmaConfig> + Clone {
    /// The shapes multiplying `lhs` by `rhs` into `acc`, the accumulate element: a half-precision
    /// output accumulates in `f32` and casts on drain, so asking with the stored type hides the
    /// `f16 × f16 → f32` shape a product wants.
    fn for_triple(self, lhs: ElemType, rhs: ElemType, acc: ElemType) -> impl Instructions {
        self.filter(move |c| c.a_type == lhs && c.b_type == rhs && c.cd_type == acc)
    }

    /// The shapes an attention contracts both its products on, for operands of `elem`:
    /// `elem × elem → f32`. The score is that triple because its operands are; the mix is that
    /// triple because the probabilities are cast to the operands' element before it, the softmax
    /// staying at `f32`.
    fn for_attention(self, elem: ElemType) -> impl Instructions {
        self.for_triple(elem, elem, ElemType::Float(FloatKind::F32))
    }

    /// Whether every float element an attention reads, `f16`, `bf16` and `f32`, is offered a
    /// shape for both its contractions.
    fn serves_attention_in_every_float(self) -> bool {
        [FloatKind::F16, FloatKind::BF16, FloatKind::F32]
            .into_iter()
            .all(|kind| {
                self.clone()
                    .for_attention(ElemType::Float(kind))
                    .next()
                    .is_some()
            })
    }

    /// The largest block offered, by area and then by depth: the reuse a block buys grows with
    /// the block. `None` where nothing is offered.
    fn largest(self) -> Option<MmaConfig> {
        self.max_by_key(|shape| (shape.m as usize * shape.n as usize, shape.k))
    }

    /// Every shape, the one that contracts the most per call first: by `m · n · k`, then the
    /// squarest, then by `m`, `n` and `k`. Among shapes contracting as much, a square one reads a
    /// rhs laid along `k` as well as one laid along `n`, where a wide one reads it in narrow
    /// strips: on an RTX 2060, `16×16×16` beats `8×32×16` on a col weight and ties it on a row
    /// one. Each next shape is the least one ranked after the last, so nothing is allocated.
    fn largest_first(self) -> impl Iterator<Item = MmaConfig> {
        let first = self.clone().min_by_key(size_rank);
        core::iter::successors(first, move |last| {
            self.clone()
                .filter(|shape| size_rank(shape) > size_rank(last))
                .min_by_key(size_rank)
        })
    }

    /// These shapes less every one offered beside a deeper one of the same tile and elements: the
    /// deeper contracts the tile in fewer calls. The shallower stays in the device's set for what
    /// reads its operand from an accumulator of that tile, as an attention's value from its score.
    fn deepest(self) -> impl Instructions {
        let offered = self.clone();
        self.filter(move |shape| {
            !offered.clone().any(|other| {
                other.k > shape.k
                    && MmaConfig {
                        k: shape.k,
                        ..other
                    } == *shape
            })
        })
    }
}

impl<I: Iterator<Item = MmaConfig> + Clone> Instructions for I {}

/// [`Instructions::largest_first`]'s order, the shape itself breaking a tie so every shape has
/// its own place.
#[allow(clippy::type_complexity)]
fn size_rank(shape: &MmaConfig) -> (Reverse<usize>, usize, u32, u32, u32, MmaConfig) {
    let (m, n, k) = (shape.m as usize, shape.n as usize, shape.k as usize);
    (
        Reverse(m * n * k),
        m.abs_diff(n),
        shape.m,
        shape.n,
        shape.k,
        *shape,
    )
}

/// The fragment contractions cubek's own transport over the matrix intrinsics (`mma`) runs on
/// `hw`: the runtime's `mma` set, and its block-scaled instructions over NVFP4, `e2m1` operands
/// under an `e4m3` scale every [`NVFP4_BLOCK`] values of `k`, which the transport runs on the
/// stored words as they lie.
///
/// The runtime states a block-scaled instruction's operands as the `e2m1x2` pairs a register
/// holds; here they are the `e2m1` values a work item stores. An `e2m1` value means nothing
/// without the scale of its block, so the `mma` set's unscaled `e2m1` shapes are left out.
pub fn manual_mma(hw: &Hardware) -> impl Instructions + '_ {
    let matmul = &hw.properties.features.matmul;
    let e2m1 = ElemType::Float(FloatKind::E2M1);
    matmul
        .mma
        .iter()
        .copied()
        .filter(move |config| config.a_type != e2m1 && config.b_type != e2m1)
        .chain(matmul.scaled_mma.iter().filter_map(nvfp4_instruction))
}

/// Whether `hw` contracts NVFP4 as it is stored: a block-scaled instruction over `e2m1`
/// operands among its [`manual_mma`] shapes.
pub fn block_scaled_nvfp4(hw: &Hardware) -> bool {
    let e2m1 = ElemType::Float(FloatKind::E2M1);
    manual_mma(hw)
        .for_triple(e2m1, e2m1, ElemType::Float(FloatKind::F32))
        .next()
        .is_some()
}

/// The shape `config` contracts, where it is a block-scaled instruction over NVFP4.
fn nvfp4_instruction(config: &ScaledMmaConfig) -> Option<MmaConfig> {
    let fp4 = ElemType::Float(FloatKind::E2M1x2);
    let nvfp4 = config.a_type == fp4
        && config.b_type == fp4
        && config.scales_type == ElemType::Float(FloatKind::E4M3)
        && config.scales_factor as usize * NVFP4_BLOCK == config.k as usize;
    nvfp4.then_some(MmaConfig {
        a_type: ElemType::Float(FloatKind::E2M1),
        b_type: ElemType::Float(FloatKind::E2M1),
        cd_type: config.cd_type,
        m: config.m,
        n: config.n,
        k: config.k,
    })
}

/// Bytes a descriptor's start address and strides are whole multiples of, and the least a box's
/// inner row moves.
pub const TMA_ALIGNMENT_BYTES: usize = 16;

/// Bytes of a box's inner row the TMA engine copies at full speed: shorter rows land as many more,
/// smaller requests. A measurement rather than a derivation, the engine's speed being the
/// device's: on the RTX PRO 6000 a stage landed in 32-byte rows copies about 40% slower than the
/// same stage in 64-byte rows (the NVFP4 product with no contraction, Qwen3-4B's down projection
/// at 2048 rows, 126 against 89 µs). The step rule cuts a stage the engine fills to it
/// (`MatmulStep::filled_by_tma` in cubek-matmul).
pub const TMA_ROW_BYTES: usize = 64;

/// 32-bit registers one unit addresses before a kernel spills to memory: NVIDIA's 255. The NVFP4
/// settings read it, which a device offers only with NVIDIA's block-scaled instruction, and a unit's
/// register block unrolls no more cells than it.
pub const UNIT_REGISTERS: u32 = 255;

/// Values of `k` one NVFP4 block scale covers.
pub const NVFP4_BLOCK: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices;
    use cubecl::{
        features::Features,
        ir::{DeviceIdentity, HardwareProperties, MemoryDeviceProperties},
    };
    use cubecl_common::profile::TimingMethod;

    /// A device reporting `plane_size_min..=plane_size_max` planes, with plane operations or
    /// without, on `num_cpu_cores` cores where it is a CPU runtime.
    fn device(
        plane_size_min: u32,
        plane_size_max: u32,
        plane_ops: bool,
        num_cpu_cores: Option<u32>,
    ) -> Hardware {
        let mut features = Features::default();
        if plane_ops {
            features.plane.insert(Plane::Ops);
        }
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min,
            plane_size_max,
            max_bindings: 8,
            max_shared_memory_size: 64 * 1024,
            max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: num_cpu_cores.is_none().then_some(19),
            num_cpu_cores,
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: 4,
            cube_mma_reserved_shared_memory: 0,
        };
        let identity = DeviceIdentity {
            name: "fixture".into(),
            fingerprint: "fixture".into(),
            physical: None,
        };
        Hardware {
            properties: Arc::new(DeviceProperties::new(
                features,
                MemoryDeviceProperties::new(u64::MAX, 4),
                hardware,
                TimingMethod::System,
                identity,
            )),
            accumulator_as_a: false,
        }
    }

    fn shape(m: u32, n: u32, k: u32) -> MmaConfig {
        let f16 = ElemType::Float(FloatKind::F16);
        MmaConfig {
            a_type: f16,
            b_type: f16,
            cd_type: ElemType::Float(FloatKind::F32),
            m,
            n,
            k,
        }
    }

    /// Runs start an odd number of transactions apart: a power-of-two pitch takes one
    /// transaction more, a pitch already odd keeps its length, a run shorter than a transaction
    /// takes a whole one, and a CPU's transaction is its cache line.
    #[test]
    fn a_staggered_pitch_is_an_odd_count_of_transactions() {
        let f16 = ElemType::Float(FloatKind::F16);
        let f32 = ElemType::Float(FloatKind::F32);
        let gpu = device(32, 32, true, None);
        assert_eq!(gpu.staggered_pitch(4096, f16), 4096 + 64);
        assert_eq!(gpu.staggered_pitch(12288, f16), 12288 + 64);
        assert_eq!(gpu.staggered_pitch(4096, f32), 4096 + 32);
        assert_eq!(gpu.staggered_pitch(4160, f16), 4160);
        assert_eq!(gpu.staggered_pitch(100, f16), 192);
        assert_eq!(gpu.staggered_pitch(8, f16), 64);
        let cpu = device(1, 1, false, Some(12));
        assert_eq!(cpu.staggered_pitch(4096, f16), 4096 + 32);
        for run in [1, 63, 64, 65, 1000, 4096, 12288] {
            let pitch = gpu.staggered_pitch(run, f16);
            let transactions = pitch * 2 / gpu.transaction_bytes();
            assert!(pitch >= run && transactions % 2 == 1, "{run} -> {pitch}");
        }
    }

    /// A single-unit plane is read off its properties, a fabricated width or a range is refused,
    /// and an exact width is the width.
    #[test]
    fn only_a_committed_plane_width_is_a_width() {
        assert_eq!(device(32, 32, true, None).plane_dim(), Ok(32));
        assert_eq!(device(1, 1, false, Some(12)).plane_size(), Ok(1));
        assert_eq!(
            device(8, 128, false, None).plane_dim(),
            Err(Refusal::PlanesUnavailable)
        );
        assert_eq!(
            device(32, 64, true, None).plane_dim(),
            Err(Refusal::ImprecisePlaneDim { min: 32, max: 64 })
        );
    }

    /// A fabricated width wider than the unit budget still yields a cube that launches.
    #[test]
    fn plane_count_never_collapses_to_zero() {
        let mut narrow = device(8, 128, false, None);
        Arc::make_mut(&mut narrow.properties)
            .hardware
            .max_units_per_cube = 64;
        assert!(narrow.plane_count_per_cube(1024, 128) >= 1);
    }

    /// A device that does not say how wide it is is not sized for a guess, a CPU core runs one
    /// unit, and a wider cube fills the same device in fewer of itself.
    #[test]
    fn a_fill_counts_units_on_the_workers_reported() {
        let mut unknown = device(32, 32, true, None);
        Arc::make_mut(&mut unknown.properties)
            .hardware
            .num_streaming_multiprocessors = None;
        assert_eq!(unknown.units_to_fill(), None);
        assert_eq!(unknown.cubes_to_fill(32), None);
        assert_eq!(device(1, 1, false, Some(12)).units_to_fill(), Some(12));
        let gpu = device(32, 32, true, None);
        assert_eq!(
            gpu.cubes_to_fill(32).unwrap(),
            gpu.cubes_to_fill(256).unwrap() * 8
        );
    }

    /// A shallower shape of a tile drops out where a deeper one of the same tile and elements is
    /// offered, and stays where none is.
    #[test]
    fn only_the_deepest_shape_of_a_tile_is_a_candidate() {
        let bf16 = ElemType::Float(FloatKind::BF16);
        let shallow_bf16 = MmaConfig {
            a_type: bf16,
            b_type: bf16,
            ..shape(16, 8, 8)
        };
        let offered = [shape(16, 8, 16), shape(16, 8, 8), shallow_bf16];
        assert_eq!(
            offered.into_iter().deepest().collect::<Vec<_>>(),
            vec![shape(16, 8, 16), shallow_bf16]
        );
    }

    /// The shape that contracts the most per call comes first whatever order the set holds them
    /// in; among shapes contracting as much, the square one, then the one fewest rows high.
    #[test]
    fn the_largest_shape_is_offered_first() {
        let order = |offered: [MmaConfig; 3]| -> Vec<_> {
            offered
                .into_iter()
                .largest_first()
                .map(|shape| (shape.m, shape.n, shape.k))
                .collect()
        };
        assert_eq!(
            order([shape(16, 16, 16), shape(32, 32, 8), shape(16, 16, 32)]),
            [(16, 16, 32), (32, 32, 8), (16, 16, 16)]
        );
        assert_eq!(
            order([shape(8, 32, 16), shape(32, 8, 16), shape(16, 16, 16)]),
            [(16, 16, 16), (8, 32, 16), (32, 8, 16)]
        );
    }

    /// The NVFP4 offer is the block-scaled instruction stated over `e2m1` values, and an unscaled
    /// `e2m1` shape is no offer.
    #[test]
    fn a_block_scaled_instruction_is_offered_over_its_values() {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        let unscaled = MmaConfig {
            a_type: e2m1,
            b_type: e2m1,
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 32,
        };
        let hw = devices::offering_mma(devices::h100(), [unscaled]);
        assert!(!block_scaled_nvfp4(&hw));
        let hw = devices::offering_nvfp4(hw);
        assert_eq!(
            manual_mma(&hw).collect::<Vec<_>>(),
            [MmaConfig { k: 64, ..unscaled }]
        );
        assert!(block_scaled_nvfp4(&hw));
    }
}
