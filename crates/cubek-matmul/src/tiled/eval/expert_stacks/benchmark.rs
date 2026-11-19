use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cubecl::benchmark::{Benchmark, ProfileDuration, TimingMethod};
use cubecl::client::Client;
use cubecl::future;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantScheme;
use cubecl::server::MemoryLayoutDescriptor;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::Axis;
use cubecl::tile::kind::Field;
use cubecl::zspace::metadata::Metadata;
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_std::layout::stored::Operand;
use cubek_std::layout::tile_order;
use cubek_std::matmul_axes::{K, N};
use cubek_test_utils::RunSamples;
use half::f16;

use super::problem::{ExpertProduct, ExpertStackProblem, mixed};
use super::strategy::StackLayout;
use crate::kernels::indexed::m_grouped::{self, MGroupedMatmulTensors};
use crate::kernels::indexed::routed::{self, RoutedMatmulTensors};
use crate::kernels::indexed::stack::SlabAxis;
use crate::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};
use crate::tiled::storage::{self, StoredValues};

const WORD: ElemType = ElemType::UInt(UIntKind::U32);
const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

/// NVFP4 codes one `u32` word holds.
const VALUES_PER_WORD: usize = 8;

/// Launches before timing: the first runs the autotune, the rest settle the clocks.
const WARM_UP: usize = 3;

/// The bytes a row of `row_bytes` takes in a buffer CUDA's and HIP's pitched allocator lays out
/// (cubecl's `PitchedMemoryLayoutPolicy`): rounded up to the power of two above it, clamped to
/// `16..=512` bytes.
pub(super) fn pitch(row_bytes: usize) -> usize {
    let align = row_bytes.next_power_of_two().clamp(16, 512);
    row_bytes.next_multiple_of(align)
}

pub fn bench(
    layout: &StackLayout,
    problem: &ExpertStackProblem,
    num_samples: usize,
) -> Result<RunSamples, String> {
    let bench = unwound(|| ExpertStackBench::new(*layout, problem, num_samples))?;
    let durations = unwound(|| {
        bench
            .run(cubek_test_utils::timing_method(TimingMethod::Device))
            .map_err(|e| format!("benchmark failed: {e}"))
    })?
    .durations;
    Ok(RunSamples::new(durations))
}

/// Each layout's durations over `rounds` rounds, each round one call of every layout in turn, the
/// first layout of a round moving along: the device's clocks drift over a run, and a laptop's by a
/// third, so layouts timed one after another compare clocks as much as layouts. A layout whose
/// stack the products cannot read answers why.
pub fn interleaved(
    problem: &ExpertStackProblem,
    layouts: &[StackLayout],
    rounds: usize,
) -> Vec<(StackLayout, Result<Vec<Duration>, String>)> {
    let benches: Vec<_> = layouts
        .iter()
        .map(|&layout| {
            (
                layout,
                unwound(|| ExpertStackBench::new(layout, problem, rounds)),
            )
        })
        .collect();
    let mut durations: Vec<Result<Vec<Duration>, String>> = benches
        .iter()
        .map(|(_, bench)| bench.as_ref().map(|_| Vec::new()).map_err(Clone::clone))
        .collect();
    let live: Vec<usize> = (0..benches.len())
        .filter(|&i| benches[i].1.is_ok())
        .collect();
    let timing = cubek_test_utils::timing_method(TimingMethod::Device);
    for round in 0..WARM_UP_ROUNDS + rounds {
        for turn in 0..live.len() {
            let i = live[(round + turn) % live.len()];
            let (Ok(bench), Ok(samples)) = (&benches[i].1, &mut durations[i]) else {
                continue;
            };
            match unwound(|| bench.once(timing)) {
                Ok(duration) if round >= WARM_UP_ROUNDS => samples.push(duration),
                Ok(_) => {}
                Err(why) => durations[i] = Err(why),
            }
        }
    }
    layouts.iter().copied().zip(durations).collect()
}

/// Rounds of [`interleaved`] before it times: the autotune, and the clocks rising.
const WARM_UP_ROUNDS: usize = 20;

/// `run`'s answer, or what it panicked with: a layout the products cannot read panics in a
/// launch, which is that layout's error, not the run's.
fn unwound<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|why| why.to_string()))
            .unwrap_or_default();
        Err(format!("panicked: {why}"))
    })
}

/// What one call reads: the stack whole, not yet read as its slabs, the activation, and the
/// routes or the groups' ends.
struct StackInputs {
    stack: PackedTensor,
    lhs: PlainTensor,
    /// The routed product's routes, one set a call in turn, each naming slabs no other set
    /// names: a call reads its slabs cold, as a model's next token reads another layer's.
    routes: Vec<PlainTensor>,
    /// The grouped product's cumulative group ends, one per expert.
    ends: Option<PlainTensor>,
}

impl StackInputs {
    fn new(
        client: &Client,
        layout: StackLayout,
        problem: &ExpertStackProblem,
    ) -> Result<Self, String> {
        let stack = Stack::new(client, problem, layout)?;
        let lhs = halves(client, [problem.lhs_rows(), problem.k]);
        let (routes, ends) = match problem.product {
            ExpertProduct::Routed { tokens } => (routes(client, problem, tokens), None),
            ExpertProduct::Grouped { .. } => (Vec::new(), Some(ends(client, problem))),
        };
        Ok(Self {
            stack: stack.0,
            lhs,
            routes,
            ends,
        })
    }
}

/// The stack, an NVFP4 `[k, experts · n]` weight packed down `k`, laid as a [`StackLayout`] says.
struct Stack(PackedTensor);

impl Stack {
    fn new(
        client: &Client,
        problem: &ExpertStackProblem,
        layout: StackLayout,
    ) -> Result<Self, String> {
        let scheme = tile_order::nvfp4_scheme();
        let (k, rows) = (problem.k, problem.rows());
        let global = PlainTensor::of_handle(TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&[1.0])),
            [1, 1],
            [1, 1],
            F32,
        ));
        let operand =
            Operand::of_scheme(&scheme, K, [K, N]).ok_or("NVFP4 blocks along k".to_string())?;
        let sized = [(K, k), (N, rows)];
        let in_rows = |row_bytes: fn(usize) -> usize| {
            // Words and block scales as metabolic hands them over: rows of whole words, each
            // `pitch` apart where the allocator pitched them.
            let words = words(client, rows, k, row_bytes(k / 2));
            let scales = PlainTensor::of_handle(TensorHandle::new(
                client.create_from_slice(&scale_rows(rows, k, row_bytes(k / 16))),
                [rows, k / 64],
                [row_bytes(k / 16) / 4, 1],
                WORD,
            ));
            PackedTensor::new(words, scales, Some(global.clone()), scheme, operand, sized)
        };
        let stack = match layout {
            StackLayout::PaddedRows => in_rows(pitch),
            StackLayout::DenseRows => in_rows(|bytes| bytes),
            StackLayout::AlignedRows => load_aligned(client, problem, global.clone(), operand),
            StackLayout::TileOrder => {
                in_tile_order(client, &scheme, operand, sized, global.clone())?
            }
        };
        Ok(Self(stack))
    }
}

/// The stack laid in the tile and the order this device's writer lays an NVFP4 weight in, as a
/// load stores it: its words relaid from dense rows, its block scales beside them in rows or,
/// laid block by block, in tiles of their own.
fn in_tile_order(
    client: &Client,
    scheme: &QuantScheme,
    operand: Operand,
    sized: [(Axis, usize); 2],
    global: PlainTensor,
) -> Result<PackedTensor, String> {
    let [(_, k), (_, rows)] = sized;
    let hw = Hardware::from_client(client);
    let tile = storage::stored_tile(&hw, MatmulOperand::Rhs, StoredValues::Packed(*scheme), F16)
        .ok_or("this device stores NVFP4 in no tile".to_string())?;
    let order = storage::tile_order_on(&hw, scheme, (tile.extent(K), tile.extent(N)), (k, rows))
        .ok_or("tile order stores no NVFP4 here".to_string())?;
    let words_per_load = hw.load_values(WORD);
    let blocks = k / order.block();
    let relayout = |source, axes: &[Axis], field, layout| {
        storage::tile(client, source, axes, WORD, Some(field), layout)
            .map_err(|err| format!("the stack lays out in tile order: {err:?}"))
    };
    // The values as the kernel names them, `[n, k/block, block]`, counted in values.
    let dense = words(client, rows, k, k / 2);
    let values = dense.bound_as(Metadata::new(
        [rows, blocks, order.block()],
        [k, order.block(), 1],
    ));
    let values = relayout(
        values,
        &tile_order::values_axes(N),
        Field::from(scheme.value),
        order.values_storage(N, words_per_load),
    )?;
    drop(dense);
    let scales = scale_bytes(client, rows, k);
    let scales = match order.scales_storage(N, order.rows_a_load(N, words_per_load)) {
        None => scales,
        Some(layout) => PlainTensor::of_handle(relayout(
            scales.bound_as(Metadata::new([rows, blocks], [blocks, 1])),
            &tile_order::scales_axes(N),
            Field::of_scale(scheme.scale_dtype()),
            layout,
        )?),
    };
    let words =
        PlainTensor::of_handle(values.clone()).viewed(tile_order::matrix_values(&values.metadata));
    let operand = PackedTensor::read_in_tiles(&hw, sized, operand, &words.meta())
        .map_err(|refusal| format!("{refusal}"))?;
    Ok(PackedTensor::new(
        words,
        scales,
        Some(global),
        *scheme,
        operand,
        sized,
    ))
}

/// `rows` rows of `k` NVFP4 codes, eight a word, each row `row_bytes` past the one before.
fn words(client: &Client, rows: usize, k: usize, row_bytes: usize) -> PlainTensor {
    let (row_words, pitch_words) = (k / VALUES_PER_WORD, row_bytes / 4);
    PlainTensor::of_handle(TensorHandle::new(
        client.create_from_slice(u32::as_bytes(&word_rows(rows, k, pitch_words))),
        [rows, row_words],
        [pitch_words, 1],
        WORD,
    ))
}

/// `rows` rows of `k` NVFP4 codes, eight a word, each row `pitch_words` words past the one before.
fn word_rows(rows: usize, k: usize, pitch_words: usize) -> Vec<u32> {
    let row_words = k / VALUES_PER_WORD;
    let mut host = vec![0u32; rows * pitch_words];
    for (row, words) in host.chunks_mut(pitch_words).enumerate() {
        for (j, word) in words[..row_words].iter_mut().enumerate() {
            *word = mixed(row * row_words + j) as u32;
        }
    }
    host
}

/// The stack copied to the device as a weight's load copies it, its codes and block scales in one
/// allocation, each laid out by the runtime under
/// [`LoadAligned`](cubecl::server::MemoryLayoutStrategy::LoadAligned): every row padded to whole
/// vector loads.
fn load_aligned(
    client: &Client,
    problem: &ExpertStackProblem,
    global: PlainTensor,
    operand: Operand,
) -> PackedTensor {
    let (k, rows) = (problem.k, problem.rows());
    let (row_words, blocks) = (k / VALUES_PER_WORD, k / 16);
    let codes = word_rows(rows, k, row_words);
    let scales = scale_rows(rows, k, blocks);
    let [words, scales]: [_; 2] = client
        .create_tensors_from_slices(vec![
            (
                MemoryLayoutDescriptor::load_aligned([rows, row_words].into(), 4),
                u32::as_bytes(&codes),
            ),
            (
                MemoryLayoutDescriptor::load_aligned([rows, blocks].into(), 1),
                &scales,
            ),
        ])
        .try_into()
        .expect("one layout a descriptor");
    let words = PlainTensor::of_handle(TensorHandle::new(
        words.memory,
        [rows, row_words],
        [words.strides[0], 1],
        WORD,
    ));
    // The scale bytes read as whole words, as a packed weight hands them over.
    let scales = PlainTensor::of_handle(TensorHandle::new(
        scales.memory,
        [rows, blocks / 4],
        [scales.strides[0] / 4, 1],
        WORD,
    ));
    let scheme = tile_order::nvfp4_scheme();
    PackedTensor::new(
        words,
        scales,
        Some(global),
        scheme,
        operand,
        [(K, k), (N, rows)],
    )
}

/// `rows` rows of a `ue4m3` scale every 16 of `k`, one byte each, between 0.25 and 2, each row
/// `row_bytes` past the one before.
fn scale_rows(rows: usize, k: usize, row_bytes: usize) -> Vec<u8> {
    let blocks = k / 16;
    let mut host = vec![0u8; rows * row_bytes];
    for (row, bytes) in host.chunks_mut(row_bytes).enumerate() {
        for (j, byte) in bytes[..blocks].iter_mut().enumerate() {
            *byte = 0x28 + (mixed(row * blocks + j) % 24) as u8;
        }
    }
    host
}

/// [`scale_rows`], dense, as the `ue4m3` bytes they are.
fn scale_bytes(client: &Client, rows: usize, k: usize) -> PlainTensor {
    let blocks = k / 16;
    PlainTensor::of_handle(TensorHandle::new(
        client.create_from_slice(&scale_rows(rows, k, blocks)),
        [rows, blocks],
        [blocks, 1],
        ElemType::from_scale_dtype(tile_order::nvfp4_scheme().scale_dtype()),
    ))
}

/// An `f16` activation of `shape`, row-major.
fn halves(client: &Client, shape: [usize; 2]) -> PlainTensor {
    const VALUES: [f32; 4] = [1.0, -0.5, 0.25, -1.0];
    let host: Vec<f16> = (0..shape[0] * shape[1])
        .map(|i| f16::from_f32(VALUES[mixed(i) % 4]))
        .collect();
    PlainTensor::of_handle(TensorHandle::new(
        client.create_from_slice(f16::as_bytes(&host)),
        shape,
        [shape[1], 1],
        F16,
    ))
}

/// Sets of `[tokens, picks]` routes, as many as the experts hold apart: a token's picks are
/// distinct, and no two sets name one expert. Experts are spread over the stack rather than
/// neighbours.
fn routes(client: &Client, problem: &ExpertStackProblem, tokens: usize) -> Vec<PlainTensor> {
    let per_set = tokens * problem.picks;
    let sets = (problem.experts / per_set).max(1);
    (0..sets)
        .map(|set| {
            let host: Vec<u32> = (0..per_set)
                .map(|i| (((set * per_set + i) * 167) % problem.experts) as u32)
                .collect();
            PlainTensor::of_handle(TensorHandle::new(
                client.create_from_slice(u32::as_bytes(&host)),
                [tokens, problem.picks],
                [problem.picks, 1],
                WORD,
            ))
        })
        .collect()
}

/// Each expert's cumulative end row ([`ExpertStackProblem::group_sizes`]).
fn ends(client: &Client, problem: &ExpertStackProblem) -> PlainTensor {
    let host: Vec<u32> = problem
        .group_sizes()
        .iter()
        .scan(0u32, |end, &size| {
            *end += size as u32;
            Some(*end)
        })
        .collect();
    PlainTensor::of_handle(TensorHandle::new(
        client.create_from_slice(u32::as_bytes(&host)),
        [host.len()],
        [1],
        WORD,
    ))
}

struct ExpertStackBench {
    inputs: Arc<StackInputs>,
    problem: ExpertStackProblem,
    layout: StackLayout,
    client: Client,
    device: cubecl::Device,
    samples: usize,
    /// The routes the next routed call reads.
    next_routes: AtomicUsize,
}

impl ExpertStackBench {
    /// The product `problem` states against its stack laid as `layout`, its autotune run.
    fn new(
        layout: StackLayout,
        problem: &ExpertStackProblem,
        samples: usize,
    ) -> Result<Self, String> {
        let device = cubecl::test_device();
        let client = device.client();
        let inputs = Arc::new(StackInputs::new(&client, layout, problem)?);
        let bench = Self {
            inputs,
            problem: *problem,
            layout,
            client,
            device,
            samples,
            next_routes: AtomicUsize::new(0),
        };
        for _ in 0..WARM_UP {
            bench.launch(&bench.inputs)?;
        }
        bench.sync();
        Ok(bench)
    }

    /// One call, timed as `timing` says.
    fn once(&self, timing: TimingMethod) -> Result<Duration, String> {
        let profile = match timing {
            TimingMethod::Device => self.profile(self.prepare())?,
            TimingMethod::System => self.profile_full(self.prepare())?,
        };
        future::block_on(profile.resolve())
            .map(|ticks| ticks.duration())
            .ok_or_else(|| "the profiled window carried no measurement".to_string())
    }

    /// One product through its autotune, as a model's layer calls it: the stack restated as its
    /// slabs, the tensors built, the winner launched.
    fn launch(&self, inputs: &StackInputs) -> Result<(), String> {
        let stack = StoredTensor::Packed(inputs.stack.clone());
        let refused = |refusal: cubecl::tile::launch::Refusal| format!("{refusal}");
        match self.problem.product {
            ExpertProduct::Routed { .. } => {
                let set = self.next_routes.fetch_add(1, Ordering::Relaxed) % inputs.routes.len();
                let tensors = RoutedMatmulTensors::new(
                    &self.client,
                    inputs.lhs.clone(),
                    stack,
                    inputs.routes[set].clone(),
                    self.problem.experts,
                    SlabAxis::Output,
                )
                .map_err(refused)?;
                routed::launch_tuned(&self.device, tensors, GroupPriority::DEFAULT)
                    .map_err(refused)?;
            }
            ExpertProduct::Grouped { .. } => {
                let ends = inputs.ends.clone().expect("a grouped product has ends");
                let tensors = MGroupedMatmulTensors::new(
                    &self.client,
                    inputs.lhs.clone(),
                    stack,
                    ends,
                    SlabAxis::Output,
                )
                .map_err(refused)?;
                m_grouped::launch_tuned(&self.device, tensors, GroupPriority::DEFAULT)
                    .map_err(refused)?;
            }
        }
        Ok(())
    }
}

impl Benchmark for ExpertStackBench {
    type Input = Arc<StackInputs>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        self.inputs.clone()
    }

    fn execute(&self, inputs: Self::Input) -> Result<(), String> {
        self.launch(&inputs)
    }

    fn num_samples(&self) -> usize {
        self.samples
    }

    fn name(&self) -> String {
        format!(
            "expert-stacks-{}-{:?}-{:?}-k{}-n{}",
            self.client.name(),
            self.layout,
            self.problem.product,
            self.problem.k,
            self.problem.n,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, inputs: Self::Input) -> Result<ProfileDuration, String> {
        cubek_test_utils::profile_launch(&self.client, "expert-stacks-bench", || {
            self.execute(inputs)
        })
    }
}
