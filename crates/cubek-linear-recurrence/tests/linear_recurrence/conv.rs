//! The gated delta conv over a window against a host reference in `f64`. The stores hold NaN
//! wherever the conv must not read, and the reference rounds to the activations' element where the
//! kernel does.

use super::harness::{F16, F32, activation_elements, read, round_to, served, upload};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_linear_recurrence::kernels::conv::{
    CubeQkNormedConvRoutine, GatedDeltaConvPastInputs, QkNormedConvInputs, QkNormedConvOutputs,
    QkNormedConvTensors,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_std::Hardware;
use cubek_std::routine::PreparedLaunches;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

pub const EPSILON: f32 = 1e-6;

/// A conv's shape, and the rows it convolves: each a `(slot, positions convolved already)`.
#[derive(Debug, Clone)]
pub struct ConvCase {
    pub heads: RecurrenceHeads,
    pub taps: usize,
    /// Columns a slot of the window holds.
    pub columns: usize,
    pub slots: usize,
    pub rows: Vec<(usize, usize)>,
}

/// How the window lies in its buffer: the gap after each column, the gap after each slot, and the
/// elements its view starts past its buffer's start; and the elements its stream's rows are
/// padded by.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvLayout {
    pub column_gap: usize,
    pub slot_gap: usize,
    pub start: usize,
    pub qkv_pad: usize,
}

pub fn case(
    heads: (usize, usize, usize, usize),
    taps: usize,
    columns: usize,
    slots: usize,
    rows: &[(usize, usize)],
) -> ConvCase {
    let (key_heads, heads_per_group, key_features, value_features) = heads;
    ConvCase {
        heads: RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        },
        taps,
        columns,
        slots,
        rows: rows.to_vec(),
    }
}

/// A decode step of a many-headed layer, prime heads with a group of three, a fresh row, a ring
/// that wraps and a ring as short as the columns read back.
pub fn cases() -> Vec<ConvCase> {
    vec![
        case(
            (16, 3, 128, 128),
            4,
            3,
            10,
            &[(7, 9), (3, 0), (9, 1), (1, 2)],
        ),
        case((2, 3, 37, 53), 4, 5, 6, &[(4, 9), (1, 0), (2, 1), (5, 2)]),
        case((2, 2, 12, 8), 2, 1, 3, &[(1, 5), (0, 0), (2, 1)]),
    ]
}

/// The host values of a case's operands, all rounded to the element they are uploaded at.
pub struct ConvOperands {
    pub qkv: Vec<f32>,
    pub filters: Vec<f32>,
    /// `[slot][column][channel]`, NaN wherever no row reads it.
    pub window: Vec<f32>,
}

/// The queries, keys and values a conv leaves, in `f64`.
#[derive(Debug, Default)]
pub struct HostOutputs {
    pub query: Vec<f64>,
    pub key: Vec<f64>,
    pub value: Vec<f64>,
}

/// The channels convolved and rounded as the kernel rounds them: the newest tap of `channel` reads
/// `newest(channel)`, tap `j` reads `past(j, channel)` or nothing where none is held.
pub fn convolve_channels(
    filters: &[f32],
    taps: usize,
    newest: impl Fn(usize) -> f32,
    past: impl Fn(usize, usize) -> Option<f32>,
    elem: ElemType,
) -> Vec<f64> {
    let round = |x: f64| round_to(x as f32, elem) as f64;
    (0..filters.len() / taps)
        .map(|channel| {
            let tap = |j: usize| filters[channel * taps + j] as f64;
            let mut sum = newest(channel) as f64 * tap(taps - 1);
            for j in 0..taps - 1 {
                sum += past(j, channel).map_or(0.0, |x| x as f64 * tap(j));
            }
            let mixed = round(sum);
            round(mixed / (1.0 + (-mixed).exp()))
        })
        .collect()
}

pub fn wave(len: usize, seed: f32, scale: f32, elem: ElemType) -> Vec<f32> {
    (0..len)
        .map(|i| round_to(scale * (i as f32 * 0.37 + seed).sin(), elem))
        .collect()
}

impl ConvCase {
    pub fn conv_dim(&self) -> usize {
        self.heads.conv_dim()
    }

    pub fn read_columns(&self) -> usize {
        self.taps - 1
    }

    /// The ring column a row that has convolved `positions` reads for tap `j`, and whether it
    /// holds it.
    pub fn read_of(&self, positions: usize, j: usize) -> (usize, bool) {
        let ahead = positions + j;
        let held = ahead >= self.read_columns();
        (
            (ahead + self.columns - self.read_columns()) % self.columns,
            held,
        )
    }

    /// Values for the stream and the filters; a window holding values only in the columns some
    /// row reads and in the slots no row lists, every other column NaN.
    pub fn operands(&self, elem: ElemType) -> ConvOperands {
        let conv_dim = self.conv_dim();
        let mut window = vec![f32::NAN; self.slots * self.columns * conv_dim];
        let values = wave(window.len(), 0.3, 0.8, elem);
        let at = |slot: usize, column: usize| (slot * self.columns + column) * conv_dim;
        for &(slot, positions) in &self.rows {
            for j in 0..self.read_columns() {
                let (column, held) = self.read_of(positions, j);
                if held {
                    let from = at(slot, column);
                    window[from..from + conv_dim].copy_from_slice(&values[from..from + conv_dim]);
                }
            }
        }
        for slot in (0..self.slots).filter(|slot| slot % 2 == 0) {
            if !self.rows.iter().any(|&(s, _)| s == slot) {
                let from = at(slot, 0);
                let len = self.columns * conv_dim;
                window[from..from + len].copy_from_slice(&values[from..from + len]);
            }
        }
        ConvOperands {
            qkv: wave(self.rows.len() * conv_dim, 1.1, 1.2, elem),
            filters: wave(conv_dim * self.taps, 0.1, 0.6, elem),
            window,
        }
    }

    /// Row `row`'s channels convolved, rounded as the kernel rounds them, over `window`.
    pub fn convolve(
        &self,
        o: &ConvOperands,
        window: &[f32],
        row: usize,
        elem: ElemType,
    ) -> Vec<f64> {
        let conv_dim = self.conv_dim();
        let (slot, positions) = self.rows[row];
        convolve_channels(
            &o.filters,
            self.taps,
            |channel| o.qkv[row * conv_dim + channel],
            |j, channel| {
                let (column, held) = self.read_of(positions, j);
                held.then(|| window[(slot * self.columns + column) * conv_dim + channel])
            },
            elem,
        )
    }

    /// The channels of row `row`'s key head `head`: its queries, its keys, then its value heads'
    /// values, from a row's convolved channels.
    pub fn head_channels(&self, convolved: &[f64], head: usize) -> Vec<f64> {
        let heads = &self.heads;
        let key_dim = heads.key_dim();
        let mut channels = Vec::new();
        channels.extend(&convolved[head * heads.key_features..][..heads.key_features]);
        channels.extend(&convolved[key_dim + head * heads.key_features..][..heads.key_features]);
        let values = 2 * key_dim + head * heads.heads_per_group * heads.value_features;
        channels.extend(&convolved[values..values + heads.heads_per_group * heads.value_features]);
        channels
    }

    /// The scales of a key head's queries and keys.
    pub fn scales(&self, channels: &[f64]) -> (f64, f64) {
        let kf = self.heads.key_features;
        let squares = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>() + EPSILON as f64;
        (
            1.0 / (squares(&channels[..kf]).sqrt() * (kf as f64).sqrt()),
            1.0 / squares(&channels[kf..2 * kf]).sqrt(),
        )
    }

    /// The queries, keys and values of every key head of a row whose channels are `convolved`:
    /// `[key_head]` queries then keys, `[value_head]` values.
    pub fn head_outputs(&self, convolved: &[f64]) -> HostOutputs {
        let kf = self.heads.key_features;
        let mut outputs = HostOutputs::default();
        for head in 0..self.heads.key_heads {
            let channels = self.head_channels(convolved, head);
            let (q_scale, k_scale) = self.scales(&channels);
            outputs
                .query
                .extend(channels[..kf].iter().map(|x| x * q_scale));
            outputs
                .key
                .extend(channels[kf..2 * kf].iter().map(|x| x * k_scale));
            outputs.value.extend(&channels[2 * kf..]);
        }
        outputs
    }

    /// The queries, keys and values of every row, as the conv leaves them.
    pub fn reference(&self, o: &ConvOperands, window: &[f32], elem: ElemType) -> HostOutputs {
        let mut all = HostOutputs::default();
        for row in 0..self.rows.len() {
            let row_outputs = self.head_outputs(&self.convolve(o, window, row, elem));
            all.query.extend(row_outputs.query);
            all.key.extend(row_outputs.key);
            all.value.extend(row_outputs.value);
        }
        all
    }

    /// Where element `(slot, column)` of the window starts in its buffer under `layout`.
    pub fn window_at(&self, layout: ConvLayout, slot: usize, column: usize) -> usize {
        let column_pitch = self.conv_dim() + layout.column_gap;
        let slot_pitch = self.columns * column_pitch + layout.slot_gap;
        layout.start + slot * slot_pitch + column * column_pitch
    }

    /// The window as it lies in a buffer under `layout`, gaps and start NaN.
    pub fn laid_out(&self, window: &[f32], layout: ConvLayout) -> Vec<f32> {
        let conv_dim = self.conv_dim();
        let pitch = conv_dim + layout.column_gap;
        let mut buffer =
            vec![f32::NAN; layout.start + self.slots * (self.columns * pitch + layout.slot_gap)];
        for slot in 0..self.slots {
            for column in 0..self.columns {
                let from = (slot * self.columns + column) * conv_dim;
                let to = self.window_at(layout, slot, column);
                buffer[to..to + conv_dim].copy_from_slice(&window[from..from + conv_dim]);
            }
        }
        buffer
    }
}

/// A conv launched: the window's buffer to read back, and what it held before.
pub struct Stored {
    pub raw: Handle,
    pub before: Vec<f32>,
}

/// A tensor of `shape` and `strides` over `buffer` uploaded as `elem`, its view starting `start`
/// elements in.
pub fn view(
    client: &Client,
    buffer: &[f32],
    start: usize,
    shape: &[usize],
    strides: &[usize],
    elem: ElemType,
) -> (Handle, PlainTensor) {
    let raw = upload(client, buffer, elem);
    let handle = raw.clone().offset_start((start * elem.size()) as u64);
    let tensor = PlainTensor::of_handle(TensorHandle::from_metadata(
        handle,
        Metadata::new(shape.to_vec(), strides.to_vec()),
        elem,
    ));
    (raw, tensor)
}

pub fn table(client: &Client, entries: Vec<u32>) -> PlainTensor {
    PlainTensor::of_u32s(client, &[entries.len()], &entries)
}

/// The slot and lengths tables of `inputs`, which a case over a window holds.
pub fn tables_of(inputs: &mut QkNormedConvInputs) -> (&mut PlainTensor, &mut PlainTensor) {
    let GatedDeltaConvPastInputs::Window { slots, lengths, .. } = &mut inputs.past else {
        panic!("the inputs are over a window");
    };
    (slots, lengths)
}

/// `entries` as a column of a `[rows, 2]` table, the other column holding `other`.
pub fn strided_table(client: &Client, entries: Vec<u32>, other: u32) -> PlainTensor {
    let rows = entries.len();
    let values: Vec<u32> = entries.iter().flat_map(|&entry| [entry, other]).collect();
    PlainTensor::of_u32s(client, &[rows, 2], &values).viewed(Metadata::new(vec![rows], vec![2]))
}

impl ConvCase {
    /// The inputs of the case under `layout`, and the window's buffer.
    pub fn inputs(
        &self,
        client: &Client,
        o: &ConvOperands,
        layout: ConvLayout,
        elem: ElemType,
    ) -> (QkNormedConvInputs, Stored) {
        let conv_dim = self.conv_dim();
        let rows = self.rows.len();
        let row_pitch = conv_dim + layout.qkv_pad;
        let mut stream = vec![f32::NAN; rows * row_pitch];
        for row in 0..rows {
            stream[row * row_pitch..][..conv_dim]
                .copy_from_slice(&o.qkv[row * conv_dim..(row + 1) * conv_dim]);
        }
        let (_, qkv) = view(client, &stream, 0, &[rows, conv_dim], &[row_pitch, 1], elem);
        let (_, filters) = view(
            client,
            &o.filters,
            0,
            &[conv_dim, self.taps],
            &[self.taps, 1],
            elem,
        );
        let buffer = self.laid_out(&o.window, layout);
        let column_pitch = conv_dim + layout.column_gap;
        let slot_pitch = self.columns * column_pitch + layout.slot_gap;
        let (raw, window) = view(
            client,
            &buffer,
            layout.start,
            &[self.slots, self.columns, conv_dim],
            &[slot_pitch, column_pitch, 1],
            elem,
        );
        let inputs = QkNormedConvInputs {
            client: client.clone(),
            qkv,
            filters,
            past: GatedDeltaConvPastInputs::Window {
                window,
                slots: table(client, self.rows.iter().map(|&(s, _)| s as u32).collect()),
                lengths: table(client, self.rows.iter().map(|&(_, p)| p as u32).collect()),
            },
            heads: self.heads,
            epsilon: EPSILON,
        };
        (
            inputs,
            Stored {
                raw,
                before: buffer,
            },
        )
    }
}

/// How far an output may fall from the reference's, relative to one plus its magnitude: a sum an
/// ulp of the element apart rounds to neighbouring values.
pub fn tolerance(elem: ElemType) -> f64 {
    match elem {
        F32 => 2e-4,
        F16 => 6e-3,
        _ => 5e-2,
    }
}

pub fn compare(got: &[f32], want: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: the element count");
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got as f64 - want).abs() <= tolerance * (1.0 + want.abs()),
            "{what}: element {i} is {got}, the reference {want}"
        );
    }
}

/// `outputs` against `want`: the queries, keys and values.
pub fn compare_outputs(
    client: &Client,
    outputs: &QkNormedConvOutputs,
    want: &HostOutputs,
    elem: ElemType,
    what: &str,
) {
    for (name, handle, want) in [
        ("queries", &outputs.query, &want.query),
        ("keys", &outputs.key, &want.key),
        ("values", &outputs.value, &want.value),
    ] {
        compare(
            &read(client, handle.handle.clone(), F32),
            want,
            tolerance(elem),
            &format!("{what} {elem:?}: the {name}"),
        );
    }
}

/// The conv of `inputs`, or `None` where this device declines it by the refusal admission names.
pub fn conv_or_skip(
    hw: &Hardware,
    inputs: QkNormedConvInputs,
    what: &str,
) -> Option<QkNormedConvOutputs> {
    let launched = CubeQkNormedConvRoutine.launch_on(hw, inputs.clone());
    served(what, launched, || admit(hw, inputs))
}

/// Whether `hw` admits the problem `inputs` state: what a refusal of their launch must name.
pub fn admit(hw: &Hardware, inputs: QkNormedConvInputs) -> Result<(), Refusal> {
    let tensors = QkNormedConvTensors::new(inputs).expect("the elements are served");
    tensors.problem().admit(hw)
}

/// How a conv's slot and length tables are bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tables {
    /// Dense, an entry a row.
    Dense,
    /// Columns of wider tensors: each row reads its own entry, not its neighbour's.
    Strided,
    /// Every slot entry past the store's last, which reads and writes the last.
    PastTheStore,
}

/// One conv over a window: a case at an element, its window laid out as `layout` lays it, its
/// tables bound as `tables` names.
#[derive(Debug, Clone)]
pub struct WindowConv {
    pub case: ConvCase,
    pub element: ElemType,
    pub layout: ConvLayout,
    pub tables: Tables,
}

impl WindowConv {
    pub fn new(case: ConvCase, element: ElemType, layout: ConvLayout) -> Self {
        Self {
            case,
            element,
            layout,
            tables: Tables::Dense,
        }
    }

    pub fn tables(self, tables: Tables) -> Self {
        Self { tables, ..self }
    }
}

/// The conv over a window against the host: the queries, keys and values, and the window's whole
/// buffer, in which each row writes its ring column bit for bit its newest values and nothing
/// else changes, the gaps and the unlisted slots included. A device may decline a conv only by the
/// refusal admission names. With `prepared`, every launch prepares through it.
pub struct WindowConvs {
    pub prepared: Option<PreparedLaunches<CubeQkNormedConvRoutine>>,
}

impl WindowConvs {
    /// `outputs` and their names, each contiguous tensor read as one row of its elements.
    pub fn written(
        outputs: impl IntoIterator<Item = (&'static str, TensorHandle)>,
    ) -> Vec<Written> {
        outputs
            .into_iter()
            .map(|(name, tensor)| {
                let elements = tensor.shape().num_elements();
                Written {
                    name,
                    tensor: TensorHandle::new(tensor.handle, [elements], [1], tensor.dtype),
                    host: HostDataType::F32,
                }
            })
            .collect()
    }

    /// `values` as one row of the host's, as [`written`](Self::written) reads an output.
    pub fn flat(values: Vec<f32>) -> HostData {
        let elements = values.len();
        HostData::row_major(HostDataVec::F32(values), Shape::from(vec![elements]))
    }

    /// The queries, keys and values of `want`, each as one row.
    pub fn outputs(want: &HostOutputs) -> [HostData; 3] {
        let narrowed = |values: &[f64]| values.iter().map(|&x| x as f32).collect();
        [
            Self::flat(narrowed(&want.query)),
            Self::flat(narrowed(&want.key)),
            Self::flat(narrowed(&want.value)),
        ]
    }
}

impl ScopeUnderTest for WindowConvs {
    type Case = WindowConv;

    /// # Panics
    ///
    /// On a refusal other than the one admission names.
    fn launch(&self, client: &Client, conv: &WindowConv) -> Result<Vec<Written>, Refusal> {
        let hw = Hardware::from_client(client);
        let case = &conv.case;
        let operands = case.operands(conv.element);
        let (mut inputs, stored) = case.inputs(client, &operands, conv.layout, conv.element);
        let (slots, lengths) = tables_of(&mut inputs);
        match conv.tables {
            Tables::Dense => {}
            Tables::Strided => {
                *slots = strided_table(
                    client,
                    case.rows.iter().map(|&(s, _)| s as u32).collect(),
                    0,
                );
                *lengths = strided_table(
                    client,
                    case.rows.iter().map(|&(_, p)| p as u32).collect(),
                    7,
                );
            }
            Tables::PastTheStore => {
                *slots = table(client, vec![case.slots as u32 + 5; case.rows.len()]);
            }
        }
        let launched = match &self.prepared {
            Some(prepared) => {
                CubeQkNormedConvRoutine.launch_prepared_on(prepared, &hw, inputs.clone())
            }
            None => CubeQkNormedConvRoutine.launch_on(&hw, inputs.clone()),
        };
        let outputs = launched.inspect_err(|why| {
            assert_eq!(
                admit(&hw, inputs),
                Err(why.clone()),
                "{conv:?}: refused but admitted"
            );
        })?;
        let window = TensorHandle::new(stored.raw, [stored.before.len()], [1], conv.element);
        Ok(Self::written([
            ("queries", outputs.query),
            ("keys", outputs.key),
            ("values", outputs.value),
            ("window", window),
        ]))
    }

    fn reference(&self, conv: &WindowConv) -> Vec<HostData> {
        let case = &conv.case;
        let operands = case.operands(conv.element);
        let want = case.reference(&operands, &operands.window, conv.element);
        let conv_dim = case.conv_dim();
        let mut window: Vec<f32> = case
            .laid_out(&operands.window, conv.layout)
            .into_iter()
            .map(|value| round_to(value, conv.element))
            .collect();
        for (row, &(slot, positions)) in case.rows.iter().enumerate() {
            let to = case.window_at(conv.layout, slot, positions % case.columns);
            window[to..to + conv_dim]
                .copy_from_slice(&operands.qkv[row * conv_dim..(row + 1) * conv_dim]);
        }
        let mut reference = Self::outputs(&want).to_vec();
        reference.push(Self::flat(window));
        reference
    }

    fn tolerance(&self, conv: &WindowConv) -> f32 {
        tolerance(conv.element) as f32
    }

    /// What the conv does not write, and the newest values it copies, must hold bit for bit.
    fn tolerance_of(&self, conv: &WindowConv, name: &'static str) -> f32 {
        match name {
            "window" => 0.,
            _ => self.tolerance(conv),
        }
    }
}

fn layouts() -> Vec<ConvLayout> {
    vec![
        ConvLayout::default(),
        ConvLayout {
            column_gap: 3,
            slot_gap: 5,
            start: 1,
            qkv_pad: 1,
        },
    ]
}

/// Every case at every element the device runs, over a window packed and over one with gaps, a
/// start off a whole read and padded stream rows: the conv matches the reference and writes
/// nothing but each row's ring column.
#[test]
fn the_conv_matches_the_reference() {
    let client = cubecl::test_device().client();
    let mut convs = Vec::new();
    for elem in activation_elements(&client) {
        for layout in layouts() {
            for case in cases() {
                convs.push(WindowConv::new(case, elem, layout));
            }
        }
    }
    WindowConvs { prepared: None }.check(convs);
}

/// Tables that are columns of wider tensors: each row reads its own entry, not its neighbour's.
#[test]
fn tables_at_a_stride_are_read_by_it() {
    let case = case((2, 3, 37, 53), 4, 5, 6, &[(4, 9), (1, 0), (2, 1), (5, 2)]);
    WindowConvs { prepared: None }
        .check([WindowConv::new(case, F32, ConvLayout::default()).tables(Tables::Strided)]);
}

/// A slot past the store's last reads and writes the last: the row behaves as one listing the last
/// slot, and leaves every other slot as it was.
#[test]
fn a_slot_past_the_store_is_the_last_slot() {
    let case = case((2, 3, 37, 53), 4, 5, 4, &[(3, 7)]);
    WindowConvs { prepared: None }
        .check([WindowConv::new(case, F32, ConvLayout::default()).tables(Tables::PastTheStore)]);
}

/// Two launches of one prepared problem with different slot tables each leave the results of
/// their own rows.
#[test]
fn a_prepared_problem_is_right_for_each_slot_table() {
    let tables: [&[(usize, usize)]; 2] = [&[(4, 9), (1, 0), (2, 1)], &[(0, 3), (3, 0), (5, 8)]];
    WindowConvs {
        prepared: Some(PreparedLaunches::default()),
    }
    .check(tables.map(|rows| {
        WindowConv::new(
            case((2, 3, 37, 53), 4, 5, 6, rows),
            F32,
            ConvLayout::default(),
        )
    }));
}
