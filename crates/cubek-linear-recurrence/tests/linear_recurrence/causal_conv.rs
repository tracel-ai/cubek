//! The causal conv with bias against a host reference in `f64`, over a window and over a prompt:
//! the ring's wrap, a row with no history, prompts and windows packed and pitched, rows right padded
//! to unequal lengths, and a prompt handed on to window steps. The rings hold NaN wherever the
//! conv must not read.

use super::harness::{
    F16, F32, compare, convolve_channels, elements, read, served, strided_table, table, tolerance,
    view, wave,
};
use super::ring::{PromptLayout, RingCase, RingPadding};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubek_linear_recurrence::kernels::SlotStore;
use cubek_linear_recurrence::kernels::conv::{
    CausalConvInputs, CausalConvOutputs, CausalConvPastInputs, CausalConvRoutine,
    CausalConvTensors, CubeCausalConvRoutine,
};
use cubek_linear_recurrence::kernels::refusals::{
    CAUSAL_CONV_CHANNELS_EMPTY, CAUSAL_CONV_NOTHING_TO_CONVOLVE, CAUSAL_CONV_OUTPUT_UNSUPPORTED,
    CAUSAL_CONV_TAPS_PAST_RING,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;
use cubek_std::routine::PreparedLaunches;

/// A causal conv's rows, over a window when `window` holds, else a prompt.
#[derive(Debug, Clone)]
struct CausalCase {
    ring: RingCase,
    window: bool,
}

/// The host values of a case.
struct Operands {
    sequences: Vec<Vec<f32>>,
    filters: Vec<f32>,
    bias: Vec<f32>,
}

/// Whether the conv has a bias, and the element it writes.
#[derive(Debug, Clone, Copy)]
struct Setting {
    biased: bool,
    output: ElemType,
}

const BIASED_F32: Setting = Setting {
    biased: true,
    output: F32,
};

fn window_case(
    channels: usize,
    taps: usize,
    columns: usize,
    rows: &[(usize, usize)],
) -> CausalCase {
    let rows: Vec<_> = rows
        .iter()
        .map(|&(slot, before)| (slot, before, 1))
        .collect();
    CausalCase {
        ring: RingCase::new(1, channels, taps, columns, &rows),
        window: true,
    }
}

fn prompt_case(
    positions: usize,
    channels: usize,
    taps: usize,
    columns: usize,
    rows: &[(usize, usize, usize)],
) -> CausalCase {
    CausalCase {
        ring: RingCase::new(positions, channels, taps, columns, rows),
        window: false,
    }
}

fn pitched(mut case: CausalCase, padding: RingPadding) -> CausalCase {
    case.ring = case.ring.pitched(padding);
    case
}

/// A window step with a ring that wraps, a fresh row and a ring as short as the taps read back, a
/// long prompt, a prompt of one position, prime channels and the widths of a wide layer; prompts
/// whose rows start from nothing, from exactly the taps read back, and from a ring they have gone
/// round; a window and a prompt over a store whose columns and slots are padded.
fn cases() -> Vec<CausalCase> {
    vec![
        window_case(1280, 4, 3, &[(7, 9), (3, 0), (4, 1), (1, 2)]),
        window_case(37 * 3, 4, 5, &[(4, 9), (1, 0), (2, 3)]),
        window_case(12, 2, 1, &[(1, 5), (0, 0), (2, 1)]),
        prompt_case(257, 1280, 4, 3, &[(0, 0, 257), (1, 3, 257)]),
        prompt_case(1, 37 * 3, 4, 5, &[(2, 9, 1), (0, 0, 1)]),
        prompt_case(5, 12, 2, 1, &[(0, 1, 5), (1, 6, 5), (2, 0, 5)]),
        prompt_case(6, 40, 4, 3, &[(0, 3, 6), (1, 9, 6)]),
        pitched(
            window_case(40, 4, 3, &[(2, 7), (0, 1), (1, 0)]),
            RingPadding { column: 5, slot: 3 },
        ),
        pitched(
            prompt_case(6, 40, 4, 3, &[(1, 3, 6), (0, 9, 4)]),
            RingPadding { column: 5, slot: 3 },
        ),
    ]
}

impl CausalCase {
    fn operands(&self, elem: ElemType) -> Operands {
        Operands {
            sequences: self.ring.sequences(elem),
            filters: wave(self.ring.channels * self.ring.taps, 0.1, 0.6, elem),
            bias: wave(self.ring.channels, 0.7, 0.5, elem),
        }
    }

    /// The inputs of the case over `ring`, the store of its rings.
    fn inputs(
        &self,
        client: &Client,
        o: &Operands,
        ring: PlainTensor,
        layout: PromptLayout,
        setting: Setting,
        elem: ElemType,
    ) -> CausalConvInputs {
        let channels = self.ring.channels;
        let [slots, lengths, valid] = self.ring.tables(client);
        let ring = SlotStore::new(ring, slots, lengths);
        let past = if self.window {
            CausalConvPastInputs::Window { ring }
        } else {
            CausalConvPastInputs::Prompt { ring, valid }
        };
        let shape = [channels, self.ring.taps];
        let filters = view(client, &o.filters, 0, &shape, &[self.ring.taps, 1], elem).1;
        let bias = setting
            .biased
            .then(|| view(client, &o.bias, 0, &[channels], &[1], elem).1);
        CausalConvInputs {
            client: client.clone(),
            input: self.ring.prompt_input(client, &o.sequences, layout, elem),
            filters,
            bias,
            past,
            output_element: setting.output,
        }
    }

    /// The channels of one row's `position` convolved over `sequence`, the positions before the
    /// sequence zero.
    fn convolved(
        &self,
        o: &Operands,
        sequence: &[f32],
        position: usize,
        biased: bool,
        elem: ElemType,
    ) -> Vec<f64> {
        let (channels, taps) = (self.ring.channels, self.ring.taps);
        let at = |position: usize, channel: usize| sequence[position * channels + channel];
        convolve_channels(
            &o.filters,
            biased.then_some(o.bias.as_slice()),
            taps,
            |channel| at(position, channel),
            |j, channel| (position + j + 1 >= taps).then(|| at(position + j + 1 - taps, channel)),
            elem,
        )
    }

    /// Every output position of every row, `[row][position][channel]`.
    fn reference(&self, o: &Operands, biased: bool, elem: ElemType) -> Vec<f64> {
        let mut all = Vec::new();
        for (row, sequence) in self.ring.rows.iter().zip(&o.sequences) {
            for t in 0..self.ring.positions {
                all.extend(self.convolved(o, sequence, row.before + t, biased, elem));
            }
        }
        all
    }
}

/// The conv of `inputs`, or `None` where this device declines it by the refusal admission names.
fn launch(hw: &Hardware, inputs: CausalConvInputs, what: &str) -> Option<CausalConvOutputs> {
    let launched = CubeCausalConvRoutine.launch_on(hw, inputs.clone());
    served(what, launched, || admit(hw, inputs))
}

/// Whether `hw` admits the problem `inputs` state: what a refusal of their launch must name.
fn admit(hw: &Hardware, inputs: CausalConvInputs) -> Result<(), cubecl::tile::launch::Refusal> {
    let tensors = CausalConvTensors::new(inputs).expect("the elements are served");
    tensors.problem().admit(hw)
}

fn outputs_of(client: &Client, outputs: &CausalConvOutputs, element: ElemType) -> Vec<f32> {
    read(client, outputs.output.handle.clone(), element)
}

/// Every case at every element the device runs, with and without a bias, written at the
/// activations' element and at `f32`, over packed operands and over pitched ones starting off a
/// whole read: the outputs match the reference, and the ring holds each row's last valid positions
/// and nothing else changes.
#[test]
fn the_causal_conv_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layouts = [PromptLayout::default(), PromptLayout { pad: 3, start: 1 }];
    for elem in elements(&client) {
        for layout in layouts {
            for biased in [false, true] {
                for output in [elem, F32] {
                    for case in cases() {
                        let o = case.operands(elem);
                        let ring = case.ring.ring(&o.sequences);
                        let (raw, store) = case.ring.ring_tensor(&client, &ring, elem);
                        let setting = Setting { biased, output };
                        let inputs = case.inputs(&client, &o, store, layout, setting, elem);
                        let what = format!("{layout:?} {setting:?} {elem:?} {case:?}");
                        if let Some(outputs) = launch(&hw, inputs, &what) {
                            let want = case.reference(&o, biased, elem);
                            let got = outputs_of(&client, &outputs, output);
                            compare(&got, &want, tolerance(elem), &what);
                            case.ring
                                .check_ring(&client, raw, &o.sequences, &ring, elem, &what);
                        }
                    }
                }
            }
        }
    }
}

/// A row with no history reads zeros for its taps behind, however the ring holds NaN.
#[test]
fn a_row_with_no_history_reads_zeros() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for case in [
        window_case(96, 4, 3, &[(1, 0)]),
        prompt_case(2, 96, 4, 3, &[(0, 0, 2)]),
    ] {
        let o = case.operands(F32);
        let ring = vec![f32::NAN; case.ring.slots * case.ring.columns * case.ring.channels];
        let (_, store) = case.ring.ring_tensor(&client, &ring, F32);
        let layout = PromptLayout::default();
        let inputs = case.inputs(&client, &o, store, layout, BIASED_F32, F32);
        if let Some(outputs) = launch(&hw, inputs, "no history") {
            let got = outputs_of(&client, &outputs, F32);
            let want = case.reference(&o, true, F32);
            compare(&got, &want, tolerance(F32), "no history");
        }
    }
}

/// A prompt of `prompt` positions then `steps` window steps over the ring the prompt fills, a ring
/// of `columns` columns: the prompt and every step match the reference over the whole sequence, so
/// the ring the prompt leaves is the one a longer prompt would have read and it wraps right.
fn handed_on(prompt: usize, steps: usize, columns: usize) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let (channels, taps) = (60, 4);
    let layout = PromptLayout::default();
    'elements: for elem in elements(&client) {
        let first = prompt_case(prompt, channels, taps, columns, &[(0, 0, prompt)]);
        let mut o = first.operands(elem);
        let ring = first.ring.ring(&o.sequences);
        let (_, store) = first.ring.ring_tensor(&client, &ring, elem);
        let inputs = first.inputs(&client, &o, store.clone(), layout, BIASED_F32, elem);
        let Some(outputs) = launch(&hw, inputs, "the prompt") else {
            continue;
        };
        let got = outputs_of(&client, &outputs, F32);
        compare(
            &got,
            &first.reference(&o, true, elem),
            tolerance(elem),
            "the prompt",
        );
        for step in 0..steps {
            let length = o.sequences[0].len() / channels;
            o.sequences[0].extend(wave(channels, 3.5 + step as f32, 1.0, elem));
            let case = window_case(channels, taps, columns, &[(0, length)]);
            let inputs = case.inputs(&client, &o, store.clone(), layout, BIASED_F32, elem);
            let what = format!("step {step} after {prompt} positions at {elem:?}");
            let Some(outputs) = launch(&hw, inputs, &what) else {
                continue 'elements;
            };
            let got = outputs_of(&client, &outputs, F32);
            compare(
                &got,
                &case.reference(&o, true, elem),
                tolerance(elem),
                &what,
            );
        }
    }
}

/// A prompt handed on to window steps: with a ring as short as the taps read back, a prompt
/// shorter than the ring, a long prompt, one step after a prompt, and a prompt of one position
/// followed by more steps than the ring has columns.
#[test]
fn a_prompt_handed_on_to_window_steps_matches_the_reference() {
    handed_on(9, 4, 3);
    handed_on(2, 5, 4);
    handed_on(257, 2, 3);
    handed_on(7, 1, 3);
    handed_on(1, 8, 3);
}

/// Rows right padded to unequal lengths, one shorter than the taps read back and one of none, then
/// a window step over the rings the prompt left: each row's step matches its valid positions alone.
#[test]
fn right_padded_rows_seed_the_ring_for_a_window_step() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for elem in elements(&client) {
        let rows = [(1, 0, 9), (0, 0, 5), (2, 2, 2), (3, 7, 0)];
        let case = prompt_case(9, 64, 4, 3, &rows);
        let o = case.operands(elem);
        let ring = case.ring.ring(&o.sequences);
        let layout = PromptLayout::default();
        let (raw, store) = case.ring.ring_tensor(&client, &ring, elem);
        let inputs = case.inputs(&client, &o, store.clone(), layout, BIASED_F32, elem);
        let Some(outputs) = launch(&hw, inputs, "padded rows") else {
            continue;
        };
        let got = outputs_of(&client, &outputs, F32);
        let want = case.reference(&o, true, elem);
        for (i, row) in case.ring.rows.iter().enumerate() {
            let (from, len) = (i * case.ring.positions * 64, row.valid * 64);
            compare(
                &got[from..from + len],
                &want[from..from + len],
                tolerance(elem),
                "padded rows",
            );
        }
        case.ring
            .check_ring(&client, raw, &o.sequences, &ring, elem, "padded rows");

        let steps: Vec<(usize, usize)> = rows.iter().map(|&(slot, b, v)| (slot, b + v)).collect();
        let step = window_case(64, 4, 3, &steps);
        let known: Vec<Vec<f32>> = (case.ring.rows.iter().zip(&o.sequences))
            .enumerate()
            .map(|(i, (row, sequence))| {
                let mut known = sequence[..(row.before + row.valid) * 64].to_vec();
                known.extend(wave(64, 2.5 + i as f32, 1.0, elem));
                known
            })
            .collect();
        let stepped = Operands {
            sequences: known,
            filters: o.filters.clone(),
            bias: o.bias.clone(),
        };
        let inputs = step.inputs(&client, &stepped, store, layout, BIASED_F32, elem);
        let Some(outputs) = launch(&hw, inputs, "the step after padded rows") else {
            continue;
        };
        let got = outputs_of(&client, &outputs, F32);
        let want = step.reference(&stepped, true, elem);
        compare(&got, &want, tolerance(elem), "padded rows");
    }
}

/// What the conv declines by name: a prompt with no position, no row, no channel, a ring shorter than the taps read
/// back, and an output that is neither `f32` nor the activations'.
#[test]
fn what_the_conv_cannot_run_is_declined_by_name() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layout = PromptLayout::default();
    let declined = |case: CausalCase, output: ElemType| {
        let o = case.operands(F32);
        let ring = vec![0.0; case.ring.slots * case.ring.columns * case.ring.channels];
        let (_, store) = case.ring.ring_tensor(&client, &ring, F32);
        let setting = Setting {
            biased: false,
            output,
        };
        let inputs = case.inputs(&client, &o, store, layout, setting, F32);
        CubeCausalConvRoutine.launch_on(&hw, inputs).err()
    };
    assert_eq!(
        declined(prompt_case(0, 32, 4, 3, &[(0, 0, 0)]), F32),
        Some(CAUSAL_CONV_NOTHING_TO_CONVOLVE)
    );
    assert_eq!(
        declined(prompt_case(1, 32, 4, 3, &[]), F32),
        Some(CAUSAL_CONV_NOTHING_TO_CONVOLVE)
    );
    assert_eq!(
        declined(prompt_case(1, 0, 4, 3, &[(0, 0, 1)]), F32),
        Some(CAUSAL_CONV_CHANNELS_EMPTY)
    );
    assert_eq!(
        declined(window_case(32, 4, 2, &[(0, 0)]), F32),
        Some(CAUSAL_CONV_TAPS_PAST_RING)
    );
    assert_eq!(
        declined(window_case(32, 4, 3, &[(0, 0)]), F16),
        Some(CAUSAL_CONV_OUTPUT_UNSUPPORTED)
    );
}

/// The window case the slot tests run over: rows at slots of a store of six.
fn slotted(rows: &[(usize, usize)]) -> CausalCase {
    let mut case = window_case(37 * 3, 4, 5, rows);
    case.ring.slots = 6;
    case
}

/// One launch of `case` at `f32`, its outputs and ring checked: `tables` replaces the slot and
/// length tables the case holds.
fn slot_launch(
    case: &CausalCase,
    tables: impl FnOnce(&Client) -> (PlainTensor, PlainTensor),
    what: &str,
) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let o = case.operands(F32);
    let ring = case.ring.ring(&o.sequences);
    let (raw, store) = case.ring.ring_tensor(&client, &ring, F32);
    let mut inputs = case.inputs(
        &client,
        &o,
        store.clone(),
        PromptLayout::default(),
        BIASED_F32,
        F32,
    );
    let (slots, lengths) = tables(&client);
    inputs.past = CausalConvPastInputs::Window {
        ring: SlotStore::new(store, slots, lengths),
    };
    if let Some(outputs) = launch(&hw, inputs, what) {
        let got = outputs_of(&client, &outputs, F32);
        compare(&got, &case.reference(&o, true, F32), tolerance(F32), what);
        case.ring
            .check_ring(&client, raw, &o.sequences, &ring, F32, what);
    }
}

/// Tables that are columns of wider tensors: each row reads its own entry, not its neighbour's.
#[test]
fn tables_at_a_stride_are_read_by_it() {
    let case = slotted(&[(4, 9), (1, 0), (2, 1), (5, 2)]);
    let rows = case.ring.rows.clone();
    slot_launch(
        &case,
        |client| {
            let slots = rows.iter().map(|r| r.slot as u32).collect();
            let lengths = rows.iter().map(|r| r.before as u32).collect();
            (
                strided_table(client, slots, 0),
                strided_table(client, lengths, 7),
            )
        },
        "tables at a stride",
    );
}

/// A slot past the store's last reads and writes the last: the row behaves as one listing the last
/// slot, and leaves every other slot as it was.
#[test]
fn a_slot_past_the_store_is_the_last_slot() {
    let case = slotted(&[(5, 7)]);
    slot_launch(
        &case,
        |client| (table(client, vec![5 + 5]), table(client, vec![7])),
        "a slot past the store",
    );
}

/// Two launches of one prepared problem with different slot tables each leave the results of
/// their own rows, over a window and over a prompt.
#[test]
fn a_prepared_problem_is_right_for_each_slot_table() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let windows: [&[(usize, usize)]; 2] = [&[(4, 9), (1, 0), (2, 1)], &[(0, 3), (3, 0), (5, 8)]];
    let prompts: [&[(usize, usize, usize)]; 2] = [
        &[(4, 9, 5), (1, 0, 5), (2, 1, 5)],
        &[(0, 3, 5), (3, 0, 5), (5, 8, 5)],
    ];
    let prompt = |rows: &[(usize, usize, usize)]| {
        let mut case = prompt_case(5, 37 * 3, 4, 5, rows);
        case.ring.slots = 6;
        case
    };
    let cases: Vec<CausalCase> = (windows.iter().map(|rows| slotted(rows)))
        .chain(prompts.iter().map(|rows| prompt(rows)))
        .collect();
    let prepared = [PreparedLaunches::default(), PreparedLaunches::default()];
    for (i, case) in cases.iter().enumerate() {
        let o = case.operands(F32);
        let ring = case.ring.ring(&o.sequences);
        let (raw, store) = case.ring.ring_tensor(&client, &ring, F32);
        let inputs = case.inputs(&client, &o, store, PromptLayout::default(), BIASED_F32, F32);
        let launched = CubeCausalConvRoutine.launch_prepared_on(
            &prepared[usize::from(!case.window)],
            &hw,
            inputs.clone(),
        );
        let what = format!("prepared {i}");
        let outputs = served(&what, launched, || admit(&hw, inputs));
        if let Some(outputs) = outputs {
            let got = outputs_of(&client, &outputs, F32);
            compare(&got, &case.reference(&o, true, F32), tolerance(F32), &what);
            case.ring
                .check_ring(&client, raw, &o.sequences, &ring, F32, &what);
        }
    }
}
