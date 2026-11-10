//! The gated delta conv of a prompt against a host reference in `f64`: rows that have convolved
//! some positions already, so their rings seed the first taps, prompts padded and off a whole read
//! in their buffers, and rows right padded to unequal lengths. The rings the prompt leaves are read
//! back, and decode steps run on them.

use super::conv::{
    ConvCase, EPSILON, HostOutputs, admit, case, compare_outputs, conv_or_skip, convolve_channels,
    table, view, wave,
};
use super::harness::{F32, activation_elements, served};
use super::ring::{PromptLayout, RingCase};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubek_linear_recurrence::kernels::conv::{
    CausalConvPastInputs, CubeQkNormedConvRoutine, QkNormedConvInputs,
};
use cubek_linear_recurrence::kernels::refusals::CONV_NOTHING_TO_CONVOLVE;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;

/// A prompt's shape: the heads and the rows' rings.
#[derive(Debug, Clone)]
struct PromptCase {
    conv: ConvCase,
    ring: RingCase,
}

fn prompt_case(
    positions: usize,
    heads: (usize, usize, usize, usize),
    taps: usize,
    columns: usize,
    rows: &[(usize, usize, usize)],
) -> PromptCase {
    let ring = RingCase::new(positions, 0, taps, columns, rows);
    let conv = case(heads, taps, columns, ring.slots, &[]);
    let channels = conv.conv_dim();
    PromptCase {
        conv,
        ring: RingCase { channels, ..ring },
    }
}

/// A long prompt, a prompt of one position, prime heads, a ring that wraps, and rows that start
/// from nothing, from exactly the taps read back, and from a ring they have gone round.
fn cases() -> Vec<PromptCase> {
    vec![
        prompt_case(512, (16, 1, 128, 128), 4, 3, &[(0, 0, 512), (1, 3, 512)]),
        prompt_case(1, (3, 3, 37, 53), 4, 5, &[(2, 9, 1), (0, 0, 1)]),
        prompt_case(5, (2, 2, 12, 8), 2, 1, &[(0, 1, 5), (1, 6, 5), (2, 0, 5)]),
    ]
}

/// The values of a case: each row's positions `[position][channel]`, its history first, and the
/// filters.
struct PromptValues {
    sequences: Vec<Vec<f32>>,
    filters: Vec<f32>,
}

impl PromptCase {
    fn values(&self, elem: ElemType) -> PromptValues {
        PromptValues {
            sequences: self.ring.sequences(elem),
            filters: wave(self.conv.conv_dim() * self.conv.taps, 0.1, 0.6, elem),
        }
    }

    /// The inputs of the prompt, and the ring's buffer.
    fn inputs(
        &self,
        client: &Client,
        values: &PromptValues,
        ring: &[f32],
        layout: PromptLayout,
        elem: ElemType,
    ) -> (QkNormedConvInputs, Handle) {
        let conv_dim = self.conv.conv_dim();
        let qkv = self
            .ring
            .prompt_input(client, &values.sequences, layout, elem);
        let (_, filters) = view(
            client,
            &values.filters,
            0,
            &[conv_dim, self.conv.taps],
            &[self.conv.taps, 1],
            elem,
        );
        let (raw, store) = self.ring.ring_tensor(client, ring, elem);
        let [slots, lengths, valid] = self.ring.tables(client);
        let inputs = QkNormedConvInputs {
            client: client.clone(),
            qkv,
            filters,
            past: CausalConvPastInputs::Prompt {
                ring: store,
                slots,
                lengths,
                valid,
            },
            heads: self.conv.heads,
            epsilon: EPSILON,
        };
        (inputs, raw)
    }

    /// The queries, keys and values of every position of every row, `[row][position]`.
    fn reference(&self, values: &PromptValues, elem: ElemType) -> HostOutputs {
        let mut all = HostOutputs::default();
        for (row, sequence) in self.ring.rows.iter().zip(&values.sequences) {
            for t in 0..self.ring.positions {
                let position = row.before + t;
                let outputs = self.position_outputs(sequence, &values.filters, position, elem);
                all.query.extend(outputs.query);
                all.key.extend(outputs.key);
                all.value.extend(outputs.value);
            }
        }
        all
    }

    /// The outputs at `position` of one row's `sequence`, the positions before the sequence zero.
    fn position_outputs(
        &self,
        sequence: &[f32],
        filters: &[f32],
        position: usize,
        elem: ElemType,
    ) -> HostOutputs {
        let (conv_dim, taps) = (self.conv.conv_dim(), self.conv.taps);
        let at = |position: usize, channel: usize| sequence[position * conv_dim + channel];
        let convolved = convolve_channels(
            filters,
            taps,
            |channel| at(position, channel),
            |j, channel| (position + j + 1 >= taps).then(|| at(position + j + 1 - taps, channel)),
            elem,
        );
        self.conv.head_outputs(&convolved)
    }
}

/// Every case at every element the device runs, over a packed prompt and over one padded and
/// starting off a whole read: the outputs match the reference, and the ring holds each row's last
/// valid positions and nothing else changes.
#[test]
fn the_prompt_conv_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layouts = [PromptLayout::default(), PromptLayout { pad: 3, start: 1 }];
    for elem in activation_elements(&client) {
        for layout in layouts {
            for case in cases() {
                let values = case.values(elem);
                let ring = case.ring.ring(&values.sequences);
                let (inputs, raw) = case.inputs(&client, &values, &ring, layout, elem);
                let what = format!("{layout:?} {case:?}");
                if let Some(outputs) = conv_or_skip(&hw, inputs, &what) {
                    let want = case.reference(&values, elem);
                    compare_outputs(&client, &outputs, &want, elem, &what);
                    case.ring
                        .check_ring(&client, raw, &values.sequences, &ring, elem, &what);
                }
            }
        }
    }
}

/// A prompt with no position has nothing to convolve: the conv declines it by name.
#[test]
fn a_prompt_with_no_position_is_declined() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = prompt_case(0, (2, 2, 12, 8), 4, 3, &[(0, 0, 0)]);
    let values = case.values(F32);
    let ring = case.ring.ring(&values.sequences);
    let (inputs, _) = case.inputs(&client, &values, &ring, PromptLayout::default(), F32);
    assert_eq!(
        CubeQkNormedConvRoutine.launch_on(&hw, inputs).err(),
        Some(CONV_NOTHING_TO_CONVOLVE)
    );
}

/// The ring store of a prompt's inputs, which a decode step reads as the prompt left it.
fn ring_of(inputs: &QkNormedConvInputs) -> PlainTensor {
    match &inputs.past {
        CausalConvPastInputs::Prompt { ring, .. } => ring.clone(),
        CausalConvPastInputs::Window { .. } => panic!("the inputs are over a prompt"),
    }
}

/// One decode step of the rows of `case` after their prompts, over the ring the prompt left: row
/// `r` has convolved `lengths[r]` positions of `values` and takes `newest[r]`. `None` where the
/// device declines the step.
fn decode_step(
    (client, hw): (&Client, &Hardware),
    case: &PromptCase,
    ring: PlainTensor,
    newest: &[Vec<f32>],
    lengths: Vec<u32>,
    values: &PromptValues,
    elem: ElemType,
) -> Option<()> {
    let conv_dim = case.conv.conv_dim();
    let rows = case.ring.rows.len();
    let flat: Vec<f32> = newest.concat();
    let (_, qkv) = view(client, &flat, 0, &[rows, conv_dim], &[conv_dim, 1], elem);
    let (_, filters) = view(
        client,
        &values.filters,
        0,
        &[conv_dim, case.conv.taps],
        &[case.conv.taps, 1],
        elem,
    );
    let inputs = QkNormedConvInputs {
        client: client.clone(),
        qkv,
        filters,
        past: CausalConvPastInputs::Window {
            ring,
            slots: table(
                client,
                case.ring.rows.iter().map(|r| r.slot as u32).collect(),
            ),
            lengths: table(client, lengths.clone()),
        },
        heads: case.conv.heads,
        epsilon: EPSILON,
    };
    let what = "a decode step";
    let outputs = served(
        what,
        CubeQkNormedConvRoutine.launch_on(hw, inputs.clone()),
        || admit(hw, inputs),
    )?;
    let mut want = HostOutputs::default();
    for (r, sequence) in values.sequences.iter().enumerate() {
        let length = lengths[r] as usize;
        let mut known = sequence[..length * conv_dim].to_vec();
        known.extend(&newest[r]);
        let outputs = case.position_outputs(&known, &values.filters, length, elem);
        want.query.extend(outputs.query);
        want.key.extend(outputs.key);
        want.value.extend(outputs.value);
    }
    compare_outputs(client, &outputs, &want, elem, what);
    Some(())
}

/// Rows right padded to unequal lengths, one of them shorter than the taps read back and one of
/// none, then a decode step over the rings the prompt left: each row's step matches a sequence of
/// its valid positions alone, which a copy of the prompt's tail would not give.
#[test]
fn right_padded_rows_seed_the_ring_for_a_decode_step() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for elem in activation_elements(&client) {
        let case = prompt_case(
            9,
            (3, 2, 20, 12),
            4,
            3,
            &[(1, 0, 9), (0, 0, 5), (2, 2, 2), (3, 7, 0)],
        );
        let values = case.values(elem);
        let ring = case.ring.ring(&values.sequences);
        let (inputs, raw) = case.inputs(&client, &values, &ring, PromptLayout::default(), elem);
        let ring_view = ring_of(&inputs);
        if conv_or_skip(&hw, inputs, "padded rows").is_none() {
            return;
        }
        case.ring
            .check_ring(&client, raw, &values.sequences, &ring, elem, "padded rows");
        let newest: Vec<Vec<f32>> = (0..case.ring.rows.len())
            .map(|r| wave(case.conv.conv_dim(), 2.5 + r as f32, 1.0, elem))
            .collect();
        let lengths = (case.ring.rows.iter())
            .map(|r| (r.before + r.valid) as u32)
            .collect();
        decode_step(
            (&client, &hw),
            &case,
            ring_view,
            &newest,
            lengths,
            &values,
            elem,
        );
    }
}

/// A prompt of `prompt` positions and `steps` decode steps after it, over a ring of `columns`
/// columns the prompt fills itself: every output matches the reference over the whole sequence.
fn handed_on(prompt: usize, steps: usize, columns: usize, elem: ElemType) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = prompt_case(prompt, (3, 2, 20, 12), 4, columns, &[(0, 0, prompt)]);
    let conv_dim = case.conv.conv_dim();
    let mut values = case.values(elem);
    let ring = case.ring.ring(&values.sequences);
    let (inputs, _) = case.inputs(&client, &values, &ring, PromptLayout::default(), elem);
    let ring_view = ring_of(&inputs);
    let Some(outputs) = conv_or_skip(&hw, inputs, "the prompt") else {
        return;
    };
    let want = case.reference(&values, elem);
    compare_outputs(&client, &outputs, &want, elem, "the prompt");
    for step in 0..steps {
        let newest = wave(conv_dim, 3.5 + step as f32, 1.0, elem);
        let length = values.sequences[0].len() / conv_dim;
        let stepped = decode_step(
            (&client, &hw),
            &case,
            ring_view.clone(),
            std::slice::from_ref(&newest),
            vec![length as u32],
            &values,
            elem,
        );
        if stepped.is_none() {
            return;
        }
        values.sequences[0].extend(&newest);
    }
}

/// A prompt handed on to decode, with a ring as short as the columns read back and with a prompt
/// shorter than the ring.
#[test]
fn a_prompt_handed_on_to_decode_matches_the_reference() {
    let client = cubecl::test_device().client();
    for elem in activation_elements(&client) {
        handed_on(9, 4, 3, elem);
        handed_on(2, 5, 4, elem);
    }
}
