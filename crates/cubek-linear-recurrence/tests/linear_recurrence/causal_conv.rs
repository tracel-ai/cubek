//! The causal conv with bias against a host reference in `f64`, over a window and over a prompt:
//! the ring's wrap, a row with no history, prompts and windows packed and pitched, rows right padded
//! to unequal lengths, and a prompt handed on to a window step. The rings hold NaN wherever the
//! conv must not read.

use super::conv::{compare, convolve_channels, tolerance, view, wave};
use super::harness::{F16, F32, activation_elements, read, served};
use super::ring::{PromptLayout, RingCase};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubek_linear_recurrence::kernels::conv::{
    CausalConvInputs, CausalConvOutputs, CausalConvPastInputs, CausalConvTensors,
    CubeCausalConvRoutine,
};
use cubek_linear_recurrence::kernels::refusals::{
    CAUSAL_CONV_OUTPUT_UNSUPPORTED, CONV_NOTHING_TO_CONVOLVE, CONV_TAPS_PAST_RING,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_std::Hardware;

/// A causal conv's rows, over a window when `positions` is one and `window` holds, else a prompt.
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

/// A decode step with a ring that wraps, a fresh row and a ring as short as the taps read back, a
/// long prompt, a prompt of one position, prime channels and the widths of a wide layer.
fn cases() -> Vec<CausalCase> {
    vec![
        window_case(1280, 4, 3, &[(7, 9), (3, 0), (4, 1), (1, 2)]),
        window_case(37 * 3, 4, 5, &[(4, 9), (1, 0), (2, 3)]),
        window_case(12, 2, 1, &[(1, 5), (0, 0), (2, 1)]),
        prompt_case(257, 1280, 4, 3, &[(0, 0, 257), (1, 3, 257)]),
        prompt_case(1, 37 * 3, 4, 5, &[(2, 9, 1), (0, 0, 1)]),
        prompt_case(5, 12, 2, 1, &[(0, 1, 5), (1, 6, 5), (2, 0, 5)]),
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

    /// The input as it lies in its buffer under `layout`: a window's `[rows, channels]` newest
    /// values, or the prompts.
    fn input(
        &self,
        client: &Client,
        sequences: &[Vec<f32>],
        layout: PromptLayout,
        elem: ElemType,
    ) -> PlainTensor {
        if !self.window {
            return self.ring.prompt_input(client, sequences, layout, elem);
        }
        let channels = self.ring.channels;
        let pitch = channels + layout.pad;
        let rows = self.ring.rows.len();
        let mut buffer = vec![f32::NAN; layout.start + rows * pitch];
        for (b, (row, sequence)) in self.ring.rows.iter().zip(sequences).enumerate() {
            buffer[layout.start + b * pitch..][..channels]
                .copy_from_slice(&sequence[row.before * channels..][..channels]);
        }
        view(
            client,
            &buffer,
            layout.start,
            &[rows, channels],
            &[pitch, 1],
            elem,
        )
        .1
    }

    /// The inputs of the case, and the ring's buffer.
    fn inputs(
        &self,
        client: &Client,
        o: &Operands,
        ring: &[f32],
        layout: PromptLayout,
        setting: Setting,
        elem: ElemType,
    ) -> (CausalConvInputs, Handle) {
        let channels = self.ring.channels;
        let (raw, store) = self.ring.ring_tensor(client, ring, elem);
        let [slots, lengths, valid] = self.ring.tables(client);
        let past = if self.window {
            CausalConvPastInputs::Window {
                ring: store,
                slots,
                lengths,
            }
        } else {
            CausalConvPastInputs::Prompt {
                ring: store,
                slots,
                lengths,
                valid,
            }
        };
        let shape = [channels, self.ring.taps];
        let filters = view(client, &o.filters, 0, &shape, &[self.ring.taps, 1], elem).1;
        let bias = setting
            .biased
            .then(|| view(client, &o.bias, 0, &[channels], &[1], elem).1);
        let inputs = CausalConvInputs {
            client: client.clone(),
            input: self.input(client, &o.sequences, layout, elem),
            filters,
            bias,
            past,
            output_element: setting.output,
        };
        (inputs, raw)
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
    served(what, launched, || {
        let tensors = CausalConvTensors::new(inputs).expect("the elements are served");
        tensors.problem().admit(hw)
    })
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
    for elem in activation_elements(&client) {
        for layout in layouts {
            for biased in [false, true] {
                for output in [elem, F32] {
                    for case in cases() {
                        let o = case.operands(elem);
                        let ring = case.ring.ring(&o.sequences);
                        let setting = Setting { biased, output };
                        let (inputs, raw) = case.inputs(&client, &o, &ring, layout, setting, elem);
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
        let setting = Setting {
            biased: true,
            output: F32,
        };
        let layout = PromptLayout::default();
        let (inputs, _) = case.inputs(&client, &o, &ring, layout, setting, F32);
        if let Some(outputs) = launch(&hw, inputs, "no history") {
            let got = outputs_of(&client, &outputs, F32);
            compare(
                &got,
                &case.reference(&o, true, F32),
                tolerance(F32),
                "no history",
            );
        }
    }
}

/// Window steps over one ring, one position each, for more steps than the ring has columns: each
/// step matches the reference over the sequence so far, so the ring wraps right.
#[test]
fn the_ring_wraps_over_several_steps() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for elem in activation_elements(&client) {
        let (channels, taps, columns, steps) = (60, 4, 3, 8);
        let full = window_case(channels, taps, columns, &[(0, steps)]);
        let o = full.operands(elem);
        let ring = vec![f32::NAN; columns * channels];
        let (_, store) = full.ring.ring_tensor(&client, &ring, elem);
        for step in 0..steps {
            let case = window_case(channels, taps, columns, &[(0, step)]);
            let one = Operands {
                sequences: vec![o.sequences[0][..(step + 1) * channels].to_vec()],
                filters: o.filters.clone(),
                bias: o.bias.clone(),
            };
            let (mut inputs, _) = case.inputs(
                &client,
                &one,
                &ring,
                PromptLayout::default(),
                Setting {
                    biased: true,
                    output: F32,
                },
                elem,
            );
            let CausalConvPastInputs::Window { ring, .. } = &mut inputs.past else {
                unreachable!("a window case");
            };
            *ring = store.clone();
            let what = format!("step {step}");
            let Some(outputs) = launch(&hw, inputs, &what) else {
                return;
            };
            let got = outputs_of(&client, &outputs, F32);
            compare(
                &got,
                &case.reference(&one, true, elem),
                tolerance(elem),
                &what,
            );
        }
    }
}

/// A prompt of `L` positions then one window step gives what a prompt of `L + 1` gives at its
/// last position: the ring the prompt leaves is the one a longer prompt would have read.
#[test]
fn a_prompt_then_a_window_step_equals_a_longer_prompt() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let setting = Setting {
        biased: true,
        output: F32,
    };
    let (channels, taps, columns, length) = (96, 4, 3, 11);
    for elem in activation_elements(&client) {
        let longer = prompt_case(length + 1, channels, taps, columns, &[(0, 0, length + 1)]);
        let o = longer.operands(elem);
        let layout = PromptLayout::default();
        let ring = longer.ring.ring(&o.sequences);
        let (inputs, _) = longer.inputs(&client, &o, &ring, layout, setting, elem);
        let Some(whole) = launch(&hw, inputs, "the longer prompt") else {
            return;
        };
        let last = outputs_of(&client, &whole, F32)[length * channels..].to_vec();

        let prompt = prompt_case(length, channels, taps, columns, &[(0, 0, length)]);
        let head = Operands {
            sequences: vec![o.sequences[0][..length * channels].to_vec()],
            filters: o.filters.clone(),
            bias: o.bias.clone(),
        };
        let (inputs, _) = prompt.inputs(&client, &head, &ring, layout, setting, elem);
        let CausalConvPastInputs::Prompt { ring: window, .. } = inputs.past.clone() else {
            unreachable!("a prompt case");
        };
        if launch(&hw, inputs, "the prompt").is_none() {
            return;
        }
        let step = window_case(channels, taps, columns, &[(0, length)]);
        let (mut inputs, _) = step.inputs(&client, &o, &ring, layout, setting, elem);
        let CausalConvPastInputs::Window { ring: store, .. } = &mut inputs.past else {
            unreachable!("a window case");
        };
        *store = window;
        // The step's input is the sequence's position `length`, which the longer prompt holds.
        let Some(stepped) = launch(&hw, inputs, "the step") else {
            return;
        };
        let got = outputs_of(&client, &stepped, F32);
        let want: Vec<f64> = last.iter().map(|&x| x as f64).collect();
        compare(&got, &want, 1e-6, "the step after a prompt");
    }
}

/// Rows right padded to unequal lengths, one shorter than the taps read back and one of none, then
/// a window step over the rings the prompt left: each row's step matches its valid positions alone.
#[test]
fn right_padded_rows_seed_the_ring_for_a_window_step() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let setting = Setting {
        biased: true,
        output: F32,
    };
    for elem in activation_elements(&client) {
        let rows = [(1, 0, 9), (0, 0, 5), (2, 2, 2), (3, 7, 0)];
        let case = prompt_case(9, 64, 4, 3, &rows);
        let o = case.operands(elem);
        let ring = case.ring.ring(&o.sequences);
        let layout = PromptLayout::default();
        let (inputs, raw) = case.inputs(&client, &o, &ring, layout, setting, elem);
        let CausalConvPastInputs::Prompt { ring: window, .. } = inputs.past.clone() else {
            unreachable!("a prompt case");
        };
        if launch(&hw, inputs, "padded rows").is_none() {
            return;
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
        let (mut inputs, _) = step.inputs(&client, &stepped, &ring, layout, setting, elem);
        let CausalConvPastInputs::Window { ring: store, .. } = &mut inputs.past else {
            unreachable!("a window case");
        };
        *store = window;
        let Some(outputs) = launch(&hw, inputs, "the step after padded rows") else {
            return;
        };
        let got = outputs_of(&client, &outputs, F32);
        compare(
            &got,
            &step.reference(&stepped, true, elem),
            tolerance(elem),
            "padded rows",
        );
    }
}

/// What the conv declines by name: a prompt with no position, a ring shorter than the taps read
/// back, and an output that is neither `f32` nor the activations'.
#[test]
fn what_the_conv_cannot_run_is_declined_by_name() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layout = PromptLayout::default();
    let declined = |case: CausalCase, output: ElemType| {
        let elem = F32;
        let o = case.operands(elem);
        let ring = vec![0.0; case.ring.slots * case.ring.columns * case.ring.channels];
        let setting = Setting {
            biased: false,
            output,
        };
        let (inputs, _) = case.inputs(&client, &o, &ring, layout, setting, elem);
        CubeCausalConvRoutine.launch_on(&hw, inputs).err()
    };
    assert_eq!(
        declined(prompt_case(0, 32, 4, 3, &[(0, 0, 0)]), F32),
        Some(CONV_NOTHING_TO_CONVOLVE)
    );
    assert_eq!(
        declined(window_case(32, 4, 2, &[(0, 0)]), F32),
        Some(CONV_TAPS_PAST_RING)
    );
    assert_eq!(
        declined(window_case(32, 4, 3, &[(0, 0)]), F16),
        Some(CAUSAL_CONV_OUTPUT_UNSUPPORTED)
    );
}
