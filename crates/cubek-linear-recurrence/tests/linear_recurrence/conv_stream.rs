//! The gated delta conv of a stream, read by tap, against a host reference in `f64`: a long
//! prompt, a prompt of one position, prime heads, and streams padded and off a whole read in their
//! buffers. The last test hands a prompt on to decode: the stream conv of the prompt, then window
//! conv steps over a window the stream's tail seeds.

use super::conv::{
    ConvCase, EPSILON, HostOutputs, admit, case, compare_outputs, conv_or_skip, convolve_channels,
    table, view, wave,
};
use super::harness::{F32, activation_elements, served};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubek_linear_recurrence::kernels::conv::{
    CubeQkNormedConvRoutine, GatedDeltaConvPastInputs, QkNormedConvInputs,
};
use cubek_linear_recurrence::kernels::refusals::QK_NORMED_CONV_NOTHING_TO_CONVOLVE;
use cubek_std::Hardware;

/// A stream's shape.
#[derive(Debug, Clone)]
struct StreamCase {
    conv: ConvCase,
    batch: usize,
    positions: usize,
}

fn stream_case(
    batch: usize,
    positions: usize,
    heads: (usize, usize, usize, usize),
    taps: usize,
) -> StreamCase {
    StreamCase {
        conv: case(heads, taps, taps - 1, 1, &[]),
        batch,
        positions,
    }
}

fn cases() -> Vec<StreamCase> {
    vec![
        stream_case(1, 512, (16, 1, 128, 128), 4),
        stream_case(2, 1, (3, 3, 37, 53), 4),
        stream_case(3, 5, (2, 2, 12, 8), 2),
    ]
}

/// The elements each stream position is padded by, and the elements the view starts past its
/// buffer's start.
#[derive(Debug, Clone, Copy, Default)]
struct StreamLayout {
    pad: usize,
    start: usize,
}

impl StreamCase {
    fn stream_positions(&self) -> usize {
        self.positions + self.conv.taps - 1
    }

    /// The stream `[batch][stream position][channel]` and the filters `[channel][tap]`.
    fn operands(&self, elem: ElemType) -> (Vec<f32>, Vec<f32>) {
        let conv_dim = self.conv.conv_dim();
        let stream = wave(
            self.batch * self.stream_positions() * conv_dim,
            0.9,
            1.1,
            elem,
        );
        (stream, wave(conv_dim * self.conv.taps, 0.1, 0.6, elem))
    }

    fn inputs(
        &self,
        client: &Client,
        stream: &[f32],
        filters: &[f32],
        layout: StreamLayout,
        elem: ElemType,
    ) -> QkNormedConvInputs {
        let conv_dim = self.conv.conv_dim();
        let rows = self.stream_positions();
        let pitch = conv_dim + layout.pad;
        let mut buffer = vec![f32::NAN; layout.start + self.batch * rows * pitch];
        for position in 0..self.batch * rows {
            buffer[layout.start + position * pitch..][..conv_dim]
                .copy_from_slice(&stream[position * conv_dim..(position + 1) * conv_dim]);
        }
        let (_, qkv) = view(
            client,
            &buffer,
            layout.start,
            &[self.batch, rows, conv_dim],
            &[rows * pitch, pitch, 1],
            elem,
        );
        let (_, filters) = view(
            client,
            filters,
            0,
            &[conv_dim, self.conv.taps],
            &[self.conv.taps, 1],
            elem,
        );
        QkNormedConvInputs {
            client: client.clone(),
            qkv,
            filters,
            past: GatedDeltaConvPastInputs::Stream,
            heads: self.conv.heads,
            epsilon: EPSILON,
        }
    }

    /// The queries, keys and values of every position of every sequence, `[batch][position]`.
    fn reference(&self, stream: &[f32], filters: &[f32], elem: ElemType) -> HostOutputs {
        let (conv_dim, taps) = (self.conv.conv_dim(), self.conv.taps);
        let mut all = HostOutputs::default();
        for b in 0..self.batch {
            for t in 0..self.positions {
                let at = |j: usize, channel: usize| {
                    stream[(b * self.stream_positions() + t + j) * conv_dim + channel]
                };
                let convolved = convolve_channels(
                    filters,
                    taps,
                    |channel| at(taps - 1, channel),
                    |j, channel| Some(at(j, channel)),
                    elem,
                );
                let position = self.conv.head_outputs(&convolved);
                all.query.extend(position.query);
                all.key.extend(position.key);
                all.value.extend(position.value);
            }
        }
        all
    }
}

/// Every case at every element the device runs, over a packed stream and over one padded and
/// starting off a whole read: the outputs match the reference.
#[test]
fn the_stream_conv_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layouts = [StreamLayout::default(), StreamLayout { pad: 3, start: 1 }];
    for elem in activation_elements(&client) {
        for layout in layouts {
            for case in cases() {
                let (stream, filters) = case.operands(elem);
                let inputs = case.inputs(&client, &stream, &filters, layout, elem);
                let what = format!("{layout:?}");
                if let Some(outputs) = conv_or_skip(&hw, inputs, &what) {
                    let want = case.reference(&stream, &filters, elem);
                    compare_outputs(&client, &outputs, &want, elem, &format!("{what} {case:?}"));
                }
            }
        }
    }
}

/// A stream of the context columns alone has nothing to convolve: the conv declines it by name.
#[test]
fn a_stream_with_no_position_is_declined() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = stream_case(1, 0, (2, 2, 12, 8), 4);
    let (stream, filters) = case.operands(F32);
    let inputs = case.inputs(&client, &stream, &filters, StreamLayout::default(), F32);
    assert_eq!(
        CubeQkNormedConvRoutine.launch_on(&hw, inputs).err(),
        Some(QK_NORMED_CONV_NOTHING_TO_CONVOLVE)
    );
}

/// A prompt of `prompt` positions and `steps` decode steps after it, over a window of `columns`
/// columns seeded from the stream's tail: every output matches the reference over the whole
/// sequence as one stream.
fn handed_on(prompt: usize, steps: usize, columns: usize, elem: ElemType) {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let sequence = stream_case(1, prompt + steps, (3, 2, 20, 12), 4);
    let (taps, conv_dim) = (sequence.conv.taps, sequence.conv.conv_dim());
    let (mut stream, filters) = sequence.operands(elem);
    // A sequence starts from nothing: a window reads the columns it has not yet held as zeros.
    stream[..(taps - 1) * conv_dim].fill(0.0);
    let want = sequence.reference(&stream, &filters, elem);
    let per_position = |outputs: &[f64], width: usize, from: usize, count: usize| {
        outputs[from * width..(from + count) * width].to_vec()
    };
    let heads = sequence.conv.heads;
    let widths = (
        heads.key_dim(),
        heads.key_dim(),
        heads.value_heads() * heads.value_features,
    );
    let slice = |from: usize, count: usize| HostOutputs {
        query: per_position(&want.query, widths.0, from, count),
        key: per_position(&want.key, widths.1, from, count),
        value: per_position(&want.value, widths.2, from, count),
    };

    let leading = stream_case(1, prompt, (3, 2, 20, 12), 4);
    let inputs = leading.inputs(
        &client,
        &stream[..leading.stream_positions() * conv_dim],
        &filters,
        StreamLayout::default(),
        elem,
    );
    let Some(outputs) = conv_or_skip(&hw, inputs, "the prompt") else {
        return;
    };
    compare_outputs(&client, &outputs, &slice(0, prompt), elem, "the prompt");

    // The stream's position `prompt + j` is what tap `j` of the first step reads back, at ring
    // column `(prompt + j - (taps - 1)) % columns`.
    let mut window = vec![f32::NAN; columns * conv_dim];
    for j in 0..taps - 1 {
        let column = (prompt + j + columns - (taps - 1)) % columns;
        window[column * conv_dim..(column + 1) * conv_dim]
            .copy_from_slice(&stream[(prompt + j) * conv_dim..][..conv_dim]);
    }
    let (_, store) = view(
        &client,
        &window,
        0,
        &[1, columns, conv_dim],
        &[columns * conv_dim, conv_dim, 1],
        elem,
    );
    let (_, filters_view) = view(&client, &filters, 0, &[conv_dim, taps], &[taps, 1], elem);
    for step in 0..steps {
        let position = prompt + step;
        let newest = &stream[(position + taps - 1) * conv_dim..][..conv_dim];
        let (_, qkv) = view(&client, newest, 0, &[1, conv_dim], &[conv_dim, 1], elem);
        let inputs = QkNormedConvInputs {
            client: client.clone(),
            qkv,
            filters: filters_view.clone(),
            past: GatedDeltaConvPastInputs::Window {
                window: store.clone(),
                slots: table(&client, vec![0]),
                lengths: table(&client, vec![position as u32]),
            },
            heads,
            epsilon: EPSILON,
        };
        let what = format!("step {step} after a prompt of {prompt}");
        let Some(outputs) = served(
            &what,
            CubeQkNormedConvRoutine.launch_on(&hw, inputs.clone()),
            || admit(&hw, inputs),
        ) else {
            return;
        };
        compare_outputs(&client, &outputs, &slice(position, 1), elem, &what);
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
