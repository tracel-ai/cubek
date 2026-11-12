//! The rows of a prompt and the rings they leave. A row has convolved `before` positions of its
//! sequence, its ring holding the last `taps - 1` of them, and brings a prompt of which `valid`
//! positions are real.

use super::harness::{read, round_to, table, view, wave};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubek_matmul::kernels::tensor::PlainTensor;

#[derive(Debug, Clone, Copy)]
pub struct PromptRow {
    pub slot: usize,
    pub before: usize,
    pub valid: usize,
}

/// The elements each prompt position is padded by, and the elements the view starts past its
/// buffer's start.
#[derive(Debug, Clone, Copy, Default)]
pub struct PromptLayout {
    pub pad: usize,
    pub start: usize,
}

/// The rings of a prompt's rows, dense or pitched.
#[derive(Debug, Clone)]
pub struct RingCase {
    pub channels: usize,
    pub taps: usize,
    pub columns: usize,
    pub slots: usize,
    pub positions: usize,
    pub rows: Vec<PromptRow>,
    /// The elements a ring column and a ring slot are padded by in the store.
    pub padding: RingPadding,
}

/// The elements the store pads each column and each slot by.
#[derive(Debug, Clone, Copy, Default)]
pub struct RingPadding {
    pub column: usize,
    pub slot: usize,
}

impl RingCase {
    /// Rows of `(slot, before, valid)` over a store of as many slots as the last row's slot needs.
    pub fn new(
        positions: usize,
        channels: usize,
        taps: usize,
        columns: usize,
        rows: &[(usize, usize, usize)],
    ) -> Self {
        Self {
            channels,
            taps,
            columns,
            slots: rows.iter().map(|&(slot, ..)| slot + 1).max().unwrap_or(1),
            positions,
            rows: rows
                .iter()
                .map(|&(slot, before, valid)| PromptRow {
                    slot,
                    before,
                    valid,
                })
                .collect(),
            padding: RingPadding::default(),
        }
    }

    /// The case with a store whose columns and slots are padded by `padding`.
    pub fn pitched(mut self, padding: RingPadding) -> Self {
        self.padding = padding;
        self
    }

    fn column_pitch(&self) -> usize {
        self.channels + self.padding.column
    }

    fn slot_pitch(&self) -> usize {
        self.columns * self.column_pitch() + self.padding.slot
    }

    /// Where the ring column `column` of `slot` starts in the store.
    fn column_start(&self, slot: usize, column: usize) -> usize {
        slot * self.slot_pitch() + column * self.column_pitch()
    }

    /// The positions of the row the ring takes: `valid`, at most the prompt's.
    pub fn real(&self, row: &PromptRow) -> usize {
        row.valid.min(self.positions)
    }

    /// The taps read back from the past.
    pub fn read_columns(&self) -> usize {
        self.taps - 1
    }

    /// Each row's sequence `[position][channel]`: its history, then its prompt.
    pub fn sequences(&self, elem: ElemType) -> Vec<Vec<f32>> {
        (self.rows.iter().enumerate())
            .map(|(i, row)| {
                let len = (row.before + self.positions) * self.channels;
                wave(len, 0.9 + i as f32, 1.1, elem)
            })
            .collect()
    }

    /// The ring store before the launch: each row's last `taps - 1` history positions at their
    /// columns, NaN elsewhere.
    pub fn ring(&self, sequences: &[Vec<f32>]) -> Vec<f32> {
        let mut ring = vec![f32::NAN; self.slots * self.slot_pitch()];
        for (row, sequence) in self.rows.iter().zip(sequences) {
            for position in row.before.saturating_sub(self.read_columns())..row.before {
                let to = self.column_start(row.slot, position % self.columns);
                ring[to..to + self.channels].copy_from_slice(
                    &sequence[position * self.channels..(position + 1) * self.channels],
                );
            }
        }
        ring
    }

    /// The ring's buffer and the tensor over it.
    pub fn ring_tensor(
        &self,
        client: &Client,
        ring: &[f32],
        elem: ElemType,
    ) -> (Handle, PlainTensor) {
        let shape = [self.slots, self.columns, self.channels];
        let strides = [self.slot_pitch(), self.column_pitch(), 1];
        view(client, ring, 0, &shape, &strides, elem)
    }

    /// The slot, length and valid tables of the rows.
    pub fn tables(&self, client: &Client) -> [PlainTensor; 3] {
        let column = |entry: fn(&PromptRow) -> usize| {
            table(client, self.rows.iter().map(|r| entry(r) as u32).collect())
        };
        [
            column(|r| r.slot),
            column(|r| r.before),
            column(|r| r.valid),
        ]
    }

    /// The rows' prompts `[rows, positions, channels]`, padded and started off a whole read as
    /// `layout` says.
    pub fn prompt_input(
        &self,
        client: &Client,
        sequences: &[Vec<f32>],
        layout: PromptLayout,
        elem: ElemType,
    ) -> PlainTensor {
        let pitch = self.channels + layout.pad;
        let rows = self.rows.len();
        let mut buffer = vec![f32::NAN; (layout.start + rows * self.positions * pitch).max(1)];
        for (b, (row, sequence)) in self.rows.iter().zip(sequences).enumerate() {
            for t in 0..self.positions {
                let from = (row.before + t) * self.channels;
                buffer[layout.start + (b * self.positions + t) * pitch..][..self.channels]
                    .copy_from_slice(&sequence[from..from + self.channels]);
            }
        }
        let shape = [rows, self.positions, self.channels];
        let strides = [self.positions * pitch, pitch, 1];
        view(client, &buffer, layout.start, &shape, &strides, elem).1
    }

    /// The ring after the launch: every column a row's last valid positions fill holds them bit
    /// for bit, and every other element is as it was uploaded.
    pub fn check_ring(
        &self,
        client: &Client,
        raw: Handle,
        sequences: &[Vec<f32>],
        before: &[f32],
        elem: ElemType,
        what: &str,
    ) {
        let after = read(client, raw, elem);
        let mut written = vec![false; after.len()];
        for (row, sequence) in self.rows.iter().zip(sequences) {
            let valid = self.real(row);
            for t in valid.saturating_sub(self.read_columns())..valid {
                let position = row.before + t;
                let to = self.column_start(row.slot, position % self.columns);
                for channel in 0..self.channels {
                    written[to + channel] = true;
                    assert_eq!(
                        after[to + channel].to_bits(),
                        sequence[position * self.channels + channel].to_bits(),
                        "{what}: ring of slot {} at position {position}, channel {channel}",
                        row.slot
                    );
                }
            }
        }
        for (i, &written) in written.iter().enumerate() {
            if !written {
                assert_eq!(
                    after[i].to_bits(),
                    round_to(before[i], elem).to_bits(),
                    "{what}: element {i} of the ring the prompt does not write changed"
                );
            }
        }
    }
}
