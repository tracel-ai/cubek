//! The gated delta walk in a store's slots against the host reference: each row's state is read
//! from, and written back to, the slot its table names.
//!
//! The store holds NaN wherever the walk must not read it: the slot of a row that has walked no
//! position, and the gaps between its slots, heads and rows. The slots no row holds and the gaps
//! must come back bit for bit.

use super::harness::{F32, elements, read, round_to};
use super::walk::{
    Case, Layout, Operands, assert_close, assert_close_within, output_tolerance, tuned_or_skip,
    walk_or_skip,
};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::metadata::Metadata;
use cubek_linear_recurrence::kernels::SlotStore;
use cubek_linear_recurrence::kernels::walk::{
    CubeWalkRoutineSettings, RecurrenceWalkInputs, RecurrenceWalkOutputs, RecurrenceWalkStateInputs,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_std::Hardware;

/// How a store lies in its buffer: the gap after each state row, the gap after each head's state,
/// the gap after each slot's heads, and the elements its view starts past its buffer's start.
#[derive(Debug, Clone, Copy, Default)]
struct Pitch {
    row_gap: usize,
    head_gap: usize,
    slot_gap: usize,
    start: usize,
}

/// Stores with a gap after each slot, after each head, after each row, and a start off their
/// buffer's.
const STRIDED_PITCHES: [Pitch; 6] = [
    Pitch {
        row_gap: 0,
        head_gap: 0,
        slot_gap: 3,
        start: 0,
    },
    Pitch {
        row_gap: 0,
        head_gap: 1,
        slot_gap: 0,
        start: 2,
    },
    Pitch {
        row_gap: 0,
        head_gap: 4,
        slot_gap: 8,
        start: 4,
    },
    Pitch {
        row_gap: 0,
        head_gap: 0,
        slot_gap: 5,
        start: 1,
    },
    Pitch {
        row_gap: 3,
        head_gap: 1,
        slot_gap: 2,
        start: 0,
    },
    Pitch {
        row_gap: 4,
        head_gap: 0,
        slot_gap: 0,
        start: 0,
    },
];

/// A walk of `rows`, each a `(slot, positions walked already)`, over a store of `slots`. Rows
/// `parked` share a slot with others, so their output and their slot are whatever the race left.
#[derive(Debug, Clone)]
struct SlotCase {
    case: Case,
    slots: usize,
    rows: Vec<(usize, usize)>,
    parked: Vec<usize>,
}

impl SlotCase {
    fn new(
        positions: usize,
        heads: (usize, usize, usize, usize),
        slots: usize,
        rows: &[(usize, usize)],
        masked: &[usize],
    ) -> Self {
        let (key_heads, heads_per_group, key_features, value_features) = heads;
        Self {
            case: Case {
                batch: rows.len(),
                positions,
                heads: RecurrenceHeads {
                    key_heads,
                    heads_per_group,
                    key_features,
                    value_features,
                },
                kept: 1,
                rule: UpdateRule::Delta,
                masked: masked.to_vec(),
            },
            slots,
            rows: rows.to_vec(),
            parked: Vec::new(),
        }
    }

    fn parked(self, parked: &[usize]) -> Self {
        Self {
            parked: parked.to_vec(),
            ..self
        }
    }

    fn cells(&self) -> usize {
        self.case.heads.key_features * self.case.heads.value_features
    }

    /// The slots an unlisted row leaves alone, and a listed one rewrites.
    fn is_listed(&self, slot: usize) -> bool {
        self.rows.iter().any(|&(s, _)| s == slot)
    }
}

/// A decode step of a many-headed layer on shuffled slots with a fresh row, three positions on
/// shuffled slots, a group of three over primes, heads that are their own key heads, an odd value
/// width, masked positions, and more slots than any row holds, each walked by both update rules.
fn cases() -> Vec<SlotCase> {
    let delta = delta_cases();
    let additive = delta
        .iter()
        .cloned()
        .map(|slotted| SlotCase {
            case: Case {
                rule: UpdateRule::Additive,
                ..slotted.case.clone()
            },
            ..slotted
        })
        .collect::<Vec<_>>();
    delta.into_iter().chain(additive).collect()
}

fn delta_cases() -> Vec<SlotCase> {
    vec![
        SlotCase::new(1, (16, 3, 128, 128), 6, &[(4, 9), (1, 3), (2, 1)], &[]),
        SlotCase::new(1, (4, 3, 128, 128), 5, &[(3, 2), (1, 0), (4, 7)], &[]),
        SlotCase::new(1, (4, 3, 64, 64), 1, &[(0, 2)], &[]),
        SlotCase::new(1, (4, 3, 64, 64), 6, &[(5, 3)], &[]),
        SlotCase::new(
            3,
            (4, 3, 128, 128),
            7,
            &[(5, 4), (2, 0), (6, 1), (1, 9)],
            &[],
        ),
        SlotCase::new(3, (3, 3, 37, 53), 4, &[(3, 17), (1, 0)], &[]),
        SlotCase::new(2, (1, 2, 16, 37), 3, &[(2, 1), (1, 5)], &[]),
        SlotCase::new(4, (2, 2, 40, 36), 5, &[(4, 2), (2, 0), (3, 6)], &[2]),
        SlotCase::new(
            1,
            (2, 3, 100, 12),
            9,
            &[(8, 1), (3, 0), (5, 4), (1, 2)],
            &[],
        ),
    ]
}

/// The store and what the walk is checked against, built from a case.
struct Slotted {
    operands: Operands,
    /// The store's buffer as uploaded, gaps and the start included.
    before: Vec<f32>,
    /// The buffer and the view over it.
    raw: Handle,
    store: PlainTensor,
    /// Where each state's rows start in the buffer: `[slot][value_head]`.
    at: Vec<Vec<usize>>,
    /// The elements from one state row's start to the next's.
    row_pitch: usize,
    slots: PlainTensor,
    lengths: PlainTensor,
}

impl SlotCase {
    /// The case's operands, its store uploaded as `pitch` lays it out, and its tables.
    fn slotted(&self, client: &Client, pitch: Pitch) -> Slotted {
        let heads = self.case.value_heads();
        let cells = self.cells();
        let mut operands = self.case.operands();
        let (key_features, value_features) =
            (self.case.heads.key_features, self.case.heads.value_features);
        let row_pitch = value_features + pitch.row_gap;
        let head_pitch = key_features * row_pitch + pitch.head_gap;
        let slot_pitch = heads * head_pitch + pitch.slot_gap;
        let at: Vec<Vec<usize>> = (0..self.slots)
            .map(|slot| {
                (0..heads)
                    .map(|head| pitch.start + slot * slot_pitch + head * head_pitch)
                    .collect()
            })
            .collect();
        let mut before = vec![f32::NAN; pitch.start + self.slots * slot_pitch];
        // The states a row resumes from are the case's; a row that has walked nothing resumes from
        // zeros, and its slot holds NaN. A slot nobody lists holds values or NaN, alternately.
        let mut gathered = vec![0.0f32; operands.state.len()];
        for slot in 0..self.slots {
            for head in 0..heads {
                let cells_at = at[slot][head];
                let mut rows_of = |values: &dyn Fn(usize) -> f32| {
                    for cell in 0..cells {
                        let (row, column) = (cell / value_features, cell % value_features);
                        before[cells_at + row * row_pitch + column] = values(cell);
                    }
                };
                let held = self
                    .rows
                    .iter()
                    .position(|&(s, walked)| s == slot && walked > 0);
                match held {
                    Some(row) => {
                        let from = (row * heads + head) * cells;
                        rows_of(&|cell| operands.state[from + cell]);
                        gathered[from..from + cells]
                            .copy_from_slice(&operands.state[from..from + cells]);
                    }
                    None if !self.is_listed(slot) && slot % 2 == 0 => {
                        rows_of(&|cell| 0.3 - 0.001 * (cell + head + slot) as f32);
                    }
                    None => {}
                }
            }
        }
        operands.state = gathered;
        let raw = client.create(Bytes::from_elems(before.clone()));
        let handle = raw
            .clone()
            .offset_start((pitch.start * size_of::<f32>()) as u64);
        let store = PlainTensor::of_handle(TensorHandle::from_metadata(
            handle,
            Metadata::new(
                [self.slots, heads, key_features, value_features],
                [slot_pitch, head_pitch, row_pitch, 1],
            ),
            F32,
        ));
        let table = |entries: Vec<u32>| PlainTensor::of_u32s(client, &[entries.len()], &entries);
        Slotted {
            operands,
            before,
            raw,
            store,
            at,
            row_pitch,
            slots: table(self.rows.iter().map(|&(slot, _)| slot as u32).collect()),
            lengths: table(self.rows.iter().map(|&(_, walked)| walked as u32).collect()),
        }
    }

    /// The walk's inputs over `slotted`, the other operands the case's.
    fn inputs(
        &self,
        client: &Client,
        slotted: &Slotted,
        element: ElemType,
    ) -> RecurrenceWalkInputs {
        RecurrenceWalkInputs {
            state: RecurrenceWalkStateInputs::Slots(SlotStore::new(
                slotted.store.clone(),
                slotted.slots.clone(),
                slotted.lengths.clone(),
            )),
            ..self
                .case
                .inputs_laid_out(client, &slotted.operands, Layout::default(), element)
        }
    }

    /// Walks the case as `settings` places it, and checks it where the device admits it.
    fn walk_and_check(
        &self,
        hw: &Hardware,
        client: &Client,
        element: ElemType,
        settings: &CubeWalkRoutineSettings,
        pitch: Pitch,
    ) {
        let slotted = self.slotted(client, pitch);
        let inputs = self.inputs(client, &slotted, element);
        let what = format!("{settings} {pitch:?} {self:?}");
        if let Some(walked) = walk_or_skip(hw, inputs, settings, &what) {
            self.check(client, &slotted, walked, element, &what);
        }
    }

    /// The output and the store against the reference: each row not parked, its slot holds its
    /// state after the walk, and everything else in the buffer is bit for bit what was uploaded.
    fn check(
        &self,
        client: &Client,
        slotted: &Slotted,
        walked: RecurrenceWalkOutputs,
        element: ElemType,
        what: &str,
    ) {
        let parked = &self.parked;
        assert!(
            walked.kept.is_none(),
            "{what}: a store keeps its states in place"
        );
        let heads = self.case.value_heads();
        let cells = self.cells();
        let value_features = self.case.heads.value_features;
        let place = |at: usize, cell: usize| {
            at + cell / value_features * slotted.row_pitch + cell % value_features
        };
        let (out, kept) = self.case.reference(&slotted.operands);
        let got = read(client, walked.out.handle.clone(), element);
        let per_row = out.len() / self.rows.len();
        for row in (0..self.rows.len()).filter(|row| !parked.contains(row)) {
            let rounded: Vec<f64> = out[row * per_row..(row + 1) * per_row]
                .iter()
                .map(|&x| round_to(x as f32, element) as f64)
                .collect();
            assert_close_within(
                got[row * per_row..(row + 1) * per_row].to_vec(),
                &rounded,
                &format!("{what} {element:?}: row {row}'s output"),
                output_tolerance(element),
            );
        }
        let after = read(client, slotted.raw.clone(), F32);
        let mut claimed = vec![false; after.len()];
        for (row, &(slot, _)) in self.rows.iter().enumerate() {
            if parked.contains(&row) {
                continue;
            }
            for head in 0..heads {
                let from = (row * heads + head) * cells;
                let to = slotted.at[slot][head];
                assert_close(
                    (0..cells).map(|cell| after[place(to, cell)]).collect(),
                    &kept[from..from + cells],
                    &format!("{what}: row {row} head {head} of slot {slot}"),
                );
            }
        }
        for slot in 0..self.slots {
            for head in 0..heads {
                let to = slotted.at[slot][head];
                for cell in 0..cells {
                    claimed[place(to, cell)] = self.is_listed(slot);
                }
            }
        }
        for (i, claimed) in claimed.iter().enumerate() {
            if !claimed {
                assert_eq!(
                    after[i].to_bits(),
                    slotted.before[i].to_bits(),
                    "{what}: element {i} outside the listed slots changed"
                );
            }
        }
    }
}

/// Every setting the cube routine lists, and the floor, walks every case in the slots of a store
/// at every element the device runs, and leaves what no row holds as it was.
#[test]
fn the_slots_walk_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let settings: Vec<_> = CubeWalkRoutineSettings::listed()
        .into_iter()
        .chain([CubeWalkRoutineSettings::FLOOR])
        .collect();
    for element in elements(&client) {
        for settings in &settings {
            for case in cases() {
                case.walk_and_check(&hw, &client, element, settings, Pitch::default());
            }
        }
    }
}

/// A store with a gap between its rows, its heads and its slots, starting off its buffer's start:
/// the walk reads and writes the states where they lie, at every width the layout leaves it, and
/// leaves the gaps alone.
#[test]
fn the_slots_walk_reads_a_strided_store_where_it_lies() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for element in elements(&client) {
        for pitch in STRIDED_PITCHES {
            for settings in [
                CubeWalkRoutineSettings::DEFAULT,
                CubeWalkRoutineSettings::FLOOR,
            ] {
                for case in cases() {
                    case.walk_and_check(&hw, &client, element, &settings, pitch);
                }
            }
        }
    }
}

/// Rows parked on the null slot, slot zero, race on it, and no row that holds a slot of its own
/// reads or writes anything but its own.
#[test]
fn rows_parked_on_the_null_slot_leave_the_others_whole() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = SlotCase::new(
        1,
        (4, 3, 128, 128),
        6,
        &[(0, 0), (3, 4), (0, 2), (5, 1), (0, 0)],
        &[],
    )
    .parked(&[0, 2, 4]);
    for settings in [
        CubeWalkRoutineSettings::DEFAULT,
        CubeWalkRoutineSettings::FLOOR,
    ] {
        case.walk_and_check(&hw, &client, F32, &settings, Pitch::default());
    }
}

/// The autotune runs the walk and leaves the call's store advanced by exactly one walk, its
/// rounds writing stores of their own, whether the call's store lies dense or strided: the key a
/// dense store tuned is met again by the strided ones.
#[test]
fn the_tuned_walk_advances_the_store_once() {
    let device = cubecl::test_device();
    let client = device.client();
    let hw = Hardware::from_client(&client);
    for element in elements(&client) {
        for case in cases() {
            for pitch in [Pitch::default(), STRIDED_PITCHES[0], STRIDED_PITCHES[2]] {
                let slotted = case.slotted(&client, pitch);
                let inputs = case.inputs(&client, &slotted, element);
                let what = format!("the table {pitch:?} {case:?}");
                if let Some(walked) = tuned_or_skip(&hw, &device, inputs, &what) {
                    case.check(&client, &slotted, walked, element, &what);
                }
            }
        }
    }
}
