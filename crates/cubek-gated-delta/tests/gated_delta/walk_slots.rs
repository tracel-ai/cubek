//! The gated delta walk in a store's slots against the host reference: each row's state is read
//! from, and written back to, the slot its table names.
//!
//! The store holds NaN wherever the walk must not read it, in the slots no row reads, in the slot
//! of a row that has walked no position, and in the gaps between its slots and heads; the slots no
//! row holds and the gaps must come back bit for bit.

use super::harness::{F16, F32, elements, read, round_to};
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
use cubek_gated_delta::kernels::walk::{
    CubeGatedDeltaWalkRoutine, CubeGatedDeltaWalkRoutineSettings, GatedDeltaWalkInputs,
    GatedDeltaWalkOutputs, GatedDeltaWalkStateInputs,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::gated_delta::GatedDeltaHeads;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// How a store lies in its buffer: the gap after each head's state, the gap after each slot's
/// heads, and the elements its view starts past its buffer's start.
#[derive(Debug, Clone, Copy, Default)]
struct Pitch {
    head_gap: usize,
    slot_gap: usize,
    start: usize,
}

/// A walk of `rows`, each a `(slot, positions walked already)`, over a store of `slots`.
#[derive(Debug, Clone)]
struct SlotCase {
    case: Case,
    slots: usize,
    rows: Vec<(usize, usize)>,
}

impl SlotCase {
    fn new(
        positions: usize,
        heads: (usize, usize, usize, usize),
        slots: usize,
        rows: &[(usize, usize)],
        masked: &[usize],
    ) -> Self {
        let (key_heads, group, key_features, value_features) = heads;
        Self {
            case: Case {
                batch: rows.len(),
                positions,
                heads: GatedDeltaHeads {
                    key_heads,
                    group,
                    key_features,
                    value_features,
                },
                kept: 1,
                masked: masked.to_vec(),
            },
            slots,
            rows: rows.to_vec(),
        }
    }

    fn cells(&self) -> usize {
        self.case.heads.key_features * self.case.heads.value_features
    }

    /// The slots a row reads: those of rows that have walked a position.
    fn is_read(&self, slot: usize) -> bool {
        self.rows.iter().any(|&(s, walked)| s == slot && walked > 0)
    }

    /// The slots an unlisted row leaves alone, and a listed one rewrites.
    fn is_listed(&self, slot: usize) -> bool {
        self.rows.iter().any(|&(s, _)| s == slot)
    }
}

/// A decode step of a many-headed layer on shuffled slots with a fresh row, three positions on
/// shuffled slots, a group of three over primes, heads that are their own key heads, an odd value
/// width, masked positions, and more slots than any row holds.
fn cases() -> Vec<SlotCase> {
    vec![
        SlotCase::new(1, (16, 3, 128, 128), 6, &[(4, 9), (1, 3), (2, 1)], &[]),
        SlotCase::new(1, (4, 3, 128, 128), 5, &[(3, 2), (1, 0), (4, 7)], &[]),
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
    /// Where each state's `cells` start in the buffer: `[slot][value_head]`.
    at: Vec<Vec<usize>>,
    slots: PlainTensor,
    positions: PlainTensor,
}

impl SlotCase {
    /// The case's operands, its store uploaded as `pitch` lays it out, and its tables.
    fn slotted(&self, client: &Client, pitch: Pitch) -> Slotted {
        let heads = self.case.value_heads();
        let cells = self.cells();
        let mut operands = self.case.operands();
        let head_pitch = cells + pitch.head_gap;
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
                let held = self
                    .rows
                    .iter()
                    .position(|&(s, walked)| s == slot && walked > 0);
                match held {
                    Some(row) => {
                        let from = (row * heads + head) * cells;
                        before[cells_at..cells_at + cells]
                            .copy_from_slice(&operands.state[from..from + cells]);
                        gathered[from..from + cells]
                            .copy_from_slice(&operands.state[from..from + cells]);
                    }
                    None if !self.is_listed(slot) && slot % 2 == 0 => {
                        for (cell, value) in
                            before[cells_at..cells_at + cells].iter_mut().enumerate()
                        {
                            *value = 0.3 - 0.001 * (cell + head + slot) as f32;
                        }
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
                [self.slots, heads, 1, cells],
                [slot_pitch, head_pitch, head_pitch, 1],
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
            slots: table(self.rows.iter().map(|&(slot, _)| slot as u32).collect()),
            positions: table(self.rows.iter().map(|&(_, walked)| walked as u32).collect()),
        }
    }

    /// The walk's inputs over `slotted`, the other operands the case's.
    fn inputs(
        &self,
        client: &Client,
        slotted: &Slotted,
        element: ElemType,
    ) -> GatedDeltaWalkInputs {
        GatedDeltaWalkInputs {
            state: GatedDeltaWalkStateInputs::Slots {
                store: slotted.store.clone(),
                slots: slotted.slots.clone(),
                positions: slotted.positions.clone(),
            },
            ..self
                .case
                .inputs_laid_out(client, &slotted.operands, Layout::default(), element)
        }
    }

    /// The output and the store against the reference: each row not `parked`, its slot holds its state after the walk, and everything else in the buffer is bit for bit
    /// what was uploaded.
    fn check(
        &self,
        client: &Client,
        slotted: &Slotted,
        walked: GatedDeltaWalkOutputs,
        element: ElemType,
        parked: &[usize],
        what: &str,
    ) {
        assert!(
            walked.kept.is_none(),
            "{what}: a store keeps its states in place"
        );
        let heads = self.case.value_heads();
        let cells = self.cells();
        let (out, kept) = self.case.reference(&slotted.operands);
        // A parked row races other rows on its slot, so its output is whatever it read.
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
                &format!("{what} {element:?} {self:?}: row {row}'s output"),
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
                    after[to..to + cells].to_vec(),
                    &kept[from..from + cells],
                    &format!("{what} {self:?}: row {row} head {head} of slot {slot}"),
                );
            }
        }
        for slot in 0..self.slots {
            for head in 0..heads {
                let to = slotted.at[slot][head];
                claimed[to..to + cells].fill(self.is_listed(slot));
            }
        }
        for (i, claimed) in claimed.iter().enumerate() {
            if !claimed {
                assert_eq!(
                    after[i].to_bits(),
                    slotted.before[i].to_bits(),
                    "{what} {self:?}: element {i} outside the listed slots changed"
                );
            }
        }
    }
}

/// Every setting the cube routine lists walks every case in the slots of a store at every element
/// the device runs, and leaves what no row holds as it was.
#[test]
fn the_slots_walk_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for element in elements(&client) {
        for settings in CubeGatedDeltaWalkRoutine.settings(&hw) {
            for case in cases() {
                let slotted = case.slotted(&client, Pitch::default());
                let inputs = case.inputs(&client, &slotted, element);
                let what = format!("{settings} {case:?}");
                if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                    case.check(
                        &client,
                        &slotted,
                        walked,
                        element,
                        &[],
                        &settings.to_string(),
                    );
                }
            }
        }
    }
}

/// The floor, which serves every admitted problem, walks every case.
#[test]
fn the_slots_floor_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let settings = CubeGatedDeltaWalkRoutineSettings::FLOOR;
    for case in cases() {
        let slotted = case.slotted(&client, Pitch::default());
        let inputs = case.inputs(&client, &slotted, F32);
        let what = format!("{settings} {case:?}");
        if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
            case.check(&client, &slotted, walked, F32, &[], "the floor");
        }
    }
}

/// A row that has walked no position starts from zeros: its slot holds NaN, which reaches no sum.
#[test]
fn a_fresh_row_reads_nothing_of_its_slot() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = SlotCase::new(1, (4, 3, 128, 128), 4, &[(2, 0), (3, 0), (1, 5)], &[]);
    for settings in [
        CubeGatedDeltaWalkRoutineSettings::DEFAULT,
        CubeGatedDeltaWalkRoutineSettings::FLOOR,
    ] {
        let slotted = case.slotted(&client, Pitch::default());
        assert!(slotted.before[slotted.at[2][0]].is_nan() && !case.is_read(2));
        let inputs = case.inputs(&client, &slotted, F32);
        let what = format!("{settings} {case:?}");
        if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
            case.check(&client, &slotted, walked, F32, &[], "a fresh row");
        }
    }
}

/// A walk of three positions reads each row's state once, at the start, and writes it once, at
/// the end.
#[test]
fn three_positions_in_slots_advance_each_state_once() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let case = SlotCase::new(
        3,
        (4, 3, 64, 64),
        6,
        &[(5, 4), (2, 0), (4, 1), (1, 2)],
        &[1],
    );
    for element in elements(&client) {
        for settings in CubeGatedDeltaWalkRoutine.settings(&hw) {
            let slotted = case.slotted(&client, Pitch::default());
            let inputs = case.inputs(&client, &slotted, element);
            let what = format!("{settings} {case:?}");
            if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                case.check(
                    &client,
                    &slotted,
                    walked,
                    element,
                    &[],
                    &settings.to_string(),
                );
            }
        }
    }
}

/// A store with a gap between its heads and between its slots, starting off its buffer's start:
/// the walk reads and writes the states where they lie, at every width the layout leaves it, and
/// leaves the gaps alone.
#[test]
fn the_slots_walk_reads_a_strided_store_where_it_lies() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let pitches = [
        Pitch {
            slot_gap: 3,
            ..Pitch::default()
        },
        Pitch {
            head_gap: 1,
            start: 2,
            ..Pitch::default()
        },
        Pitch {
            head_gap: 4,
            slot_gap: 8,
            start: 4,
        },
        Pitch {
            slot_gap: 5,
            start: 1,
            ..Pitch::default()
        },
    ];
    for pitch in pitches {
        for settings in [
            CubeGatedDeltaWalkRoutineSettings::DEFAULT,
            CubeGatedDeltaWalkRoutineSettings::FLOOR,
        ] {
            for case in cases() {
                let slotted = case.slotted(&client, pitch);
                let inputs = case.inputs(&client, &slotted, F16);
                let what = format!("{settings} {case:?}");
                if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                    case.check(
                        &client,
                        &slotted,
                        walked,
                        F16,
                        &[],
                        &format!("{pitch:?} {settings}"),
                    );
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
    );
    let parked = [0, 2, 4];
    for settings in [
        CubeGatedDeltaWalkRoutineSettings::DEFAULT,
        CubeGatedDeltaWalkRoutineSettings::FLOOR,
    ] {
        let slotted = case.slotted(&client, Pitch::default());
        let inputs = case.inputs(&client, &slotted, F32);
        let what = format!("{settings} {case:?}");
        if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
            case.check(&client, &slotted, walked, F32, &parked, "parked rows");
        }
    }
}

/// The autotune runs the walk and leaves the call's store advanced by exactly one walk: its rounds
/// write stores of their own.
#[test]
fn the_tuned_walk_advances_the_store_once() {
    let device = cubecl::test_device();
    let client = device.client();
    let hw = Hardware::from_client(&client);
    for element in elements(&client) {
        for case in cases() {
            let slotted = case.slotted(&client, Pitch::default());
            let inputs = case.inputs(&client, &slotted, element);
            if let Some(walked) = tuned_or_skip(&hw, &device, inputs, "the table") {
                case.check(&client, &slotted, walked, element, &[], "the table");
            }
        }
    }
}

/// A key tuned over a dense store is met again by calls whose store lies strided: the winner
/// serves them, and each call advances its own store once.
#[test]
fn the_table_serves_a_strided_store_after_a_dense_one() {
    let device = cubecl::test_device();
    let client = device.client();
    let hw = Hardware::from_client(&client);
    let case = SlotCase::new(1, (4, 3, 128, 128), 6, &[(4, 9), (1, 3), (2, 1)], &[]);
    for pitch in [
        Pitch::default(),
        Pitch {
            slot_gap: 3,
            start: 1,
            ..Pitch::default()
        },
        Pitch {
            head_gap: 4,
            ..Pitch::default()
        },
    ] {
        let slotted = case.slotted(&client, pitch);
        let inputs = case.inputs(&client, &slotted, F32);
        if let Some(walked) = tuned_or_skip(&hw, &device, inputs, "the table") {
            case.check(&client, &slotted, walked, F32, &[], &format!("{pitch:?}"));
        }
    }
}
