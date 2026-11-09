//! The routines of the walk's two update rules against a host reference, at shapes chosen to be
//! awkward.
//!
//! The reference walks the recurrence in `f64` from the operands' values. The shapes hold a decode
//! step of a many-headed layer and several sequences of it, a prompt keeping a ring of states, a
//! prime count of positions with every state kept, heads that are their own key heads, a head of an
//! odd value width read a column at a time, key rows no unit takes whole, a state resumed from
//! values other than zero, and masked positions. The keys and queries are normalized, as the layer
//! hands them, so the state stays bounded over a long walk.

use super::harness::{F16, F32, elements, read, round_to};
use core::num::NonZeroUsize;
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::metadata::Metadata;
use cubek_linear_recurrence::kernels::walk::{
    CubeAdditiveWalkRoutine, CubeDeltaWalkRoutine, CubeWalkRoutineSettings, RecurrenceWalkInputs,
    RecurrenceWalkOutputs, RecurrenceWalkProblem, RecurrenceWalkRoutine, RecurrenceWalkStateInputs,
    RecurrenceWalkTensors, UpdateRuleInputs, launch_tuned,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_microkernel::linear_recurrence::walk::UpdateRule;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;

/// A walk's shape, and the positions whose decay and strength are zero.
#[derive(Debug, Clone)]
pub struct Case {
    pub batch: usize,
    pub positions: usize,
    pub heads: RecurrenceHeads,
    pub kept: usize,
    pub rule: UpdateRule,
    pub masked: Vec<usize>,
}

/// A decode step of a many-headed layer and eight sequences of it, a prompt of it keeping a ring of
/// four, a prime walk keeping every state, heads that are their own key heads, an odd value width,
/// key rows of three lines, key rows that no unit takes whole, and masked positions; then the
/// shapes of a layer with many value heads a key head, 128 x 80, 128 x 64, 256 x 64 and an odd 16 x
/// 37 over groups of 16, 8, 32 and 3 heads. Every case is walked by both update rules.
pub fn cases() -> Vec<Case> {
    let delta = delta_cases();
    let additive = delta
        .iter()
        .cloned()
        .map(|case| Case {
            rule: UpdateRule::Additive,
            ..case
        })
        .collect::<Vec<_>>();
    delta.into_iter().chain(additive).collect()
}

fn delta_cases() -> Vec<Case> {
    let case = |batch,
                positions,
                key_heads,
                heads_per_group,
                key_features,
                value_features,
                kept,
                masked: &[usize]| Case {
        batch,
        positions,
        heads: RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        },
        kept,
        rule: UpdateRule::Delta,
        masked: masked.to_vec(),
    };
    vec![
        case(1, 1, 16, 3, 128, 128, 1, &[]),
        case(8, 1, 16, 3, 128, 128, 1, &[]),
        case(1, 67, 4, 3, 128, 128, 4, &[]),
        case(2, 13, 2, 2, 64, 64, 13, &[]),
        case(3, 5, 4, 1, 32, 32, 2, &[]),
        case(1, 9, 1, 2, 16, 37, 1, &[]),
        case(2, 6, 2, 3, 12, 8, 3, &[]),
        case(2, 8, 2, 2, 40, 36, 3, &[2, 5, 7]),
        case(2, 5, 2, 3, 37, 53, 5, &[0]),
        case(1, 6, 2, 2, 100, 12, 2, &[1, 4]),
        case(1, 5, 2, 16, 128, 80, 3, &[1]),
        case(2, 4, 2, 8, 128, 64, 1, &[]),
        case(1, 3, 1, 32, 256, 64, 2, &[2]),
        case(2, 6, 1, 3, 16, 37, 4, &[0, 3]),
    ]
}

/// How operands lie in their buffers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Layout {
    /// Elements each row of an operand is padded by.
    pub pad: usize,
    /// Elements every buffer starts past its own start.
    pub offset: usize,
    /// A stride on each axis of one element that no index can reach, as a decode step's swapped
    /// view leaves.
    pub loose_units: bool,
    /// The log decays in one plane of a buffer and the strengths in the next, which the rates
    /// view as `[.., 2]` along a strided axis.
    pub rate_planes: bool,
    /// The skip on every other element of its buffer.
    pub strided_skip: bool,
}

impl Layout {
    /// `values`, row-major in `shape`, uploaded as this layout lays them out, the gaps and the
    /// start holding NaN: an element the walk reads that it should not poisons its result.
    fn upload(&self, client: &Client, shape: &[usize], values: &[f32]) -> PlainTensor {
        let rank = shape.len();
        let mut strides = vec![1; rank];
        let mut row = shape[rank - 1] + self.pad;
        for axis in (0..rank - 1).rev() {
            strides[axis] = row;
            row *= shape[axis];
        }
        if self.loose_units {
            for axis in 0..rank - 1 {
                if shape[axis] == 1 {
                    strides[axis] = strides[axis] * 3 + 5;
                }
            }
        }
        let len = self.offset
            + 1
            + shape
                .iter()
                .zip(&strides)
                .map(|(extent, stride)| (extent - 1) * stride)
                .sum::<usize>();
        let mut buffer = vec![f32::NAN; len];
        for (flat, &value) in values.iter().enumerate() {
            let (mut rest, mut at) = (flat, self.offset);
            for axis in (0..rank).rev() {
                at += rest % shape[axis] * strides[axis];
                rest /= shape[axis];
            }
            buffer[at] = value;
        }
        let handle = client
            .create(Bytes::from_elems(buffer))
            .offset_start((self.offset * size_of::<f32>()) as u64);
        PlainTensor::of_handle(TensorHandle::from_metadata(
            handle,
            Metadata::new(shape.to_vec(), strides),
            F32,
        ))
    }

    /// The rates `[b, t, h, 2]` of `decay` and `strength`: interleaved, or one plane each when
    /// `rate_planes` is set, the gaps and the start holding NaN as `upload` leaves them.
    fn upload_rates(
        &self,
        client: &Client,
        shape: [usize; 4],
        decay: &[f32],
        strength: &[f32],
    ) -> PlainTensor {
        if !self.rate_planes {
            let rates: Vec<f32> = decay
                .iter()
                .zip(strength)
                .flat_map(|(&decay, &strength)| [decay, strength])
                .collect();
            return self.upload(client, &shape, &rates);
        }
        let plane = decay.len();
        let mut buffer = vec![f32::NAN; self.offset];
        buffer.extend_from_slice(decay);
        buffer.extend_from_slice(strength);
        let [_, t, h, _] = shape;
        let strides = vec![t * h, h, 1, plane];
        let handle = client
            .create(Bytes::from_elems(buffer))
            .offset_start((self.offset * size_of::<f32>()) as u64);
        PlainTensor::of_handle(TensorHandle::from_metadata(
            handle,
            Metadata::new(shape.to_vec(), strides),
            F32,
        ))
    }

    /// The skip `[h]`: contiguous, or on every other element of its buffer when `strided_skip` is
    /// set.
    fn upload_skip(&self, client: &Client, skip: &[f32]) -> PlainTensor {
        if !self.strided_skip {
            return self.upload(client, &[skip.len()], skip);
        }
        let mut buffer = vec![f32::NAN; 2 * skip.len()];
        for (at, &value) in skip.iter().enumerate() {
            buffer[2 * at] = value;
        }
        PlainTensor::of_handle(TensorHandle::from_metadata(
            client.create(Bytes::from_elems(buffer)),
            Metadata::new(vec![skip.len()], vec![2]),
            F32,
        ))
    }
}

/// The operands of a case, as `f32` host values.
pub struct Operands {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub decay: Vec<f32>,
    pub strength: Vec<f32>,
    pub state: Vec<f32>,
    /// The additive rule's weight of each value head's value in its output.
    pub skip: Vec<f32>,
}

impl Case {
    pub fn value_heads(&self) -> usize {
        self.heads.value_heads()
    }

    /// Deterministic operands: unit-norm queries and keys, the queries scaled as the layer scales
    /// them, values of either sign, a decay in `(-1, 0)`, a strength in `(0, 1)`, a state of small
    /// values; a masked position decays by nothing and writes nothing.
    pub fn operands(&self) -> Operands {
        let wave = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32 * 0.7311 + seed).sin() * 43758.547).fract() - 0.5)
                .collect()
        };
        let key_features = self.heads.key_features;
        let normalized = |raw: Vec<f32>, scale: f32| -> Vec<f32> {
            raw.chunks(key_features)
                .flat_map(|row| {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    row.iter()
                        .map(move |x| x / norm * scale)
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let keyed = self.batch * self.positions * self.heads.key_heads * key_features;
        let scalars = self.batch * self.positions * self.value_heads();
        let unmasked = |values: Vec<f32>| -> Vec<f32> {
            values
                .into_iter()
                .enumerate()
                .map(|(i, value)| {
                    let position = i / self.value_heads() % self.positions;
                    if self.masked.contains(&position) {
                        0.0
                    } else {
                        value
                    }
                })
                .collect()
        };
        Operands {
            query: normalized(wave(keyed, 1.0), 1.0 / (key_features as f32).sqrt()),
            key: normalized(wave(keyed, 2.0), 1.0),
            value: wave(scalars * self.heads.value_features, 3.0),
            decay: unmasked(
                wave(scalars, 4.0)
                    .iter()
                    .map(|x| -(x + 0.5) * 0.9 - 0.05)
                    .collect(),
            ),
            strength: unmasked(wave(scalars, 5.0).iter().map(|x| x + 0.5).collect()),
            skip: wave(self.value_heads(), 7.0)
                .iter()
                .map(|x| x + 1.0)
                .collect(),
            state: wave(
                self.batch * self.value_heads() * key_features * self.heads.value_features,
                6.0,
            )
            .iter()
            .map(|x| x * 0.2)
            .collect(),
        }
    }

    pub fn inputs(
        &self,
        client: &Client,
        operands: &Operands,
        element: ElemType,
    ) -> RecurrenceWalkInputs {
        self.inputs_laid_out(client, operands, Layout::default(), element)
    }

    /// The operands uploaded as `layout` lays them out in their buffers.
    pub fn inputs_laid_out(
        &self,
        client: &Client,
        operands: &Operands,
        layout: Layout,
        element: ElemType,
    ) -> RecurrenceWalkInputs {
        let heads = &self.heads;
        let (b, t, kh, h) = (
            self.batch,
            self.positions,
            heads.key_heads,
            self.value_heads(),
        );
        let up = |shape: &[usize], values: &[f32]| layout.upload(client, shape, values);
        RecurrenceWalkInputs {
            client: client.clone(),
            query: up(&[b, t, kh, heads.key_features], &operands.query),
            key: up(&[b, t, kh, heads.key_features], &operands.key),
            value: up(&[b, t, h, heads.value_features], &operands.value),
            rates: layout.upload_rates(client, [b, t, h, 2], &operands.decay, &operands.strength),
            state: RecurrenceWalkStateInputs::Owned {
                state: up(
                    &[b, h, heads.key_features, heads.value_features],
                    &operands.state,
                ),
                kept: NonZeroUsize::new(self.kept).expect("a state is kept"),
            },
            update: match self.rule {
                UpdateRule::Delta => UpdateRuleInputs::Delta,
                UpdateRule::Additive => UpdateRuleInputs::Additive {
                    skip: layout.upload_skip(client, &operands.skip),
                },
            },
            output_element: element,
        }
    }

    /// The recurrence of the case's rule walked in `f64`: what each position reads, and the kept
    /// states.
    pub fn reference(&self, o: &Operands) -> (Vec<f64>, Vec<f64>) {
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features: kd,
            value_features: vd,
        } = self.heads;
        let (t_len, h_len) = (self.positions, self.value_heads());
        let mut out = vec![0.0f64; self.batch * t_len * h_len * vd];
        let mut kept = vec![0.0f64; self.kept * self.batch * h_len * kd * vd];
        let first = t_len - self.kept;
        for b in 0..self.batch {
            for h in 0..h_len {
                let kh = h / heads_per_group;
                let at = ((b * h_len + h) * kd) * vd;
                let mut s: Vec<f64> = o.state[at..at + kd * vd]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                for t in 0..t_len {
                    let keyed = ((b * t_len + t) * key_heads + kh) * kd;
                    let k = &o.key[keyed..keyed + kd];
                    let q = &o.query[keyed..keyed + kd];
                    let scalar = (b * t_len + t) * h_len + h;
                    let (g, beta) = ((o.decay[scalar] as f64).exp(), o.strength[scalar] as f64);
                    let v = &o.value[scalar * vd..(scalar + 1) * vd];
                    s.iter_mut().for_each(|x| *x *= g);
                    for j in 0..vd {
                        let recalled: f64 = match self.rule {
                            UpdateRule::Delta => (0..kd).map(|r| s[r * vd + j] * k[r] as f64).sum(),
                            UpdateRule::Additive => 0.0,
                        };
                        let delta = (v[j] as f64 - recalled) * beta;
                        for r in 0..kd {
                            s[r * vd + j] += k[r] as f64 * delta;
                        }
                        let read: f64 = (0..kd).map(|r| s[r * vd + j] * q[r] as f64).sum();
                        out[scalar * vd + j] = match self.rule {
                            UpdateRule::Delta => read,
                            UpdateRule::Additive => read + o.skip[h] as f64 * v[j] as f64,
                        };
                    }
                    if t >= first {
                        let slot = (((t - first) * self.batch + b) * h_len + h) * kd * vd;
                        kept[slot..slot + kd * vd].copy_from_slice(&s);
                    }
                }
            }
        }
        (out, kept)
    }

    /// `walked` against the reference, the output rounded to `element` as the kernel rounds it.
    pub fn check(
        &self,
        client: &Client,
        o: &Operands,
        walked: RecurrenceWalkOutputs,
        element: ElemType,
        what: &str,
    ) {
        let (out, kept) = self.reference(o);
        let out: Vec<f64> = out
            .iter()
            .map(|&x| round_to(x as f32, element) as f64)
            .collect();
        assert_close_within(
            read(client, walked.out.handle.clone(), element),
            &out,
            &format!("{what} {element:?} {self:?}: the output"),
            output_tolerance(element),
        );
        let walked_kept = walked.kept.expect("a walk over owned states keeps them");
        assert_close(
            read(client, walked_kept.handle.clone(), F32),
            &kept,
            &format!("{what} {self:?}: the kept states"),
        );
    }
}

/// How far the output may fall from the reference's rounding: a sum an ulp of the element apart
/// rounds to neighbouring values.
pub fn output_tolerance(element: ElemType) -> f64 {
    match element {
        F16 => 1.5e-3,
        F32 => 1e-4,
        _ => 1.5e-2,
    }
}

/// The problem `inputs` state, which a device admits or declines.
fn problem_of(inputs: &RecurrenceWalkInputs) -> RecurrenceWalkProblem {
    RecurrenceWalkTensors::new(inputs.clone())
        .expect("the harness hands elements the device runs")
        .problem()
        .clone()
}

/// The walk of `inputs` at `settings`, or `None` where this device declines the problem by the
/// refusal admission names: a head wider than its cube holds, or too few bindings. Any other
/// refusal fails the test.
pub fn walk_or_skip(
    hw: &Hardware,
    inputs: RecurrenceWalkInputs,
    settings: &CubeWalkRoutineSettings,
    what: &str,
) -> Option<RecurrenceWalkOutputs> {
    let problem = problem_of(&inputs);
    let launched = match problem.rule {
        UpdateRule::Delta => CubeDeltaWalkRoutine.launch_on(hw, inputs, settings),
        UpdateRule::Additive => CubeAdditiveWalkRoutine.launch_on(hw, inputs, settings),
    };
    match launched {
        Ok(walked) => Some(walked),
        Err(why) => {
            eprintln!("skipped, the device declines {what}: {why:?}");
            assert_eq!(problem.admit(hw), Err(why), "{what}");
            None
        }
    }
}

/// The walk of `inputs` through the table, or `None` where this device declines the problem.
pub fn tuned_or_skip(
    hw: &Hardware,
    device: &cubecl::Device,
    inputs: RecurrenceWalkInputs,
    what: &str,
) -> Option<RecurrenceWalkOutputs> {
    if let Err(why) = problem_of(&inputs).admit(hw) {
        eprintln!("skipped, the device declines {what}: {why:?}");
        return None;
    }
    Some(launch_tuned(device, inputs, GroupPriority::DEFAULT))
}

/// Every value within an `f32` sum's rounding of the reference's, relative to the largest magnitude
/// it holds.
pub fn assert_close(got: Vec<f32>, want: &[f64], what: &str) {
    assert_close_within(got, want, what, 1e-4);
}

/// Every value within `tolerance` of the reference's, relative to the largest magnitude it holds.
pub fn assert_close_within(got: Vec<f32>, want: &[f64], what: &str, tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{what}: the element count");
    let scale = want.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (*got as f64 - want).abs() <= tolerance * scale,
            "{what}: element {i} is {got}, the reference {want}"
        );
    }
}

/// Every setting the cube routine lists runs every case at every element the device runs, and
/// matches the reference.
#[test]
fn the_cube_walk_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for element in elements(&client) {
        for settings in CubeWalkRoutineSettings::listed() {
            for case in cases() {
                let operands = case.operands();
                let inputs = case.inputs(&client, &operands, element);
                let what = format!("{settings} {case:?}");
                if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                    case.check(&client, &operands, walked, element, &settings.to_string());
                }
            }
        }
    }
}

/// The floor, which serves every admitted problem, matches the reference.
#[test]
fn the_floor_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let settings = CubeWalkRoutineSettings::FLOOR;
    for element in elements(&client) {
        for case in cases() {
            let operands = case.operands();
            let inputs = case.inputs(&client, &operands, element);
            let what = format!("the floor {case:?}");
            if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                case.check(&client, &operands, walked, element, "the floor");
            }
        }
    }
}

/// Every case through the table, which runs whichever setting its round elected.
#[test]
fn the_table_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let device = cubecl::test_device();
    for element in elements(&client) {
        for case in cases() {
            let operands = case.operands();
            let inputs = case.inputs(&client, &operands, element);
            let what = format!("the table {case:?}");
            if let Some(walked) = tuned_or_skip(&hw, &device, inputs, &what) {
                case.check(&client, &operands, walked, element, "the table");
            }
        }
    }
}

/// A walk whose every position is masked leaves the state it resumed from, at every kept position,
/// under either update rule.
#[test]
fn a_masked_walk_passes_the_state_through() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    for rule in [UpdateRule::Delta, UpdateRule::Additive] {
        for settings in CubeWalkRoutineSettings::listed() {
            let case = Case {
                batch: 2,
                positions: 4,
                heads: RecurrenceHeads {
                    key_heads: 2,
                    heads_per_group: 2,
                    key_features: 40,
                    value_features: 36,
                },
                kept: 3,
                rule,
                masked: vec![0, 1, 2, 3],
            };
            let operands = case.operands();
            let inputs = case.inputs(&client, &operands, F32);
            let what = format!("{rule:?} {settings}");
            let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) else {
                continue;
            };
            let kept = read(&client, walked.kept.unwrap().handle.clone(), F32);
            let state: Vec<f64> = operands.state.iter().map(|&x| x as f64).collect();
            for slab in kept.chunks(state.len()) {
                assert_close(slab.to_vec(), &state, &format!("{what}: the kept state"));
            }
        }
    }
}

/// Operands that lie padded, off a whole read of their buffers, with the strides a decode step's
/// swapped view leaves on its extent-one axes, the rates as two planes and the skip on every other
/// element: the walk reads them where they lie, at every width the layouts leave it: 4, 2 and 1.
#[test]
fn the_walk_reads_operands_where_they_lie() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let layouts = [
        Layout {
            pad: 3,
            ..Layout::default()
        },
        Layout {
            pad: 2,
            ..Layout::default()
        },
        Layout {
            offset: 1,
            ..Layout::default()
        },
        Layout {
            pad: 1,
            offset: 3,
            ..Layout::default()
        },
        Layout {
            loose_units: true,
            ..Layout::default()
        },
        Layout {
            rate_planes: true,
            ..Layout::default()
        },
        Layout {
            rate_planes: true,
            offset: 1,
            ..Layout::default()
        },
        Layout {
            strided_skip: true,
            ..Layout::default()
        },
    ];
    for element in elements(&client) {
        for layout in layouts {
            for settings in [
                CubeWalkRoutineSettings::DEFAULT,
                CubeWalkRoutineSettings::FLOOR,
            ] {
                for case in cases() {
                    let operands = case.operands();
                    let inputs = case.inputs_laid_out(&client, &operands, layout, element);
                    let what = format!("{layout:?} {settings} {case:?}");
                    if let Some(walked) = walk_or_skip(&hw, inputs, &settings, &what) {
                        let what = format!("{layout:?} {settings}");
                        case.check(&client, &operands, walked, element, &what);
                    }
                }
            }
        }
    }
}

/// A key tuned on an aligned call is met again by calls that bind narrower: the winner serves them,
/// in the table of either update rule.
#[test]
fn the_table_serves_a_narrower_call_after_an_aligned_one() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let device = cubecl::test_device();
    for rule in [UpdateRule::Delta, UpdateRule::Additive] {
        let case = Case {
            batch: 1,
            positions: 8,
            heads: RecurrenceHeads {
                key_heads: 4,
                heads_per_group: 3,
                key_features: 128,
                value_features: 128,
            },
            kept: 2,
            rule,
            masked: vec![],
        };
        for layout in [
            Layout::default(),
            Layout {
                offset: 1,
                ..Layout::default()
            },
            Layout {
                pad: 3,
                ..Layout::default()
            },
        ] {
            let operands = case.operands();
            let inputs = case.inputs_laid_out(&client, &operands, layout, F32);
            let what = format!("the table {rule:?} {layout:?}");
            if let Some(walked) = tuned_or_skip(&hw, &device, inputs, &what) {
                case.check(&client, &operands, walked, F32, &what);
            }
        }
    }
}
