//! The gated delta walk's routines against a host reference, at shapes chosen to be awkward.
//!
//! The reference walks the recurrence in `f64` from the operands' values. The shapes hold a decode
//! step of a many-headed layer and several sequences of it, a prompt keeping a ring of states, a
//! prime count of positions with every state kept, heads that are their own key heads, a head of an
//! odd value width read a column at a time, key rows no unit takes whole, a state resumed from
//! values other than zero, and masked positions. The keys and queries are normalized, as the layer
//! hands them, so the state stays bounded over a long walk.

use super::harness::{F16, F32, elements, round_to};
use core::num::NonZeroUsize;
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_linear_recurrence::kernels::walk::{
    CubeDeltaWalkRoutine, CubeDeltaWalkRoutineSettings, RecurrenceWalkInputs,
    RecurrenceWalkOutputs, RecurrenceWalkProblem, RecurrenceWalkRoutine, RecurrenceWalkStateInputs,
    RecurrenceWalkTensors, launch_tuned,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::linear_recurrence::RecurrenceHeads;
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

/// A walk's shape, and the positions whose decay and strength are zero.
#[derive(Debug, Clone)]
pub struct Case {
    pub batch: usize,
    pub positions: usize,
    pub heads: RecurrenceHeads,
    pub kept: usize,
    pub masked: Vec<usize>,
}

/// A decode step of a many-headed layer and eight sequences of it, a prompt of it keeping a ring
/// of four, a prime walk keeping every state, heads that are their own key heads, an odd value
/// width, key rows of three lines, key rows that no unit takes whole, and masked positions.
pub fn cases() -> Vec<Case> {
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
}

/// The operands of a case, as `f32` host values.
pub struct Operands {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub decay: Vec<f32>,
    pub strength: Vec<f32>,
    pub state: Vec<f32>,
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
            decay: up(&[b, t, h], &operands.decay),
            strength: up(&[b, t, h], &operands.strength),
            state: RecurrenceWalkStateInputs::Owned {
                state: up(
                    &[b, h, heads.key_features, heads.value_features],
                    &operands.state,
                ),
                kept: NonZeroUsize::new(self.kept).expect("a state is kept"),
            },
            output_element: element,
        }
    }

    /// The recurrence walked in `f64`: what each position reads, and the kept states.
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
                        let recalled: f64 = (0..kd).map(|r| s[r * vd + j] * k[r] as f64).sum();
                        let delta = (v[j] as f64 - recalled) * beta;
                        for r in 0..kd {
                            s[r * vd + j] += k[r] as f64 * delta;
                        }
                        out[scalar * vd + j] = (0..kd).map(|r| s[r * vd + j] * q[r] as f64).sum();
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
}

/// How far the output may fall from the reference's rounding: a sum an ulp of the element apart
/// rounds to neighbouring values.
pub fn output_tolerance(element: ElemType) -> f64 {
    match element {
        F16 => 1.5e-3,
        F32 => 1e-4,
        _ => 4e-3,
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
    settings: &CubeDeltaWalkRoutineSettings,
    what: &str,
) -> Option<RecurrenceWalkOutputs> {
    let problem = problem_of(&inputs);
    match CubeDeltaWalkRoutine.launch_on(hw, inputs, settings) {
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

/// How a walk is reached: the cube routine at one of its settings, or the table, which runs
/// whichever setting its round elected.
#[derive(Debug, Clone)]
pub enum Reach {
    Setting(CubeDeltaWalkRoutineSettings),
    Table,
}

/// One walk: a case at an element, its operands laid out as `layout` lays them, reached as
/// `reach` names.
#[derive(Debug, Clone)]
pub struct WalkCase {
    pub case: Case,
    pub element: ElemType,
    pub layout: Layout,
    pub reach: Reach,
}

impl WalkCase {
    /// Each of `cases` at each of `elements`, laid out as each of `layouts` lays them, reached as
    /// each of `reaches` names.
    pub fn every(
        cases: &[Case],
        elements: &[ElemType],
        layouts: &[Layout],
        reaches: &[Reach],
    ) -> Vec<Self> {
        let mut every = Vec::new();
        for &element in elements {
            for &layout in layouts {
                for reach in reaches {
                    for case in cases {
                        every.push(Self {
                            case: case.clone(),
                            element,
                            layout,
                            reach: reach.clone(),
                        });
                    }
                }
            }
        }
        every
    }
}

/// The walk against the host: its output rounded to the element as the kernel rounds it, and the
/// states it keeps, in `f32`. A device may decline a walk only by the refusal admission names: a
/// head wider than its cube holds, or too few bindings.
pub struct Walks;

impl ScopeUnderTest for Walks {
    type Case = WalkCase;

    /// # Panics
    ///
    /// On a refusal other than the one admission names.
    fn launch(&self, client: &Client, walk: &WalkCase) -> Result<Vec<Written>, Refusal> {
        let hw = Hardware::from_client(client);
        let operands = walk.case.operands();
        let inputs = walk
            .case
            .inputs_laid_out(client, &operands, walk.layout, walk.element);
        let admitted = problem_of(&inputs).admit(&hw);
        let walked = match &walk.reach {
            Reach::Setting(settings) => CubeDeltaWalkRoutine.launch_on(&hw, inputs, settings),
            Reach::Table => admitted
                .clone()
                .map(|()| launch_tuned(&cubecl::test_device(), inputs, GroupPriority::DEFAULT)),
        };
        let walked = walked.inspect_err(|why| {
            assert_eq!(admitted, Err(why.clone()), "{walk:?}: refused but admitted");
        })?;
        Ok(vec![
            Written {
                name: "output",
                tensor: walked.out,
                host: HostDataType::F32,
            },
            Written {
                name: "kept states",
                tensor: walked.kept.expect("a walk over owned states keeps them"),
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, walk: &WalkCase) -> Vec<HostData> {
        let Case {
            batch,
            positions,
            heads,
            kept,
            ..
        } = walk.case.clone();
        let (out, kept_states) = walk.case.reference(&walk.case.operands());
        let out = out
            .iter()
            .map(|&x| round_to(x as f32, walk.element))
            .collect();
        let value_heads = walk.case.value_heads();
        let out_shape = [batch, positions, value_heads, heads.value_features];
        let kept_shape = [
            kept,
            batch,
            value_heads,
            heads.key_features,
            heads.value_features,
        ];
        vec![
            HostData::row_major(HostDataVec::F32(out), Shape::from(out_shape.to_vec())),
            HostData::row_major(
                HostDataVec::F32(kept_states.iter().map(|&x| x as f32).collect()),
                Shape::from(kept_shape.to_vec()),
            ),
        ]
    }

    /// A sum an ulp of the element apart rounds to neighbouring values.
    fn tolerance(&self, walk: &WalkCase) -> f32 {
        output_tolerance(walk.element) as f32
    }

    /// The kept states are summed and stored in `f32`.
    fn tolerance_of(&self, walk: &WalkCase, name: &'static str) -> f32 {
        match name {
            "kept states" => 1e-4,
            _ => self.tolerance(walk),
        }
    }
}

/// Every setting the cube routine lists runs every case at every element the device runs, and
/// matches the reference.
#[test]
fn the_cube_walk_matches_the_reference() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let reaches: Vec<Reach> = CubeDeltaWalkRoutine
        .settings(&hw)
        .into_iter()
        .map(Reach::Setting)
        .collect();
    Walks.check(WalkCase::every(
        &cases(),
        &elements(&client),
        &[Layout::default()],
        &reaches,
    ));
}

/// The floor, which serves every admitted problem, matches the reference.
#[test]
fn the_floor_matches_the_reference() {
    let client = cubecl::test_device().client();
    Walks.check(WalkCase::every(
        &cases(),
        &elements(&client),
        &[Layout::default()],
        &[Reach::Setting(CubeDeltaWalkRoutineSettings::FLOOR)],
    ));
}

/// Every case through the table, which runs whichever setting its round elected.
#[test]
fn the_table_matches_the_reference() {
    let client = cubecl::test_device().client();
    Walks.check(WalkCase::every(
        &cases(),
        &elements(&client),
        &[Layout::default()],
        &[Reach::Table],
    ));
}

/// A walk whose every position is masked leaves the state it resumed from, at every kept position.
#[test]
fn a_masked_walk_passes_the_state_through() {
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let masked = Case {
        batch: 2,
        positions: 4,
        heads: RecurrenceHeads {
            key_heads: 2,
            heads_per_group: 2,
            key_features: 40,
            value_features: 36,
        },
        kept: 3,
        masked: vec![0, 1, 2, 3],
    };
    let reaches: Vec<Reach> = CubeDeltaWalkRoutine
        .settings(&hw)
        .into_iter()
        .map(Reach::Setting)
        .collect();
    Walks.check(WalkCase::every(
        &[masked],
        &[F32],
        &[Layout::default()],
        &reaches,
    ));
}

/// Operands that lie padded, off a whole read of their buffers, and with the strides a decode
/// step's swapped view leaves on its extent-one axes: the walk reads them where they lie, at every
/// width the layouts leave it: 4, 2 and 1.
#[test]
fn the_walk_reads_operands_where_they_lie() {
    let client = cubecl::test_device().client();
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
    ];
    Walks.check(WalkCase::every(
        &cases(),
        &elements(&client),
        &layouts,
        &[
            Reach::Setting(CubeDeltaWalkRoutineSettings::DEFAULT),
            Reach::Setting(CubeDeltaWalkRoutineSettings::FLOOR),
        ],
    ));
}

/// A key tuned on an aligned call is met again by calls that bind narrower: the winner serves them.
#[test]
fn the_table_serves_a_narrower_call_after_an_aligned_one() {
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
        masked: vec![],
    };
    let layouts = [
        Layout::default(),
        Layout {
            offset: 1,
            ..Layout::default()
        },
        Layout {
            pad: 3,
            ..Layout::default()
        },
    ];
    Walks.check(WalkCase::every(&[case], &[F32], &layouts, &[Reach::Table]));
}
