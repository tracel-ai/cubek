//! The cases the suite runs, in blocks of runs, and the host reference they are checked against,
//! in `f64`.

use cubecl::ir::{ElemType, FloatKind};

pub const EPSILON: f32 = 1e-6;
pub const F32: ElemType = ElemType::Float(FloatKind::F32);
pub const F16: ElemType = ElemType::Float(FloatKind::F16);
pub const BF16: ElemType = ElemType::Float(FloatKind::BF16);

/// One launch's elements, and where its gate enters.
#[derive(Debug, Clone, Copy)]
pub struct Run {
    pub element: ElemType,
    pub output_element: ElemType,
}

/// Runs over cases, at every setting a routine lists or at its default one.
pub struct Block {
    pub every_setting: bool,
    pub runs: Vec<Run>,
    pub cases: Vec<Case>,
}

/// `rows` rows of `dim` channels, scaled by one weight row or one per row.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub rows: usize,
    pub dim: usize,
    pub weight_a_row: bool,
}

impl Case {
    pub fn new(rows: usize, dim: usize) -> Self {
        Self {
            rows,
            dim,
            weight_a_row: false,
        }
    }

    pub fn input(&self) -> Vec<f32> {
        (0..self.rows * self.dim).map(|i| wobble(i, 1)).collect()
    }

    pub fn weight(&self) -> Vec<f32> {
        let rows = if self.weight_a_row { self.rows } else { 1 };
        (0..rows * self.dim).map(|i| wobble(i, 2) / 2.0).collect()
    }

    /// The norm of the input's rows scaled by the weight, with both as `element` holds them.
    pub fn reference(&self, element: ElemType) -> Vec<f64> {
        let held = |values: Vec<f32>| -> Vec<f64> {
            values.into_iter().map(|x| round_to(x, element)).collect()
        };
        let (input, weight) = (held(self.input()), held(self.weight()));
        let mut out = Vec::with_capacity(input.len());
        for row in 0..self.rows {
            let values = &input[row * self.dim..(row + 1) * self.dim];
            let mean = values.iter().map(|x| x * x).sum::<f64>() / self.dim as f64;
            let scale = 1.0 / (mean + EPSILON as f64).sqrt();
            let weights = if self.weight_a_row { row } else { 0 } * self.dim;
            for (d, x) in values.iter().enumerate() {
                out.push(x * scale * weight[weights + d]);
            }
        }
        out
    }
}

/// `value` as `element` holds it.
pub fn round_to(value: f32, element: ElemType) -> f64 {
    match element {
        ElemType::Float(FloatKind::F16) => half::f16::from_f32(value).to_f32() as f64,
        ElemType::Float(FloatKind::BF16) => half::bf16::from_f32(value).to_f32() as f64,
        _ => value as f64,
    }
}

/// Values in `[-2, 2)`, not periodic in any extent a case uses.
fn wobble(i: usize, salt: usize) -> f32 {
    ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
}

/// The runs of `pairs`, each under every order of `orders`.
fn runs(pairs: &[(ElemType, ElemType)]) -> Vec<Run> {
    pairs
        .iter()
        .map(|&(element, output_element)| Run {
            element,
            output_element,
        })
        .collect()
}

/// What each setting runs, and what the default one runs.
pub fn blocks() -> Vec<Block> {
    let mut a_row = Case::new(6, 128);
    a_row.weight_a_row = true;
    vec![
        // Every setting's partitioning: a decode row, a prompt's rows, a prime row that binds one
        // element at a time, a row a line past a power of two, one longer than a plane's
        // registers hold, and a weight a row.
        Block {
            every_setting: true,
            runs: runs(&[(F16, F16)]),
            cases: vec![
                Case::new(1, 2560),
                Case::new(300, 128),
                Case::new(2, 111),
                Case::new(1, 4100),
                Case::new(2, 8192),
                a_row,
            ],
        },
        // Each order at each element pair, plain and grouped.
        Block {
            every_setting: false,
            runs: runs(&[
                (F32, F32),
                (F16, F16),
                (BF16, BF16),
                (F16, F32),
                (BF16, F32),
            ]),
            cases: vec![Case::new(16, 128)],
        },
    ]
}
