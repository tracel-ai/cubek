//! The cases the suite runs, in blocks of runs, and the host reference they are checked against,
//! in `f64`.

use cubecl::ir::{ElemType, FloatKind};

pub const EPSILON: f32 = 1e-6;

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
        _ => value as f64,
    }
}

/// Values in `[-2, 2)`, not periodic in any extent a case uses.
fn wobble(i: usize, salt: usize) -> f32 {
    ((i.wrapping_mul(2654435761) + salt * 40503) % 2048) as f32 / 512.0 - 2.0
}

pub fn cases() -> Vec<Case> {
    let mut a_row = Case::new(6, 128);
    a_row.weight_a_row = true;
    vec![
        // Decode steps: the model's row, and a head norm's rows.
        Case::new(1, 2560),
        Case::new(16, 128),
        // A prompt's rows of the model and of its heads.
        Case::new(37, 1024),
        Case::new(300, 128),
        // A short row, a prime one that binds one element at a time, and one a line past a power
        // of two.
        Case::new(3, 96),
        Case::new(2, 111),
        Case::new(1, 4100),
        // A row longer than a plane's registers hold, which only the cube and the floor take.
        Case::new(2, 8192),
        // No rows: nothing to launch.
        Case::new(0, 128),
        // A weight a row, as the query and key heads of a decode step share their norms' weights.
        a_row,
    ]
}
