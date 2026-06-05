use crate::{BoundaryMode, Footprint, Placement, Semiring, Weights};
use cubecl::prelude::*;

#[derive(Debug, Clone)]
pub struct LocalOp {
    pub semiring: Semiring,
    pub weights: Weights,
    pub placement: Placement,
    pub boundary: BoundaryMode,
}
impl LocalOp {
    /// Step 1: Resolve the abstract mathematical definition into discrete
    /// memory addresses and weights for a given output coordinate `p`.
    pub fn footprint(&self, p: usize, len_in: usize) -> Footprint {
        // Implementation generates the continuous mapping based on `self.placement`,
        // samples the continuous curve defined by `self.weights`,
        // normalizes weights if necessary, and returns the discrete indices/weights.
        unimplemented!("Projection and sampling engine goes here")
    }

    /// Step 2: The Core Execution Engine.
    /// This directly illustrates what the Semirings actually DO mathematically.
    pub fn execute_at(&self, p: usize, input: &[f32]) -> f32 {
        let foot = self.footprint(p, input.len());
        let len = foot.indices.len();

        match self.semiring {
            // y = Σ (w_i * x_i)
            Semiring::Linear => {
                let mut acc = 0.0;
                for i in 0..len {
                    let val = self.read_bounded(input, foot.indices[i]);
                    acc += val * foot.weights[i];
                }
                acc
            }

            // y = max (x_i + w_i)
            Semiring::TropicalMax => {
                let mut acc = std::f32::NEG_INFINITY;
                for i in 0..len {
                    let val = self.read_bounded(input, foot.indices[i]);
                    acc = acc.max(val + foot.weights[i]);
                }
                acc
            }

            // y = min (x_i + w_i)
            Semiring::TropicalMin => {
                let mut acc = std::f32::INFINITY;
                for i in 0..len {
                    let val = self.read_bounded(input, foot.indices[i]);
                    acc = acc.min(val + foot.weights[i]);
                }
                acc
            }

            // y = max (x_i * w_i)
            Semiring::MaxTimes => {
                let mut acc = 0.0;
                for i in 0..len {
                    let val = self.read_bounded(input, foot.indices[i]);
                    acc = acc.max(val * foot.weights[i]);
                }
                acc
            }

            // Log-Sum-Exp: y = log( Σ exp(x_i + w_i) )
            Semiring::Log => {
                let mut max_val = std::f32::NEG_INFINITY;
                let mut vals = SmallVec::<[f32; 8]>::with_capacity(len);

                // First pass for numerical stability (find max)
                for i in 0..len {
                    let val = self.read_bounded(input, foot.indices[i]) + foot.weights[i];
                    vals.push(val);
                    max_val = max_val.max(val);
                }

                // Second pass to sum exps
                let mut sum_exp = 0.0;
                for v in vals {
                    sum_exp += (v - max_val).exp();
                }
                max_val + sum_exp.ln()
            }
        }
    }

    /// Step 3: Edge Case Handling
    /// Reads from the input tensor while respecting the boundary conditions.
    fn read_bounded(&self, input: &[f32], idx: i64) -> f32 {
        let len = input.len() as i64;

        // Fast path: fully inside the tensor
        if idx >= 0 && idx < len {
            return input[idx as usize];
        }

        // Edge cases
        match self.boundary {
            BoundaryMode::Constant => {
                // The "identity" element changes based on the algebra used!
                match self.semiring {
                    Semiring::TropicalMax => std::f32::NEG_INFINITY,
                    Semiring::TropicalMin => std::f32::INFINITY,
                    _ => 0.0,
                }
            }
            BoundaryMode::Replicate => {
                let clamped = idx.clamp(0, len - 1);
                input[clamped as usize]
            }
            BoundaryMode::Reflect => {
                // Simplified reflection logic
                let mut r = idx.abs();
                if r >= len {
                    r = len - 1;
                }
                input[r as usize]
            }
            BoundaryMode::Circular => {
                let r = idx.rem_euclid(len);
                input[r as usize]
            }
        }
    }
}
