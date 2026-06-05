/// Defines the reduction operator (⊕) and the combination operator (⊗).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Semiring {
    /// (+, ×). Linear operator: y = Σ (w_i * x_i).
    Linear,
    /// (max, +). Tropical Max operator: y = max (w_i + x_i).
    TropicalMax,
    /// (min, +). Tropical Min operator: y = min (w_i + x_i).
    TropicalMin,
}
