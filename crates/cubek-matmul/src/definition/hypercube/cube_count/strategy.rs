use crate::definition::SmAllocation;

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Front-facing configuration when crafting a TilingBlueprint
/// Allows choosing a strategy before knowing actual values
pub enum CubeCountStrategy {
    #[default]
    /// X: num cubes in m, Y: num cubes in n, Z: num cubes in batch
    FromProblem,

    /// If not cubes_first: X: num SMs, Y: num cubes per SM
    /// If cubes_first: X: num cubes per SM, Y: num SMs
    Sm {
        cubes_first: bool,
        num_sms: u32,
        sm_usage: SmAllocation,
    },

    /// X: total cubes flattened (num SMs * num cubes per SM)
    Flattened,

    /// Heuristically find a balance for X, Y, Z that respects hardware limits
    Spread,
}
