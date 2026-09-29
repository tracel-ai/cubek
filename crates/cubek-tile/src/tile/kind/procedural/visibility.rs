use cubecl::prelude::*;

use crate::Axis;

use super::{Reads, Recipe, RecipeCoords, RecipeExpand};

/// Which keys each query sees, as the bias an attention's scores are summed with: zero where the
/// key is seen, the type's minimum where it is not.
///
/// A query sees the keys below `keys`, the attended length. Under `causal` it sees none past its
/// own position, aligned at the bottom right: the last of `queries` queries sees the last key, so
/// query `q` sees key `s` where `s + queries ≤ q + keys`.
#[derive(CubeType, Clone)]
pub struct KeyVisibility {
    pub keys: u32,
    pub queries: u32,
    #[cube(comptime)]
    pub query: Axis,
    #[cube(comptime)]
    pub key: Axis,
    #[cube(comptime)]
    pub causal: bool,
}

#[cube]
impl<T: Numeric> Recipe<T> for KeyVisibility {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let s = coordinates.along(self.key);
        let mut seen = s < self.keys;
        if comptime!(self.causal) {
            seen = seen && s + self.queries <= coordinates.along(self.query) + self.keys;
        }
        select(seen, T::from_int(0), T::min_value())
    }
}

impl Reads for KeyVisibilityExpand {
    fn reads(&self, _scope: &Scope, axis: Axis) -> bool {
        axis == self.key || (self.causal && axis == self.query)
    }
}
