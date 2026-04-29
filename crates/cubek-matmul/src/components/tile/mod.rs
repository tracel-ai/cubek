mod matmul;
mod setup;

// Internal-only re-export of cubek-std tile types so existing crate-relative paths keep
// working. Not exported — external crates should import these directly from cubek-std.
pub(crate) use cubek_std::tile::{
    Plane, Scope, Tile, Unit, Value, cmma::*, interleaved::*, mma::*,
    plane_vec_mat_inner_product::*, register::*,
};
pub use matmul::*;
pub use setup::*;
