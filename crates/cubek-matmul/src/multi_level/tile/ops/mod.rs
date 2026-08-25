//! Cross-variant `Tile` op dispatchers. Each is a thin match on `self.kind`
//! delegating to a per-variant method.

mod copy;
mod elementwise;
mod mask;
mod matmul;
mod partition;
mod rowwise;
mod softmax;

pub use matmul::*;
pub use partition::*;
pub use softmax::*;
