//! One walk's launch arguments.

use cubecl::tile::Launcher;
use cubecl::tile::launch::Bound;
use cubek_microkernel::attention_backward::AttentionBackwardOperands;

/// The arguments of one walk's launch, its operands bound under its launcher. Not `Debug`: the
/// tile DSL's `Launcher` is not.
pub(crate) struct AttentionBackwardBinding {
    pub launcher: Launcher,
    pub operands: AttentionBackwardOperands<Bound>,
}
