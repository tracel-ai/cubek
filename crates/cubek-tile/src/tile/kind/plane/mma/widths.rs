//! The widths a manual-mma fragment's registers take in each role, bound at allocation.

use cubecl::prelude::*;

// Per-role fragment register widths, bound at allocation to `def.vector_size(role)`.
define_size!(pub(crate) NL);
define_size!(pub(crate) NR);
define_size!(pub(crate) NA);
// A block-scaled fragment's widths: its values' per role, and its scales'.
define_size!(pub(crate) NLB);
define_size!(pub(crate) NRB);
define_size!(pub(crate) NSB);
