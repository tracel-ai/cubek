use cubek_std::stage::SwizzleMode;

use crate::definition::{StageIdent, SwizzleModes};

pub(crate) fn swizzle_for_ident(modes: SwizzleModes, ident: StageIdent) -> SwizzleMode {
    match ident {
        StageIdent::Lhs => modes.lhs,
        StageIdent::Rhs => modes.rhs,
        StageIdent::Acc => modes.acc,
        StageIdent::Out => modes.out,
    }
}
