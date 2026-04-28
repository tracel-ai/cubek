use crate::stage::SwizzleMode;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageIdent {
    Lhs,
    Rhs,
    Acc,
    Out,
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SwizzleModes {
    pub lhs: SwizzleMode,
    pub rhs: SwizzleMode,
    pub acc: SwizzleMode,
    pub out: SwizzleMode,
}

impl SwizzleModes {
    pub fn has_swizzle(&self) -> bool {
        self.lhs != SwizzleMode::None
            || self.rhs != SwizzleMode::None
            || self.acc != SwizzleMode::None
            || self.out != SwizzleMode::None
    }
}
