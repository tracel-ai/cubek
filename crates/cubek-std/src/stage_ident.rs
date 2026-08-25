#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageIdent {
    Lhs,
    Rhs,
    Acc,
    Out,
}
