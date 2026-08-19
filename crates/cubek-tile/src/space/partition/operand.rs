//! An operand of a [`Tiling::over`](crate::Tiling::over) build: the axes it spans, plus the
//! per-level residences the level closures write. Each level states
//! [`stage`](Operand::stage) for the operands it materializes; an operand a level leaves
//! unstated is [`InPlace`](Residence::InPlace) there. Direction is not the operand's to say:
//! whether a residence fills or drains is stated by the op call that consumes the tile (a
//! written operand is the call's `&mut` receiver).

use crate::{Axis, Residence};

/// One operand's space-independent spec: the axes it spans, and the residences accumulated
/// while the levels are declared ([`stage`](Operand::stage)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Operand {
    axes: Vec<Axis>,
    residences: Vec<Residence>,
    sealed: bool,
}

impl Operand {
    pub fn new(axes: &[Axis]) -> Self {
        Operand {
            axes: axes.to_vec(),
            residences: Vec::new(),
            sealed: false,
        }
    }

    /// State where this operand lives at the level currently being declared. Sayable once per
    /// level, inside that level's closure; a level that says nothing leaves the operand
    /// [`InPlace`](Residence::InPlace).
    pub fn stage(&mut self, residence: Residence) {
        assert!(
            !self.sealed,
            "Operand::stage: stated after the space was built"
        );
        self.residences.push(residence);
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// The residences, one per level, coarse to fine (unstated levels
    /// [`InPlace`](Residence::InPlace)). Complete once the build has run, which is what seals
    /// the operand against further [`stage`](Operand::stage) statements.
    pub fn residences(&self) -> &[Residence] {
        &self.residences
    }

    /// Seal level `index`: reject a double statement, pad an omission to
    /// [`InPlace`](Residence::InPlace). Run by the builder as each level closure returns.
    pub(crate) fn close_level(&mut self, index: usize) {
        assert!(
            self.residences.len() <= index + 1,
            "Operand::stage: {:?} stated more than one residence at level {index}",
            self.axes
        );
        if self.residences.len() == index {
            self.residences.push(Residence::InPlace);
        }
    }

    /// Freeze the operand once its space is built: every later [`stage`](Operand::stage)
    /// panics, so the residences cannot drift from the space they were stated against.
    pub(crate) fn seal(&mut self) {
        self.sealed = true;
    }
}

/// The operands a [`Tiling::over`](crate::Tiling::over) build threads through its level
/// closures. Each routine defines its own struct of [`Operand`] fields and lists them here, so
/// stating a residence is a field access and the builder can seal every operand per level.
/// A field left out of [`each`](OperandSet::each) is invisible to the builder: it is neither
/// padded nor sealed, and only a consumer's length check can catch it.
pub trait OperandSet {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand>;
}
