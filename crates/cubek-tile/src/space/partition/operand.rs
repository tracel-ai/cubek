//! An operand of a [`Tiling::over`](crate::Tiling::over) build: what it is before the space
//! exists (role, axes), plus the residence ladder the level closures write. Each level states
//! [`stage`](Operand::stage) for the operands it materializes; an operand a level leaves
//! unstated is [`InPlace`](Residence::InPlace) there.

use crate::{Axis, Residence};

/// Which way an operand crosses a residence rung: an input fills descending, an output drains
/// ascending.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Role {
    Input,
    Output,
}

/// One operand's space-independent spec: its [`Role`], the axes it spans, and the residence
/// ladder accumulated while the levels are declared ([`stage`](Operand::stage)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Operand {
    role: Role,
    axes: Vec<Axis>,
    ladder: Vec<Residence>,
}

impl Operand {
    pub fn input(axes: &[Axis]) -> Self {
        Operand::new(Role::Input, axes)
    }

    pub fn output(axes: &[Axis]) -> Self {
        Operand::new(Role::Output, axes)
    }

    fn new(role: Role, axes: &[Axis]) -> Self {
        Operand {
            role,
            axes: axes.to_vec(),
            ladder: Vec::new(),
        }
    }

    /// State where this operand lives at the level currently being declared. Sayable once per
    /// level, inside that level's closure; a level that says nothing leaves the operand
    /// [`InPlace`](Residence::InPlace).
    pub fn stage(&mut self, residence: Residence) {
        self.ladder.push(residence);
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// The full ladder, one residence per level, coarse to fine (unstated levels
    /// [`InPlace`](Residence::InPlace)). Complete once the build has run.
    pub fn residences(&self) -> &[Residence] {
        &self.ladder
    }

    /// Seal level `index`: reject a double statement, pad an omission to
    /// [`InPlace`](Residence::InPlace). Run by the builder as each level closure returns.
    pub(crate) fn close_level(&mut self, index: usize) {
        assert!(
            self.ladder.len() <= index + 1,
            "Operand::stage: {:?} {:?} stated more than one residence at level {index}",
            self.role,
            self.axes
        );
        if self.ladder.len() == index {
            self.ladder.push(Residence::InPlace);
        }
    }
}

/// The operands a [`Tiling::over`](crate::Tiling::over) build threads through its level
/// closures. Each routine defines its own struct of [`Operand`] fields and lists them here, so
/// stating a residence is a field access and the builder can seal every ladder per level.
pub trait OperandSet {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand>;
}
