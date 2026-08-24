//! An operand of a [`Tiling::over`](crate::Tiling::over) build: the facts of the global tensor
//! (axes, element type, quantization scheme), plus the typed stages the level closures write.
//! Each level states [`stage`](Operand::stage) (a move) or [`stage_as`](Operand::stage_as)
//! (a conversion) for the operands it materializes; an operand a level leaves unstated is
//! [`InPlace`](Residence::InPlace) there. Read one operand's stated types top to bottom and
//! you have its type flow: unchanged through every move, re-typed at each conversion, meeting
//! the instruction's port at the bottom. Direction is not the operand's to say: whether a
//! stage fills or drains is stated by the op call that consumes the tile (a written operand
//! is the call's `&mut` receiver).
//!
//! A conversion's extra inputs must be facts of the operand: a quantized type decodes with the
//! scales its own binding carries, read in place from global memory (cache-served, never
//! staged), addressed by the region the stage covers. An edge whose inputs would have to be
//! computed (fresh scales for a re-scaling) has no vocabulary here and must be rejected where
//! edges are lowered.

use cubecl::ir::ElemType;
use cubecl::quant::scheme::QuantScheme;

use crate::{Axis, Residence};

/// One stage of an operand: where the cells sit while the level below runs, and the
/// element type they hold there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Stage {
    pub residence: Residence,
    pub dtype: ElemType,
}

/// One operand's space-independent spec: the facts of its global tensor (axes, the element
/// type memory holds, its quantization scheme), and the typed stages accumulated while the
/// levels are declared.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Operand {
    axes: Vec<Axis>,
    dtype: ElemType,
    quant: Option<QuantScheme>,
    stages: Vec<Stage>,
    sealed: bool,
}

impl Operand {
    /// An operand over `axes` whose global tensor holds `dtype`.
    pub fn new(axes: &[Axis], dtype: ElemType) -> Self {
        Operand {
            axes: axes.to_vec(),
            dtype,
            quant: None,
            stages: Vec::new(),
            sealed: false,
        }
    }

    /// State that the global tensor is quantized: values plus scales, decoded per `scheme`.
    /// A fact of the binding, and what a conversion off this operand's type decodes with.
    pub fn quantized(mut self, scheme: QuantScheme) -> Self {
        self.quant = Some(scheme);
        self
    }

    /// Stage this operand at the level currently being declared, keeping its current type:
    /// a move. Sayable once per level, inside that level's closure; a level that says nothing
    /// leaves the operand [`InPlace`](Residence::InPlace).
    pub fn stage(&mut self, residence: Residence) {
        let dtype = self.current_dtype();
        self.push(Stage { residence, dtype });
    }

    /// Stage this operand at the level currently being declared, converting it to `dtype`.
    /// The written type is the statement that work happens here; a conversion to the type the
    /// operand already holds is a move wearing a conversion's clothes, and is rejected.
    pub fn stage_as(&mut self, residence: Residence, dtype: ElemType) {
        assert_ne!(
            dtype,
            self.current_dtype(),
            "Operand::stage_as: {:?} already holds {dtype:?}; use stage",
            self.axes
        );
        self.push(Stage { residence, dtype });
    }

    fn push(&mut self, stage: Stage) {
        assert!(
            !self.sealed,
            "Operand::stage: stated after the space was built"
        );
        self.stages.push(stage);
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    /// The element type the global tensor holds.
    pub fn dtype(&self) -> ElemType {
        self.dtype
    }

    /// The quantization scheme, for an operand whose binding is values plus scales.
    pub fn quant(&self) -> Option<QuantScheme> {
        self.quant
    }

    /// The stages, one per level, coarse to fine (unstated levels
    /// [`InPlace`](Residence::InPlace) at the type the operand held there). Complete once the
    /// build has run, which is what seals the operand against further statements.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// The element type this operand holds below every stage stated so far: the last stated
    /// type, else the global tensor's.
    pub fn current_dtype(&self) -> ElemType {
        self.stages.last().map_or(self.dtype, |stage| stage.dtype)
    }

    /// The residence column of the operand, one entry per level coarse to fine: what a
    /// [`TileSpec`](crate::TileSpec) stamps as its per-level residences.
    pub fn residences(&self) -> Vec<Residence> {
        self.stages.iter().map(|stage| stage.residence).collect()
    }

    /// Whether this operand stated a residence at level `index`, asked before
    /// [`close_level`](Self::close_level) pads an omission.
    pub(crate) fn stated_at(&self, index: usize) -> bool {
        self.stages.len() > index
    }

    /// Seal level `index`: reject a double statement, pad an omission to an
    /// [`InPlace`](Residence::InPlace) move. Run by the builder as each level closure returns.
    pub(crate) fn close_level(&mut self, index: usize) {
        assert!(
            self.stages.len() <= index + 1,
            "Operand::stage: {:?} stated more than one stage at level {index}",
            self.axes
        );
        if self.stages.len() == index {
            let dtype = self.current_dtype();
            self.stages.push(Stage {
                residence: Residence::InPlace,
                dtype,
            });
        }
    }

    /// Drop the stages of the levels the build did not keep, `kept` holding one flag per
    /// declared level. Only a level no operand stated anything at is dropped, so every stage
    /// this removes is the [`InPlace`](Residence::InPlace) padding, and the column stays one
    /// entry per level of the space that was built.
    pub(crate) fn keep_levels(&mut self, kept: &[bool]) {
        // A flag per stage, or the flags line up against the wrong levels and every residence
        // below the first drop silently shifts a level.
        assert_eq!(
            kept.len(),
            self.stages.len(),
            "Operand::keep_levels: {:?} holds {} stages against {} levels",
            self.axes,
            self.stages.len(),
            kept.len()
        );
        let mut level = 0;
        self.stages.retain(|_| {
            let keep = kept[level];
            level += 1;
            keep
        });
    }

    /// Freeze the operand once its space is built: every later statement panics, so the
    /// stages cannot drift from the space they were stated against.
    pub(crate) fn seal(&mut self) {
        self.sealed = true;
    }
}

// Small tuples of operands as an [`OperandSet`], for callers (tests foremost) whose set has
// no name of its own: `Tiling::over((a, b, out), …)` with `o.0`/`o.1`/`o.2` in the closures.
impl OperandSet for (Operand,) {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.0].into_iter()
    }
}

impl OperandSet for (Operand, Operand) {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.0, &mut self.1].into_iter()
    }
}

impl OperandSet for (Operand, Operand, Operand) {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.0, &mut self.1, &mut self.2].into_iter()
    }
}

impl OperandSet for (Operand, Operand, Operand, Operand) {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.0, &mut self.1, &mut self.2, &mut self.3].into_iter()
    }
}

/// The operands a [`Tiling::over`](crate::Tiling::over) build threads through its level
/// closures. Each routine defines its own struct of [`Operand`] fields and lists them here, so
/// stating a stage is a field access and the builder can seal every operand per level.
/// A field left out of [`each`](OperandSet::each) is invisible to the builder: it is neither
/// padded nor sealed, and only a consumer's length check can catch it.
pub trait OperandSet {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand>;
}
