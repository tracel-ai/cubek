//! A table these values index: stored fields that are positions rather than numbers
//! ([`Tile::lookup`](crate::Tile::lookup)), read back as `table[index]` where the kernel copies
//! them, and nowhere else.
//!
//! Like a [`Factor`], it rides the values it was attached to, and the table's element is the
//! table's business: it is erased while the kernel is expanded and read back as `f32`.

use std::sync::Arc;

use cubecl::{ir::Scope, prelude::*, std::tensor::layout::CoordsDyn, unexpanded};

use crate::*;

/// A table as the values read it.
pub(crate) trait CodebookRead {
    /// The table's entry at `index`, widened to `f32`.
    fn entry(&self, scope: &Scope, index: NativeExpand<u32>) -> NativeExpand<f32>;
}

impl<S: Numeric> CodebookRead for TileExpand<S> {
    fn entry(&self, scope: &Scope, index: NativeExpand<u32>) -> NativeExpand<f32> {
        table_entry::expand::<S>(scope, self, index)
    }
}

/// `table[index]`, the table read as the one-axis tile it is.
#[cube]
fn table_entry<S: Numeric>(table: &Tile<S>, index: u32) -> f32 {
    let entries = table.nd_packed::<Const<1>>(comptime!(Guard::Checked));
    let mut at = CoordsDyn::new();
    at.push(index);
    f32::cast_from(entries.read(at).extract(0usize))
}

/// The table these values index, if any. Empty is values that are numbers.
#[derive(Clone)]
pub(crate) struct Codebook;

#[derive(Clone, Default)]
pub(crate) struct CodebookExpand {
    table: Option<Arc<dyn CodebookRead>>,
}

impl Codebook {
    /// No table: what every operand carries until [`Tile::lookup`](crate::Tile::lookup) says
    /// otherwise.
    pub(crate) fn none() -> Codebook {
        unexpanded!()
    }

    /// The entry the stored `index` names.
    pub(crate) fn entry(&self, _index: u32) -> f32 {
        unexpanded!()
    }

    pub(crate) fn __expand_none(_scope: &Scope) -> CodebookExpand {
        CodebookExpand::default()
    }
}

impl CodebookExpand {
    /// A codebook reading `table`.
    pub(crate) fn of<S: Numeric>(table: &TileExpand<S>) -> Self {
        CodebookExpand {
            table: Some(Arc::new(table.clone())),
        }
    }

    /// Whether these values index a table at all.
    pub(crate) fn present(&self) -> bool {
        self.table.is_some()
    }

    pub(crate) fn __expand_entry_method(
        &self,
        scope: &Scope,
        index: NativeExpand<u32>,
    ) -> NativeExpand<f32> {
        self.table
            .as_ref()
            .expect("Codebook::entry: these values index no table")
            .entry(scope, index)
    }
}

impl CubeType for Codebook {
    type ExpandType = CodebookExpand;
}
impl IntoExpand for CodebookExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for CodebookExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for CodebookExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for CodebookExpand {}
impl AsRefExpand for CodebookExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for CodebookExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
