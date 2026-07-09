//! [`Resident`]: the accumulator's register-tier decorator, [`Staging`](crate::Staging)'s
//! write-side dual. A memory accumulator whose operation runs register-resident is
//! initialized from the accumulator once, accumulated across the whole walk, and written
//! back once (the classic global matmul's `init_accumulator` / epilogue). Where `Staging`
//! refills per region, residency brackets the whole operation. What "register form" means
//! (the fragment grid, its windows) is the backing store's business
//! ([`fragments_like`](Tile), [`copy_from`](Tile::copy_from)), not ours.
//!
//! `contract` is a hand-written expand method for the same reason as `Staging`'s
//! `fill`/`consume`: the write-back must follow the caller-defined body, a `Drop` guard
//! can't emit ops in cubecl, and `#[cube]` rejects `impl Trait` args.

use cubecl::{prelude::*, unexpanded};

use crate::*;

/// The register-resident form of a memory accumulator: the register tile standing in
/// for the tile it decorates. Built by [`promote`](Tile), consumed by
/// [`contract`](Resident::contract).
#[derive(CubeType)]
pub struct Resident<T: Numeric> {
    acc: Tile<T>,
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Promote this accumulator to its register form, initialized from the delivered
    /// values so the operation accumulates onto them. `k` is the operation's contraction
    /// depth (this tile's own axes give `m`/`n` but not `k`).
    pub fn promote(&self, #[comptime] k: usize) -> Resident<Acc> {
        let mut acc = self.fragments_like(k);
        acc.copy_from(self);
        Resident::<Acc> { acc }
    }
}

#[cube]
impl<Acc: Numeric> Resident<Acc> {
    /// Write the resident accumulator back into the tile it decorates.
    fn writeback(&self, out: &mut Tile<Acc>) {
        out.copy_from(&self.acc);
    }
}

impl<Acc: Numeric> Resident<Acc> {
    /// Run `compute` on the resident accumulator, then write it back to `out` — the
    /// closure-scoped lifecycle, mirroring [`Staging`](crate::Staging)'s `fill`/`consume`.
    /// See [`ResidentExpand::__expand_contract_method`].
    pub fn contract(&mut self, _out: &mut Tile<Acc>, _compute: impl FnOnce(&mut Tile<Acc>)) {
        unexpanded!()
    }
}

impl<Acc: Numeric> ResidentExpand<Acc> {
    pub fn __expand_contract_method<F>(
        &mut self,
        scope: &Scope,
        out: &mut TileExpand<Acc>,
        compute: F,
    ) where
        F: FnOnce(&Scope, &mut TileExpand<Acc>),
    {
        compute(scope, &mut self.acc);
        self.__expand_writeback_method(scope, out);
    }
}
