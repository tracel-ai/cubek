//! A tile's rows scaled by per-row factors its holder keeps in registers.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl<E: Float> Tile<E> {
    /// `self[r, ..] *= factors[r]`, where a row is this tile's cells along its innermost axis and
    /// every other axis indexes the rows: what an online softmax's correction does to whatever was
    /// summed against its earlier blocks.
    ///
    /// Scaled where its holder keeps it: a unit's register block in its registers; a plane's
    /// fragments bounced one at a time through the plane's scratch, met on `sync_plane`; a plane's
    /// window of shared memory by the plane's units in turns. Every unit of a plane calls it for
    /// the plane's own tiles, and the plane holds every row's factor.
    pub fn mul_rows(&mut self, factors: &Array<E>) {
        let space = comptime!(self.place.space.clone());
        let holder = comptime!(self.place.holder());
        match &mut self.kind {
            TileKind::PlaneTile(tile) => tile.mul_rows(factors, 0usize),
            TileKind::PlanePartition(partition) => partition.mul_rows(factors),
            TileKind::Memory(window) => {
                comptime!(assert!(
                    holder == ComputeScope::Plane,
                    "Tile::mul_rows: a window of shared memory is scaled by the plane that holds \
                     it; this one is held by {holder:?}"
                ));
                let columns = comptime!(space.extent_at(space.rank() - 1));
                let cells = comptime!(space.cells());
                let size!(W) = 1usize;
                let mut view = window.flat_mut::<W>();
                let mut at = UNIT_POS_PLANE as usize;
                while at < cells {
                    view.write(at, view.read(at) * Vector::cast_from(factors[at / columns]));
                    at += PLANE_DIM as usize;
                }
            }
            TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::mul_rows: a register block, a plane's fragments, or a plane's window")
            }
        }
    }
}
