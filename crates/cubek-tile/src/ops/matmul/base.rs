//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]).

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use super::leaf::memory;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final register-resident tile.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        self.init_identity(comptime!(semiring.add()));
        self.mma(lhs, rhs, semiring);
    }

    /// `c += a · b` at a final register-resident tile, folding onto what `c` holds.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        mma_leaf(self, lhs, rhs, semiring)
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mm_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let init_from = self.request_init_from(comptime!(InitFrom::Identity));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
        }
        self.mma_with(lhs, rhs, config, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c += a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mma_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let space = comptime!(self.place.space.clone());
        match &mut self.kind {
            TileKind::Memory(g) => {
                memory::contract::<Acc, Lhs, Rhs>(g, lhs, rhs, space, config, semiring)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::mma_with: the software instruction contracts into a memory accumulator; a \
                 register accumulator carries its own block (Tile::block_accumulator)"
            ),
        }
    }
}

/// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form.
#[cube]
pub(crate) fn mma_leaf<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.place.space.clone());
    let tile_kind = &mut acc.kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space, semiring),
        // A partition reaching a final tile carries exactly one tile.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space, semiring)
        }
        TileKind::Memory(_) => panic!(
            "mma_leaf: a Gmem/Smem accumulator contracts through the software instruction, which \
             runs under a register block; state it with Tile::mma_with(lhs, rhs, config, semiring)"
        ),
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) | TileKind::Lines(_) => {
            panic!("mma: a procedural tile and the plane's units are not an accumulator sink")
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile, each factor times whatever scales it carries.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(lhs, rhs);
                strided_2d(lhs, rhs, comptime!(out.clone()), transposed);
                hardware_semiring(semiring);
                d.mma(lhs, rhs, out)
            }
            PlaneTile::Mma(d) => {
                // The manual-mma instruction reads registers, where a scaled factor cannot land.
                lhs.refuse_factor("PlaneTile::Mma");
                rhs.refuse_factor("PlaneTile::Mma");
                flattened_k(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Registers(d) => match &lhs.kind {
                TileKind::PlaneTile(held) => match held {
                    PlaneTile::Registers(held) => d.mma_held_lhs(held, rhs, semiring),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::PlanePartition(held) => match held.fragment() {
                    PlaneTile::Registers(held) => d.mma_held_lhs(&held, rhs, semiring),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::Memory(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    let folded = comptime!(d.fold > 1);
                    strided_2d(lhs, rhs, comptime!(out.clone()), folded);
                    d.mma(lhs, rhs, out, semiring)
                }
            },
        }
    }
}

/// Asserts that the algebra is one a hardware instruction implements (multiply-add).
#[cube]
fn hardware_semiring(#[comptime] semiring: Semiring) {
    comptime!(assert!(
        semiring == Semiring::SUM_PROD,
        "mma: a hardware instruction contracts under the sum-product semiring alone, not \
         {semiring:?}; contract in memory or in a register block to fold under another"
    ));
}

/// Asserts that operands are not gathered and read as one matrix each.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] out: Space,
    #[comptime] rhs_along_k: bool,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let kc = Space::merge(&[&lhs.place.space, &rhs.place.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.place.space);
        let cols = axes.cols(&out);
        let rhs_matrix = if rhs_along_k {
            MatrixAxes::new(&rhs.place.space, cols, kc)
        } else {
            MatrixAxes::new(&rhs.place.space, kc, cols)
        };
        MatrixAxes::new(&lhs.place.space, axes.rows(&out), kc).is_ok() && rhs_matrix.is_ok()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `memory::memory` arm of `mma_leaf`"
    ));
}

/// Whether `rhs` is read col-major: a col-major cmma fragment or a staged `(col, k)` window.
#[cube]
fn transposed_rhs<EL: Numeric, ER: Numeric>(
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) -> comptime_type!(bool) {
    match &rhs.kind {
        TileKind::PlaneTile(t) => match t {
            PlaneTile::Cmma(d) => comptime!(d.layout == MatrixLayout::ColMajor),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => comptime!(false),
        },
        // The contracted axis is the lhs's trailing one; not every axis the output lacks is.
        // Only a shared stage can answer column-major; a gmem rhs never does.
        TileKind::Memory(m) => comptime!(
            m.address == AddressSpace::Shared
                && crate::ops::matmul::leaf::rhs_layout(
                    &rhs.place.space,
                    lhs.place.space.axis_at(lhs.place.space.rank() - 1)
                ) == MatrixLayout::ColMajor
        ),
        TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => comptime!(false),
    }
}

/// Asserts that operands contract their shared axes in the same order.
#[cube]
fn flattened_k<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.place.space, &rhs.place.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.place.space.difference(&out),
        rhs.place.space.difference(&out)
    ));
}
