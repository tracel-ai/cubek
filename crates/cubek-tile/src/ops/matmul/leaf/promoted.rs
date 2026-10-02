//! The promoted accumulator form of the register nest: the `mr × nr` block is the accumulator,
//! kept in `T` across calls and drained once.

use cubecl::prelude::*;

use super::registers;
use super::scale::Side;
use crate::*;

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` over the block, one rank-1 update per `K` step, with scales folded in.
    /// The block must be opened against the rhs it contracts (width and fold).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        let semiring = comptime!(self.accumulation.semiring("RegisterData::mma"));
        // A load stored across several columns reads as their runs along the contraction.
        let rhs_load = rhs.vector_tile();
        let vw = comptime!(rhs_load.run_length());
        let lw = lhs.vector_size();
        let fold = comptime!(self.fold);
        // A packed rhs is served at its packing factor, so this checks the block was opened by it:
        // its lines are the rhs's, or it folds the rhs's runs into one-wide cells.
        comptime!(assert!(
            (fold == 1 && vw == self.vector_size) || (fold == vw && self.vector_size == 1),
            "RegisterData::mma: the block's lines are {} wide folding {fold} values a step, but the \
             rhs serves runs of {vw}; a packed rhs serves its packing factor, so open the block \
             against the rhs it contracts",
            self.vector_size
        ));
        let lined_along_k = comptime!(
            lhs.place
                .space
                .contains(rhs.place.space.axis_at(rhs.place.space.rank() - 1))
        );
        comptime!(assert!(
            lined_along_k == (fold > 1),
            "RegisterData::mma: the block's lines hold {fold} partials of a cell but the rhs lines \
             along {}; open the block against the operands it contracts",
            if lined_along_k {
                "the contraction"
            } else {
                "the accumulator"
            }
        ));
        comptime!(assert!(
            fold == 1 || lw == fold,
            "RegisterData::mma: a step consumes {fold} contracted values, so the lhs must line \
             along the contraction at that width (it is {lw} wide)"
        ));
        let size!(L) = lw;

        let operands = comptime!(Space::merge(&[&lhs.place.space, &rhs.place.space]));
        let kc = comptime!(operands.contracted_extent(&out));
        let (mr, nr) = comptime!((self.mr, self.nr));

        // Read the column edge off the operands so a split column group stays one edge.
        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs.place.space));
        let cols = comptime!(acc_axes.cols(&out));
        let lhs_axes =
            comptime!(MatrixAxes::new(&lhs.place.space, mr, kc).unwrap_or_else(|e| panic!("{e}")));
        let rhs_axes = comptime!(if fold > 1 {
            MatrixAxes::new(&rhs.place.space, cols, kc).unwrap_or_else(|e| panic!("{e}"))
        } else {
            MatrixAxes::new(&rhs.place.space, kc, cols).unwrap_or_else(|e| panic!("{e}"))
        });

        let size!(RL) = comptime!(rhs_load.values());

        let config = comptime!(self.config);
        // The scalars the block holds: a line a cell, one value where it folds.
        let unroll = comptime!(mr * nr * self.vector_size <= config.budget);
        let component_fanout = comptime!(config.component_fanout);

        let lhs_mat = lhs.matrix_packed::<L>(lhs_axes, 0usize);
        let rhs_mat = rhs.matrix_packed::<RL>(rhs_axes, 0usize);
        let lhs_scales = lhs.reader(
            lhs_axes,
            0usize,
            comptime!(Side::Lhs),
            comptime!(out.clone()),
            comptime!(acc_axes),
        );
        let rhs_scales = rhs.reader(
            rhs_axes,
            0usize,
            comptime!(Side::Rhs),
            comptime!(out.clone()),
            comptime!(acc_axes),
        );

        registers::contract::<T, EL, L, ER, RA, RL>(
            &lhs_mat,
            &lhs_scales,
            &rhs_mat,
            &rhs_scales,
            &mut self.data,
            lw,
            fold,
            comptime!(rhs_load.values() / vw),
            mr,
            nr,
            kc,
            unroll,
            component_fanout,
            semiring,
        );
    }
}

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// `self += lhs · rhs` where `lhs` is a register block over the same rows as this one: each
    /// of its cells is a factor read in registers, and `rhs` is read one step of the contraction
    /// at a time. What an attention's unit does with the probabilities it just computed,
    /// contracting them with the values without leaving registers.
    pub(crate) fn mma_block<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &RegisterData<EL>,
        rhs: &Tile<ER>,
    ) {
        let semiring = comptime!(self.accumulation.semiring("RegisterData::mma_block"));
        comptime!(assert!(
            lhs.fold == 1 && self.fold == 1,
            "RegisterData::mma_block: both blocks hold whole cells, a line of neighbours"
        ));
        comptime!(assert!(
            lhs.mr == self.mr,
            "RegisterData::mma_block: the lhs block has {} rows and this one {}; they are \
             the same rows",
            lhs.mr,
            self.mr
        ));
        let vw = rhs.vector_size();
        comptime!(assert!(
            vw == self.vector_size,
            "RegisterData::mma_block: the block's lines are {} wide but the rhs serves {vw}",
            self.vector_size
        ));
        let (mr, nr, lw) = (self.mr, self.nr, lhs.vector_size);
        let kc = comptime!(lhs.nr * lw);
        let rhs_axes = comptime!(
            MatrixAxes::new(&rhs.place.space, kc, nr * vw).unwrap_or_else(|e| panic!("{e}"))
        );
        let rhs_mat = rhs.matrix_packed::<RA>(rhs_axes, 0usize);
        let mut b = Array::<Vector<T, RA>>::new(nr);
        #[unroll]
        for k in 0..kc {
            #[unroll]
            for n in 0..nr {
                b[n] = Vector::<T, RA>::cast_from(
                    rhs_mat.read(((k as u32).runtime(), (n as u32).runtime())),
                );
            }
            #[unroll]
            for i in 0..mr {
                let line = lhs.data[i * lhs.nr + k / lw];
                let a = Vector::<T, RA>::cast_from(line.extract(comptime!(k % lw)));
                #[unroll]
                for n in 0..nr {
                    let at = i * nr + n;
                    self.data[at] = semiring.step::<Vector<T, RA>>(a, b[n], self.data[at]);
                }
            }
        }
    }
}
