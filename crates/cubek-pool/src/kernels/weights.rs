//! An average's weights: a procedural tile over the window, one factor per pooled axis, each
//! normalized to sum to one over the taps the average counts.

use crate::kernels::AverageCount;
use crate::kernels::space::{BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W, WindowAxis};
use cubecl::prelude::*;
use cubecl::tile::procedural::{
    AffineCoordinate, Constant, DivGuard, Factors, Procedural, Reads, Recipe, RecipeCoords,
    RecipeExpand, Sum, TapSupport, affine_along, sum_of,
};
use cubecl::tile::{Axis, Partitioning, Tile};

/// Where tap `t` of output position `o` lands in the padded input: `o · stride + t · dilation`.
type PaddedPosition<E> = Sum<AffineCoordinate<E>, AffineCoordinate<E>>;

/// One where a tap lands before `limit` in the padded input, zero past it.
#[derive(CubeType, Clone)]
pub struct InsidePadding<E: Float> {
    position: PaddedPosition<E>,
    limit: E,
}

#[cube]
impl<E: Float> Recipe<E> for InsidePadding<E> {
    fn evaluate(&self, coordinates: &RecipeCoords) -> E {
        select(
            self.position.evaluate(coordinates) < self.limit,
            E::new(1.0),
            E::new(0.0),
        )
    }
}

impl<E: Float> Reads for InsidePaddingExpand<E> {
    fn reads(&self, scope: &Scope, axis: Axis) -> bool {
        self.position.reads(scope, axis)
    }
}

/// The weights of an average over `partitioning`'s space: one over the taps it counts. Where no
/// tap leaves the input, that is a constant, and nothing is normalized per window.
#[cube]
pub fn average_weights<E: Float>(
    partitioning: &Partitioning,
    #[comptime] windows: [WindowAxis; 2],
    #[comptime] input: [usize; 2],
    #[comptime] count: AverageCount,
) -> Tile {
    let space = comptime!(
        partitioning
            .space()
            .subspace(&[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W])
    );
    let guard = comptime!(DivGuard::default());
    match comptime!(count) {
        AverageCount::Window => {
            let mut factors = Sequence::new();
            factors.push(Constant::<E> {
                value: E::new(1.0 / windows[0].taps as f64),
            });
            factors.push(Constant::<E> {
                value: E::new(1.0 / windows[1].taps as f64),
            });
            Procedural::separable::<E, Factors<Constant<E>>>(space, Factors::new(factors))
                .tile::<E>()
        }
        AverageCount::InBounds => {
            let mut factors = Sequence::new();
            factors.push(Constant::<E> { value: E::new(1.0) });
            factors.push(Constant::<E> { value: E::new(1.0) });
            Procedural::separable::<E, Factors<Constant<E>>>(space, Factors::new(factors))
                .normalized(comptime!(TapSupport::InBounds), guard)
                .tile::<E>()
        }
        AverageCount::Padded => {
            let mut factors = Sequence::new();
            factors.push(inside_padding::<E>(OUTPUT_H, TAP_H, windows[0], input[0]));
            factors.push(inside_padding::<E>(OUTPUT_W, TAP_W, windows[1], input[1]));
            Procedural::separable::<E, Factors<InsidePadding<E>>>(space, Factors::new(factors))
                .normalized(comptime!(TapSupport::Whole), guard)
                .tile::<E>()
        }
    }
}

/// [`InsidePadding`] along one pooled axis, its window `window` over an input `input` long.
#[cube]
fn inside_padding<E: Float>(
    #[comptime] output: Axis,
    #[comptime] tap: Axis,
    #[comptime] window: WindowAxis,
    #[comptime] input: usize,
) -> InsidePadding<E> {
    InsidePadding::<E> {
        position: sum_of(
            affine_along(output, E::new(0.0), E::new(window.stride as f64)),
            affine_along(tap, E::new(0.0), E::new(window.dilation as f64)),
        ),
        limit: E::new((input + 2 * window.padding) as f64),
    }
}
