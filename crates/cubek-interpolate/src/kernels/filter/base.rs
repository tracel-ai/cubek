use crate::definition::{InterpolateMode, ModeProperties, mode_properties};
use crate::kernels::coordinate::Rational;
use crate::kernels::space::{BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W};
use cubecl::prelude::*;
use cubecl::tile::procedural::{
    AffineCoordinate, Constant, DivGuard, Factors, Phase, Procedural, Reads, Recipe, Sum,
    TapSupport, affine_along, sum_of,
};
use cubecl::tile::{Axis, Partitioning, Tile};
use cubecl_common::Ratio;

use super::{Cubic, Lanczos, Linear};

pub type TapDistance<E> = Sum<AffineCoordinate<E>, Phase<E>>;
type NearestAxis<E> = Constant<E>;
type BilinearAxis<E> = Linear<TapDistance<E>>;
type BicubicAxis<E> = Cubic<TapDistance<E>>;
type Lanczos3Axis<E> = Lanczos<TapDistance<E>>;

#[cube]
pub trait SeparableFilter<E: Float>: Send + std::marker::Sync + 'static
where
    <Self::Axis as CubeType>::ExpandType: Reads,
{
    type Axis: Recipe<E> + 'static;
    fn along(distance: TapDistance<E>) -> Self::Axis;
}

/// One factor per resampled axis, in the order the contraction walks the tap axes.
pub type SeparableWeights<E, F> = Factors<<F as SeparableFilter<E>>::Axis>;

/// Bridges host-side mode selection to the element type selected when a kernel is launched.
pub trait SeparableFilterFamily: Send + std::marker::Sync + 'static {
    type Filter<E: Float>: SeparableFilter<E>;
    const MODE: InterpolateMode;
    const NORMALIZATION: Option<(TapSupport, DivGuard)> = None;

    fn mode_properties() -> ModeProperties {
        mode_properties(Self::MODE)
    }

    fn radius() -> usize {
        (Self::mode_properties().taps - 1) / 2
    }
}

/// The weights of `F` over the output positions and taps of `partitioning`'s space, both axes
/// resampled as `row` and `col` map an output position to its source.
#[cube]
pub fn separable_weights<E: Float, F: SeparableFilterFamily>(
    partitioning: &Partitioning,
    #[comptime] row: Rational,
    #[comptime] col: Rational,
) -> Tile {
    let radius = comptime!(F::radius());
    let mut factors = Sequence::new();
    factors.push(F::Filter::<E>::along(tap_distance(
        TAP_H, OUTPUT_H, row, radius,
    )));
    factors.push(F::Filter::<E>::along(tap_distance(
        TAP_W, OUTPUT_W, col, radius,
    )));
    let weights = Procedural::separable::<E, SeparableWeights<E, F::Filter<E>>>(
        comptime!(
            partitioning
                .space()
                .subspace(&[BATCH, OUTPUT_H, OUTPUT_W, TAP_H, TAP_W])
        ),
        Factors::new(factors),
    );
    match comptime!(F::NORMALIZATION) {
        Some((taps, guard)) => weights.normalized(comptime!(taps), comptime!(guard)),
        None => weights,
    }
    .tile::<E>()
}

/// The distance from a tap to the source coordinate the output position lands on.
///
/// The rational is comptime because it is the same one the gather's projection already carries
/// ([`Rational::tap_axis`]), and the kernel is already compiled per shape. Handed as launch
/// scalars instead, its divisor reaches [`Phase`] as a runtime value: nothing is computed at
/// compile time, and every output position pays a true integer division for the residue and a
/// float division to normalize it, per resampled axis.
#[cube]
fn tap_distance<E: Float>(
    #[comptime] tap: Axis,
    #[comptime] output: Axis,
    #[comptime] source: Rational,
    #[comptime] radius: usize,
) -> TapDistance<E> {
    sum_of(
        affine_along(tap, E::new(-(radius as f64)), E::new(1.0)),
        Phase::<E> {
            coefficient: E::new(-1.0),
            numerator_scale: comptime!(source.scale as u32),
            numerator_offset: comptime!(source.offset as i32),
            divisor: comptime!(source.divisor as u32),
            axis: output,
        },
    )
}

pub struct NearestFilter;
impl SeparableFilterFamily for NearestFilter {
    type Filter<E: Float> = Self;
    const MODE: InterpolateMode = InterpolateMode::Nearest(crate::definition::NearestMode::Exact);
}
#[cube]
impl<E: Float> SeparableFilter<E> for NearestFilter {
    type Axis = NearestAxis<E>;
    fn along(_distance: TapDistance<E>) -> Self::Axis {
        NearestAxis::<E> { value: E::new(1.0) }
    }
}

pub struct BilinearFilter;
impl SeparableFilterFamily for BilinearFilter {
    type Filter<E: Float> = Self;
    const MODE: InterpolateMode = InterpolateMode::Bilinear;
}
#[cube]
impl<E: Float> SeparableFilter<E> for BilinearFilter {
    type Axis = BilinearAxis<E>;
    fn along(distance: TapDistance<E>) -> Self::Axis {
        BilinearAxis::<E> {
            coordinate: distance,
        }
    }
}

pub struct BicubicFilter;
impl SeparableFilterFamily for BicubicFilter {
    type Filter<E: Float> = Self;
    const MODE: InterpolateMode = InterpolateMode::Bicubic;
}
#[cube]
impl<E: Float> SeparableFilter<E> for BicubicFilter {
    type Axis = BicubicAxis<E>;
    fn along(distance: TapDistance<E>) -> Self::Axis {
        BicubicAxis::<E> {
            coordinate: distance,
            a: comptime!(Ratio::new(-3, 4)),
        }
    }
}

/// Six-tap Lanczos filtering.
pub struct Lanczos3Filter;
impl SeparableFilterFamily for Lanczos3Filter {
    type Filter<E: Float> = Self;
    const MODE: InterpolateMode = InterpolateMode::Lanczos3;
    const NORMALIZATION: Option<(TapSupport, DivGuard)> = Some((
        TapSupport::InBounds,
        DivGuard {
            epsilon: 1e-7,
            fallback: 0.0,
        },
    ));
}
#[cube]
impl<E: Float> SeparableFilter<E> for Lanczos3Filter {
    type Axis = Lanczos3Axis<E>;
    fn along(distance: TapDistance<E>) -> Self::Axis {
        Lanczos3Axis::<E> {
            coordinate: distance,
            lobes: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_mode_properties<F: SeparableFilterFamily>(mode: InterpolateMode) {
        assert_eq!(mode_properties(mode), F::mode_properties());
    }

    #[test]
    fn filters_use_the_mode_properties() {
        assert_mode_properties::<NearestFilter>(InterpolateMode::Nearest(
            crate::definition::NearestMode::Exact,
        ));
        assert_mode_properties::<BilinearFilter>(InterpolateMode::Bilinear);
        assert_mode_properties::<BicubicFilter>(InterpolateMode::Bicubic);
        assert_mode_properties::<Lanczos3Filter>(InterpolateMode::Lanczos3);
    }
}
