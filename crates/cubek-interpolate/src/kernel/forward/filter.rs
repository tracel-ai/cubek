use crate::definition::{InterpolateMode, ModeProperties, mode_properties};
use cubecl::prelude::*;
use cubecl_common::Ratio;
use cubek_tile::{
    AffineCoordinate, Constant, Cubic, DivGuard, Lanczos, Linear, Phase, Recipe, SeparableProduct,
    Sum, TapMask,
};

pub type TapDistance<E> = Sum<AffineCoordinate<E>, Phase<E>>;
type NearestAxis<E> = Constant<E>;
type BilinearAxis<E> = Linear<TapDistance<E>>;
type BicubicAxis<E> = Cubic<TapDistance<E>>;
type Lanczos3Axis<E> = Lanczos<TapDistance<E>>;

#[cube]
pub trait SeparableFilter<E: Float>: Send + std::marker::Sync + 'static {
    type Axis: Recipe<E> + 'static;
    fn along(distance: TapDistance<E>) -> Self::Axis;
}

/// One factor per resampled axis, in the order the contraction walks the tap axes.
pub type SeparableWeights<E, F> = SeparableProduct<<F as SeparableFilter<E>>::Axis>;

/// Bridges host-side mode selection to the element type selected when a kernel is launched.
pub trait SeparableFilterFamily: Send + std::marker::Sync + 'static {
    type Filter<E: Float>: SeparableFilter<E>;
    const MODE: InterpolateMode;
    const NORMALIZATION: Option<(TapMask, DivGuard)> = None;

    fn mode_properties() -> ModeProperties {
        mode_properties(Self::MODE)
    }

    fn radius() -> usize {
        (Self::mode_properties().taps - 1) / 2
    }
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
        NearestAxis::<E> {
            value: E::new(1.0_f32),
        }
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
    const NORMALIZATION: Option<(TapMask, DivGuard)> = Some((
        TapMask::Masked,
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
