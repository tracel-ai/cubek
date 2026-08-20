use cubecl::prelude::*;
use cubecl_common::Ratio;
use cubek_tile::*;

pub type TapDistance<E> = Sum<AffineCoordinate<E>, Phase<E>>;
type NearestAxis<E> = Constant<E>;
type BilinearAxis<E> = Linear<TapDistance<E>>;
type BicubicAxis<E> = Cubic<TapDistance<E>>;
type Lanczos3Axis<E> = Lanczos<TapDistance<E>>;

pub trait TapSupport {
    const TAPS: usize;
    const BOUNDARY: Boundary;
    fn radius() -> usize {
        (Self::TAPS - 1) / 2
    }
}

#[cube]
pub trait SeparableFilter<E: Float>: TapSupport + Send + std::marker::Sync + 'static {
    type Axis: Recipe<E> + 'static;
    fn along(distance: TapDistance<E>) -> Self::Axis;
}

/// One factor per resampled axis, in the order the contraction walks the tap axes.
pub type SeparableWeights<E, F> = SeparableProduct<<F as SeparableFilter<E>>::Axis>;

/// Bridges host-side mode selection to the element type selected when a kernel is launched.
pub trait SeparableFilterFamily: TapSupport + Send + std::marker::Sync + 'static {
    type Filter<E: Float>: SeparableFilter<E>;
}

pub struct NearestFilter;
impl TapSupport for NearestFilter {
    const TAPS: usize = 1;
    const BOUNDARY: Boundary = Boundary::Clamp;
}
impl SeparableFilterFamily for NearestFilter {
    type Filter<E: Float> = Self;
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
impl TapSupport for BilinearFilter {
    const TAPS: usize = 2;
    const BOUNDARY: Boundary = Boundary::Clamp;
}
impl SeparableFilterFamily for BilinearFilter {
    type Filter<E: Float> = Self;
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
impl TapSupport for BicubicFilter {
    const TAPS: usize = 4;
    const BOUNDARY: Boundary = Boundary::Clamp;
}
impl SeparableFilterFamily for BicubicFilter {
    type Filter<E: Float> = Self;
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

/// Six-tap Lanczos filtering. At image edges this intentionally zero-pads without renormalizing
/// the surviving weights, so its border pixels differ from the CPU reference.
pub struct Lanczos3Filter;
impl TapSupport for Lanczos3Filter {
    const TAPS: usize = 6;
    const BOUNDARY: Boundary = Boundary::Zero;
}
impl SeparableFilterFamily for Lanczos3Filter {
    type Filter<E: Float> = Self;
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
