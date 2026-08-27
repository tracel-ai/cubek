use crate::definition::Transform;
use cubek_tile::{Axis, PhysicalAxisMap};

/// A source coordinate `(output * scale + offset) / divisor`, in lowest terms.
#[derive(Clone, Copy)]
pub struct Rational {
    pub scale: usize,
    pub offset: isize,
    pub divisor: usize,
}

impl Rational {
    pub fn of(transform: Transform) -> Self {
        let offset_denominator = transform.offset_denominator;
        assert!(offset_denominator > 0, "a source coordinate runs forwards");
        let offset_denominator = offset_denominator as usize;
        let scale = transform.scale_numerator * offset_denominator;
        let offset = transform.offset_numerator * transform.scale_denominator as isize;
        let divisor = transform.scale_denominator * offset_denominator;
        let common = gcd(gcd(scale, divisor), offset.unsigned_abs());
        Self {
            scale: scale / common,
            offset: offset / common as isize,
            divisor: divisor / common,
        }
    }

    /// Map a tap range whose zero is `radius` samples before `floor(coordinate)`.
    ///
    /// When `scale` is 0 (e.g., input extent of 1 under `align_corners`), the source coordinate
    /// does not move with the output. The output term is omitted since [`PhysicalAxisMap`] rejects
    /// zero coefficients.
    pub fn tap_axis(self, output: Axis, tap: Axis, radius: usize) -> PhysicalAxisMap {
        let moving = [(output, self.scale), (tap, self.divisor)];
        let terms: &[(Axis, usize)] = if self.scale == 0 {
            &moving[1..]
        } else {
            &moving
        };
        let offset = self.offset - (radius * self.divisor) as isize;

        PhysicalAxisMap::affine_with_offset(terms, offset).over(self.divisor)
    }
}

pub fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}
