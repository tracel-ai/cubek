//! Where a [`Projection`]'s runtime coefficients sit once they are packed for launch.
//!
//! A projection's scales, offsets and divisors are each [`Static`](Scale::Static) or
//! [`Dynamic`](Scale::Dynamic). The dynamic ones travel to the kernel in two flat carriers, one
//! unsigned (coefficients and divisors) and one signed (offsets), and every reader has to agree
//! on the order: the projection's own, physical axis major, terms within an axis, each axis's
//! divisor after its terms. That agreement is what this file is; the launch fills the carriers by
//! walking the maps in order, and the kernel indexes them with the same functions.

use super::Projection;

impl Projection {
    /// Where physical axis `pa`'s term `t` sits in the runtime coefficient carrier, or `None` when
    /// it is [`Static`](Scale::Static). The order is the projection's own, physical axis major and
    /// term order within, each axis's [`Dynamic`](Divisor::Dynamic) divisor last, so a caller fills
    /// the carrier by walking the maps in order.
    pub fn dynamic_scale_index(&self, pa: usize, t: usize) -> Option<usize> {
        if !self.physical_axis(pa).terms()[t].scale.is_dynamic() {
            return None;
        }
        let within = self.physical_axis(pa).terms()[..t]
            .iter()
            .filter(|term| term.scale.is_dynamic())
            .count();
        Some(self.coefficient_base(pa) + within)
    }

    /// Where physical axis `pa`'s divisor sits in the runtime coefficient carrier, or `None` when
    /// it is [`Static`](Divisor::Static). Divisors share the carrier with coefficients: both are
    /// unsigned values of the same combination, and an axis's divisor follows its own terms.
    pub fn dynamic_divisor_index(&self, pa: usize) -> Option<usize> {
        if !self.physical_axis(pa).divisor().is_dynamic() {
            return None;
        }
        Some(self.coefficient_base(pa) + self.physical_axis(pa).dynamic_scale_count())
    }

    /// Where physical axis `pa`'s entries start in the runtime coefficient carrier; at the physical
    /// rank, the carrier's whole length.
    fn coefficient_base(&self, pa: usize) -> usize {
        (0..pa)
            .map(|i| self.physical_axis(i))
            .map(|m| m.dynamic_scale_count() + m.divisor().is_dynamic() as usize)
            .sum()
    }

    /// The length of the runtime coefficient carrier: every [`Dynamic`](Scale::Dynamic) coefficient
    /// and every [`Dynamic`](Divisor::Dynamic) divisor.
    pub(crate) fn dynamic_coefficient_count(&self) -> usize {
        self.coefficient_base(self.physical_rank())
    }

    /// Whether any physical axis's divisor is only known at runtime.
    pub(crate) fn has_dynamic_divisors(&self) -> bool {
        self.axis_maps().any(|m| m.divisor().is_dynamic())
    }

    /// Where physical axis `pa`'s offset sits in the runtime offset carrier, or `None` when it is
    /// [`Static`](Offset::Static). Offsets ride their own signed carrier, so this order is
    /// independent of [`dynamic_scale_index`](Self::dynamic_scale_index)'s.
    pub(crate) fn dynamic_offset_index(&self, pa: usize) -> Option<usize> {
        if !self.physical_axis(pa).offset().is_dynamic() {
            return None;
        }
        Some(
            (0..pa)
                .map(|i| self.physical_axis(i))
                .filter(|m| m.offset().is_dynamic())
                .count(),
        )
    }

    /// How many offsets are [`Dynamic`](Offset::Dynamic): the length of the offset carrier.
    pub(crate) fn dynamic_offset_count(&self) -> usize {
        self.axis_maps().filter(|m| m.offset().is_dynamic()).count()
    }

    /// Whether any coefficient, offset or divisor is only known at runtime.
    pub(crate) fn has_dynamic(&self) -> bool {
        self.has_dynamic_scales() || self.dynamic_offset_count() > 0 || self.has_dynamic_divisors()
    }

    /// Whether any coefficient is only known at runtime.
    pub(crate) fn has_dynamic_scales(&self) -> bool {
        self.axis_maps().any(|m| m.has_dynamic_scale())
    }
}
