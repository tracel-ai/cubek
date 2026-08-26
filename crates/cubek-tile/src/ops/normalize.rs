//! Normalization policy and helpers shared by procedural filters.

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// How division handles a denominator whose magnitude is too small to divide by. Both fields are
/// comptime kernel constants. The fallback is the reciprocal multiplier, so the default maps a
/// guarded result to zero.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DivGuard {
    pub epsilon: f32,
    pub fallback: f32,
}

pub(crate) fn validate_guard(guard: DivGuard) {
    assert!(
        guard.epsilon.is_finite() && guard.epsilon >= 0.0,
        "DivGuard: epsilon must be finite and non-negative"
    );
    assert!(
        guard.fallback.is_finite(),
        "DivGuard: fallback must be finite"
    );
}

/// Which taps contribute to the sum of a normalized separable filter factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TapMask {
    /// Sum only taps whose projected input sample is in bounds, removing edge darkening. The
    /// contraction must read the original source window in place; a shared-memory stage no longer
    /// records which staged zeros came from outside that window and is rejected when launching.
    #[default]
    Masked,
    /// Sum the full mathematical support, preserving the fade of zero padding at an edge.
    Unmasked,
}

/// A guarded reciprocal that preserves the sign of a valid denominator. Substitution keeps the
/// discarded division finite even though `select` evaluates both arms; NaN fails the comparison
/// and takes the fallback.
///
/// The contraction remains generic over numeric weights for unnormalized recipes. Only the
/// float-only public normalization surface can set the flag that reaches this helper.
#[cube]
pub(crate) fn guarded_recip_numeric<E: Numeric>(d: E, #[comptime] guard: DivGuard) -> E {
    let epsilon = E::cast_from(comptime!(guard.epsilon));
    let fallback = E::cast_from(comptime!(guard.fallback));
    let valid = d.abs() > epsilon;
    let safe = select(valid, d, E::from_int(1));
    select(valid, E::from_int(1) / safe, fallback)
}

impl<T: Float> Tile<T> {
    /// Normalize a separable procedural tile's factor runs where the gather contraction evaluates
    /// them. This is deliberately refused for opaque recipes and backed tiles: silently routing
    /// either through a post-pass would conceal an extra contraction or memory walk. A masked
    /// normalization also requires the contraction's rhs to remain at its original source window;
    /// staging that rhs in shared memory is rejected rather than adding boundary tracking to the
    /// normal staging path.
    pub fn normalized(self, _mask: TapMask, _guard: DivGuard) -> Tile<T> {
        unexpanded!()
    }
}

impl<T: Float> TileExpand<T> {
    pub fn __expand_normalized_method(
        mut self,
        scope: &Scope,
        mask: TapMask,
        guard: DivGuard,
    ) -> TileExpand<T> {
        validate_guard(guard);
        match &mut self.tile_kind {
            TileKindExpand::Procedural(data) => {
                assert!(
                    data.factor_count(scope).is_some(),
                    "Tile::normalized: the procedural recipe states no separable factorization"
                );
                data.normalization = Some((mask, guard, data.space.clone()));
            }
            TileKindExpand::Gmem(_)
            | TileKindExpand::Smem(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_) => {
                panic!("Tile::normalized: only a separable procedural tile has factor runs")
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_guard_accepts_finite_non_negative_thresholds() {
        validate_guard(DivGuard::default());
        validate_guard(DivGuard {
            epsilon: 1.0e-7,
            fallback: -1.0,
        });
    }

    #[test]
    fn division_guard_rejects_invalid_thresholds() {
        for epsilon in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| validate_guard(DivGuard {
                    epsilon,
                    fallback: 0.0,
                }))
                .is_err(),
                "epsilon {epsilon:?} should be rejected"
            );
        }
        for fallback in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| validate_guard(DivGuard {
                    epsilon: 1.0e-7,
                    fallback,
                }))
                .is_err(),
                "fallback {fallback:?} should be rejected"
            );
        }
    }

    fn test_scope() -> Scope {
        Scope::root(cubecl::ir::settings::KernelSettings::new(
            cubecl::ir::settings::Dim3::new_single(),
            cubecl::ir::settings::ExecutionMode::Checked,
            cubecl::ir::AddressType::U32,
        ))
    }

    #[test]
    #[should_panic(expected = "the procedural recipe states no separable factorization")]
    fn normalized_rejects_an_opaque_procedural_recipe() {
        let scope = test_scope();
        let tile = Tile::<f32>::__expand_zeros(&scope, Space::new(&[(Axis(0), 4)]));
        tile.__expand_normalized_method(&scope, TapMask::Unmasked, DivGuard::default());
    }

    #[test]
    #[should_panic(expected = "only a separable procedural tile has factor runs")]
    fn normalized_rejects_a_non_procedural_tile() {
        let scope = test_scope();
        let plane_tile = PlaneTile::<f32>::__expand_acc(
            &scope,
            Instruction::Cmma,
            8,
            8,
            8,
            1,
            LaneShare::Whole,
            Monoid::Sum,
        );
        let tile = TileExpand::<f32> {
            tile_kind: TileKindExpand::PlaneTile(plane_tile),
            space: Space::new(&[(Axis(0), 4)]),
        };
        tile.__expand_normalized_method(&scope, TapMask::Unmasked, DivGuard::default());
    }
}
