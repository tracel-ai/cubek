//! A level-centric builder for a multi-level [`Space`]: [`Tiling::over`] declares the axes and
//! their extents, then one [`level`](Tiling::level) per decomposition states who works on which
//! axes ([`LevelCuts`]), coarse to fine. Each level is a [`Level`] the [`Walk`](crate::Walk)
//! consumes; no transpose.
//!
//! Geometry only. How a level is walked (its order, how deep its stages are buffered), where an
//! operand is materialized and what runs on the cells are the kernel's to write, level by level,
//! against this space.

use crate::{Axis, Extent, Space};

use super::{Level, LevelCuts};

/// Builds a [`Space`] one level at a time. Start with [`over`](Tiling::over) (static extents)
/// or [`axes`](Tiling::axes) (extents resolved in-kernel), add levels with
/// [`level`](Tiling::level), then [`build`](Tiling::build).
pub struct Tiling {
    extents: Vec<(Axis, Extent)>,
    levels: Vec<Level>,
}

impl Tiling {
    /// Declare every axis and its top extent, fixing the canonical axis order; cuts may come in
    /// any order and are realigned to it.
    pub fn over(extents: &[(Axis, usize)]) -> Tiling {
        Tiling {
            extents: extents
                .iter()
                .map(|&(axis, extent)| (axis, Extent::Static(extent)))
                .collect(),
            levels: Vec::new(),
        }
    }

    /// [`over`](Tiling::over) with every top extent [`Dynamic`](Extent::Dynamic): the kernel
    /// form, resolved in-kernel from the tensors, so one compiled kernel serves every shape. The
    /// launch stamps the real extents back on with [`Space::with_extents`].
    pub fn axes(axes: &[Axis]) -> Tiling {
        Tiling {
            extents: axes.iter().map(|&axis| (axis, Extent::Dynamic)).collect(),
            levels: Vec::new(),
        }
    }

    /// Add a decomposition level (coarse to fine): `f` states who works on which axes, on the
    /// collector ([`Level::cuts`]). A level that cuts nothing is one region, and is kept: the
    /// kernel walks what was stated, level for level.
    pub fn level(mut self, f: impl FnOnce(&mut LevelCuts)) -> Self {
        let axes: Vec<Axis> = self.extents.iter().map(|&(a, _)| a).collect();
        self.levels.push(Level::cuts(&axes, f));
        self
    }

    /// Build the [`Space`]: the extents and the stack of levels.
    pub fn build(self) -> Space {
        let mut space = Space::from_extents(&self.extents);
        for level in self.levels {
            space = space.with_level(level);
        }
        space
    }
}
