//! The extents and strides an operand is addressed by, on the host.

use core::fmt::{self, Display, Formatter};

use cubecl::prelude::*;

/// One operand's physical extents and strides, in scalars, one entry per physical dim.
///
/// The two are one value because they are never separately true. Held apart they are two
/// `Vec<usize>` a caller can state at two ranks, and two same-typed arguments a call site can
/// swap in silence; here a dim is an `(extent, stride)` pair and neither is expressible. A bound
/// operand takes its geometry off its binding ([`From`]), an unbound one states it, and the
/// derivation reaches both the same way.
///
/// The kernel-side twin is [`RuntimeGeometry`](crate::RuntimeGeometry), which is what the
/// derivation's settled geometry is handed to a tile as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Geometry {
    /// One `(extent, stride)` per physical dim, coarsest first, both in scalars.
    pub fn of_dims(dims: &[(usize, usize)]) -> Self {
        Self {
            shape: dims.iter().map(|&(extent, _)| extent).collect(),
            strides: dims.iter().map(|&(_, stride)| stride).collect(),
        }
    }

    /// The extents, coarsest first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The strides that step them, in scalars.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// How many physical dims the operand has.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// The dims, coarsest first.
    pub fn dims(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.shape.iter().copied().zip(self.strides.iter().copied())
    }

    /// Whether this operand can be served in `vector_size`-wide lines.
    ///
    /// The kernel re-expresses the geometry in lines rather than scalars: [`Tile::of`] counts the
    /// innermost extent in lines and divides every coarser stride by the served width, so a width
    /// that does not divide them truncates — in bounds, no fault, addressing a fraction of the
    /// operand. [`Launcher::vector_size`](crate::Launcher::vector_size) reads this to *pick* a
    /// width; a caller that states one is *refused* by it. One answer, so the two cannot drift
    /// apart about what a servable width is.
    ///
    /// [`Tile::of`]: crate::Tile::of
    pub(crate) fn serves_lines(&self, vector_size: usize) -> Result<(), LineMisfit> {
        if vector_size == 1 {
            return Ok(());
        }
        if self.rank() == 0 {
            return Err(LineMisfit::NoDims);
        }
        let last = self.rank() - 1;
        if self.strides[last] != 1 {
            return Err(LineMisfit::InnermostStrided(self.strides[last]));
        }
        if !self.shape[last].is_multiple_of(vector_size) {
            return Err(LineMisfit::PartialLine(self.shape[last]));
        }
        match self.strides[..last]
            .iter()
            .find(|stride| !stride.is_multiple_of(vector_size))
        {
            Some(&stride) => Err(LineMisfit::StrideInsideLine(stride)),
            None => Ok(()),
        }
    }
}

/// Why a [`Geometry`] cannot be served at some width — the value that decided it, so a message
/// names the number a reader has to go looking for otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineMisfit {
    /// The innermost dim's own stride, when it is not 1: consecutive values are not one line.
    InnermostStrided(usize),
    /// The innermost extent, when it is not a whole number of lines.
    PartialLine(usize),
    /// A coarser stride, when re-expressing it as `stride / width` would land inside a line.
    StrideInsideLine(usize),
    /// No dims at all: there is no innermost extent to count in lines, so no width but `1`
    /// describes it. Carries nothing, because the misfit is the absence.
    NoDims,
}

impl Display for LineMisfit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InnermostStrided(stride) => write!(
                f,
                "its innermost dim steps by {stride} rather than 1, so consecutive values are \
                 not one line"
            ),
            Self::PartialLine(extent) => write!(
                f,
                "its innermost extent is {extent}, which is not a whole number of lines"
            ),
            Self::StrideInsideLine(stride) => write!(
                f,
                "its stride {stride} is not a whole number of lines, so a coarser step lands \
                 inside a line"
            ),
            Self::NoDims => write!(
                f,
                "it has no dims, so it has no innermost extent to count in lines"
            ),
        }
    }
}

impl<R: Runtime> From<&TensorBinding<R>> for Geometry {
    fn from(binding: &TensorBinding<R>) -> Self {
        Self {
            shape: binding.shape.to_vec(),
            strides: binding.strides.to_vec(),
        }
    }
}
