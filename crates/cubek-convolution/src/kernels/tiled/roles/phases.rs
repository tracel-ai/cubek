use crate::components::ConvolutionOperation;
use crate::kernels::tiled::ConvDimension;
use crate::launch::SpatialWindow;
use cubecl::tile::Dimension;

/// Whether the strided spatial dims of a data backward are split into phases: the input gradient
/// at stride `s` is `s` stride-1 convolutions of the output gradient, one for each residue of the
/// tap, written to interleaved cells. A phase is a dimension of its own, free like the group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvPhasing {
    /// No dim is split into phases.
    Whole,
    /// The stride of each spatial dim, 1 for a dim that has no phases; at least one is above 1.
    Strided(Vec<usize>),
}

impl ConvPhasing {
    /// The phasing of `operation` over `windows`, one for each spatial dim: only a data backward
    /// reads the output gradient at fewer cells than it writes.
    pub fn of(
        operation: ConvolutionOperation,
        windows: impl IntoIterator<Item = SpatialWindow>,
    ) -> Self {
        let strides: Vec<usize> = windows.into_iter().map(|window| window.stride).collect();
        let strided = strides.iter().any(|&stride| stride > 1);
        match (operation, strided) {
            (ConvolutionOperation::BackwardData, true) => Self::Strided(strides),
            _ => Self::Whole,
        }
    }

    /// The stride of spatial dim `dim`, 1 where it has no phases.
    pub fn stride(&self, dim: usize) -> usize {
        match self {
            Self::Whole => 1,
            Self::Strided(strides) => strides[dim],
        }
    }

    /// Whether spatial dim `dim` is split into phases.
    pub fn is_phased(&self, dim: usize) -> bool {
        self.stride(dim) > 1
    }

    /// The phase dimension of spatial dim `dim`, where it is phased.
    pub fn phase(&self, dim: usize) -> Option<Dimension> {
        self.is_phased(dim)
            .then(|| ConvDimension::Phase(dim).dimension())
    }

    /// The phase dimensions, in the order of the spatial dims.
    pub fn dimensions(&self) -> Vec<Dimension> {
        match self {
            Self::Whole => Vec::new(),
            Self::Strided(strides) => (0..strides.len())
                .filter_map(|dim| self.phase(dim))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ConvolutionOperation::{BackwardData, BackwardWeight, Forward};

    fn windows(strides: &[usize]) -> Vec<SpatialWindow> {
        (strides.iter())
            .map(|&stride| SpatialWindow {
                stride,
                padding: 1,
                dilation: 1,
            })
            .collect()
    }

    #[test]
    fn only_a_strided_data_backward_is_phased() {
        let strided = windows(&[2, 1]);
        assert_eq!(
            ConvPhasing::of(BackwardData, strided.iter().copied()),
            ConvPhasing::Strided(vec![2, 1])
        );
        for operation in [Forward, BackwardWeight] {
            let phasing = ConvPhasing::of(operation, strided.iter().copied());
            assert_eq!(phasing, ConvPhasing::Whole, "{operation:?}");
        }
        let unit = windows(&[1, 1]);
        let phasing = ConvPhasing::of(BackwardData, unit.iter().copied());
        assert_eq!(phasing, ConvPhasing::Whole);
    }

    #[test]
    fn a_phase_dimension_exists_for_each_strided_dim_only() {
        let phasing = ConvPhasing::Strided(vec![2, 1, 3]);
        assert_eq!(
            phasing.dimensions(),
            [
                ConvDimension::Phase(0).dimension(),
                ConvDimension::Phase(2).dimension()
            ]
        );
        assert_eq!(phasing.phase(1), None);
        assert_eq!(phasing.stride(2), 3);
        assert_eq!(ConvPhasing::Whole.stride(5), 1);
        assert!(ConvPhasing::Whole.dimensions().is_empty());
    }
}
