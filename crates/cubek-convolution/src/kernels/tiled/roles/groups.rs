use crate::components::ConvolutionOperation;
use crate::kernels::tiled::{ConvDecline, G};
use cubecl::tile::Dimension;
use cubek_microkernel::matmul::{MatmulOperand, MatmulOperands};

/// Whether the channels are split into groups, each a convolution of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvGrouping {
    Whole,
    /// More than one group, as many as the count holds.
    Grouped(usize),
}

impl ConvGrouping {
    /// The grouping of a convolution of `groups` groups. A count of zero never reaches it:
    /// [`ConvChannelOperands::groups`] declines the operands that state one.
    pub fn of(groups: usize) -> Self {
        match groups {
            1 => Self::Whole,
            _ => Self::Grouped(groups),
        }
    }

    /// The dimensions this grouping adds ahead of the free ones: the group, where there is one.
    pub fn dimensions(self) -> &'static [Dimension] {
        match self {
            Self::Whole => &[],
            Self::Grouped(_) => &[G],
        }
    }
}

/// Where a tensor's channel axis is, the one a split by group cuts in two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAxis {
    /// A weight's filters, or a bias.
    First,
    /// An activation's channels.
    Last,
}

/// Which operand of an operation holds the input channels, which the output channels and which is
/// the weight (or its gradient): where the group count is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvChannelOperands {
    /// The activation whose channels are the input's.
    pub input: MatmulOperand,
    /// The activation whose channels are the output's.
    pub output: MatmulOperand,
    /// The weight or its gradient, whose first dim is the filters and whose last is the input
    /// channels of one group.
    pub weight: MatmulOperand,
}

/// The channel counts a group count is read from.
struct GroupChannels {
    input: usize,
    output: usize,
    filters: usize,
    per_group: usize,
}

/// What the channel counts of a convolution state about its groups, the cases in the order they
/// are told apart.
#[derive(Debug, PartialEq, Eq)]
enum GroupSplit {
    /// No input channels and none per group: nothing to split.
    Empty,
    /// Input channels that are none, or not a whole number of the weight's, or a weight of none.
    Indivisible,
    /// Filters that the groups do not divide.
    FiltersIndivisible,
    /// Output channels that the groups do not divide.
    OutputIndivisible,
    /// Groups of one input and one output channel each.
    Depthwise,
    /// This many groups, one or more.
    Grouped(usize),
}

impl GroupSplit {
    fn of(channels: &GroupChannels) -> Self {
        let GroupChannels {
            input,
            output,
            filters,
            per_group,
        } = *channels;
        if per_group == 0 && input == 0 {
            return Self::Empty;
        }
        if per_group == 0 || input == 0 || input % per_group != 0 {
            return Self::Indivisible;
        }
        let groups = input / per_group;
        if filters % groups != 0 {
            return Self::FiltersIndivisible;
        }
        if output % groups != 0 {
            return Self::OutputIndivisible;
        }
        if groups > 1 && per_group == 1 && filters == groups {
            return Self::Depthwise;
        }
        Self::Grouped(groups)
    }
}

impl ConvChannelOperands {
    /// Where `operation` keeps its channels.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::Transposed`], for a transposed forward.
    pub fn of(operation: ConvolutionOperation) -> Result<Self, ConvDecline> {
        use MatmulOperand::{Lhs, Out, Rhs};
        match operation {
            ConvolutionOperation::Forward => Ok(Self {
                input: Lhs,
                output: Out,
                weight: Rhs,
            }),
            ConvolutionOperation::BackwardWeight => Ok(Self {
                input: Lhs,
                output: Rhs,
                weight: Out,
            }),
            ConvolutionOperation::BackwardData => Ok(Self {
                input: Out,
                output: Lhs,
                weight: Rhs,
            }),
            ConvolutionOperation::ForwardTransposed => Err(ConvDecline::Transposed),
        }
    }

    /// Where `operand`'s channel axis is.
    pub fn axis(&self, operand: MatmulOperand) -> ChannelAxis {
        match operand == self.weight {
            true => ChannelAxis::First,
            false => ChannelAxis::Last,
        }
    }

    /// The group count `shapes`, the operands in their own layouts, state: the input channels over
    /// the weight's channels per group.
    ///
    /// # Errors
    ///
    /// [`ConvDecline::Groups`], where the input channels or the weight's filters are not whole
    /// groups of the weight's input channels. [`ConvDecline::OutChannels`], where the output's
    /// channels do not split into those groups. [`ConvDecline::Depthwise`], for a group of one input
    /// and one output channel.
    pub fn groups(&self, shapes: &MatmulOperands<&[usize]>) -> Result<usize, ConvDecline> {
        let channels = |operand| {
            shapes
                .get_ref(operand)
                .last()
                .copied()
                .expect("an operand has a channels dim")
        };
        let counts = GroupChannels {
            input: channels(self.input),
            output: channels(self.output),
            filters: shapes.get_ref(self.weight)[0],
            per_group: channels(self.weight),
        };
        match GroupSplit::of(&counts) {
            GroupSplit::Empty => Ok(1),
            GroupSplit::Indivisible | GroupSplit::FiltersIndivisible => Err(ConvDecline::Groups),
            GroupSplit::OutputIndivisible => Err(ConvDecline::OutChannels),
            GroupSplit::Depthwise => Err(ConvDecline::Depthwise),
            GroupSplit::Grouped(groups) => Ok(groups),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ConvolutionOperation::{BackwardWeight, Forward};

    fn counts(input: usize, output: usize, filters: usize, per_group: usize) -> GroupChannels {
        GroupChannels {
            input,
            output,
            filters,
            per_group,
        }
    }

    #[test]
    fn group_split_tells_each_case_apart() {
        let cases = [
            (counts(0, 8, 8, 0), GroupSplit::Empty),
            (counts(0, 8, 8, 4), GroupSplit::Indivisible),
            (counts(8, 8, 8, 0), GroupSplit::Indivisible),
            (counts(10, 8, 8, 4), GroupSplit::Indivisible),
            (counts(8, 8, 9, 4), GroupSplit::FiltersIndivisible),
            (counts(8, 9, 8, 4), GroupSplit::OutputIndivisible),
            (counts(8, 8, 8, 1), GroupSplit::Depthwise),
            (counts(8, 16, 16, 1), GroupSplit::Grouped(8)),
            (counts(8, 8, 8, 8), GroupSplit::Grouped(1)),
            (counts(8, 6, 6, 4), GroupSplit::Grouped(2)),
        ];
        for (channels, expected) in cases {
            assert_eq!(GroupSplit::of(&channels), expected);
        }
    }

    #[test]
    fn groups_read_from_the_operands_in_their_own_layouts() {
        let forward = ConvChannelOperands::of(Forward).expect("a served operation");
        let shapes = |lhs, rhs, out| MatmulOperands { lhs, rhs, out };
        let grouped = shapes(&[1, 6, 6, 64][..], &[32, 3, 3, 16][..], &[1, 6, 6, 32][..]);
        assert_eq!(forward.groups(&grouped), Ok(4));
        let whole = shapes(&[1, 6, 6, 64][..], &[32, 3, 3, 64][..], &[1, 6, 6, 32][..]);
        assert_eq!(forward.groups(&whole), Ok(1));
        let weight = ConvChannelOperands::of(BackwardWeight).expect("a served operation");
        let grad = shapes(&[1, 6, 6, 64][..], &[1, 6, 6, 32][..], &[32, 3, 3, 16][..]);
        assert_eq!(weight.groups(&grad), Ok(4));
    }

    #[test]
    fn groups_decline_by_name() {
        let forward = ConvChannelOperands::of(Forward).expect("a served operation");
        let declined = |lhs: &[usize], rhs: &[usize], out: &[usize]| {
            forward.groups(&MatmulOperands { lhs, rhs, out })
        };
        assert_eq!(
            declined(&[1, 6, 6, 64], &[32, 3, 3, 24], &[1, 6, 6, 32]),
            Err(ConvDecline::Groups)
        );
        assert_eq!(
            declined(&[1, 6, 6, 64], &[30, 3, 3, 16], &[1, 6, 6, 30]),
            Err(ConvDecline::Groups)
        );
        assert_eq!(
            declined(&[1, 6, 6, 64], &[32, 3, 3, 16], &[1, 6, 6, 30]),
            Err(ConvDecline::OutChannels)
        );
        assert_eq!(
            declined(&[1, 6, 6, 16], &[16, 3, 3, 1], &[1, 6, 6, 16]),
            Err(ConvDecline::Depthwise)
        );
    }

    #[test]
    fn a_group_count_names_its_grouping() {
        assert_eq!(ConvGrouping::of(1), ConvGrouping::Whole);
        assert_eq!(ConvGrouping::of(3), ConvGrouping::Grouped(3));
    }
}
