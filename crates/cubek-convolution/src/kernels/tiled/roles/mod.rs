mod base;
mod dimensions;
mod extents;
mod gather;
mod groups;
mod views;

pub use base::ConvRoles;
pub use gather::ConvGather;
pub use groups::{ChannelAxis, ConvChannelOperands, ConvGrouping};
pub use views::ViewOrder;
