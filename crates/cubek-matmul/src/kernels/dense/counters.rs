use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::BufferArg;
use cubecl::std::tensor::TensorHandle;

/// Counters the cubes of a launch meet through, one `u32` each, holding zero: a relay's turns
/// ([`Partitioning::relay_counters`](cubek_tile::Partitioning::relay_counters)), or a merge's
/// arrivals ([`Partitioning::arrival_counters`](cubek_tile::Partitioning::arrival_counters)).
/// The cubes leave them holding zero, so they serve the next launch as they arrived.
pub struct ZeroedCounters {
    counters: TensorHandle,
}

impl ZeroedCounters {
    /// `count` zeroed counters on `client`'s device.
    pub fn new(count: usize, client: &Client) -> Self {
        let counters = TensorHandle::zeros(client, [count], ElemType::UInt(UIntKind::U32));
        Self { counters }
    }

    /// The counters as a kernel's argument.
    pub fn into_arg(self) -> BufferArg {
        self.counters.into_arg().into_buffer_arg()
    }
}
