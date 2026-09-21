use cubecl::{
    ir::{FloatKind, VectorRegisters},
    prelude::*,
};

/// A loop that folds one running value per vector lane, and the vector sizes worth building it at.
///
/// A fold is latency-bound, so each register its vectors span adds an independent chain, until
/// the vectors a step keeps live outnumber the device's registers and every use becomes a load.
#[derive(Debug, Clone, Copy)]
pub struct Accumulation<'a> {
    /// The element the loop loads, whose load widths stand in on a device with no register set.
    pub load: ElemType,
    /// The element types among the vectors a step keeps live.
    pub live_elems: &'a [ElemType],
    /// How many vectors one step keeps live at its peak.
    pub live_vectors: usize,
}

impl Accumulation<'_> {
    /// Vector sizes, widest first, at which every live vector stays in a register.
    pub fn vector_sizes(
        &self,
        client: &Client,
    ) -> impl Iterator<Item = VectorSize> + Clone + use<> {
        let hardware = &client.properties().hardware;
        let widest = match VectorRegisters::of(hardware, self.lane_size()) {
            Some(registers) => registers.widest_lanes(self.live_vectors),
            None => client
                .io_optimized_vector_sizes(self.load.size())
                .max()
                .unwrap_or(1),
        };

        (0..=widest.ilog2()).rev().map(|power| 1 << power)
    }

    fn lane_size(&self) -> usize {
        self.live_elems
            .iter()
            .map(|elem| match elem {
                // A host without half arithmetic evaluates a half float in f32 registers.
                ElemType::Float(FloatKind::F16 | FloatKind::BF16) => size_of::<f32>(),
                elem => elem.size(),
            })
            .max()
            .unwrap_or(self.load.size())
    }
}
