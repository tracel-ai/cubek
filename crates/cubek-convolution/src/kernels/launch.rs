use cubek_matmul::launch::{AcceleratedTileKind, ReadingStrategy};

#[derive(Clone)]
pub struct ConvolutionArgs<const N_SPATIAL: usize> {
    pub stride: [usize; N_SPATIAL],
    pub padding: [usize; N_SPATIAL],
    pub dilation: [usize; N_SPATIAL],
}

pub enum Strategy {
    Simple {
        read_strategy: ReadingStrategy,
        tile_kind: AcceleratedTileKind,
    },
}
