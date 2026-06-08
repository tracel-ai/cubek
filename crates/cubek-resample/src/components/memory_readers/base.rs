use crate::components::{AccessPattern, NdLayout};
use cubecl::{prelude::*, std::tensor::layout::Coordinates};

#[cube]
pub trait MemoryReader<C: Numeric, P: AccessPattern>: CubeType {
    type Coords: Coordinates;

    fn init(out_coord: Self::Coords, args: &P) -> Self;
    fn read_next(reader: &mut Self, input: &Tensor<C>, in_layout: &NdLayout) -> Option<(C, C)>;
}

#[derive(CubeType)]
pub struct GlobalReader<C: Numeric, P: AccessPattern> {
    pub tap_idx: u32,
    pub num_taps: u32,
    pub out_coord: P::Coord,
    pub args: P,

    #[cube(comptime)]
    pub _type_marker: core::marker::PhantomData<C>,
}

#[cube]
impl<C: Numeric, P: AccessPattern> MemoryReader<C, P> for GlobalReader<C, P> {
    type Coords = P::Coord;

    fn init(out_coord: Self::Coords, args: &P) -> Self {
        Self {
            tap_idx: 0,
            num_taps: P::footprint_size(args, ()),
            out_coord,
            args: *args,
            _type_marker: core::marker::PhantomData,
        }
    }

    fn read_next(reader: &mut Self, input: &Tensor<C>, in_layout: &NdLayout) -> Option<(C, C)> {
        if reader.tap_idx >= reader.num_taps {
            None
        } else {
            let in_coord = P::map_coord(reader.out_coord.clone(), reader.tap_idx, &reader.args, ());
            let in_idx = in_layout.to_source_pos(in_coord);

            let x = if in_idx < input.len() {
                input[in_idx]
            } else {
                C::cast_from(0)
            };

            let w: C = P::eval_weight(reader.out_coord.clone(), reader.tap_idx, &reader.args, ());

            reader.tap_idx += 1;
            Some((x, w))
        }
    }
}
