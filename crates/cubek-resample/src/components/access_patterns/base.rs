use cubecl::{prelude::*, std::tensor::layout::Coordinates};

#[cube]
pub trait AccessPattern: CubeType {
    type Coord: Coordinates;

    fn footprint_size(args: &Self, #[comptime] config: ()) -> u32;
    fn map_coord(
        out_coord: Self::Coord,
        tap_idx: u32,
        args: &Self,
        #[comptime] config: (),
    ) -> Self::Coord;
    fn eval_weight<C: Numeric>(
        out_coord: Self::Coord,
        tap_idx: u32,
        args: &Self,
        #[comptime] config: (),
    ) -> C;
}

#[derive(CubeType)]
pub struct ReduceAxisPattern<Coord: Coordinates> {
    pub reduce_axis: u32,
    pub reduce_size: u32,

    #[cube(comptime)]
    pub _coord_marker: core::marker::PhantomData<Coord>,
}

#[cube]
impl<Coord: Coordinates> AccessPattern for ReduceAxisPattern<Coord> {
    type Coord = Coord;

    fn footprint_size(args: &Self, #[comptime] _config: ()) -> u32 {
        args.reduce_size
    }

    fn map_coord(out_coord: Coord, _tap_idx: u32, _args: &Self, #[comptime] _config: ()) -> Coord {
        out_coord
    }

    fn eval_weight<C: Numeric>(
        _out_coord: Coord,
        _tap_idx: u32,
        _args: &Self,
        #[comptime] _config: (),
    ) -> C {
        C::from_int(1)
    }
}
