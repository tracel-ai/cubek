use super::CubeStage;
use cubecl::tile::launch::{Destination, DestinationLaunch};
use cubecl::tile::{Partitioning, Region, StageStorage, Tile, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube, prelude::*};

/// A cube's window of a producer's output, and where the cube's owners write their lines: the
/// window itself, or a stage of the cube's box that [`finish`](Self::finish) encodes into the
/// window once every owner has written its run.
///
/// The stage is the cube's alone, laid out in rows over its box, so an owner's run of it starts
/// where the owners before it end: the offset an owner hands [`owner`](Self::owner).
#[derive(CubeType)]
pub struct CubeStore {
    out: Tile,
    stage: Tile,
    #[cube(comptime)]
    form: CubeStage,
}

#[cube]
impl CubeStore {
    /// The store of `out`, the cube's window of the output, its lines `N` elements of `E` wide,
    /// written as `form` says.
    pub fn new<E: Float, N: Size>(out: Tile, #[comptime] form: CubeStage) -> CubeStore {
        let stage = match comptime!(form.clone()) {
            CubeStage::Direct => out.clone(),
            CubeStage::Encoded { cube_box } => {
                Tile::smem::<E>(cube_box, N::value(), StageStorage::Strided, 0usize)
            }
        };
        CubeStore { out, stage, form }
    }

    /// Where the owner of `region` writes its lines: its window of the output, or its run of the
    /// stage, which starts `first` lines into it.
    pub fn owner(&self, region: &Region, first: usize) -> OwnerLines {
        match comptime!(self.form.clone()) {
            CubeStage::Direct => OwnerLines::window(self.out.at(region)),
            CubeStage::Encoded { .. } => OwnerLines {
                tile: self.stage.clone(),
                first,
            },
        }
    }

    /// Once every owner has written its lines: nothing more, or the cube meets and encodes its
    /// stage into its window, calibrating each block's scale from its largest magnitude.
    pub fn finish(&mut self) {
        if comptime!(matches!(self.form, CubeStage::Encoded { .. })) {
            sync_cube();
            self.out.quantize_from(&self.stage);
        }
    }
}

/// Where one owner writes the lines of its box, in the order its box lays them out: a window it
/// writes at its own positions, or a run of a cube's stage starting `first` lines in.
#[derive(CubeType)]
pub struct OwnerLines {
    tile: Tile,
    first: usize,
}

#[cube]
impl OwnerLines {
    /// The lines of `window`, an owner's window of an output, written where they lie.
    pub fn window(window: Tile) -> OwnerLines {
        OwnerLines {
            tile: window,
            first: 0usize,
        }
    }

    /// Line `line` of the owner's box written as `value`.
    pub fn write<E: Float, N: Size>(&mut self, line: usize, value: Vector<E, N>) {
        let mut lines = self.tile.flat_mut::<E, N>();
        lines.write(self.first + line, value);
    }
}

/// A packed output a producer's cubes encode their rows into ([`CubeStore`]): bound as one tile of
/// its words under their block scales, as a product reads its packed lhs.
pub struct Encoded;

#[cube]
impl Destination for Encoded {
    type Arg<E: Numeric> = TileTensor<'static>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        arg.tile(partitioning)
    }
}

impl DestinationLaunch for Encoded {
    type Operand = TileTensorLaunch;

    fn arg<E: Numeric>(operand: TileTensorLaunch) -> TileTensorLaunch {
        operand
    }
}
