// use cubecl::{prelude::*, std::tensor::layout::*};

// /// Layout for tiled input with halo padding
// /// Maps tile coordinates (with halo) to global input coordinates
// #[derive(CubeType, Clone, Copy)]
// pub struct InputTiledLayout {
//     tile_h: u32,
//     tile_w: u32,
//     halo: u32,
//     input_h: u32,
//     input_w: u32,
// }

// #[cube]
// impl InputTiledLayout {
//     pub fn new(tile_h: usize, tile_w: usize, halo: usize, input_h: u32, input_w: u32) -> Self {
//         InputTiledLayout {
//             tile_h: tile_h as u32,
//             tile_w: tile_w as u32,
//             halo: halo as u32,
//             input_h,
//             input_w,
//         }
//     }
// }

// #[cube]
// impl Layout for InputTiledLayout {
//     type Coordinates = Coords2d;
//     type SourceCoordinates = Coords2d;

//     fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
//         let (row, col) = pos;

//         let global_y = row as i32 - (self.halo as i32 / 2);
//         let global_x = col as i32 - (self.halo as i32 / 2);

//         (global_y.max(0) as u32, global_x.max(0) as u32)
//     }

//     fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
//         let (row, col) = pos;
//         let global_y = row as i32 - (self.halo as i32 / 2);
//         let global_x = col as i32 - (self.halo as i32 / 2);

//         let valid = global_y >= 0
//             && global_y < self.input_h as i32
//             && global_x >= 0
//             && global_x < self.input_w as i32;

//         ((global_y.max(0) as u32, global_x.max(0) as u32), valid)
//     }

//     fn shape(&self) -> Self::Coordinates {
//         // Return the shape of the haloed tile
//         let haloed_h = self.tile_h + self.halo;
//         let haloed_w = self.tile_w + self.halo;
//         (haloed_h, haloed_w)
//     }

//     fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
//         let (row, col) = pos;
//         row < (self.tile_h + self.halo) && col < (self.tile_w + self.halo)
//     }
// }

// /// Layout for tiled output
// /// Maps output tile coordinates to global output coordinates
// #[derive(CubeType, Clone, Copy)]
// pub struct OutputTiledLayout {
//     tile_h: u32,
//     tile_w: u32,
//     output_h: u32,
//     output_w: u32,
// }

// #[cube]
// impl OutputTiledLayout {
//     pub fn new(tile_h: usize, tile_w: usize, output_h: u32, output_w: u32) -> Self {
//         OutputTiledLayout {
//             tile_h: tile_h as u32,
//             tile_w: tile_w as u32,
//             output_h,
//             output_w,
//         }
//     }
// }

// #[cube]
// impl Layout for OutputTiledLayout {
//     type Coordinates = Coords2d;
//     type SourceCoordinates = Coords2d;

//     fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
//         // Output tile coordinates are already in local tile space
//         // For a single tile kernel, they map 1:1 to output space
//         pos
//     }

//     fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
//         let (row, col) = pos;
//         let valid = row < self.tile_h && col < self.tile_w;
//         (pos, valid)
//     }

//     fn shape(&self) -> Self::Coordinates {
//         (self.tile_h, self.tile_w)
//     }

//     fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
//         let (row, col) = pos;
//         row < self.tile_h && col < self.tile_w
//     }
// }
