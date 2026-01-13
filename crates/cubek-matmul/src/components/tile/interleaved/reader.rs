use crate::components::tile::StridedTile;
use crate::components::tile::interleaved::config::InterleavedMatmulConfig;
use crate::components::tile::interleaved::{InterleavedAccumulator, InterleavedFragment};
use crate::definition::{MatrixLayout, StageIdent};
use cubecl::prelude::*;

#[derive(CubeType)]
pub struct InterleavedStageReader {}

#[cube]
impl InterleavedStageReader {
    pub fn load_fragment<E: Numeric, V: Numeric>(
        tile: &StridedTile<V>,
        fragment: &mut InterleavedFragment<E>,
        #[comptime] ident: StageIdent,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
        let (m, n, k_local) = (
            config.elements_per_unit_m(),
            config.elements_per_unit_n(),
            config.elements_per_unit_k(),
        );
        let layout = comptime!(tile.layout);
        let line_size = comptime!(tile.line_size as usize);

        let unit_id = UNIT_POS_X as usize;
        let k_offset = k_local * unit_id;

        let (strided_dim_count, contiguous_dim_count) = match (layout, ident) {
            (MatrixLayout::RowMajor, StageIdent::Lhs) => (m, k_local),
            (MatrixLayout::RowMajor, StageIdent::Rhs) => (k_local, n),
            (MatrixLayout::ColMajor, StageIdent::Lhs) => (k_local, m),
            (MatrixLayout::ColMajor, StageIdent::Rhs) => (n, k_local),
            _ => unreachable!(),
        };

        let (strided_dim_offset, contiguous_dim_offset) = match (layout, ident) {
            // k is contiguous dim
            (MatrixLayout::RowMajor, StageIdent::Lhs)
            | (MatrixLayout::ColMajor, StageIdent::Rhs) => (0, k_offset / line_size),
            // k is not contiguous dim
            (MatrixLayout::RowMajor, StageIdent::Rhs)
            | (MatrixLayout::ColMajor, StageIdent::Lhs) => (k_offset, 0),
            _ => unreachable!(),
        };

        assert!(contiguous_dim_count % line_size == 0);
        let line_count_in_dim = contiguous_dim_count / line_size;

        for i in 0..strided_dim_count {
            for j in 0..line_count_in_dim {
                let line = Line::cast_from(tile.get_line(
                    (i + strided_dim_offset) as u32,
                    (j + contiguous_dim_offset) as u32,
                ));

                let line_start = i * contiguous_dim_count + j * line_size;
                for l in 0..line_size {
                    fragment.array[line_start + l] = line[l];
                }
            }
        }
    }

    pub fn load_accumulator<A: Numeric, V: Numeric>(
        value: &V,
        fragment: &mut InterleavedAccumulator<A>,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
        let size = config.elements_per_unit_m() * config.elements_per_unit_n();

        for i in 0..size {
            fragment.array[i] = A::cast_from(*value);
        }
    }
}
