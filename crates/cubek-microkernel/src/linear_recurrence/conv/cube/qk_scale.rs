use super::convolved_head::ConvolvedHead;
use crate::linear_recurrence::HEAD_CHANNEL;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// The scales of a head's queries (by the L2 norm and the root of the width) and keys (by the L2
/// norm).
#[derive(CubeType)]
pub(super) struct QkScale {
    q: f32,
    k: f32,
}

#[cube]
impl QkScale {
    /// The scales of `convolved`'s head, summed in line order once the cube has met.
    pub fn of<N: Size>(
        convolved: &ConvolvedHead<N>,
        epsilon: f32,
        #[comptime] key_features: usize,
    ) -> QkScale {
        let width = N::value();
        let lines = comptime!(key_features / width);
        let mut query_squares = 0.0f32;
        let mut key_squares = 0.0f32;
        for part in 0..lines {
            query_squares += convolved.query_squares(part);
            key_squares += convolved.key_squares(part);
        }
        QkScale {
            q: 1.0f32
                / ((query_squares + epsilon).sqrt()
                    * comptime!((key_features as f64).sqrt() as f32)),
            k: 1.0f32 / (key_squares + epsilon).sqrt(),
        }
    }

    /// The head's queries and keys scaled, written by the units that own the lines.
    pub fn write<N: Size>(
        &self,
        cube: &Region,
        convolved: &ConvolvedHead<N>,
        query: &mut Tile,
        key: &mut Tile,
        #[comptime] key_features: usize,
    ) {
        let width = N::value();
        let key_lines = comptime!(key_features / width);
        let q_all = Vector::<f32, N>::cast_from(self.q);
        let k_all = Vector::<f32, N>::cast_from(self.k);
        let mut queries = query.flat_mut::<f32, N>();
        let mut keys = key.flat_mut::<f32, N>();
        for share in cube {
            let part = share.coord(HEAD_CHANNEL);
            if part < key_lines {
                queries.write(part, convolved.query_line(part) * q_all);
            } else if part < 2 * key_lines {
                keys.write(
                    part - key_lines,
                    convolved.key_line(part - key_lines) * k_all,
                );
            }
        }
    }
}
