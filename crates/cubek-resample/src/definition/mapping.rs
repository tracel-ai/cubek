use cubecl::prelude::*;

#[cube]
pub trait IndexMapper: Send + Sync + 'static {
    /// Maps the output and reduction coordinates to a specific tensor coordinate.
    fn map(
        out_coord: Sequence<u32>,
        reduction_coord: Sequence<u32>,
        scales: Sequence<f32>,
    ) -> Sequence<u32>;
}

/// For nearest neighbor, it scales the specified axis and floors it.
#[derive(Clone, Copy)]
pub struct NearestMapper;

#[cube]
impl IndexMapper for NearestMapper {
    fn map(
        out_coord: Sequence<u32>,
        _reduction_coord: Sequence<u32>,
        scales: Sequence<f32>,
    ) -> Sequence<u32> {
        let mut mapped = Sequence::<u32>::new();
        let rank = out_coord.len();

        #[unroll]
        for i in 0..rank {
            let scale = if i < scales.len() { scales[i] } else { 1.0f32 };
            mapped.push(f32::floor(f32::cast_from(out_coord[i]) * scale) as u32);
        }
        mapped
    }
}

/// Identity mapper.
#[derive(Clone, Copy)]
pub struct IdentityMapper;

#[cube]
impl IndexMapper for IdentityMapper {
    fn map(
        out_coord: Sequence<u32>,
        _reduction_coord: Sequence<u32>,
        _scales: Sequence<f32>,
    ) -> Sequence<u32> {
        // Requires cloning because sequence is moved, but we can't easily clone in cube.
        // We recreate it.
        let mut mapped = Sequence::<u32>::new();
        let rank = out_coord.len();
        #[unroll]
        for i in 0..rank {
            mapped.push(out_coord[i]);
        }
        mapped
    }
}
