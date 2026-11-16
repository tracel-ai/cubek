use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::G;

/// Every plane of the launch whose cube took the first head of its group casting its keys' and
/// values' gradients out of the sums every cube of the group added into: the other cubes of the
/// group cover the same keys.
#[cube(launch_unchecked)]
pub fn cast_key_gradient_kernel(
    key_sum: &TileTensor<'_>,
    value_sum: &TileTensor<'_>,
    key_gradient: &TileTensor<'_>,
    value_gradient: &TileTensor<'_>,
    partitioning: Partitioning,
) {
    let (key_sum, value_sum) = (key_sum.tile(&partitioning), value_sum.tile(&partitioning));
    let (d_k, d_v) = (
        key_gradient.tile(&partitioning),
        value_gradient.tile(&partitioning),
    );

    for plane in &partitioning {
        if plane.origin(G) == 0 {
            let mut d_k = d_k.at(&plane);
            d_k.copy_cast_from(&key_sum.at(&plane));

            let mut d_v = d_v.at(&plane);
            d_v.copy_cast_from(&value_sum.at(&plane));
        }
    }
}

/// Every plane of the launch casting its queries' gradient out of the sum every cube added into.
#[cube(launch_unchecked)]
pub fn cast_query_gradient_kernel(
    sum: &TileTensor<'_>,
    query_gradient: &TileTensor<'_>,
    partitioning: Partitioning,
) {
    let (sum, d_q) = (sum.tile(&partitioning), query_gradient.tile(&partitioning));

    for plane in &partitioning {
        let mut d_q = d_q.at(&plane);
        d_q.copy_cast_from(&sum.at(&plane));
    }
}
