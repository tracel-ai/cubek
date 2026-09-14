//! A [`grouped`](Level::grouped) cube level deals its first two entries from one linear id, in
//! the order the doc states — and its two ends are the two plain orders.
//!
//! Each cube writes its own linear id into its box, so the host can read which box every id
//! took: the mapping is the whole contract, and a decode off by one shows up as another cube's
//! id rather than as a near miss.

use cubecl::{prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const TILES_M: usize = 6;
const TILES_N: usize = 4;

/// Every cube stamps its box with its position on the grid's `x`.
#[cube(launch)]
fn stamp_kernel<E: Float>(
    out: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let out = out.tile(comptime!(space.clone()));
    for cube in space {
        let mut dst = out.at(&cube);
        dst.init(E::cast_from(CUBE_POS_X));
    }
}

/// Where the doc says cube `linear` lands, `(m, n)`, on a `TILES_M × TILES_N` grid dealt in
/// groups of `width` along `m`.
fn expected(linear: usize, width: usize) -> (usize, usize) {
    let per_group = width * TILES_N;
    let group = linear / per_group;
    let in_group = linear % per_group;
    let start = group * width;
    let size = width.min(TILES_M - start);
    (start + in_group % size, in_group / size)
}

fn stamped(width: usize) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, TILES_M), (N, TILES_N)]),
            vec![Level::cubes(&[(M, 1), (N, 1)]).grouped(width)],
        ),
        KernelForm::Static,
    );
    assert!(
        matches!(launcher.cube_count(), CubeCount::Static(24, 1, 1)),
        "the two entries ride `x` as one index"
    );
    let output = TestInput::builder(client.clone(), shape![TILES_M, TILES_N])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    stamp_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );
    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

/// Every linear id lands where the order says, at every width: whole groups, a short last
/// group (`6` tiles in groups of `4`), and the two ends.
#[test]
fn a_grouped_grid_deals_its_ids_in_the_stated_order() {
    for width in [1, 2, 4, 6, 8] {
        let stamps = stamped(width);
        let mut seen = [false; TILES_M * TILES_N];
        for linear in 0..TILES_M * TILES_N {
            let (m, n) = expected(linear, width);
            assert_eq!(
                stamps.get_f32(&[m, n]) as usize,
                linear,
                "width {width}: cube {linear} did not land at ({m}, {n})"
            );
            seen[m * TILES_N + n] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "width {width}: a box was never visited"
        );
    }
}

/// At a width of the first entry's whole count the first entry runs fastest, which is the
/// plain grid's own order under a linear id; at one, the second entry does.
#[test]
fn the_two_ends_are_the_two_plain_orders() {
    for linear in 0..TILES_M * TILES_N {
        assert_eq!(
            expected(linear, TILES_M),
            (linear % TILES_M, linear / TILES_M)
        );
        assert_eq!(expected(linear, 1), (linear / TILES_N, linear % TILES_N));
    }
}
