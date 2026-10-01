//! A loop down to a partitioning's leaves, whatever levels lie between.
//!
//! [`Region::leaves`] walks every level below a region, so a kernel reads the same whether its
//! partitioning stacks no level, one, or several between the region it holds and the leaf.

use super::{Form, implied};
use cubecl::{prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const M: Axis = Axis(0);

const CELLS: usize = 6;

/// `out[m] = m + 1` for every leaf, reached from the outermost level through `leaves`.
#[cube(launch)]
fn mark_leaves(out: &mut Tensor<f32>, space: Partitioning) {
    for outer in space {
        for leaf in outer.leaves() {
            let m = leaf.origin(M);
            out[m] = f32::cast_from(m) + 1.0;
        }
    }
}

/// Every cell marked exactly once over `levels`, whose leaf is one cell.
fn assert_every_leaf_visited(levels: Levels) {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, CELLS)]), levels.build()),
        Form::Static,
    );
    // Unwritten cells keep a value no leaf writes, so a missed leaf fails.
    let out = TestInput::builder(client.clone(), Shape::new([CELLS]))
        .dtype(f32::elem_type_native())
        .custom(vec![-1.0; CELLS])
        .generate_without_host_data();

    mark_leaves::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        out.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for m in 0..CELLS {
        assert_eq!(got.get_f32(&[m]), m as f32 + 1.0, "cell {m}");
    }
}

#[test]
fn the_outer_level_is_itself_the_leaf() {
    assert_every_leaf_visited(Levels::leaf(&[(M, 1)]).walk(&[(M, CELLS)]));
}

#[test]
fn leaves_walk_the_level_below() {
    assert_every_leaf_visited(Levels::leaf(&[(M, 1)]).walk(&[(M, 2)]).walk(&[(M, 3)]));
}

#[test]
fn leaves_walk_every_level_below() {
    assert_every_leaf_visited(
        Levels::leaf(&[(M, 1)])
            .walk(&[(M, 2)])
            .walk(&[(M, 1)])
            .walk(&[(M, 3)]),
    );
}
