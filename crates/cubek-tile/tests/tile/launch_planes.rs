//! A plane level spanning the launch ([`Levels::launch_planes`]): the outermost level, with no
//! cube level above it, its tiles dealt to the planes of the whole launch in a [`PlaneOrder`].
//!
//! Every case adds its products atomically into a zeroed output, so a tile no plane took reads
//! zero and a tile two planes took reads twice its product: a plane past the last tile walks one
//! past the grid, which the binding masks, not one another plane holds.
//!
//! Every plane the launch spawns walks one tile, so the planes of a cube all reach a barrier any
//! of them meets; a plane that walked none would leave the others waiting at it.
#![allow(non_snake_case)]

use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

use cubek_tile::kind::{Boundary, Write};
use cubek_tile::launch::{BoundaryPolicy, Buffered, Destination, DestinationLaunch, Grid};
use cubek_tile::space::{CubeOrder, PlaneOrder};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// A batch the inputs do not span: every batch's output is the same product.
const B: Axis = Axis(3);

/// Rows, columns and depth of the block one unit owns.
const BLOCK: [usize; 3] = [2, 2, 4];

/// Units of a plane, each a block along `n`.
const UNITS: usize = 2;

/// `c += a · b`, every plane of the launch its tile.
#[cube(launch)]
fn launch_plane_matmul<O: Destination>(
    a: &TileArg<'_, f32, Const<1>>,
    b: &TileArg<'_, f32, Const<1>>,
    c: &O::Arg<f32, Const<1>>,
    space: Partitioning,
) {
    let (a, b) = (a.tile(&space), b.tile(&space));
    let c = O::tile::<f32, Const<1>>(c, &space);
    for plane in &space {
        plane_product(&plane, &a, &b, &c);
    }
}

/// `c += a · b`, every plane of every cube its tile: the control.
#[cube(launch)]
fn cube_plane_matmul<O: Destination>(
    a: &TileArg<'_, f32, Const<1>>,
    b: &TileArg<'_, f32, Const<1>>,
    c: &O::Arg<f32, Const<1>>,
    space: Partitioning,
) {
    let (a, b) = (a.tile(&space), b.tile(&space));
    let c = O::tile::<f32, Const<1>>(c, &space);
    for cube in &space {
        for plane in cube {
            plane_product(&plane, &a, &b, &c);
        }
    }
}

/// How many tiles every plane of the launch walks, at the plane's place in the launch: its cube's,
/// read off the grid's `x` and `y`, then its own in the cube.
#[cube(launch)]
fn count_walks(walks: &mut Tensor<f32>, space: Partitioning) {
    let mut taken = 0.0f32;
    for _plane in &space {
        taken += 1.0;
    }
    if UNIT_POS_X == 0 {
        let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
        walks[(cube * CUBE_DIM_Y + UNIT_POS_Y) as usize] = taken;
    }
}

/// `plane`'s tile of `c += a · b`, every unit of it a block summed in registers.
#[cube]
fn plane_product(plane: &Region, a: &Tile<f32>, b: &Tile<f32>, c: &Tile<f32>) {
    for unit in plane {
        let (a_unit, b_unit, c_unit) = (a.at(&unit), b.at(&unit), c.at(&unit));
        let sum = c_unit.accumulator::<f32, f32, f32>(
            &a_unit,
            &b_unit,
            comptime!(Instruction::Registers {
                config: RegisterBlock::new(BLOCK[0] * BLOCK[1])
            }),
            Semiring::SUM_PROD,
        );
        for step in &unit {
            let mut sum_step = sum.at(&step);
            sum_step.mma(&a_unit.at(&step), &b_unit.at(&step));
        }
        sum.drained_into(&c_unit);
    }
}

/// Who takes a plane's worth of tiles.
#[derive(Clone, Debug)]
enum Planes {
    /// The planes of each cube, a cube level holding a `2 × 2` plane level: the control.
    OfEachCube,
    /// The planes of the launch, in this order.
    OfTheLaunch(PlaneOrder),
}

impl Planes {
    /// The levels the planes take a plane's `2 × 4` tile at.
    fn levels(&self) -> Vec<Level> {
        let [rows, columns, depth] = BLOCK;
        let units = Levels::leaf(&[(M, rows), (N, columns), (K, depth)])
            .walk_every(&[K])
            .units(&[(N, UNITS)]);
        match self {
            Planes::OfEachCube => units
                .planes(&[(M, 2), (N, 2)])
                .cubes(&[M, N])
                .batches(&[B])
                .build(),
            Planes::OfTheLaunch(order) => units.launch_planes(&[M, N, B], order.clone()).build(),
        }
    }
}

/// An `m × n × k` product, written into each of `batches` batches of the output.
#[derive(Clone, Copy, Debug)]
struct Problem {
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
}

impl Problem {
    const fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            batches: 1,
        }
    }

    fn batched(self, batches: usize) -> Self {
        Self { batches, ..self }
    }
}

fn lhs(i: usize, p: usize, k: usize) -> f32 {
    ((i * k + p) % 7) as f32 - 3.0
}

fn rhs(p: usize, j: usize, n: usize) -> f32 {
    ((p * n + j) % 5) as f32 - 2.0
}

/// Whether the test device adds `f32`s atomically, which every case's output takes.
fn adds_atomically(client: &cubecl::client::Client) -> bool {
    let usage = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)));
    let adds = usage.contains(AtomicUsage::Add);
    if !adds {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
    }
    adds
}

/// `problem`'s product, its tiles taken by `planes`; `None` where the device cannot add into the
/// output.
fn run(problem: Problem, planes: &Planes) -> Option<HostData> {
    let Problem { m, n, k, batches } = problem;
    let client = cubecl::test_device().client();
    if !adds_atomically(&client) {
        return None;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..m * k).map(|i| lhs(i / k, i % k, k)).collect();
    let b: Vec<f32> = (0..k * n).map(|i| rhs(i / n, i % n, n)).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![batches, m, n])
        .dtype(dtype)
        .custom(vec![0.0; batches * m * n])
        .generate_without_host_data();

    let partitioning = Partitioning::new(
        Space::new(&[(B, batches), (M, m), (N, n), (K, k)]),
        planes.levels(),
    );
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: partitioning.cube_dim(UNITS as u32),
    };
    let space = partitioning.space().clone();
    let launcher = Launcher::new(&client, partitioning, &space, grid).unwrap();
    let policy = BoundaryPolicy::Every(Boundary::Zero);
    let bound = launcher
        .arg(out.clone().binding())
        .axes(&[B, M, N])
        .build()
        .unwrap();

    let a_arg = TileArgLaunch::new(
        a_handle.binding().into_tensor_arg(),
        TileSpec::direct(&[M, K]).boundary(policy),
    );
    let b_arg = TileArgLaunch::new(
        b_handle.binding().into_tensor_arg(),
        TileSpec::direct(&[K, N]).boundary(policy),
    );
    let c_arg = Buffered::arg((bound, Write::Accumulate));
    match planes {
        Planes::OfTheLaunch(_) => launch_plane_matmul::launch::<Buffered>(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a_arg,
            b_arg,
            c_arg,
            launcher.partitioning_arg(),
        ),
        Planes::OfEachCube => cube_plane_matmul::launch::<Buffered>(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            a_arg,
            b_arg,
            c_arg,
            launcher.partitioning_arg(),
        ),
    }
    Some(HostData::from_tensor_handle(
        &client,
        out,
        HostDataType::F32,
    ))
}

/// Every cell of every batch of the output is its product once: no tile left out, none taken
/// twice.
fn assert_every_tile_once(got: &HostData, problem: Problem, what: &str) {
    let Problem { m, n, k, batches } = problem;
    for batch in 0..batches {
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| lhs(i, p, k) * rhs(p, j, n)).sum();
                assert_eq!(
                    got.get_f32(&[batch, i, j]),
                    want,
                    "{what}: at ({batch}, {i}, {j})"
                );
            }
        }
    }
}

/// A plane's tile is `2 × 4`: two rows of units' blocks, two units of `2 × 2` along `n`.
const PROBLEMS: [Problem; 3] = [
    Problem::new(8, 16, 8),
    Problem::new(7, 29, 11),
    Problem::new(13, 5, 6),
];

/// Every problem's product, its tiles taken by `planes`.
fn assert_takes_every_tile_once(planes: Planes, what: &str) {
    for problem in PROBLEMS {
        if let Some(got) = run(problem, &planes) {
            assert_every_tile_once(&got, problem, what);
        }
    }
}

/// The control: a cube level holding a `2 × 2` plane level.
#[test]
fn a_cube_level_holding_planes_takes_every_tile_once() {
    assert_takes_every_tile_once(Planes::OfEachCube, "cubes holding planes");
}

/// One block a cube, a `2 × 2` box of tiles: what a cube level holding a plane level of
/// `2 × 2` deals, on problems whose edges cut the boxes short.
#[test]
fn one_block_a_cube_takes_every_tile_once() {
    let order = PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor);
    assert_takes_every_tile_once(Planes::OfTheLaunch(order), "one block a cube");
}

/// One plane a block, three planes a cube: the planes run straight across the tiles, and the
/// last cube holds planes past the last tile, which walk tiles past the grid and add nothing.
#[test]
fn planes_past_the_last_tile_take_none() {
    let order = PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3);
    assert_takes_every_tile_once(Planes::OfTheLaunch(order), "three planes a cube");
}

/// A block of four tiles along `n` held across two cubes of two planes.
#[test]
fn a_block_spans_several_cubes() {
    let order = PlaneOrder::new(&[(N, 4)], CubeOrder::RowMajor).cubes_of(2);
    assert_takes_every_tile_once(Planes::OfTheLaunch(order), "a block across two cubes");
}

/// The blocks taken in a swizzle, ragged where its strips do not divide the blocks.
#[test]
fn swizzled_blocks_take_every_tile_once() {
    for block_order in [CubeOrder::SwizzleRow(2), CubeOrder::SwizzleCol(3)] {
        let order = PlaneOrder::new(&[(M, 2), (N, 1)], block_order);
        assert_takes_every_tile_once(Planes::OfTheLaunch(order), &format!("{block_order:?}"));
    }
}

/// A batch axis beside `m` and `n`: the blocks of the two run fastest, a batch's after another's,
/// and the last cube's planes past the last batch take none.
#[test]
fn the_batches_ride_outermost() {
    for order in [
        PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor),
        PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3),
        PlaneOrder::new(&[(M, 2), (N, 1)], CubeOrder::SwizzleRow(2)),
    ] {
        let planes = Planes::OfTheLaunch(order);
        for problem in PROBLEMS.map(|problem| problem.batched(3)) {
            if let Some(got) = run(problem, &planes) {
                assert_every_tile_once(&got, problem, &format!("three batches, {planes:?}"));
            }
        }
    }
}

/// The cubes the levels ask for: every block's planes, a cube's worth at a time.
#[test]
fn the_launch_holds_every_block() {
    let partitioning = |order| {
        Partitioning::new(
            Space::new(&[(B, 1), (M, 7), (N, 29), (K, 11)]),
            Levels::leaf(&[(M, 2), (N, 2), (K, 4)])
                .walk_every(&[K])
                .units(&[(N, UNITS)])
                .launch_planes(&[M, N, B], order)
                .build(),
        )
    };
    let cubes = |partitioning: &Partitioning| match partitioning.cube_count() {
        CubeCount::Static(x, y, z) => (x, y, z),
        CubeCount::Dynamic(_) => panic!("a static space spawns a static grid"),
    };
    // Tiles of 2 × 4: 4 along m, 8 along n.
    let one_block_a_cube = partitioning(PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor));
    assert_eq!(cubes(&one_block_a_cube), (8, 1, 1));
    assert_eq!(one_block_a_cube.planes_per_cube(), 4);
    assert_eq!(one_block_a_cube.cube_dim(2), CubeDim::new_2d(2, 4));
    let three_a_cube = partitioning(PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3));
    assert_eq!(cubes(&three_a_cube), (11, 1, 1));
}

/// The orders whose launches hold planes past the last tile, beside one whose blocks the tiles
/// fill: blocks the edge cuts short, cubes past the last block, and both in a swizzle.
fn orders() -> [PlaneOrder; 4] {
    [
        PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor),
        PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3),
        PlaneOrder::new(&[(N, 4)], CubeOrder::RowMajor).cubes_of(2),
        PlaneOrder::new(&[(M, 2), (N, 1)], CubeOrder::SwizzleCol(3)),
    ]
}

/// Every plane the launch spawns walks exactly one tile, a plane past the last tile one past the
/// grid: the planes of a cube take the same steps, so none skips a barrier the others meet.
#[test]
fn every_plane_of_the_launch_walks_one_tile() {
    let client = cubecl::test_device().client();
    for order in orders() {
        for problem in PROBLEMS.map(|problem| problem.batched(2)) {
            let Problem { m, n, k, batches } = problem;
            let partitioning = Partitioning::new(
                Space::new(&[(B, batches), (M, m), (N, n), (K, k)]),
                Planes::OfTheLaunch(order.clone()).levels(),
            );
            let cube_count = partitioning.cube_count();
            let CubeCount::Static(x, y, z) = cube_count else {
                panic!("a static space spawns a static grid");
            };
            let planes = (x * y * z * partitioning.planes_per_cube()) as usize;
            let walks = TestInput::builder(client.clone(), shape![planes])
                .dtype(f32::elem_type_native())
                .custom(vec![0.0; planes])
                .generate_without_host_data();
            let grid = Grid::Stated {
                cube_count,
                cube_dim: partitioning.cube_dim(UNITS as u32),
            };
            let space = partitioning.space().clone();
            let launcher = Launcher::new(&client, partitioning, &space, grid).unwrap();
            count_walks::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                walks.clone().binding().into_tensor_arg(),
                launcher.partitioning_arg(),
            );
            let got = HostData::from_tensor_handle(&client, walks, HostDataType::F32);
            for plane in 0..planes {
                assert_eq!(
                    got.get_f32(&[plane]),
                    1.0,
                    "{order:?} on {problem:?}: plane {plane} of {planes}"
                );
            }
        }
    }
}

/// The binding masks an axis where the launch's planes run past its tiles, and no other: along
/// an axis whose blocks the edge cuts short, and along the axis the blocks past the last land
/// past, the outermost of three.
#[test]
fn the_binding_masks_where_the_planes_run_past_the_tiles() {
    let masked = |order: PlaneOrder, (m, n): (usize, usize)| {
        let partitioning = Partitioning::new(
            Space::new(&[(B, 1), (M, m), (N, n), (K, 8)]),
            Planes::OfTheLaunch(order).levels(),
        );
        [M, N, B].map(|axis| partitioning.overhangs(axis))
    };
    let blocks = || PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor);
    // Tiles of 2 × 4: 4 × 4 of them fill the 2 × 2 blocks.
    assert_eq!(masked(blocks(), (8, 16)), [false, false, false]);
    // 4 × 5: the last block along `n` holds one tile of two.
    assert_eq!(masked(blocks(), (8, 20)), [false, true, false]);
    // 16 tiles in cubes of three planes: the last cube's two spare planes land past the batch.
    let three_a_cube = PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3);
    assert_eq!(masked(three_a_cube, (8, 16)), [false, false, true]);
    // 15 tiles in cubes of three: none spare.
    let three_a_cube = PlaneOrder::new(&[], CubeOrder::RowMajor).cubes_of(3);
    assert_eq!(masked(three_a_cube, (6, 20)), [false, false, false]);
}

/// A kernel compiled for every batch count asks the static axes alone: the blocks the edge cuts
/// short decide them, and the batch, which the spare blocks land past, is the launch's.
#[test]
fn a_static_axis_is_answered_beside_a_dynamic_batch() {
    let partitioning = |n| {
        Partitioning::new(
            Space::new(&[(B, 1), (M, 8), (N, n), (K, 8)]).with_dynamic(&[B]),
            Planes::OfTheLaunch(PlaneOrder::new(&[(M, 2), (N, 2)], CubeOrder::RowMajor)).levels(),
        )
    };
    assert!(!partitioning(16).overhangs(M));
    assert!(!partitioning(16).overhangs(N));
    assert!(partitioning(20).overhangs(N));
}
