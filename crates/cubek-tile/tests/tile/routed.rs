//! Expert routing as a coordinate the kernel states.
//!
//! A token's expert is a value read from a table, not a loop coordinate. [`Walk::routed`] takes
//! one step along the expert axis at that value, so the axis keeps the extent it truly has while
//! the walk visits one of them; the weights operand carries nothing, and `at` is the same call.
//!
//! The leaf sees one value of the axis, so it contracts nothing, and the block it opens folds
//! along `K` exactly as an unrouted one does.
#![allow(non_snake_case)]

use cubecl::{client::Client, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, skip_unless_plane_holds};
use cubek_tile::*;

/// Token.
const M: Axis = Axis(0);
/// Output feature.
const N: Axis = Axis(1);
/// Contracted feature.
const K: Axis = Axis(2);
/// Which expert. Three of them, and the space says so.
const EXPERT: Axis = Axis(3);

const TOKENS: usize = 4;
const FEATURES: usize = 4;
const EXPERTS: usize = 3;
/// The contraction the folded leaf walks: two lines rather than one, so the walk has steps.
const DEPTH: usize = 8;
/// The line a folded step serves off each operand.
const LINE: usize = 4;

/// Which expert each token routes to. Distinct enough that reusing one token's expert for
/// another cannot land on the right answer.
const ROUTES: [u32; TOKENS] = [2, 0, 2, 1];

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `x[m, k]`.
fn activations(depth: usize) -> Vec<f32> {
    (0..TOKENS * depth).map(|i| (i % 5) as f32).collect()
}

/// `w[e, k, n]`. Each expert's weights are its own: expert `e` scales by `e + 1`, so contracting a
/// token against the wrong slab is off by a whole factor.
fn weights(depth: usize) -> Vec<f32> {
    (0..EXPERTS * depth * FEATURES)
        .map(|i| {
            let e = i / (depth * FEATURES);
            let rest = i % (depth * FEATURES);
            ((e + 1) * (rest % 3 + 1)) as f32
        })
        .collect()
}

/// [`weights`] with each slab transposed to `w[e, n, k]`, which is how an operand lined along the
/// contraction stores them.
fn weights_lined_along_k(depth: usize) -> Vec<f32> {
    let along_n = weights(depth);
    let mut along_k = vec![0.0; along_n.len()];
    for e in 0..EXPERTS {
        for k in 0..depth {
            for n in 0..FEATURES {
                along_k[(e * FEATURES + n) * depth + k] = along_n[(e * depth + k) * FEATURES + n];
            }
        }
    }
    along_k
}

/// Each token against its own expert, folded on the host.
fn expected(routes: &[u32], depth: usize) -> Vec<Vec<f32>> {
    let (x, w) = (activations(depth), weights(depth));
    (0..TOKENS)
        .map(|m| {
            let e = routes[m] as usize;
            (0..FEATURES)
                .map(|n| {
                    (0..depth)
                        .map(|k| x[m * depth + k] * w[(e * depth + k) * FEATURES + n])
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Every cell against the host's own fold, under the table the launch was given.
fn assert_routed(got: &HostData, routes: &[u32], depth: usize) {
    for (m, row) in expected(routes, depth).iter().enumerate() {
        for (n, cell) in row.iter().enumerate() {
            assert_eq!(
                got.get_f32(&[m, n]),
                *cell,
                "token {m} feature {n}: expert {}",
                routes[m]
            );
        }
    }
}

/// The table, one `u32` expert per token.
fn routing_table(client: &Client, routes: &[u32]) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new([TOKENS]))
        .dtype(u32::elem_type_native())
        .custom(routes.iter().map(|&r| r as f32).collect())
        .generate_without_host_data()
}

/// The output, filled with a value no product lands on, so a launch that never ran fails here
/// rather than passing.
fn unwritten_output(client: &Client) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32::elem_type_native())
        .custom(vec![-1.0; TOKENS * FEATURES])
        .generate_without_host_data()
}

/// `out[m] = x[m] · W[routes[m]]`: one token per step, its expert's slab picked by the table.
#[cube(launch)]
fn routed_matmul_kernel<E: Numeric>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    routes: &Tensor<u32>,
    space: Partitioning,
    #[comptime] token: Level,
    #[comptime] expert: Level,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for tok in space.over(&token) {
        let e = routes[tok.coord(M)] as usize;

        // The whole of it: the expert axis takes one step, at the coordinate the table named.
        for slab in tok.over(&expert).routed(EXPERT, e) {
            let mut o = out.at(&slab);
            o.mm_with(
                &x.at(&slab),
                &w.at(&slab),
                REGISTER_BLOCK,
                Semiring::SUM_PROD,
            );
        }
    }
}

/// The routed weights staged in shared memory rather than read where they lie: the ring is built
/// over the routed walk, so what it fills is the expert the table named.
#[cube(launch)]
fn routed_staged_matmul_kernel<E: Numeric>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    routes: &Tensor<u32>,
    space: Partitioning,
    #[comptime] token: Level,
    #[comptime] expert: Level,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for tok in space.over(&token) {
        let e = routes[tok.coord(M)] as usize;
        let experts = tok.over(&expert).routed(EXPERT, e);

        let mut ring = Ring::smem_single(&experts, &w, StageStorage::Strided, 1usize);
        pipelined(experts, &mut ring, |slot, slab| {
            let mut o = out.at(slab);
            slot.consume(|w_s| {
                o.mm_with(&x.at(slab), w_s, REGISTER_BLOCK, Semiring::SUM_PROD);
            });
        });
    }
}

/// What the token-per-step kernels above launch with: one token and one expert per leaf, the
/// expert axis at its true extent, and the weights stored `[e, k, n]`.
///
/// Nothing here says a token uses one expert; the walk does.
fn per_token_operands(client: &Client) -> (Launcher, TensorHandle, TensorHandle) {
    let f32_ty = f32::elem_type_native();
    let launcher = Launcher::implied(
        client,
        Partitioning::new(
            Space::new(&[(M, TOKENS), (N, FEATURES), (K, FEATURES), (EXPERT, EXPERTS)]),
            Tiling::leaf(&[(M, 1), (EXPERT, 1)])
                .walk_every(&[EXPERT])
                .walk_every(&[M])
                .levels(),
        ),
        KernelForm::Static,
    );

    let (x, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(activations(FEATURES))
        .generate_with_f32_host_data();
    let (w, _) = TestInput::builder(client.clone(), Shape::new([EXPERTS, FEATURES, FEATURES]))
        .dtype(f32_ty)
        .custom(weights(FEATURES))
        .generate_with_f32_host_data();

    (launcher, x, w)
}

fn run(routes: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let (launcher, x, w) = per_token_operands(&client);
    let table = routing_table(&client, routes);
    let out = unwritten_output(&client);

    routed_matmul_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(x.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(
            w.binding().into_tensor_arg(),
            TileSpec::direct(&[EXPERT, K, N]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        table.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.level(0),
        launcher.level(1),
        f32::elem_type_native(),
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

fn run_staged(routes: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let (launcher, x, w) = per_token_operands(&client);
    let table = routing_table(&client, routes);
    let out = unwritten_output(&client);

    routed_staged_matmul_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(x.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(
            w.binding().into_tensor_arg(),
            TileSpec::direct(&[EXPERT, K, N]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        table.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.level(0),
        launcher.level(1),
        f32::elem_type_native(),
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// Two tables over the same weights: the answer follows the table. A route that did nothing could
/// not, since it would fold every expert under both of them.
#[test]
fn a_routed_walk_contracts_each_token_against_the_expert_its_table_names() {
    for routes in [ROUTES, [0, 0, 0, 0]] {
        assert_routed(&run(&routes), &routes, FEATURES);
    }
}

/// Staging under a routed coordinate: the stage is filled from the expert the route named, not
/// from the first one and then reused. A window displacement living on the operand would owe a
/// refusal (a staged operand inherits it); here the coordinate is the walk's, the stage per region.
#[test]
fn a_routed_operand_stages_the_expert_the_table_named() {
    assert_routed(&run_staged(&ROUTES), &ROUTES, FEATURES);
}

/// A routing table names a tile the axis does not have. The coordinate came from data, so this is
/// not a mistake the caller can be told about (a refusal inside a cube verb dies on a worker
/// thread). The walk clamps instead, so the read stays inside the weights, on the last expert.
#[test]
fn a_route_past_the_last_expert_clamps_to_it() {
    const OVER: [u32; TOKENS] = [99, 0, 2, 1];
    const CLAMPED: [u32; TOKENS] = [(EXPERTS - 1) as u32, 0, 2, 1];
    assert_routed(&run(&OVER), &CLAMPED, FEATURES);
}

/// `out[m] = x[m] · W[routes[m]]` with the partials in registers for the whole `K` walk: the block
/// is opened inside the route, so everything it reads is already the one expert the table named.
#[cube(launch)]
fn routed_block_matmul_kernel<E: Numeric>(
    x: &TileArg<'_, E, Const<4>>,
    w: &TileArg<'_, E, Const<4>>,
    out: &TileArg<'_, E, Const<1>>,
    routes: &Tensor<u32>,
    space: Partitioning,
    #[comptime] token: Level,
    #[comptime] expert: Level,
    #[comptime] depth: Level,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for tok in space.over(&token) {
        let e = routes[tok.coord(M)] as usize;

        for slab in tok.over(&expert).routed(EXPERT, e) {
            let out_s = out.at(&slab);
            let x_s = x.at(&slab);
            let w_s = w.at(&slab);
            let mut acc = out_s.block_accumulator::<E, E, E>(
                &x_s,
                &w_s,
                comptime!(Fragments::below(&out_s, &x_s)),
                REGISTER_BLOCK,
                Monoid::Sum,
            );
            acc.zero();
            for step in slab.over(&depth) {
                let mut acc_s = acc.at(&step);
                acc_s.mma(&x_s.at(&step), &w_s.at(&step), Semiring::SUM_PROD);
            }
            for cell in out_s.walk().unrolled() {
                let mut o = out_s.at(&cell);
                o.copy_cast_from(&acc.at(&cell));
            }
        }
    }
}

fn run_block(routes: &[u32]) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, TOKENS), (N, FEATURES), (K, DEPTH), (EXPERT, EXPERTS)]),
            Tiling::leaf(&[(M, 1), (EXPERT, 1), (K, LINE)])
                .walk_every(&[K])
                .walk_every(&[EXPERT])
                .walk_every(&[M])
                .levels(),
        ),
        KernelForm::Static,
    );

    let (x, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, DEPTH]))
        .dtype(f32_ty)
        .custom(activations(DEPTH))
        .generate_with_f32_host_data();
    let (w, _) = TestInput::builder(client.clone(), Shape::new([EXPERTS, FEATURES, DEPTH]))
        .dtype(f32_ty)
        .custom(weights_lined_along_k(DEPTH))
        .generate_with_f32_host_data();
    let table = routing_table(&client, routes);
    let out = unwritten_output(&client);

    routed_block_matmul_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(x.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(
            w.binding().into_tensor_arg(),
            TileSpec::direct(&[EXPERT, N, K]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        table.binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.level(0),
        launcher.level(1),
        launcher.level(2),
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// A routed walk contracting in a register block at a folded step: the weights line along `K`, so
/// a step consumes a whole line of each operand and the block's lanes are one cell's partials.
///
/// The expert axis holds one value under the route, so it contracts nothing. Counted as a
/// contracted axis it becomes the fastest one, not the axis the operands line along, and the fold
/// every `K`-stored weight needs is gone: a route could only run one scalar cell at a time.
///
/// A refusal inside a cube verb dies on the expansion worker, so a regression here reads as the
/// output keeping its fill value rather than as a message.
#[test]
fn a_routed_walk_folds_its_contraction_into_a_register_block() {
    assert_routed(&run_block(&ROUTES), &ROUTES, DEPTH);
}

/// Counts the steps of a walk routed on `axis`, which the refusal test hands an axis the space
/// does not carry.
#[cube(launch)]
fn routed_axis_kernel(
    k: &TileArg<'_, f32, Const<1>>,
    out: &mut Tensor<f32>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] axis: Axis,
) {
    let k = k.tile(comptime!(space.clone()));
    let mut steps = 0u32;
    for _region in k.over(&level).routed(axis, 0usize) {
        steps += 1;
    }
    out[0] = f32::cast_from(steps);
}

fn launch_routed_on(axis: Axis) -> f32 {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, TOKENS), (EXPERT, EXPERTS)]),
            Tiling::leaf(&[(EXPERT, 1)]).walk_every(&[EXPERT]).levels(),
        ),
        KernelForm::Static,
    );

    let (k_handle, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, EXPERTS]))
        .dtype(f32_ty)
        .custom(vec![0.0; TOKENS * EXPERTS])
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(f32_ty)
        .custom(vec![-7.0])
        .generate_without_host_data();

    routed_axis_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[M, EXPERT]),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.level(0),
        axis,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32).get_f32(&[0])
}

/// A comptime refusal inside a `#[cube]` verb does NOT reach the caller: expansion runs on a
/// worker thread, so `Walk::routed`'s assert panics there and the launch returns as if nothing
/// happened (see `CUBECL_DEBUG_LOG`; measured 2026-09-08). The routed refusal must be host-side.
#[test]
fn routing_an_axis_of_the_space_is_the_only_case_checked_here() {
    assert_eq!(launch_routed_on(EXPERT), 1.0);
}

/// Each lane writes the expert coordinate its region carried, so a walk that folds the lane's own
/// position into a routed axis is visible as lanes disagreeing.
#[cube(launch)]
fn routed_lanes_kernel(
    out: &mut Tensor<f32>,
    space: Partitioning,
    #[comptime] lanes: Level,
    #[comptime] target: usize,
) {
    for region in space
        .over(&lanes)
        .routed(EXPERT, comptime!(target).runtime())
    {
        out[UNIT_POS_X as usize] = f32::cast_from(region.coord(EXPERT) as u32);
    }
}

/// A routed axis spread across the plane's lanes: naming a coordinate names it for every lane,
/// since a lane's own share of the axis is what the route replaces rather than shifts. Without
/// that, lane `l` would read expert `target + l` off one shared operand.
#[test]
fn a_routed_axis_reads_the_same_coordinate_in_every_lane() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    // A unit level must partition the plane exactly, so the axis is as wide as the plane.
    let lanes = client.properties().hardware.plane_size_max as usize;
    let target = 2usize;
    // The axis is as wide as the plane, so the plane must hold the expert the route names.
    if skip_unless_plane_holds(&client, target as u32 + 1) {
        return;
    }

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(EXPERT, lanes)]),
            Tiling::leaf(&[(EXPERT, 1)])
                .lanes(&[(EXPERT, lanes)])
                .levels(),
        ),
        KernelForm::Static,
    );

    let out_handle = TestInput::builder(client.clone(), Shape::new([lanes]))
        .dtype(f32_ty)
        .custom(vec![-1.0; lanes])
        .generate_without_host_data();

    routed_lanes_kernel::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_2d(lanes as u32, 1),
        out_handle.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.level(0),
        target,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for lane in 0..lanes {
        assert_eq!(
            got.get_f32(&[lane]),
            target as f32,
            "lane {lane} read a different expert than the one the route named"
        );
    }
}
