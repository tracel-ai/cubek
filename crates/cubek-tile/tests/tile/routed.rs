//! Expert routing as a coordinate the kernel states.
//!
//! A token's expert is a value read from a table, not a loop coordinate. [`Walk::routed`] takes
//! one step along the expert axis at that value, so the axis keeps the extent it truly has (three
//! experts, three experts) while the walk visits one of them. Everything below reads the ordinary
//! coordinate it is: the weights operand carries nothing, and `at` is the same call it is
//! everywhere else.
#![allow(non_snake_case)]

use cubecl::{prelude::*, zspace::Shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
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

/// Which expert each token routes to. Distinct enough that reusing one token's expert for
/// another cannot land on the right answer.
const ROUTES: [u32; TOKENS] = [2, 0, 2, 1];

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `out[m] = x[m] · W[routes[m]]`: one token per step, its expert's slab picked by the table.
#[cube(launch)]
fn moe_kernel<E: Numeric>(
    x: &TileArg<'_, E, Const<1>>,
    w: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    routes: &Tensor<u32>,
    space: Space,
    #[comptime] token: Level,
    #[comptime] expert: Level,
    #[comptime] route: bool,
    #[define(E)] _dtype: ElemType,
) {
    let x = x.tile(comptime!(space.clone()));
    let w = w.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));

    for tok in space.level(comptime!(token.clone())) {
        let m = tok.coord(M);
        let e = routes[m] as usize;

        // The whole of it: the expert axis takes one step, at the coordinate the table named.
        // `route` off walks the axis instead, which is the control the test breaks it with.
        let experts = tok.level(comptime!(expert.clone()));
        let experts = match comptime!(route) {
            true => experts.routed(EXPERT, e),
            false => experts,
        };

        for slab in experts {
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

fn x_values() -> Vec<f32> {
    (0..TOKENS * FEATURES).map(|i| (i % 5) as f32).collect()
}

/// Each expert's weights are its own: expert `e` scales by `e + 1`, so contracting a token
/// against the wrong slab is off by a whole factor.
fn w_values() -> Vec<f32> {
    (0..EXPERTS * FEATURES * FEATURES)
        .map(|i| {
            let e = i / (FEATURES * FEATURES);
            let rest = i % (FEATURES * FEATURES);
            ((e + 1) * (rest % 3 + 1)) as f32
        })
        .collect()
}

fn run(route: bool) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            // EXPERT at its true extent: the walk visits one of the three, the space still holds
            // three. Nothing here says a token uses one expert; the walk does.
            Space::new(&[
                (M, TOKENS),
                (N, FEATURES),
                (K, FEATURES),
                (EXPERT, EXPERTS),
            ]),
            vec![Level::walk(&[(M, 1)]), Level::walk(&[(EXPERT, 1)])],
        ),
        KernelForm::Static,
    );

    let (x_handle, _) = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(x_values())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(
        client.clone(),
        Shape::new([EXPERTS, FEATURES, FEATURES]),
    )
    .dtype(f32_ty)
    .custom(w_values())
    .generate_with_f32_host_data();
    let routes_handle = TestInput::builder(client.clone(), Shape::new([TOKENS]))
        .dtype(u32_ty)
        .custom(ROUTES.iter().map(|&r| r as f32).collect())
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([TOKENS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; TOKENS * FEATURES])
        .generate_without_host_data();

    moe_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            x_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            w_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[EXPERT, K, N]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        routes_handle.binding().into_tensor_arg(),
        launcher.space_arg(),
        launcher.level(0),
        launcher.level(1),
        route,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

/// The reference: each token against its own expert, folded on the host.
fn expected() -> Vec<Vec<f32>> {
    let (x, w) = (x_values(), w_values());
    (0..TOKENS)
        .map(|m| {
            let e = ROUTES[m] as usize;
            (0..FEATURES)
                .map(|n| {
                    (0..FEATURES)
                        .map(|k| {
                            x[m * FEATURES + k]
                                * w[e * FEATURES * FEATURES + k * FEATURES + n]
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_routed_walk_contracts_each_token_against_its_own_expert() {
    let got = run(true);
    let want = expected();
    for (m, row) in want.iter().enumerate() {
        for (n, cell) in row.iter().enumerate() {
            assert_eq!(
                got.get_f32(&[m, n]),
                *cell,
                "token {m} feature {n}: expert {}",
                ROUTES[m]
            );
        }
    }
}

/// Without the routed coordinate the same kernel walks all three experts and folds every one of
/// them into the token, so the answer is a sum over experts rather than a pick of one. The
/// mechanism is what the test above measures, not the arithmetic around it.
#[test]
fn without_the_route_the_walk_folds_every_expert() {
    let got = run(false);
    let want = expected();
    let differs = (0..TOKENS)
        .flat_map(|m| (0..FEATURES).map(move |n| (m, n)))
        .any(|(m, n)| got.get_f32(&[m, n]) != want[m][n]);
    assert!(
        differs,
        "dropping the route changed nothing, so the routed coordinate is not what places the \
         weights window"
    );
}
