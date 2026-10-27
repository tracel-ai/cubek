use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubek_test_utils::client;

use super::{GroupedRows, PaddedRows, RouteSort, upload};
use crate::kernels::tensor::PlainTensor;

/// `values` uploaded as a `u32` vector.
fn u32s(client: &Client, values: &[u32]) -> PlainTensor {
    PlainTensor::of_handle(upload(client, values))
}

/// `values` uploaded as a row-major `f32` tensor of `shape`.
fn f32s(client: &Client, shape: [usize; 2], values: &[f32]) -> TensorHandle {
    TensorHandle::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(values)),
        ElemType::Float(FloatKind::F32),
    )
}

/// A row-major `u32` tensor read back.
fn read_u32(client: &Client, tensor: TensorHandle) -> Vec<u32> {
    let bytes = client
        .read_one(tensor.handle)
        .expect("the tensor reads back");
    u32::from_bytes(&bytes).to_vec()
}

/// A row-major `f32` tensor read back.
fn read_f32(client: &Client, tensor: TensorHandle) -> Vec<f32> {
    let bytes = client
        .read_one(tensor.handle)
        .expect("the tensor reads back");
    f32::from_bytes(&bytes).to_vec()
}

/// Each group's tiles are `starts[g]..starts[g + 1]`: ragged groups, an
/// empty one taking no tile, and one spanning three, at four rows a tile.
#[test]
fn starts_name_each_groups_first_tile() {
    let client = client();
    let (sizes, align) = ([5usize, 0, 11, 2], 4usize);
    let rows: usize = sizes.iter().sum();
    let ends: Vec<u32> = sizes
        .iter()
        .scan(0, |end, size| {
            *end += *size as u32;
            Some(*end)
        })
        .collect();
    let starts = GroupedRows::new(&client, u32s(&client, &ends), rows, align).starts();
    assert_eq!(read_u32(&client, starts), vec![0, 2, 2, 5, 6]);
}

/// The layout laid out on the host is the one `group_tiles` lays out on the device: ragged
/// groups, an empty one, ends that run backwards, a last end short of the rows and one past
/// them. Held through what reads it: the table, the starts, the rows it pads, and a gate
/// reading them from it.
#[test]
fn an_uploaded_layout_is_the_launched_one() {
    let client = client();
    let cases: [(&[u32], usize, usize); 4] = [
        (&[5, 5, 16, 18], 18, 4),
        (&[9, 3, 12], 12, 4),
        (&[2, 7], 10, 3),
        (&[5, 100], 10, 4),
    ];
    for (ends, rows, align) in cases {
        let launched = GroupedRows::new(&client, u32s(&client, ends), rows, align);
        let uploaded = GroupedRows::uploaded(&client, ends, rows, align);
        // Each row padded by its index, so the padded rows name where each tile read from.
        let indices: Vec<f32> = (0..rows).map(|row| row as f32).collect();
        let padded = |layout: &GroupedRows| {
            let rows = PlainTensor::of_handle(f32s(&client, [rows, 1], &indices));
            read_f32(&client, layout.pad(rows))
        };
        assert_eq!(
            read_u32(&client, launched.table()),
            read_u32(&client, uploaded.table()),
            "table of {ends:?}"
        );
        assert_eq!(
            read_u32(&client, launched.starts()),
            read_u32(&client, uploaded.starts()),
            "starts of {ends:?}"
        );
        assert_eq!(
            padded(&launched),
            padded(&uploaded),
            "padded rows of {ends:?}"
        );
        let both = f32s(
            &client,
            [launched.padded_rows(), 2],
            &values(launched.padded_rows() * 2, 1),
        );
        let gated = |from: GroupedRows| {
            let gated = launched.gate(&PaddedRows::new(both.clone(), from));
            read_f32(&client, gated)
        };
        assert_eq!(gated(launched.clone()), gated(uploaded), "gate of {ends:?}");
    }
}

/// How many tokens route how many times over how many experts.
struct Spread {
    tokens: usize,
    slots: usize,
    experts: usize,
}

/// Seven tokens of three routes over five experts: ragged groups, one spanning tiles.
const SPREAD: Spread = Spread {
    tokens: 7,
    slots: 3,
    experts: 5,
};
/// Rows a tile, fewer than the busiest expert's.
const ALIGN: usize = 4;

/// Tokens routed to experts and sorted by expert stably, as the host computes it: the
/// reference every routed kernel here is held to.
struct Routing {
    tokens: usize,
    slots: usize,
    /// `[tokens · slots]`, each route's expert.
    routes: Vec<u32>,
    ends: Vec<u32>,
    route_at_place: Vec<u32>,
    place_of_route: Vec<u32>,
}

impl Routing {
    /// `tokens` tokens of `slots` routes over `experts` experts, the routes spread so some
    /// experts take none and one takes most.
    fn new(
        Spread {
            tokens,
            slots,
            experts,
        }: Spread,
    ) -> Self {
        let routes: Vec<u32> = (0..tokens * slots)
            .map(|route| match route % 5 {
                0 | 3 => 1,
                _ => ((route * 7 + 3) % experts) as u32,
            })
            .collect();
        let named = &routes;
        let route_at_place: Vec<u32> = (0..experts as u32)
            .flat_map(|expert| {
                (0..named.len() as u32).filter(move |&route| named[route as usize] == expert)
            })
            .collect();
        let mut place_of_route = vec![0u32; routes.len()];
        for (place, &route) in route_at_place.iter().enumerate() {
            place_of_route[route as usize] = place as u32;
        }
        let ends = (0..experts as u32)
            .scan(0u32, |end, expert| {
                *end += routes.iter().filter(|&&named| named == expert).count() as u32;
                Some(*end)
            })
            .collect();
        Self {
            tokens,
            slots,
            routes,
            ends,
            route_at_place,
            place_of_route,
        }
    }
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|at| ((at * 37 + seed * 11) % 23) as f32 / 23.0 - 0.5)
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (at, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "{what} at {at}: {got} against {want}"
        );
    }
}

/// Each sorted place padded holds its route's token, whatever the width's vector: an odd
/// width reads one element a unit, and an expert no route names takes no tile.
#[test]
fn pad_routes_lays_out_each_routes_token() {
    let client = client();
    let routing = Routing::new(SPREAD);
    for width in [3usize, 8] {
        let tokens = values(routing.tokens * width, width);
        let layout = GroupedRows::new(
            &client,
            u32s(&client, &routing.ends),
            routing.routes.len(),
            ALIGN,
        );
        let padded = layout.pad_routes(
            PlainTensor::of_handle(f32s(&client, [routing.tokens, width], &tokens)),
            RouteSort {
                route_at_place: u32s(&client, &routing.route_at_place),
                slots: routing.slots,
            },
        );
        let want: Vec<f32> = routing
            .route_at_place
            .iter()
            .flat_map(|&route| {
                let token = route as usize / routing.slots;
                tokens[token * width..(token + 1) * width].to_vec()
            })
            .collect();
        let got = read_f32(&client, layout.unpad(PlainTensor::of_handle(padded)));
        assert_close(&got, &want, "pad_routes");
    }
}

/// The gated rows land in the second layout's tiles: `[gate | up]` rows padded to four rows
/// a tile, gated into a layout of two and of eight.
#[test]
fn gate_moves_the_gated_rows_between_layouts() {
    let client = client();
    let routing = Routing::new(SPREAD);
    let rows = routing.routes.len();
    let ends = u32s(&client, &routing.ends);
    for half in [3usize, 8] {
        let both = values(rows * 2 * half, half);
        let from = GroupedRows::new(&client, ends.clone(), rows, ALIGN);
        let padded = PaddedRows::new(
            from.pad(PlainTensor::of_handle(f32s(
                &client,
                [rows, 2 * half],
                &both,
            ))),
            from,
        );
        let want: Vec<f32> = (0..rows)
            .flat_map(|row| {
                let both = &both[row * 2 * half..(row + 1) * 2 * half];
                (0..half)
                    .map(|at| {
                        let (gate, up) = (both[at], both[half + at]);
                        gate / (1.0 + (-gate).exp()) * up
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for align in [2usize, 8] {
            let to = GroupedRows::new(&client, ends.clone(), rows, align);
            let gated = to.gate(&padded);
            let got = read_f32(&client, to.unpad(PlainTensor::of_handle(gated)));
            assert_close(&got, &want, "gate");
        }
    }
}

/// Every token's routes summed back under their weights out of a padded output, in the
/// order the routes name them.
#[test]
fn combine_routes_sums_each_tokens_routes() {
    let client = client();
    let routing = Routing::new(SPREAD);
    let rows = routing.routes.len();
    let shape = [routing.tokens, routing.slots];
    for width in [3usize, 8] {
        let grouped = values(rows * width, width);
        let weights = values(rows, 3);
        let layout = GroupedRows::new(&client, u32s(&client, &routing.ends), rows, ALIGN);
        let padded = PaddedRows::new(
            layout.pad(PlainTensor::of_handle(f32s(
                &client,
                [rows, width],
                &grouped,
            ))),
            layout,
        );
        let out = padded.combine_routes(
            u32s(&client, &routing.routes),
            u32s(&client, &routing.place_of_route),
            PlainTensor::of_handle(f32s(&client, shape, &weights)),
        );
        let want: Vec<f32> = (0..routing.tokens)
            .flat_map(|token| {
                (0..width)
                    .map(|at| {
                        (0..routing.slots)
                            .map(|slot| {
                                let route = token * routing.slots + slot;
                                let place = routing.place_of_route[route] as usize;
                                weights[route] * grouped[place * width + at]
                            })
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_close(&read_f32(&client, out), &want, "combine_routes");
    }
}
