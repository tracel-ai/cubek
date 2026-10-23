use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::ComptimeOptionArgs;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubecl::{CubeCount, CubeDim};
use cubek_std::Hardware;

use super::PaddedRows;
use super::kernel::{combine_routes, gate_rows, group_tiles, pad_rows, unpad_rows};
use crate::kernels::tensor::PlainTensor;

const U32: ElemType = ElemType::UInt(UIntKind::U32);

/// Rows grouped by slab, laid out for a grouped product: every group padded to
/// start on a multiple of `align` rows, and the slab of each tile of them.
///
/// Built on the device from the groups' ends, never read back: the padded
/// extent is a bound the host computes, `rows + groups · (align - 1)` rounded
/// up to whole tiles, and every tile past the real ones names no slab.
#[derive(Clone)]
pub struct GroupedRows {
    client: Client,
    ends: PlainTensor,
    table: TensorHandle,
    sources: TensorHandle,
    starts: TensorHandle,
    rows: usize,
    align: usize,
}

impl GroupedRows {
    /// The layout of `rows` rows grouped by `ends` on `client`'s device, each group's cumulative
    /// end row as a `u32`, one group a slab, padded to `align`.
    ///
    /// The ends are the caller's to get right, since checking them would read
    /// them back: non-decreasing, the last one `rows`. Wrong ones cost the
    /// answer and nothing else: a group that runs backwards takes no tile, and
    /// no row past `rows` is read or written.
    ///
    /// # Panics
    ///
    /// On ends other than one `u32` per group, at least one group, or a zero
    /// `align`.
    pub fn new(client: &Client, ends: PlainTensor, rows: usize, align: usize) -> Self {
        assert!(align > 0, "a tile holds at least one row");
        assert_eq!(ends.elem, U32, "the groups' ends are `u32` rows");
        assert_eq!(ends.shape().rank(), 1, "the groups' ends are one per group");
        let ends = ends.into_contiguous(client);
        let groups = ends.shape().num_elements();
        assert!(groups > 0, "rows are grouped into one slab at least");
        let tiles = Self::tiles(rows, groups, align);
        let table = Self::u32s(client, tiles);
        let sources = Self::u32s(client, tiles);
        let starts = Self::u32s(client, groups + 1);
        let cube_dim = Hardware::from_client(client).cube_dim_1d(256);
        group_tiles::launch(
            client,
            CubeCount::Static(groups.div_ceil(cube_dim.num_elems() as usize) as u32, 1, 1),
            cube_dim,
            ends.binding.clone().into_tensor_arg(),
            table.clone().binding().into_tensor_arg(),
            sources.clone().binding().into_tensor_arg(),
            starts.clone().binding().into_tensor_arg(),
            align as u32,
        );
        Self {
            client: client.clone(),
            ends,
            table,
            sources,
            starts,
            rows,
            align,
        }
    }

    /// [`new`](Self::new) laid out on the host and uploaded, so it holds where launches are
    /// discarded.
    ///
    /// # Panics
    ///
    /// On no group, or a zero `align`.
    pub fn uploaded(client: &Client, ends: &[u32], rows: usize, align: usize) -> Self {
        assert!(align > 0, "a tile holds at least one row");
        assert!(!ends.is_empty(), "rows are grouped into one slab at least");
        let groups = ends.len();
        let tiles = Self::tiles(rows, groups, align);
        let last = ends[groups - 1];
        let mut table = vec![groups as u32; tiles];
        let mut sources = vec![last; tiles];
        let mut starts = Vec::with_capacity(groups + 1);
        let (mut tile, mut start) = (0, 0);
        for (group, &end) in ends.iter().enumerate() {
            starts.push(tile as u32);
            let owned = (end.saturating_sub(start) as usize).div_ceil(align);
            for t in (0..owned).take_while(|t| tile + t < tiles) {
                table[tile + t] = group as u32;
                sources[tile + t] = start + (t * align) as u32;
            }
            tile += owned;
            start = end;
        }
        starts.push(tile as u32);
        Self {
            client: client.clone(),
            ends: PlainTensor::of_handle(upload(client, ends)),
            table: upload(client, &table),
            sources: upload(client, &sources),
            starts: upload(client, &starts),
            rows,
            align,
        }
    }

    /// Rows a tile holds: every group starts on a multiple of it.
    pub fn align(&self) -> usize {
        self.align
    }

    /// The slab of each tile, `u32`: the sentinel `groups` past the real ones.
    pub fn table(&self) -> TensorHandle {
        self.table.clone()
    }

    /// Each group's first tile, `u32`, and one past the last group's: group
    /// `g`'s tiles are `starts[g]..starts[g + 1]`, `groups + 1` entries.
    pub fn starts(&self) -> TensorHandle {
        self.starts.clone()
    }

    /// The client the layout lives on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// `rows[rows, width]` laid out padded: `[tiles · align, width]`, padding
    /// zero.
    ///
    /// # Panics
    ///
    /// On rows other than the `rows` this layout was built for.
    pub fn pad(&self, rows: PlainTensor) -> TensorHandle {
        assert_eq!(rows.shape().rank(), 2, "grouped rows are `[rows, width]`");
        assert_eq!(rows.shape()[0], self.rows, "the rows this layout groups");
        self.pad_from(rows, None)
    }

    /// `padded[tiles · align, width]` back to the rows as they were grouped,
    /// padding dropped: `[rows, width]`.
    ///
    /// # Panics
    ///
    /// On padded rows other than this layout's.
    pub fn unpad(&self, padded: PlainTensor) -> TensorHandle {
        assert_eq!(padded.shape().rank(), 2, "padded rows are `[rows, width]`");
        assert_eq!(
            padded.shape()[0],
            self.padded_rows(),
            "the padded rows this layout lays out"
        );
        let padded = padded.into_contiguous(&self.client);
        let width = padded.shape()[1];
        let rows = self.output([self.rows, width], padded.elem);
        let flat = FlatLaunch::new(&self.client, padded.elem, [self.padded_rows(), width]);
        unpad_rows::launch(
            &self.client,
            flat.count,
            flat.cube_dim,
            flat.vector,
            padded.binding.clone().into_tensor_arg(),
            rows.clone().binding().into_tensor_arg(),
            self.ends.binding.clone().into_tensor_arg(),
            self.table.clone().binding().into_tensor_arg(),
            self.sources.clone().binding().into_tensor_arg(),
            self.align,
            padded.elem,
        );
        rows
    }

    /// How many rows are grouped, padding aside.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The rows `sort` placed, padded and gathered from their tokens in the
    /// same pass: sorted place `p` holds row `route_at_place[p] / slots` of
    /// `tokens`, `[tokens, width]`, so the sorted copy of the tokens never
    /// exists. `[tiles · align, width]`, padding zero.
    ///
    /// # Panics
    ///
    /// On a `route_at_place` other than one `u32` per row this layout groups.
    pub fn pad_routes(&self, tokens: PlainTensor, sort: RouteSort) -> TensorHandle {
        assert_eq!(tokens.shape().rank(), 2, "tokens are `[tokens, width]`");
        let route_at_place = &sort.route_at_place;
        assert_eq!(route_at_place.elem, U32, "sorted places are `u32`");
        assert_eq!(
            route_at_place.shape().num_elements(),
            self.rows,
            "one sorted place per row this layout groups"
        );
        self.pad_from(tokens, Some(sort))
    }

    /// `silu(gate) · up` of `both`, a product's `[gate | up]` rows padded by
    /// another layout over the same groups, padded by this one: the gated
    /// activation moves the rows between two products' tiles in the pass it
    /// makes anyway. `[tiles · align, width / 2]`, padding zero.
    ///
    /// # Panics
    ///
    /// On `both` grouping other rows than this layout, or a width that is not
    /// two halves.
    pub fn gate(&self, both: &PaddedRows) -> TensorHandle {
        let from = both.layout();
        assert_eq!(from.rows, self.rows, "both layouts group the same rows");
        let rows = PlainTensor::of_handle(both.rows().clone()).into_contiguous(&self.client);
        let width = rows.shape()[1];
        assert!(
            width.is_multiple_of(2),
            "`[gate | up]` rows hold two halves"
        );
        let half = width / 2;
        let gated = self.output([self.padded_rows(), half], rows.elem);
        let flat = FlatLaunch::new(&self.client, rows.elem, [self.padded_rows(), half]);
        gate_rows::launch(
            &self.client,
            flat.count,
            flat.cube_dim,
            flat.vector,
            rows.binding.clone().into_tensor_arg(),
            gated.clone().binding().into_tensor_arg(),
            self.ends.binding.clone().into_tensor_arg(),
            self.table.clone().binding().into_tensor_arg(),
            self.sources.clone().binding().into_tensor_arg(),
            from.starts.clone().binding().into_tensor_arg(),
            self.rows as u32,
            self.align,
            from.align,
            rows.elem,
        );
        gated
    }

    /// Every token's routes read back out of `padded`, rows in this layout,
    /// and summed under their weights:
    /// `out[t] = Σ_s weights[t, s] · padded[row of route t · slots + s]`,
    /// where `routes` names each route's group and `place_of_route` its place
    /// among the grouped rows, one `u32` per route, and `weights` is
    /// `[tokens, slots]` at the rows' element. `[tokens, width]`, summed in f32
    /// in route order: each output row has one writer.
    ///
    /// # Panics
    ///
    /// On routes or places other than one `u32` per grouped row, or weights
    /// other than `[tokens, slots]` over them at the rows' element.
    pub(super) fn combine_routes(
        &self,
        padded: &TensorHandle,
        routes: PlainTensor,
        place_of_route: PlainTensor,
        weights: PlainTensor,
    ) -> TensorHandle {
        let shape = weights.shape();
        assert_eq!(shape.rank(), 2, "weights are `[tokens, slots]`");
        let [tokens, slots] = [shape[0], shape[1]];
        assert_eq!(tokens * slots, self.rows, "one grouped row per route");
        assert_eq!(
            routes.shape().num_elements(),
            self.rows,
            "one expert per route"
        );
        assert_eq!(
            place_of_route.shape().num_elements(),
            self.rows,
            "one place per route"
        );
        assert!(
            routes.elem == U32 && place_of_route.elem == U32,
            "routes and places are `u32`"
        );
        let grouped = PlainTensor::of_handle(padded.clone()).into_contiguous(&self.client);
        assert_eq!(
            weights.elem, grouped.elem,
            "the weights are read at the rows' element"
        );
        let width = grouped.shape()[1];
        let out = self.output([tokens, width], grouped.elem);
        let flat = FlatLaunch::new(&self.client, grouped.elem, [tokens, width]);
        combine_routes::launch(
            &self.client,
            flat.count,
            flat.cube_dim,
            flat.vector,
            grouped.binding.clone().into_tensor_arg(),
            routes
                .into_contiguous(&self.client)
                .binding
                .into_tensor_arg(),
            place_of_route
                .into_contiguous(&self.client)
                .binding
                .into_tensor_arg(),
            weights.binding.into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
            self.ends.binding.clone().into_tensor_arg(),
            self.starts.clone().binding().into_tensor_arg(),
            slots as u32,
            self.align,
            grouped.elem,
        );
        out
    }

    /// Rows once every group is padded to whole tiles: `tiles · align`.
    pub fn padded_rows(&self) -> usize {
        self.table.shape().num_elements() * self.align
    }

    /// [`padded_rows`](Self::padded_rows) of `rows` rows grouped into `groups`, each padded to
    /// start on a multiple of `align` rows: the bound the layout lays out, whatever the groups
    /// turn out to hold.
    pub fn padded_extent(rows: usize, groups: usize, align: usize) -> usize {
        Self::tiles(rows, groups, align) * align
    }

    /// Tiles of `align` rows that `rows` rows grouped into `groups` take at most, each group
    /// padded to start on a tile: `rows + groups · (align - 1)` rounded up to whole tiles.
    fn tiles(rows: usize, groups: usize, align: usize) -> usize {
        (rows + groups * (align - 1)).div_ceil(align)
    }

    /// An uninitialized row-major `[rows, width]` at `elem`, on the layout's device.
    fn output(&self, [rows, width]: [usize; 2], elem: ElemType) -> TensorHandle {
        PlainTensor::empty_contiguous(&self.client, Shape::new([rows, width]), elem).0
    }

    /// An uninitialized `u32` vector of `len`, which a launch fills.
    fn u32s(client: &Client, len: usize) -> TensorHandle {
        PlainTensor::empty_contiguous(client, Shape::new([len]), U32).0
    }

    /// `source` padded, each grouped row read where it lies or, with
    /// a route sort's places and slots, out of its token.
    fn pad_from(&self, source: PlainTensor, sort: Option<RouteSort>) -> TensorHandle {
        let source = source.into_contiguous(&self.client);
        let width = source.shape()[1];
        let padded = self.output([self.padded_rows(), width], source.elem);
        let flat = FlatLaunch::new(&self.client, source.elem, [self.padded_rows(), width]);
        let (route_at_place, slots) = match sort {
            Some(RouteSort {
                route_at_place,
                slots,
            }) => (
                ComptimeOptionArgs::Some(
                    route_at_place
                        .into_contiguous(&self.client)
                        .binding
                        .into_tensor_arg(),
                ),
                slots,
            ),
            None => (ComptimeOptionArgs::None, 1),
        };
        pad_rows::launch(
            &self.client,
            flat.count,
            flat.cube_dim,
            flat.vector,
            source.binding.clone().into_tensor_arg(),
            route_at_place,
            padded.clone().binding().into_tensor_arg(),
            self.ends.binding.clone().into_tensor_arg(),
            self.table.clone().binding().into_tensor_arg(),
            self.sources.clone().binding().into_tensor_arg(),
            self.rows as u32,
            slots as u32,
            self.align,
            source.elem,
        );
        padded
    }
}

/// Where a route sort placed every route: the route each sorted place holds,
/// one `u32` per route, and the routes each token has, so place `p` reads token
/// `route_at_place[p] / slots`.
#[derive(Debug, Clone)]
pub struct RouteSort {
    pub route_at_place: PlainTensor,
    pub slots: usize,
}

/// `values` uploaded as a `u32` vector on `client`'s device. An upload is not a launch, so a
/// process mode that discards launches keeps it.
pub(crate) fn upload(client: &Client, values: &[u32]) -> TensorHandle {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorHandle::new_contiguous([values.len()], client.create_from_slice(&bytes), U32)
}

/// A flat launch over `[rows, width]` elements: one unit per vector, at the
/// widest io-optimized vector dividing `width`.
struct FlatLaunch {
    count: CubeCount,
    cube_dim: CubeDim,
    vector: usize,
}

impl FlatLaunch {
    fn new(client: &Client, elem: ElemType, [rows, width]: [usize; 2]) -> Self {
        let vector = client
            .io_optimized_vector_sizes(elem.size())
            .filter(|&v| width.is_multiple_of(v))
            .max()
            .unwrap_or(1);
        let units = (rows * width / vector).max(1);
        // A throughput preference, which `cube_dim_1d` clamps to what the device spawns.
        let cube_dim = Hardware::from_client(client).cube_dim_1d(256);
        let cubes = units.div_ceil(cube_dim.num_elems() as usize) as u32;
        Self {
            count: CubeCount::Static(cubes, 1, 1),
            cube_dim,
            vector,
        }
    }
}
