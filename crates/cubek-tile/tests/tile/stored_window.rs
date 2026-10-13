//! A window that spans several stored tiles, cut down to the tile each unit reads: the unit's
//! place inside its plane's window is a coordinate across the stored tiles, so moving there
//! carries from one tile into the next rather than continuing the first tile's stride.

use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{bytes::Bytes, prelude::*, std::tensor::TensorHandle};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::kind::Storage;
use cubek_tile::layout::{StorageLevels, StoragePartitioning};
use cubek_tile::*;

use super::quant::column_reads::offset;
use super::{Form, implied};

const N: Axis = Axis(0);
/// `K` as blocks of sixteen, and the position inside one.
const KB: Axis = Axis(1);
const KI: Axis = Axis(2);

/// Values a block of `K` holds.
const BLOCK: usize = 16;
/// Columns a stored tile holds side by side.
const TILE_COLUMNS: usize = 8;

/// `output = input`, each unit copying, cell by cell, the window it is cut down to through its
/// cube's and its plane's.
#[cube(launch)]
fn copy_per_unit<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] rank: usize,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let output = output.tile(&space);
    for cube in space {
        let input = input.at(&cube);
        let output = output.at(&cube);
        for plane in cube {
            let input = input.at(&plane);
            let output = output.at(&plane);
            for step in plane {
                let input = input.at(&step);
                let output = output.at(&step);
                for unit in step {
                    copy_cells(&input.at(&unit), &mut output.at(&unit), rank);
                }
            }
        }
    }
}

/// Every cell of `input`'s window of `rank` axes into `output`'s, one at a time through their
/// views, in row-major order.
#[cube]
fn copy_cells<E: Numeric>(input: &Tile<E>, output: &mut Tile<E>, #[comptime] rank: usize) {
    let read = input.view::<Const<1>>();
    let mut write = output.view_mut::<Const<1>>();
    let shape = read.shape();
    let mut cells = 1u32;
    #[unroll]
    for d in 0..rank {
        cells *= shape[d];
    }
    for cell in 0..cells {
        let mut at = CoordsDyn::new();
        #[unroll]
        for d in 0..rank {
            let mut finer = 1u32;
            #[unroll]
            for f in d + 1..rank {
                finer *= shape[f];
            }
            at.push((cell / finer) % shape[d]);
        }
        write.write(at.clone(), read.read(at));
    }
}

/// One level of planes over sixteen columns each, two stored tiles of eight, and under it one
/// unit a column taking four blocks of `K`, interleaved two to a column: the units' level is the
/// storage tile's, so a unit's window is reached by moving inside a plane's window that spans
/// two stored tiles.
fn launcher(client: &Client, cols: usize, blocks: usize) -> Launcher {
    implied(
        client,
        Partitioning::new(
            Space::new(&[(N, cols), (KB, blocks), (KI, BLOCK)]),
            Levels::leaf(&[(N, 1), (KB, 4), (KI, BLOCK)])
                .units(&[(N, 2 * TILE_COLUMNS), (KB, 2)])
                .interleaved(KB)
                .walk_every(&[KB])
                .planes(&[(N, 2)])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    )
}

/// A `[n, k/16, 16]` operand stored as NVFP4 tile order lays it: 8 values a run along `KI`, two
/// runs and four blocks down one column, eight columns side by side, the tiles down `K` then
/// across `N`. The unit at column 8 of a plane's window reads the second tile's first column,
/// not the first column's next tile down `K`.
#[test]
fn a_unit_inside_a_window_across_two_stored_tiles_reads_its_own_column() {
    let storage = StorageLevels::new(&[(KI, 8)])
        .tile(&[(KI, BLOCK / 8), (KB, 4), (N, TILE_COLUMNS)])
        .grid(&[KI, KB, N]);
    check_copy(storage, &[N, KB, KI]);
}

/// The block scales of the same operand, `[n, k/16]` stored four blocks a run along `KB` and
/// eight columns side by side, the tiles down `K` then across `N`: the unit at column 8 of a
/// plane's window reads the second tile's first column.
#[test]
fn a_unit_inside_a_window_across_two_stored_tiles_reads_its_own_scales() {
    let storage = StorageLevels::new(&[(KB, 4)])
        .tile(&[(N, TILE_COLUMNS)])
        .grid(&[KB, N]);
    check_copy(storage, &[N, KB]);
}

/// An operand over `axes` of 32 columns and 8 blocks of `K`, stored as `storage` states, copied
/// out unit by unit under [`launcher`]: every cell lands where its logical coordinate says.
fn check_copy(storage: StoragePartitioning, axes: &[Axis]) {
    let (cols, blocks) = (32, 8);
    let client = cubecl::test_device().client();
    // The levels put 32 units in a plane, and a grid read off them needs the plane to be exactly
    // that wide.
    let plane = client.properties().hardware.plane_size_max;
    if plane != 32 {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "the window takes a 32-unit plane; this runtime reports {plane}"
        )))
        .enforce();
        return;
    }
    let launcher = launcher(&client, cols, blocks);

    let extents: Vec<(Axis, usize)> = axes
        .iter()
        .map(|&axis| match axis {
            N => (N, cols),
            KB => (KB, blocks),
            _ => (KI, BLOCK),
        })
        .collect();
    let shape: Vec<usize> = extents.iter().map(|&(_, extent)| extent).collect();
    let cells: usize = shape.iter().product();
    let geometry = storage.physical(&extents).unwrap();
    let labels = geometry.labels(axes);

    // Each value its own row-major logical index, laid down where the storage puts it.
    let coords = |index: usize| -> Vec<usize> {
        let mut rest = index;
        let mut coords = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            coords[d] = rest % shape[d];
            rest /= shape[d];
        }
        coords
    };
    let mut stored = vec![0f32; cells];
    for index in 0..cells {
        stored[offset(&geometry, &labels, axes, &coords(index))] = index as f32;
    }
    let dtype = f32::elem_type_native();
    let input = TensorHandle::new(
        client.create(Bytes::from_elems(stored)),
        geometry.shape().to_vec(),
        geometry.strides().to_vec(),
        dtype,
    );
    // Filled with a value no cell holds, so a cell no unit wrote shows.
    let output = TestInput::builder(client.clone(), shape.clone())
        .dtype(dtype)
        .custom(vec![-1.0; cells])
        .generate_without_host_data();

    let mut binding = input.binding();
    binding.tiling = geometry.tiling();
    let input_op = launcher.arg(binding).axes(axes).build().unwrap();
    assert_eq!(
        input_op.spec.storage,
        Storage::Tiled(Some(3)),
        "the units' level is the storage tile's"
    );
    let output_op = launcher
        .arg(output.clone().binding())
        .axes(axes)
        .build()
        .unwrap();

    copy_per_unit::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input_op.arg(),
        output_op.arg(),
        launcher.partitioning_arg(),
        axes.len(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for index in 0..cells {
        let at = coords(index);
        assert_eq!(got.get_f32(&at), index as f32, "at {at:?} of {axes:?}");
    }
}
