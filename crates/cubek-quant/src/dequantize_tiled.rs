use cubecl::{
    ir::ElemType,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, ScaleDtype},
};
use cubek_tile::{
    Axis, CubeAxis, Cut, DequantAt, QuantTileArg, Schedule, Space, TileArg, Tiling, WalkOrder,
};

use crate::utils;

/// Dequantize `values` into `output` through the tile engine: the input tile serves the
/// output element and decodes on read, so the kernel body is one copy.
///
/// `values` is declared **in values**: for a packed store its shape and strides count the
/// quantized values (innermost stride 1) while its buffer is narrower by the packing factor.
/// `scales` holds one binding per scale level, innermost first; a two-level scheme's second
/// binding is its global, whole-tensor scale.
///
/// # Errors
///
/// Propagates the kernel's [`LaunchError`]. An unsupported scheme (a non-f32 param, a store
/// that is neither native nor packed-u32, a packing factor no device line covers, or a FULL
/// dim inside a block) panics on the caller's thread instead, so the plan fails loudly.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    values: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: &[TensorBinding<R>],
    scheme: &QuantScheme,
    output_dtype: ElemType,
) -> Result<(), LaunchError> {
    let input_dtype = match scheme.store {
        QuantStore::Native => {
            utils::check_i8_supported(client, scheme);
            ElemType::from_quant_value(scheme.value)
        }
        QuantStore::PackedU32(_) => utils::packed_storage_elem(scheme),
        other => panic!("dequantize_tiled: unsupported storage {other:?} (native or packed-u32)"),
    };
    assert!(
        scheme.scale_dtype() == ScaleDtype::F32,
        "only f32 scales are supported for now."
    );
    assert_eq!(
        scales.len(),
        scheme.num_levels(),
        "dequantize_tiled: one scale binding per level, innermost first"
    );
    let rank = output.shape.len();
    assert!(rank >= 1, "dequantize_tiled: a scalar binding has no axes");
    assert_eq!(
        &values.shape[..],
        &output.shape[..],
        "dequantize_tiled: the values binding is declared in values, so both shapes agree"
    );

    let inner_block = inner_block_edge(scheme, rank);
    let inner_extent = output.shape[rank - 1];
    let type_size = input_dtype.size().max(output_dtype.size());
    let width = served_width(
        client.io_optimized_vector_sizes(type_size),
        inner_extent,
        inner_block,
        scheme.num_quants(),
        &[&values.strides, &output.strides],
    );
    assert!(
        width.is_multiple_of(scheme.num_quants()),
        "dequantize_tiled: a served line may not split a u32, and no width covering the \
         {}-value packing factor qualifies here",
        scheme.num_quants()
    );
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let geometry = Geometry::of(width, plane_size, inner_extent, inner_block);

    let axes: Vec<Axis> = (0..rank).map(|p| Axis(p as u8)).collect();
    let extents: Vec<(Axis, usize)> = axes
        .iter()
        .zip(output.shape.iter())
        .map(|(&axis, &extent)| (axis, extent))
        .collect();
    let space = geometry.space(&extents);
    let launch = space.launcher(client);

    let input_op = launch
        .arg(values)
        .subspace(&axes)
        .vectorize(geometry.width)
        .quantized(scales, *scheme, DequantAt::Read)
        .build();
    let output_op = launch
        .arg(output)
        .subspace(&axes)
        .vectorize(geometry.width)
        .build();

    dequantize::launch::<R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        input_op.bound_width(),
        output_op.vector_size,
        input_op.arg(),
        output_op.arg(),
        launch.space().clone(),
        input_dtype,
        output_dtype,
    );

    Ok(())
}

#[cube(launch)]
/// The input tile serves `O` and dequantizes on read, so the body is a plain copy; `I` (the
/// storage element) only names the binding's element, the scheme recovers the served value.
/// `VI` is the binding width (served over the packing factor), `VO` the served width.
pub fn dequantize<I: Numeric, O: Numeric, VI: Size, VO: Size>(
    input: &QuantTileArg<'_, I, VI>,
    output: &TileArg<'_, O, VO>,
    #[comptime] space: Space,
    #[define(I)] _input_dtype: ElemType,
    #[define(O)] _output_dtype: ElemType,
) {
    let input = input.tile::<O>(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy(&input);
}

/// The unit count one cube aims for; plane counts derive from it per device.
const TARGET_UNITS: usize = 256;

/// One launch's innermost-axis plan. Derived host-side so the kernel only walks what was
/// proven sound: a vectorized plan never overhangs (a checked operand cannot be vectorized)
/// and every edge tiles whole scale blocks or sits inside one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    /// Served values per line, `1` when nothing wider qualifies.
    width: usize,
    /// The innermost cube-tile edge, in values.
    cube_edge: usize,
    /// A second-level plane cut raising the cube's unit count; `None` keeps one plane per cube.
    plane_edge: Option<usize>,
}

impl Geometry {
    fn of(
        width: usize,
        plane_size: usize,
        inner_extent: usize,
        inner_block: Option<usize>,
    ) -> Geometry {
        let fits = |edge: usize| {
            inner_block.is_none_or(|b| edge.is_multiple_of(b) || b.is_multiple_of(edge))
        };
        let plane_edge = plane_size * width;
        let planes = (TARGET_UNITS / plane_size).max(1);
        let preferred = planes * plane_edge;
        if planes > 1
            && inner_extent.is_multiple_of(preferred)
            && fits(preferred)
            && fits(plane_edge)
        {
            return Geometry {
                width,
                cube_edge: preferred,
                plane_edge: Some(plane_edge),
            };
        }
        if inner_extent.is_multiple_of(plane_edge) && fits(plane_edge) {
            return Geometry {
                width,
                cube_edge: plane_edge,
                plane_edge: None,
            };
        }
        if width > 1 {
            // Divides the extent and sits inside a block by served_width's own gates.
            return Geometry {
                width,
                cube_edge: width,
                plane_edge: None,
            };
        }
        // Scalar with a masked overhang; the edge still tiles whole blocks.
        let cube_edge = inner_block.map_or(plane_size, |b| b * plane_size.div_ceil(b));
        Geometry {
            width: 1,
            cube_edge,
            plane_edge: None,
        }
    }

    /// The level stack over `extents` (one axis per tensor dim, innermost last): innermost
    /// tiles ride one-per-cube on X, the next axis on Y and every remaining global axis on Z,
    /// one element per cube. The plane level, when present, cuts nothing the walk visits; it
    /// only raises the cube's unit count for the cooperative fill.
    fn space(&self, extents: &[(Axis, usize)]) -> Space {
        let rank = extents.len();
        let inner = extents[rank - 1].0;
        let global: Vec<Axis> = extents[..rank - 1].iter().map(|&(axis, _)| axis).collect();
        let mut tiling =
            Tiling::new()
                .extents(extents)
                .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                    let mut l = l.axis(inner, Cut::cube(CubeAxis::X, self.cube_edge));
                    if let Some((&y, zs)) = global.split_last() {
                        l = l
                            .axis(y, Cut::cube(CubeAxis::Y, 1))
                            .axes(zs, Cut::cube(CubeAxis::Z, 1));
                    }
                    l
                });
        if let Some(plane_edge) = self.plane_edge {
            tiling = tiling.level(WalkOrder::RowMajor, Schedule::Direct, |l| {
                l.axis(inner, Cut::plane(plane_edge))
                    .axes(&global, Cut::sequential(1))
            });
        }
        tiling.build()
    }
}

/// The widest qualifying served line: it must tile the innermost extent, ride contiguous
/// innermost dims (coarser strides re-express in lines), cover whole packed words, and sit
/// inside the innermost scale block. `1` when nothing wider qualifies.
fn served_width(
    candidates: impl Iterator<Item = usize>,
    inner_extent: usize,
    inner_block: Option<usize>,
    num_quants: usize,
    operands: &[&[usize]],
) -> usize {
    candidates
        .filter(|&v| {
            v == 1
                || (inner_extent.is_multiple_of(v)
                    && v.is_multiple_of(num_quants)
                    && inner_block.is_none_or(|b| b.is_multiple_of(v))
                    && operands.iter().all(|strides| {
                        strides.last() == Some(&1)
                            && strides[..strides.len() - 1]
                                .iter()
                                .all(|&s| s.is_multiple_of(v))
                    }))
        })
        .max()
        .unwrap_or(1)
}

/// The innermost axis's block edge in values, `None` for a per-tensor scheme. A block scheme
/// holding a FULL (0) dim would put a zero edge into the scale windowing, so it is refused.
fn inner_block_edge(scheme: &QuantScheme, rank: usize) -> Option<usize> {
    let block = scheme.block_size()?;
    let dims = block.to_dim_vec(rank);
    assert!(
        dims.iter().all(|&b| b != 0),
        "dequantize_tiled: a FULL (0) dim inside a block scheme is not supported, got {dims:?}"
    );
    Some(dims[rank - 1] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANDIDATES: [usize; 4] = [8, 4, 2, 1];

    fn width(inner_extent: usize, inner_block: Option<usize>, strides: &[usize]) -> usize {
        served_width(
            CANDIDATES.iter().copied(),
            inner_extent,
            inner_block,
            1,
            &[strides, strides],
        )
    }

    #[test]
    fn served_width_takes_the_widest_qualifying_line() {
        assert_eq!(width(128, None, &[128, 1]), 8);
    }

    /// An extent no candidate divides serves scalar; the checked path takes over from there.
    #[test]
    fn served_width_falls_to_scalar_on_an_awkward_extent() {
        assert_eq!(width(127, None, &[127, 1]), 1);
    }

    /// A line may not straddle a scale block, so the block caps the width.
    #[test]
    fn served_width_sits_inside_the_inner_block() {
        assert_eq!(width(128, Some(4), &[128, 1]), 4);
        assert_eq!(width(128, Some(6), &[128, 1]), 2);
    }

    /// A sliced view keeps its parent's coarser strides; a width they cannot re-express in
    /// lines would truncate them.
    #[test]
    fn served_width_respects_strides() {
        assert_eq!(width(128, None, &[128, 2]), 1); // innermost not contiguous
        assert_eq!(width(64, None, &[132, 1]), 4); // coarser stride caps the width
    }

    /// Packing constrains from below: a served line covers whole `u32` words or nothing.
    #[test]
    fn served_width_covers_whole_packed_words() {
        let w = served_width(CANDIDATES.iter().copied(), 128, None, 4, &[&[128, 1]]);
        assert_eq!(w, 8);
        // No candidate reaches the packing factor: scalar comes back and the caller refuses it.
        let w = served_width([2usize, 1].into_iter(), 128, None, 4, &[&[128, 1]]);
        assert_eq!(w, 1);
    }

    #[test]
    fn geometry_prefers_a_multi_plane_cube() {
        let g = Geometry::of(4, 32, 4096, None);
        assert_eq!(
            g,
            Geometry {
                width: 4,
                cube_edge: 1024,
                plane_edge: Some(128)
            }
        );
    }

    #[test]
    fn geometry_falls_back_to_one_plane_per_cube() {
        let g = Geometry::of(4, 32, 128, None);
        assert_eq!(
            g,
            Geometry {
                width: 4,
                cube_edge: 128,
                plane_edge: None
            }
        );
    }

    /// A vectorized width always has a dividing edge (itself), so a shape the plane edge does
    /// not divide narrows the cube rather than degrading to scalar.
    #[test]
    fn geometry_narrows_the_cube_before_giving_up_the_width() {
        let g = Geometry::of(4, 32, 64, None);
        assert_eq!(
            g,
            Geometry {
                width: 4,
                cube_edge: 4,
                plane_edge: None
            }
        );
    }

    #[test]
    fn geometry_scalar_fallback_overhangs_at_the_plane_size() {
        let g = Geometry::of(1, 32, 127, None);
        assert_eq!(
            g,
            Geometry {
                width: 1,
                cube_edge: 32,
                plane_edge: None
            }
        );
    }

    /// The scalar edge still tiles whole blocks: a straddling cut would misaddress scales.
    #[test]
    fn geometry_scalar_fallback_rounds_the_edge_to_whole_blocks() {
        let g = Geometry::of(1, 32, 96, Some(48));
        assert_eq!(
            g,
            Geometry {
                width: 1,
                cube_edge: 48,
                plane_edge: None
            }
        );
    }
}
