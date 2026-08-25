//! Convolution as a client of the tile DSL: a gather-reduce whose reduce axes are *abstract*.
//!
//! A convolution's window axes are not physical axes of the input. They address the same physical
//! axis its output axes address, at their own coefficient: `Ih = Oh*stride + Rh*dilation`. So the
//! input tile spans more logical axes than its buffer has physical ones, consecutive tiles overlap
//! by the receptive field,
//! and one input element is read by several output positions. That mapping is the operand's
//! [`Projection`]; everything else (the space, the walk, `Tile::at`, the leaf) is the same
//! machinery matmul runs on.
#![allow(non_snake_case)]

use cubecl::{
    Runtime, TestRuntime,
    prelude::*,
    zspace::{Shape, shape},
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

use cubek_tile::*;

// Output positions, output channels, window taps, input channels. `OW`/`RW` are the second
// spatial pair the 2-D case adds.
const OH: Axis = Axis(0);
const CO: Axis = Axis(1);
const RH: Axis = Axis(2);
const CI: Axis = Axis(3);
const OW: Axis = Axis(4);
const RW: Axis = Axis(5);

/// The same body matmul runs: the operands' spaces say what is contracted, their projections say
/// how they are addressed, and `mma` does the rest.
#[cube(launch)]
fn conv_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// [`conv_kernel`] serving the input in two-wide lines. The gathered operand lines along its
/// fastest contracted axis, which is the one the leaf splits into a line index and a lane, so the
/// width has to be a real one somewhere to exercise that fold at all.
#[cube(launch)]
fn conv_kernel_lined<E: Numeric>(
    input: &TileArg<'_, E, Const<2>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// Small integers, so the accumulation is exact in `f32` and the reference can be compared for
/// equality rather than closeness.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % period) as f32) - 1.0).collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    in_shape: Shape,
    w_shape: Shape,
    out_shape: Shape,
    in_spec: TileSpec,
    w_axes: &[Axis],
    out_spec: TileSpec,
    space: Space,
    in_v: usize,
    instruction: Instruction,
) -> (HostData, Vec<f32>, Vec<f32>) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let in_data = ramp(in_shape.num_elements(), 7);
    let w_data = ramp(w_shape.num_elements(), 5);

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // Every binding stays scalar-unit: the kernel's element type carries the width, and `Tile::of`
    // is what converts the shape and strides to lines.
    let in_binding = in_handle.binding();
    // `#[cube(launch)]` mints its own element type per kernel, so the two arms cannot share
    // pre-built arguments; only the bindings are common.
    let w_binding = w_handle.binding();
    // The two inputs of one contraction stage at the same levels here, so the weight follows the
    // gathered input's plan rather than repeating it at every call site.
    let w_spec = TileSpec::direct(w_axes).residence(&in_spec.residence);
    let out_binding = out_handle.clone().binding();
    match in_v {
        1 => conv_kernel::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
            TileArgLaunch::new(w_binding.into_tensor_arg(), w_spec),
            TileArgLaunch::new(out_binding.into_tensor_arg(), out_spec),
            space.with_instruction(instruction),
            f32_ty,
        ),
        2 => conv_kernel_lined::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
            TileArgLaunch::new(w_binding.into_tensor_arg(), w_spec),
            TileArgLaunch::new(out_binding.into_tensor_arg(), out_spec),
            space.with_instruction(instruction),
            f32_ty,
        ),
        other => panic!("conv: no kernel at input width {other}"),
    }

    (
        HostData::from_tensor_handle(&client, out_handle, HostDataType::F32),
        in_data,
        w_data,
    )
}

// ---- 1-D -------------------------------------------------------------------

/// `out[o, co] = Σ_{r, ci} w[r, ci, co] · in[o·stride + r·dilation, ci]`.
struct Conv1d {
    oh: usize,
    co: usize,
    rh: usize,
    ci: usize,
    stride: usize,
    dilation: usize,
}

impl Conv1d {
    /// The input length a valid (unpadded) convolution consumes: the last output position's last
    /// tap, plus one.
    fn in_len(&self) -> usize {
        (self.oh - 1) * self.stride + (self.rh - 1) * self.dilation + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let x = input[(o * self.stride + r * self.dilation) * self.ci + i];
                        let w = weight[(r * self.ci + i) * self.co + c];
                        acc += x * w;
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_co, 1, false, Buffering::SINGLE, &[])
    }

    /// `check` with the input served in `in_v`-wide lines and, when `checked`, both the input and
    /// the output bounds-masked: the two axes of the gather path a plain `check` leaves at their
    /// degenerate values. `buffering` says how deeply the level is buffered; `residence` says
    /// whether the leaf gathers straight out of gmem (`InPlace`) or out of a compacted stage
    /// (`Smem`).
    fn check_at(
        &self,
        tile_oh: usize,
        tile_co: usize,
        in_v: usize,
        checked: bool,
        buffering: Buffering,
        residence: &[Residence],
    ) {
        self.check_at_with_instruction(
            tile_oh,
            tile_co,
            in_v,
            checked,
            buffering,
            residence,
            Instruction::registers(16),
        )
    }

    /// `check_at` with an explicit instruction. Kept separate so the fan-out path can be traced
    /// on the CPU test runtime even though production CPU launches select the flat walk.
    #[allow(clippy::too_many_arguments)]
    fn check_at_with_instruction(
        &self,
        tile_oh: usize,
        tile_co: usize,
        in_v: usize,
        checked: bool,
        buffering: Buffering,
        residence: &[Residence],
        instruction: Instruction,
    ) {
        let space = Tiling::new()
            .extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)])
            .level(WalkOrder::RowMajor, buffering, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        // The input's one gathered physical axis: the output position at `stride`, the tap at
        // `dilation`.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.stride), (RH, self.dilation)]),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .checked(checked)
        .residence(residence);

        let (got, input, weight) = run(
            shape![self.in_len(), self.ci],
            shape![self.rh, self.ci, self.co],
            shape![self.oh, self.co],
            in_spec,
            &[RH, CI, CO],
            TileSpec::direct(&[OH, CO]).checked(checked),
            space,
            in_v,
            instruction,
        );

        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "conv1d {buffering:?} stride {} dilation {} tile {tile_oh}x{tile_co}: \
                     wrong at ({o}, {c})",
                    self.stride,
                    self.dilation
                );
            }
        }
    }
}

/// Stride 1, dilation 1: consecutive windows overlap by `rh - 1`, the plainest gather there is.
#[test]
fn conv1d_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// Stride 2: the output steps twice as fast as the input, so a tile's span is no longer its edge.
#[test]
fn conv1d_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check(4, 4);
}

/// Dilation 2: the taps themselves are spread, so the two coefficients differ.
#[test]
fn conv1d_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 2,
    }
    .check(3, 4);
}

/// Both at once, and a tap count that does not divide anything.
#[test]
fn conv1d_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check(2, 2);
}

/// A 1x1 window is a matmul wearing a projection: the gather degenerates and must still agree.
#[test]
fn conv1d_single_tap_is_a_matmul() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check(4, 4);
}

/// One output tile covering everything: the whole receptive field is one window.
#[test]
fn conv1d_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check(6, 4);
}

/// Padded 1-D convolution: taps that underflow physical bounds mask to 0.
#[test]
fn conv1d_padded_underflow_masks_to_zero() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let space = Tiling::new()
        .extents(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(OH, Cut::sequential(3))
                .axis(CO, Cut::sequential(4))
                .axis(RH, Cut::sequential(rh))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine_with_offset(
                &[(OH, stride), (RH, dilation)],
                -(padding as isize),
            ),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .checked(true);

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).checked(true),
        space,
        1,
        Instruction::registers(16),
    );

    // Reference with zero padding:
    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "padded conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// Padded 1-D convolution under `Boundary::Clamp`: taps that go past physical bounds read the
/// edge cell instead of masking to zero.
#[test]
fn conv1d_padded_underflow_clamps_to_edge() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let space = Tiling::new()
        .extents(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(OH, Cut::sequential(3))
                .axis(CO, Cut::sequential(4))
                .axis(RH, Cut::sequential(rh))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine_with_offset(
                &[(OH, stride), (RH, dilation)],
                -(padding as isize),
            ),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .with_boundary(Some(Boundary::Clamp));

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).checked(true),
        space,
        1,
        Instruction::registers(16),
    );

    // Reference clamping the tap to the input's edge row instead of masking to zero:
    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let hc = h.clamp(0, in_len as isize - 1) as usize;
                    let x = input[hc * ci + i];
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "clamped conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// The same padded convolution under a staged schedule.
#[test]
fn conv1d_padded_staged_underflow_masks_to_zero() {
    let oh = 6;
    let co = 4;
    let rh = 3;
    let ci = 2;
    let stride = 1;
    let dilation = 1;
    let padding = 1;
    let in_len = 6;

    let space = Tiling::new()
        .extents(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(OH, Cut::sequential(3))
                .axis(CO, Cut::sequential(4))
                .axis(RH, Cut::sequential(rh))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine_with_offset(
                &[(OH, stride), (RH, dilation)],
                -(padding as isize),
            ),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .checked(true)
    .residence(&[Residence::Smem]);

    let (got, input, weight) = run(
        shape![in_len, ci],
        shape![rh, ci, co],
        shape![oh, co],
        in_spec,
        &[RH, CI, CO],
        TileSpec::direct(&[OH, CO]).checked(true),
        space,
        1,
        Instruction::registers(16),
    );

    let mut want = vec![0.0f32; oh * co];
    for o in 0..oh {
        for c in 0..co {
            let mut acc = 0.0f32;
            for r in 0..rh {
                for i in 0..ci {
                    let h = (o * stride + r * dilation) as isize - padding as isize;
                    let x = if h >= 0 && (h as usize) < in_len {
                        input[(h as usize) * ci + i]
                    } else {
                        0.0
                    };
                    let w = weight[(r * ci + i) * co + c];
                    acc += x * w;
                }
            }
            want[o * co + c] = acc;
        }
    }

    for o in 0..oh {
        for c in 0..co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * co + c],
                "padded staged conv1d: wrong at ({o}, {c})"
            );
        }
    }
}

/// The gathered operand in two-wide lines. Its innermost axis is `CI`, the fastest contracted one,
/// so the leaf splits each reduce step into the line index it reads and the lane it broadcasts:
/// the one arithmetic on the gather path that a width of `1` leaves dead.
#[test]
fn conv1d_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, false, Buffering::SINGLE, &[]);
}

/// The GPU specialization uses fixed lane extracts. Exercise that path on the CPU test runtime as
/// well, even though production CPU launches select the compact flat walk.
#[test]
fn conv1d_vectorized_input_fanout() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at_with_instruction(
        4,
        4,
        2,
        false,
        Buffering::SINGLE,
        &[],
        Instruction::Registers {
            config: RegisterBlock::new(16, false, true),
        },
    );
}

/// Vectorized with stride and dilation both off `1`, so the line fold and the affine advance are
/// exercised at once rather than one at a time.
#[test]
fn conv1d_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(3, 2, 2, false, Buffering::SINGLE, &[]);
}

/// An output extent the tile edge does not divide: the last tile's receptive field runs past the
/// input's real length. The masking happens under the projection, so the tap that overhangs must
/// read `0` and the output cell that does not exist must not be written, leaving every valid
/// position equal to the reference.
#[test]
fn conv1d_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, true, Buffering::SINGLE, &[]);
}

/// The same overhang with a stride, where the window runs further past the end: the mask is on the
/// physical axis, so how far the gather reached past it must not matter.
#[test]
fn conv1d_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, true, Buffering::SINGLE, &[]);
}

// ---- through the Launcher --------------------------------------------------

impl Conv1d {
    /// [`check_at`](Conv1d::check_at) driven end to end by [`Launcher`]: the input reaches the
    /// launch through [`StridedTileSource::gathered`] instead of a hand-built [`TileSpec`], so the
    /// kernel projects from the kernel-form space and one compiled kernel serves every shape.
    /// `padding` shifts the window's origin, which the builder's derived check has to arm on by
    /// itself. `dynamic` is the axis set the kernel takes at runtime, `None` for all of them.
    fn check_launched_over(
        &self,
        tile_oh: usize,
        tile_co: usize,
        padding: usize,
        buffering: Buffering,
        dynamic: Option<&[Axis]>,
        residence: &[Residence],
    ) {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        let f32_ty = f32::elem_type_native();

        let space = Tiling::new()
            .extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)])
            .level(WalkOrder::RowMajor, buffering, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        // Padding shortens the input by exactly what it shifts the window back by, so the last
        // output position's last tap still lands on the final row.
        let in_len = self.in_len() - padding;
        let in_shape = shape![in_len, self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        // Every axis here can go runtime, including the two the input gathers over: `OH` is stated
        // by the output, which maps it identically, and `RH` by the weight. Only an axis no
        // operand witnesses has to stay static, which is what `dynamic` narrows to.
        let launch = match dynamic {
            Some(axes) => space.launcher_over(&client, axes),
            None => space.launcher(&client),
        };
        let mut in_operand = Operand::new(&[OH, RH, CI], f32_ty);
        let mut w_operand = Operand::new(&[RH, CI, CO], f32_ty);
        for &r in residence {
            in_operand.stage(r);
            w_operand.stage(r);
        }
        let in_arg = launch
            .arg(in_handle.binding())
            .operand(&in_operand)
            .gathered(Projection::new(
                &[OH, RH, CI],
                &[
                    PhysicalAxisMap::affine_with_offset(
                        &[(OH, self.stride), (RH, self.dilation)],
                        -(padding as isize),
                    ),
                    PhysicalAxisMap::of(CI),
                ],
            ))
            .build();
        let w_arg = launch
            .arg(w_handle.binding())
            .subspace(&[RH, CI, CO])
            .operand(&w_operand)
            .build();
        let out_arg = launch
            .arg(out_handle.clone().binding())
            .subspace(&[OH, CO])
            .build();

        conv_kernel::launch::<TestRuntime>(
            &client,
            launch.cube_count(),
            launch.cube_dim(),
            in_arg.arg(),
            w_arg.arg(),
            out_arg.arg(),
            launch
                .space()
                .clone()
                .with_instruction(Instruction::registers(16)),
            f32_ty,
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference_padded(&input, &weight, in_len, padding);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "launched conv1d {buffering:?} padding {padding}: wrong at ({o}, {c})"
                );
            }
        }
    }

    /// [`check_launched_over`](Conv1d::check_launched_over) with every axis runtime.
    fn check_launched(
        &self,
        tile_oh: usize,
        tile_co: usize,
        padding: usize,
        buffering: Buffering,
        residence: &[Residence],
    ) {
        self.check_launched_over(tile_oh, tile_co, padding, buffering, None, residence);
    }
}

/// The two axes the input gathers over, `OH` and `RH`, kept static while the rest goes runtime:
/// the launch a gathered operand was restricted to before any operand could state a gathered axis'
/// size. It has to keep working, since an axis no operand witnesses still has to reach the kernel
/// static.
#[test]
fn conv1d_launched_static_window() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_launched_over(3, 4, 0, Buffering::SINGLE, Some(&[CO, CI]), &[]);
}

/// The same convolution with `OH` and `RH` runtime as well, at a tiling neither divides: the walk
/// counts are `div_ceil`s of sizes read off the output (`OH`) and the weight (`RH`), and the
/// trailing partial tile is masked exactly as the static form's is.
#[test]
fn conv1d_launched_dynamic_window_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 2,
    }
    .check_launched(4, 4, 1, Buffering::SINGLE, &[]);
}

/// The plainest gather off the builder: nothing overhangs and the origin cannot go negative, so
/// the derived check is off and the kernel's own extents are all runtime.
#[test]
fn conv1d_launched_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(4, 4, 0, Buffering::SINGLE, &[]);
}

/// Stride and dilation both off `1`, staged: the affine advance and the compaction have to survive
/// the dynamic space the launcher hands the kernel.
#[test]
fn conv1d_launched_staged_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_launched(2, 2, 0, Buffering::SINGLE, &[Residence::Smem]);
}

/// A padded window with no overhang anywhere: nothing about the tiling says the operand needs a
/// check, so the mask is armed by the negative origin alone and the padded taps must read zero.
#[test]
fn conv1d_launched_padding_arms_the_check() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(3, 4, 1, Buffering::SINGLE, &[]);
}

/// An output extent the tile edge does not divide, so the overhang arms the check through the
/// affine map: the tap that runs past the input reads `0` and the row that does not exist is not
/// written.
#[test]
fn conv1d_launched_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_launched(4, 4, 0, Buffering::SINGLE, &[]);
}

// ---- runtime coefficients --------------------------------------------------

/// [`conv_kernel`] with the input's stride and dilation arriving as runtime scalars rather than
/// baked into its projection: the two `Scale::Dynamic` coefficients ride the tile.
#[cube(launch)]
fn conv_kernel_dynamic<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    stride: u32,
    dilation: u32,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    // In `Projection::dynamic_scale_index` order: physical axis 0's terms, `OH` then `RH`.
    let mut coefficients = Coords::<u32>::new();
    coefficients.push(stride);
    coefficients.push(dilation);

    let input = input.tile_gathered(comptime!(space.clone()), coefficients, Coords::new());
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

impl Conv1d {
    /// [`check`](Conv1d::check) with both coefficients `Dynamic`, so the same convolution runs off
    /// a kernel that was compiled without knowing either. Their bounds are the exact values here,
    /// the tightest a stage can be sized at; `check_dynamic_padded` covers the staged schedule.
    fn check_dynamic(&self, tile_oh: usize, tile_co: usize) {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        let f32_ty = f32::elem_type_native();

        let space = Tiling::new()
            .extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::scaled(&[
                    (OH, Scale::Dynamic { max: self.stride }),
                    (RH, Scale::Dynamic { max: self.dilation }),
                ]),
                PhysicalAxisMap::of(CI),
            ],
        ));

        let in_shape = shape![self.in_len(), self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        conv_kernel_dynamic::launch::<TestRuntime>(
            &client,
            space.cube_count(),
            space.cube_dim(&client),
            TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
            TileArgLaunch::new(
                w_handle.binding().into_tensor_arg(),
                TileSpec::direct(&[RH, CI, CO]),
            ),
            TileArgLaunch::new(
                out_handle.clone().binding().into_tensor_arg(),
                TileSpec::direct(&[OH, CO]),
            ),
            self.stride as u32,
            self.dilation as u32,
            space.with_instruction(Instruction::registers(16)),
            f32_ty,
        );

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "dynamic conv1d stride {} dilation {}: wrong at ({o}, {c})",
                    self.stride,
                    self.dilation
                );
            }
        }
    }
}

/// The degenerate coefficients, which a `Static` projection folds away entirely: the runtime path
/// has to produce the same answer without that folding.
#[test]
fn conv1d_dynamic_unit_coefficients() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic(4, 4);
}

/// A stride and a dilation that both exceed one, over several tiles: the window advance and the
/// receptive field are then runtime values, which is what the descent has to size its window from.
#[test]
fn conv1d_dynamic_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_dynamic(2, 2);
}

/// [`conv_kernel`] with the input's padding arriving as a runtime scalar: the `Offset::Dynamic`
/// rides the tile as a signed value, while the stride and the dilation stay comptime.
#[cube(launch)]
fn conv_kernel_dynamic_padding<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    offset: i32,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let mut offsets = Coords::<i32>::new();
    offsets.push(offset);

    let input = input.tile_gathered(comptime!(space.clone()), Coords::new(), offsets);
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

/// [`conv_kernel_dynamic`] with the padding runtime too: nothing of the input's affine map is
/// known when the kernel is compiled.
#[cube(launch)]
fn conv_kernel_all_dynamic<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    stride: u32,
    dilation: u32,
    offset: i32,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let mut coefficients = Coords::<u32>::new();
    coefficients.push(stride);
    coefficients.push(dilation);
    let mut offsets = Coords::<i32>::new();
    offsets.push(offset);

    let input = input.tile_gathered(comptime!(space.clone()), coefficients, offsets);
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

impl Conv1d {
    /// The padded reference: a tap outside `[0, in_len)` reads zero.
    fn reference_padded(
        &self,
        input: &[f32],
        weight: &[f32],
        in_len: usize,
        padding: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let h = (o * self.stride + r * self.dilation) as isize - padding as isize;
                        let x = if h >= 0 && (h as usize) < in_len {
                            input[(h as usize) * self.ci + i]
                        } else {
                            0.0
                        };
                        acc += x * weight[(r * self.ci + i) * self.co + c];
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    /// A padded convolution whose padding is an `Offset::Dynamic`, so the window origin is placed
    /// at runtime and the underflow guard is armed without knowing the sign. `dynamic_scales` also
    /// hands the stride and the dilation over at runtime, which forces `Residence::InPlace` (no
    /// staging); a dynamic offset alone stages, since the compaction drops it.
    #[allow(clippy::too_many_arguments)]
    fn check_dynamic_padded(
        &self,
        tile_oh: usize,
        tile_co: usize,
        padding: usize,
        in_len: usize,
        buffering: Buffering,
        dynamic_scales: bool,
        residence: &[Residence],
    ) {
        let client = <TestRuntime as Runtime>::client(&Default::default());
        let f32_ty = f32::elem_type_native();

        let space = Tiling::new()
            .extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)])
            .level(WalkOrder::RowMajor, buffering, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        let gathered = if dynamic_scales {
            PhysicalAxisMap::scaled_with_offset(
                &[
                    (OH, Scale::Dynamic { max: self.stride }),
                    (RH, Scale::Dynamic { max: self.dilation }),
                ],
                Offset::Dynamic,
            )
        } else {
            PhysicalAxisMap::affine_with_offset(
                &[(OH, self.stride), (RH, self.dilation)],
                Offset::Dynamic,
            )
        };
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[gathered, PhysicalAxisMap::of(CI)],
        ))
        .checked(true)
        .residence(residence);
        let w_spec = TileSpec::direct(&[RH, CI, CO]).residence(residence);

        let in_shape = shape![in_len, self.ci];
        let w_shape = shape![self.rh, self.ci, self.co];
        let input = ramp(in_shape.num_elements(), 7);
        let weight = ramp(w_shape.num_elements(), 5);

        let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
            .dtype(f32_ty)
            .custom(input.clone())
            .generate_with_f32_host_data();
        let (w_handle, _) = TestInput::builder(client.clone(), w_shape)
            .dtype(f32_ty)
            .custom(weight.clone())
            .generate_with_f32_host_data();
        let out_handle = TestInput::builder(client.clone(), shape![self.oh, self.co])
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data();

        let in_binding = in_handle.binding();
        let w_binding = w_handle.binding();
        let out_binding = out_handle.clone().binding();
        let offset = -(padding as i32);
        if dynamic_scales {
            conv_kernel_all_dynamic::launch::<TestRuntime>(
                &client,
                space.cube_count(),
                space.cube_dim(&client),
                TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
                TileArgLaunch::new(w_binding.into_tensor_arg(), w_spec),
                TileArgLaunch::new(
                    out_binding.into_tensor_arg(),
                    TileSpec::direct(&[OH, CO]).checked(true),
                ),
                self.stride as u32,
                self.dilation as u32,
                offset,
                space.with_instruction(Instruction::registers(16)),
                f32_ty,
            );
        } else {
            conv_kernel_dynamic_padding::launch::<TestRuntime>(
                &client,
                space.cube_count(),
                space.cube_dim(&client),
                TileArgLaunch::new(in_binding.into_tensor_arg(), in_spec),
                TileArgLaunch::new(w_binding.into_tensor_arg(), w_spec),
                TileArgLaunch::new(
                    out_binding.into_tensor_arg(),
                    TileSpec::direct(&[OH, CO]).checked(true),
                ),
                offset,
                space.with_instruction(Instruction::registers(16)),
                f32_ty,
            );
        }

        let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
        let want = self.reference_padded(&input, &weight, in_len, padding);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "dynamic padded conv1d {buffering:?} padding {padding}: wrong at ({o}, {c})"
                );
            }
        }
    }
}

/// A runtime padding under a direct schedule: the taps that underflow must still mask to zero,
/// though nothing proved at comptime that the origin was negative.
#[test]
fn conv1d_dynamic_padding_direct() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 6, Buffering::SINGLE, false, &[]);
}

/// The same padding staged: a dynamic offset costs no comptime window geometry at all, so it
/// survives the compaction into smem without even a bound to declare.
#[test]
fn conv1d_dynamic_padding_staged() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 6, Buffering::SINGLE, false, &[Residence::Smem]);
}

/// Stride, dilation and padding all runtime at once: the whole affine map arrives with the launch.
#[test]
fn conv1d_all_dynamic() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 8, Buffering::SINGLE, true, &[]);
}

/// The same thing staged: the runtime coefficients size their smem box off their declared bounds,
/// so the stage holds the window and the launch's actual stride and dilation read out of it.
#[test]
fn conv1d_all_dynamic_staged() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_dynamic_padded(3, 4, 1, 8, Buffering::SINGLE, true, &[Residence::Smem]);
}

// ---- 2-D -------------------------------------------------------------------

/// `out[oh, ow, co] = Σ_{rh, rw, ci} w[rh, rw, ci, co] · in[oh·sh + rh·dh, ow·sw + rw·dw, ci]`.
/// Two abstract dimensions, each gathering its own input physical axis.
struct Conv2d {
    oh: usize,
    ow: usize,
    co: usize,
    rh: usize,
    rw: usize,
    ci: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
}

impl Conv2d {
    fn in_h(&self) -> usize {
        (self.oh - 1) * self.sh + (self.rh - 1) * self.dh + 1
    }
    fn in_w(&self) -> usize {
        (self.ow - 1) * self.sw + (self.rw - 1) * self.dw + 1
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let (iw, ci, co) = (self.in_w(), self.ci, self.co);
        let mut out = vec![0.0f32; self.oh * self.ow * co];
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..co {
                    let mut acc = 0.0f32;
                    for rh in 0..self.rh {
                        for rw in 0..self.rw {
                            for i in 0..ci {
                                let h = oh * self.sh + rh * self.dh;
                                let w_ = ow * self.sw + rw * self.dw;
                                let x = input[(h * iw + w_) * ci + i];
                                let k = weight[((rh * self.rw + rw) * ci + i) * co + c];
                                acc += x * k;
                            }
                        }
                    }
                    out[(oh * self.ow + ow) * co + c] = acc;
                }
            }
        }
        out
    }

    fn check(&self, tile_oh: usize, tile_ow: usize, tile_co: usize) {
        self.check_at(tile_oh, tile_ow, tile_co, Buffering::SINGLE, &[])
    }

    /// `check` under `buffering`: `InPlace` gathers straight out of gmem, `Smem` compacts the two
    /// gathered physical axes into a dense stage first.
    fn check_at(
        &self,
        tile_oh: usize,
        tile_ow: usize,
        tile_co: usize,
        buffering: Buffering,
        residence: &[Residence],
    ) {
        let space = Tiling::new()
            .extents(&[
                (OH, self.oh),
                (OW, self.ow),
                (CO, self.co),
                (RH, self.rh),
                (RW, self.rw),
                (CI, self.ci),
            ])
            .level(WalkOrder::RowMajor, buffering, |l| {
                l.axis(OH, Cut::sequential(tile_oh))
                    .axis(OW, Cut::sequential(tile_ow))
                    .axis(CO, Cut::sequential(tile_co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(RW, Cut::sequential(self.rw))
                    .axis(CI, Cut::sequential(self.ci))
            })
            .build();

        // Two gathered physical axes, one per spatial axis pair; the channel axis rides identity.
        let in_spec = TileSpec::new(Projection::new(
            &[OH, OW, RH, RW, CI],
            &[
                PhysicalAxisMap::affine(&[(OH, self.sh), (RH, self.dh)]),
                PhysicalAxisMap::affine(&[(OW, self.sw), (RW, self.dw)]),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .residence(residence);

        let (got, input, weight) = run(
            shape![self.in_h(), self.in_w(), self.ci],
            shape![self.rh, self.rw, self.ci, self.co],
            shape![self.oh, self.ow, self.co],
            in_spec,
            &[RH, RW, CI, CO],
            TileSpec::direct(&[OH, OW, CO]),
            space,
            1,
            Instruction::registers(16),
        );

        let want = self.reference(&input, &weight);
        for oh in 0..self.oh {
            for ow in 0..self.ow {
                for c in 0..self.co {
                    assert_eq!(
                        got.get_f32(&[oh, ow, c]),
                        want[(oh * self.ow + ow) * self.co + c],
                        "conv2d {buffering:?}: wrong at ({oh}, {ow}, {c})"
                    );
                }
            }
        }
    }
}

#[test]
fn conv2d_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(2, 2, 4);
}

/// Different stride and dilation per spatial axis, so the two projected physical axes cannot be
/// confused.
#[test]
fn conv2d_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check(2, 3, 2);
}

/// One tile covering the whole output: every axis's window is the full receptive field.
#[test]
fn conv2d_single_tile() {
    Conv2d {
        oh: 3,
        ow: 3,
        co: 2,
        rh: 2,
        rw: 2,
        ci: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check(3, 3, 2);
}

// ---- staged ----------------------------------------------------------------

// The same convolutions with the input staged ([`Residence::Smem`]). A gathered operand's region is
// copied into a shared-memory tile shaped like the physical *window* it reads (`span(oh, rh) × ci`),
// compacted onto the lattice its stride and dilation reach, so each input element is stored once. The
// stage keeps the operand's own projection, so the leaf gathers out of smem exactly as it gathered
// out of gmem, and every case here must agree with its direct (in-place) twin.

/// Stride 1: consecutive windows overlap by `rh - 1`, so the window the stage holds
/// (`oh + rh - 1`) is smaller than the `oh × rh` logical cells reading it.
#[test]
fn conv1d_staged_dense_window() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// Stride and dilation both off `1`: the fill's per-axis advance and the receptive-field span are
/// what place each staged element, so neither can collapse to the edge.
#[test]
fn conv1d_staged_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 4,
        ci: 3,
        stride: 3,
        dilation: 2,
    }
    .check_at(2, 2, 1, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// A single tap at unit stride: the window is exactly the output's own extent, so the stage is the
/// smallest it ever gets while the projection stays affine (`Rh` is still a declared axis at
/// coefficient `1`, so the leaf stays on the N-D nest). The answer must not move.
#[test]
fn conv1d_staged_single_tap() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// One region covering everything: the walk runs once, so the fill is the whole kernel.
#[test]
fn conv1d_staged_single_tile() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(6, 4, 1, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// Staged in two-wide lines: the fill moves whole lines, and the innermost axis is the one the
/// projection is required to carry at coefficient `1` for that to be sound.
#[test]
fn conv1d_staged_vectorized_input() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// Vectorized with stride and dilation both off `1`.
#[test]
fn conv1d_staged_vectorized_strided_and_dilated() {
    Conv1d {
        oh: 6,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 2,
        dilation: 2,
    }
    .check_at(3, 2, 2, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// The overhanging last tile: the mask sits under the projection, so a tap past the input's real
/// length must stage a `0` rather than whatever the buffer holds there.
#[test]
fn conv1d_staged_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 1, true, Buffering::SINGLE, &[Residence::Smem]);
}

/// The same overhang with a stride, where the window runs further past the end.
#[test]
fn conv1d_staged_masked_overhang_strided() {
    Conv1d {
        oh: 5,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, true, Buffering::SINGLE, &[Residence::Smem]);
}

/// A single tap at stride 2 reaches only every second input position, so the stage keeps half of
/// the window it spans and the fill steps by two. One of the three non-unit-step cases, with
/// `conv1d_staged_vectorized_strided_and_dilated` (`gcd(2, 2) = 2`, above) and
/// `conv2d_staged_mixed_steps`.
#[test]
fn conv1d_staged_single_tap_strided() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 1,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(4, 4, 1, false, Buffering::SINGLE, &[Residence::Smem]);
}

/// The overhang masked *and* the input lined: the fill steps whole lines through a window whose
/// last taps read past the input, so the line fold and the mask have to hold at once.
#[test]
fn conv1d_staged_vectorized_masked_overhang() {
    Conv1d {
        oh: 7,
        co: 4,
        rh: 3,
        ci: 4,
        stride: 1,
        dilation: 1,
    }
    .check_at(4, 4, 2, true, Buffering::SINGLE, &[Residence::Smem]);
}

/// Double buffering drives the same fill from two slots on alternating regions, so a gathered
/// source has to re-window per slot rather than once per walk.
#[test]
fn conv1d_staged_double_buffered() {
    Conv1d {
        oh: 8,
        co: 4,
        rh: 3,
        ci: 2,
        stride: 2,
        dilation: 1,
    }
    .check_at(2, 4, 1, false, Buffering::DOUBLE, &[Residence::Smem]);
}

/// Two gathered physical axes staged at once: the fill decodes one destination coordinate per
/// logical axis, so both affine maps are resolved from the same index.
#[test]
fn conv2d_staged_dense_window() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 3,
        rw: 3,
        ci: 2,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
    }
    .check_at(2, 2, 4, Buffering::SINGLE, &[Residence::Smem]);
}

/// Different stride and dilation per spatial axis, so the two staged physical axes cannot be
/// confused for one another.
#[test]
fn conv2d_staged_asymmetric_stride_and_dilation() {
    Conv2d {
        oh: 4,
        ow: 3,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 1,
        dw: 2,
    }
    .check_at(2, 3, 2, Buffering::SINGLE, &[Residence::Smem]);
}

/// One compacted physical axis per step: `h` has `gcd(2, 2) = 2`, so its stage keeps every second
/// row and the fill steps through it, while `w` is dense and steps by one. The only case where
/// `StepUp` carries more than one distinct step, so a transposed or broadcast step shows up here.
#[test]
fn conv2d_staged_mixed_steps() {
    Conv2d {
        oh: 4,
        ow: 4,
        co: 4,
        rh: 2,
        rw: 3,
        ci: 2,
        sh: 2,
        sw: 1,
        dh: 2,
        dw: 1,
    }
    .check_at(2, 2, 4, Buffering::SINGLE, &[Residence::Smem]);
}

// ---- the projected 2-D view ------------------------------------------------

/// Reads a gathered operand through [`Tile::matrix`] and writes every batch matrix out flat, so
/// the host can check the projected layout against the same gather done by hand. The 2-D door for
/// an operand whose logical axes outnumber its buffer's physical ones: the matrix coordinate
/// resolves through the leading axes first, then folds onto the window through the projection.
#[cube(launch)]
fn projected_matrix_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    out: &mut Tensor<f32>,
    #[comptime] space: Space,
    #[comptime] matrices: usize,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(space);
    let size!(W) = input.vector_size();

    #[unroll]
    for m in 0..matrices {
        let view = input.matrix::<W>(m);
        #[unroll]
        for r in 0..rows {
            #[unroll]
            for c in 0..cols {
                let value = view.read((r as u32, c as u32).runtime());
                out[comptime!((m * rows + r) * cols + c)] = f32::cast_from(value);
            }
        }
    }
}

/// The one 2-D convolution both view tests read, strided and dilated on both spatial axes so no
/// two logical coordinates land on the same input cell by accident.
struct Conv2dViewSetup {
    /// `(oh, ow, rh, rw, ci)`, the logical axes in the space's own order.
    logical: (usize, usize, usize, usize, usize),
    /// `(sh, sw, dh, dw)`, the strides then the dilations.
    steps: (usize, usize, usize, usize),
    in_w: usize,
    space: Space,
    in_spec: TileSpec,
    in_data: Vec<f32>,
    in_handle: cubecl::std::tensor::TensorHandle<TestRuntime>,
}

fn setup_conv2d_view() -> Conv2dViewSetup {
    let (oh, ow, rh, rw, ci) = (3usize, 4usize, 2usize, 3usize, 2usize);
    let (sh, sw, dh, dw) = (2usize, 1usize, 1usize, 2usize);
    let in_h = (oh - 1) * sh + (rh - 1) * dh + 1;
    let in_w = (ow - 1) * sw + (rw - 1) * dw + 1;

    let space = Tiling::new()
        .extents(&[(OH, oh), (OW, ow), (RH, rh), (RW, rw), (CI, ci)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(OH, Cut::sequential(oh))
                .axis(OW, Cut::sequential(ow))
                .axis(RH, Cut::sequential(rh))
                .axis(RW, Cut::sequential(rw))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, OW, RH, RW, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, sh), (RH, dh)]),
            PhysicalAxisMap::affine(&[(OW, sw), (RW, dw)]),
            PhysicalAxisMap::of(CI),
        ],
    ));

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();
    let in_data = ramp(in_h * in_w * ci, 7);
    let (in_handle, _) = TestInput::builder(client.clone(), shape![in_h, in_w, ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();

    Conv2dViewSetup {
        logical: (oh, ow, rh, rw, ci),
        steps: (sh, sw, dh, dw),
        in_w,
        space,
        in_spec,
        in_data,
        in_handle,
    }
}

/// The 2-D view over the 2-D convolution's input: logical `[OH, OW, RH, RW, CI]` over the physical
/// `[IH, IW, CI]`, so three leading axes are pinned and the trailing `RW x CI` pair is the matrix.
/// Three pinned axes is what makes this worth running: the unravel's weights are a product of
/// several extents, and reading those off the window instead of the space would silently pick up
/// the receptive field's span (`IH`, `IW`) rather than the logical edges.
#[test]
fn conv2d_projected_matrix_view() {
    let s = setup_conv2d_view();
    let (oh, ow, rh, rw, ci) = s.logical;
    let (sh, sw, dh, dw) = s.steps;
    let (in_w, in_data, space) = (s.in_w, s.in_data, s.space);

    let matrices = oh * ow * rh;
    let (rows, cols) = (rw, ci);

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();
    let out_handle = TestInput::builder(client.clone(), shape![matrices, rows, cols])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    projected_matrix_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(s.in_handle.binding().into_tensor_arg(), s.in_spec),
        out_handle.clone().binding().into_tensor_arg(),
        space,
        matrices,
        rows,
        cols,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for m in 0..matrices {
        // The pinned axes, unraveled the way the layout unravels them: row-major over `[OH, OW, RH]`.
        let (o_h, o_w, r_h) = (m / (ow * rh), (m / rh) % ow, m % rh);
        for r_w in 0..rows {
            for c_i in 0..cols {
                let h = o_h * sh + r_h * dh;
                let w = o_w * sw + r_w * dw;
                assert_eq!(
                    got.get_f32(&[m, r_w, c_i]),
                    in_data[(h * in_w + w) * ci + c_i],
                    "projected matrix {m} ({o_h}, {o_w}, {r_h}) at ({r_w}, {c_i})"
                );
            }
        }
    }
}

/// Reads a gathered operand through [`Tile::fragment_matrix`] and writes the whole matrix out, so
/// the host can check it against the im2col expansion done by hand. This is the face an mma
/// fragment reads: the output positions flattened into the row edge, the taps and channels into
/// the column edge, resolved onto the compact window underneath.
#[cube(launch)]
fn fragment_matrix_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    out: &mut Tensor<f32>,
    #[comptime] space: Space,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(space);
    let size!(W) = input.vector_size();
    let view = input.fragment_matrix::<E, W, W>(rows, cols);

    #[unroll]
    for r in 0..rows {
        #[unroll]
        for c in 0..cols {
            let value = view.read((r as u32, c as u32).runtime());
            out[comptime!(r * cols + c)] = f32::cast_from(value);
        }
    }
}

/// The 2-D convolution input as one `(OH·OW) x (RH·RW·CI)` matrix: five logical axes flattened
/// into two edges over three physical ones. Neither edge is a single axis, which is the whole
/// point: no pinning of trailing axes produces this face, and it is the one an mma contracts.
#[test]
fn conv2d_fragment_matrix_view() {
    let s = setup_conv2d_view();
    let (oh, ow, rh, rw, ci) = s.logical;
    let (sh, sw, dh, dw) = s.steps;
    let (in_w, in_data, space) = (s.in_w, s.in_data, s.space);

    let (rows, cols) = (oh * ow, rh * rw * ci);

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();
    let out_handle = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    fragment_matrix_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(s.in_handle.binding().into_tensor_arg(), s.in_spec),
        out_handle.clone().binding().into_tensor_arg(),
        space,
        rows,
        cols,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for r in 0..rows {
        let (o_h, o_w) = (r / ow, r % ow);
        for c in 0..cols {
            // The column edge unravels row-major over `[RH, RW, CI]`, as the layout groups it.
            let (r_h, r_w, c_i) = (c / (rw * ci), (c / ci) % rw, c % ci);
            let h = o_h * sh + r_h * dh;
            let w = o_w * sw + r_w * dw;
            assert_eq!(
                got.get_f32(&[r, c]),
                in_data[(h * in_w + w) * ci + c_i],
                "fragment matrix at ({r}, {c}) = ({o_h}, {o_w}, {r_h}, {r_w}, {c_i})"
            );
        }
    }
}

// ---- the manual-mma leaf ---------------------------------------------------

/// The resident promote → zero → mma → drain kernel of the matmul tests, with a *gathered* lhs.
/// The accumulator is sized by the whole contraction (taps times channels), the input stages into
/// its compacted smem window, and the fragment load flattens the tap and channel axes back into
/// the `k` edge as it reads that window.
#[cube(launch)]
fn conv_mma_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight = weight.tile(comptime!(space.clone()));
    let out = out.tile(space);
    let mut acc = out.accumulate::<E, _>(&input, Monoid::Sum);
    acc.zero();
    acc.mma(&input, &weight);
}

#[test]
fn conv1d_mma_leaf() {
    conv1d_mma_leaf_with(MmaIOConfig::manual());
}

#[test]
fn conv1d_mma_leaf_gathered_lhs_ignores_ldmatrix() {
    conv1d_mma_leaf_with(MmaIOConfig {
        lhs_load_method: LoadMethod::LoadMatrix,
        ..MmaIOConfig::manual()
    });
}

fn conv1d_mma_leaf_with(io: MmaIOConfig) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if client.properties().features.matmul.mma.is_empty() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no manual mma (features.matmul.mma) support".to_string(),
        ))
        .enforce();
        return;
    }

    // `k = rh * ci = 8`, an 8x8x8 instruction, as the matmul mma test uses.
    let (oh, co, rh, ci) = (8usize, 8usize, 2usize, 4usize);
    let (stride, dilation) = (1usize, 1usize);
    let in_len = (oh - 1) * stride + (rh - 1) * dilation + 1;
    let instruction = Instruction::Mma { io };

    let space = Tiling::new()
        .extents(&[(OH, oh), (CO, co), (RH, rh), (CI, ci)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(OH, Cut::sequential(oh))
                .axis(CO, Cut::sequential(co))
                .axis(RH, Cut::sequential(rh))
                .axis(CI, Cut::sequential(ci))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::affine(&[(OH, stride), (RH, dilation)]),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .residence(&[Residence::Smem]);

    let f32_ty = f32::elem_type_native();
    let in_data = ramp(in_len * ci, 7);
    let w_data = ramp(rh * ci * co, 5);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![in_len, ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![rh, ci, co])
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![oh, co])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    conv_mma_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            w_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[RH, CI, CO]).residence(&[Residence::Smem]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            // The manual-mma leaf contracts register fragments, so the output states where its
            // accumulator lives: registers, for the one level there is.
            TileSpec::direct(&[OH, CO]).residence(&[Residence::Register]),
        ),
        space.with_instruction(instruction),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for o in 0..oh {
        for c in 0..co {
            let want: f32 = (0..rh)
                .flat_map(|r| (0..ci).map(move |i| (r, i)))
                .map(|(r, i)| {
                    let h = o * stride + r * dilation;
                    in_data[h * ci + i] * w_data[(r * ci + i) * co + c]
                })
                .sum();
            assert_eq!(
                got.get_f32(&[o, c]),
                want,
                "conv1d mma leaf: wrong at ({o}, {c})"
            );
        }
    }
}

// ---- rational (fractional scale) -------------------------------------------

/// `out[o, co] = Σ_{r, ci} w[r, ci, co] · in[⌊(o·scale + r·tap + offset) / divisor⌋, ci]`: the
/// continuous-resize mapping, where an output position lands between input cells so the window
/// steps a whole cell on some outputs and stays put on others.
struct Resize1d {
    oh: usize,
    co: usize,
    rh: usize,
    ci: usize,
    in_len: usize,
    scale: usize,
    tap: usize,
    offset: isize,
    divisor: usize,
}

impl Resize1d {
    /// The input cell tap `r` of output `o` reads, `None` outside the buffer, where
    /// [`Boundary::Zero`] contributes nothing.
    fn source(&self, o: usize, r: usize) -> Option<usize> {
        let numerator = (o * self.scale + r * self.tap) as isize + self.offset;
        let h = numerator.div_euclid(self.divisor as isize);
        (h >= 0 && (h as usize) < self.in_len).then_some(h as usize)
    }

    fn reference(&self, input: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.oh * self.co];
        for o in 0..self.oh {
            for c in 0..self.co {
                let mut acc = 0.0f32;
                for r in 0..self.rh {
                    for i in 0..self.ci {
                        let x = self.source(o, r).map_or(0.0, |h| input[h * self.ci + i]);
                        acc += x * weight[(r * self.ci + i) * self.co + c];
                    }
                }
                out[o * self.co + c] = acc;
            }
        }
        out
    }

    /// One level per entry, each cutting `OH` at that edge and keeping the other axes whole: two
    /// entries nest a second descent, which is where a rational window's leftover phase has to
    /// accumulate rather than restart.
    fn space(&self, oh_edges: &[usize]) -> Space {
        self.space_with_buffering(oh_edges, Buffering::SINGLE)
    }

    fn space_with_buffering(&self, oh_edges: &[usize], buffering: Buffering) -> Space {
        let mut tiling =
            Tiling::new().extents(&[(OH, self.oh), (CO, self.co), (RH, self.rh), (CI, self.ci)]);
        for (i, &edge) in oh_edges.iter().enumerate() {
            let sched = if i == 0 { buffering } else { Buffering::SINGLE };
            tiling = tiling.level(WalkOrder::RowMajor, sched, |l| {
                l.axis(OH, Cut::sequential(edge))
                    .axis(CO, Cut::sequential(self.co))
                    .axis(RH, Cut::sequential(self.rh))
                    .axis(CI, Cut::sequential(self.ci))
            });
        }
        tiling.build()
    }

    fn check(&self, oh_edges: &[usize]) {
        self.check_with(oh_edges, Buffering::SINGLE, 1, &[]);
    }

    fn check_with(
        &self,
        oh_edges: &[usize],
        buffering: Buffering,
        vector_size: usize,
        residence: &[Residence],
    ) {
        let in_spec = TileSpec::new(Projection::new(
            &[OH, RH, CI],
            &[
                PhysicalAxisMap::affine_with_offset(
                    &[(OH, self.scale), (RH, self.tap)],
                    self.offset,
                )
                .over(self.divisor),
                PhysicalAxisMap::of(CI),
            ],
        ))
        .checked(true)
        .residence(residence);

        let (got, input, weight) = run(
            shape![self.in_len, self.ci],
            shape![self.rh, self.ci, self.co],
            shape![self.oh, self.co],
            in_spec,
            &[RH, CI, CO],
            TileSpec::direct(&[OH, CO]).checked(true),
            self.space_with_buffering(oh_edges, buffering),
            vector_size,
            Instruction::registers(16),
        );

        let want = self.reference(&input, &weight);
        for o in 0..self.oh {
            for c in 0..self.co {
                assert_eq!(
                    got.get_f32(&[o, c]),
                    want[o * self.co + c],
                    "resize1d {}/{} offset {} edges {oh_edges:?} buffering {buffering:?} \
                     v {vector_size}: wrong at ({o}, {c})",
                    self.scale,
                    self.divisor,
                    self.offset
                );
            }
        }
    }
}

/// The tap axis carries the divisor as its coefficient, so it survives the division whole: the
/// resample shape, `⌊(o·4 - 2) / 6⌋ + r`. The last outputs read past the input, which
/// `Boundary::Zero` masks.
#[test]
fn resize1d_rational_static() {
    Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check(&[2]);
}

/// The same resample with `Residence::Smem`: the input tile stages uncompacted into shared
/// memory.
#[test]
fn resize1d_staged_static() {
    Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(&[2], Buffering::SINGLE, 1, &[Residence::Smem]);
}

/// Rational gather driven under double buffering on alternating slots.
#[test]
fn resize1d_staged_double_buffered() {
    Resize1d {
        oh: 8,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 6,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(&[2], Buffering::DOUBLE, 1, &[Residence::Smem]);
}

/// A tap coefficient the divisor does not cancel: the window itself is fractionally dilated, so
/// two taps can land on one input cell. Walked over two levels, where the second descent starts
/// from the phase the first left over instead of from the projection's own offset.
#[test]
fn resize1d_rational_fractional_taps() {
    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 3,
        ci: 2,
        in_len: 8,
        scale: 5,
        tap: 2,
        offset: -2,
        divisor: 3,
    };
    resize.check(&[3]);
    resize.check(&[3, 1]);
}

/// Staged rational gather with fractional taps.
#[test]
fn resize1d_staged_fractional_taps() {
    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 3,
        ci: 2,
        in_len: 8,
        scale: 5,
        tap: 2,
        offset: -2,
        divisor: 3,
    };
    resize.check_with(&[3], Buffering::SINGLE, 1, &[Residence::Smem]);
    resize.check_with(&[3, 1], Buffering::SINGLE, 1, &[Residence::Smem]);
}

/// Staged rational gather served in two-wide lines on the ungathered `CI` axis.
#[test]
fn resize1d_staged_vectorized() {
    Resize1d {
        oh: 6,
        co: 4,
        rh: 2,
        ci: 4,
        in_len: 6,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    }
    .check_with(&[2], Buffering::SINGLE, 2, &[Residence::Smem]);
}

/// A rational gathered operand whose divisor and offset arrive at runtime.
#[cube(launch)]
fn conv_kernel_rational_dynamic<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    weight: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    divisor: u32,
    offset: i32,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    // The one carried coefficient is the divisor: both scales are Static, so it is the whole
    // carrier (`Projection::dynamic_divisor_index` puts it at 0).
    let mut coefficients = Coords::<u32>::new();
    coefficients.push(divisor);
    let mut offsets = Coords::<i32>::new();
    offsets.push(offset);

    let input = input.tile_gathered(comptime!(space.clone()), coefficients, offsets);
    let weight = weight.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.zero();
    out.mma(&input, &weight);
}

#[test]
fn resize1d_rational_dynamic() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    };
    let space = resize.space(&[2]);

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::scaled_with_offset(
                &[(OH, Scale::Static(4)), (RH, Scale::Static(6))],
                Offset::Dynamic,
            )
            .over(Divisor::Dynamic {
                min: resize.divisor,
            }),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .checked(true);

    let in_data = ramp(resize.in_len * resize.ci, 7);
    let w_data = ramp(resize.rh * resize.ci * resize.co, 5);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![resize.in_len, resize.ci])
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let (w_handle, _) = TestInput::builder(client.clone(), shape![resize.rh, resize.ci, resize.co])
        .dtype(f32_ty)
        .custom(w_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![resize.oh, resize.co])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    conv_kernel_rational_dynamic::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            w_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[RH, CI, CO]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[OH, CO]).checked(true),
        ),
        resize.divisor as u32,
        resize.offset as i32,
        space.with_instruction(Instruction::registers(16)),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let want = resize.reference(&in_data, &w_data);
    for o in 0..resize.oh {
        for c in 0..resize.co {
            assert_eq!(
                got.get_f32(&[o, c]),
                want[o * resize.co + c],
                "resize1d dynamic: wrong at ({o}, {c})"
            );
        }
    }
}

/// A rational gathered stage born with dynamic coefficients can be addressed before fill
/// without tripping AxisProjection's dynamic coefficient count assert.
#[cube(launch)]
fn conv_kernel_rational_dynamic_stage_read<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    divisor: u32,
    offset: i32,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let mut coefficients = Coords::<u32>::new();
    coefficients.push(divisor);
    let mut offsets = Coords::<i32>::new();
    offsets.push(offset);

    let input = input.tile_gathered(comptime!(space.clone()), coefficients, offsets);
    let stage = MemData::smem_like(&input);
    let _view = stage.nd::<E, Const<1>, Const<1>>();
}

#[test]
fn resize1d_dynamic_stage_read_before_fill() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let resize = Resize1d {
        oh: 6,
        co: 2,
        rh: 2,
        ci: 3,
        in_len: 4,
        scale: 4,
        tap: 6,
        offset: -2,
        divisor: 6,
    };
    let space = resize.space(&[2]);

    let in_spec = TileSpec::new(Projection::new(
        &[OH, RH, CI],
        &[
            PhysicalAxisMap::scaled_with_offset(
                &[(OH, Scale::Static(4)), (RH, Scale::Static(6))],
                Offset::Dynamic,
            )
            .over(Divisor::Dynamic {
                min: resize.divisor,
            }),
            PhysicalAxisMap::of(CI),
        ],
    ))
    .checked(true);

    let (in_handle, _) = TestInput::builder(client.clone(), shape![resize.in_len, resize.ci])
        .dtype(f32_ty)
        .custom(ramp(resize.in_len * resize.ci, 7))
        .generate_with_f32_host_data();

    conv_kernel_rational_dynamic_stage_read::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        resize.divisor as u32,
        resize.offset as i32,
        space,
        f32_ty,
    );
}
