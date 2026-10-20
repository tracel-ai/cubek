use cubecl::{
    prelude::*,
    zspace::{Tiling, shape, strides},
};
use cubek_matmul::{definition::MatmulElems, launch::launch_ref, strategy::Strategy};
use cubek_std::{
    InputBinding, MatrixLayout,
    config::{CubekConfig, MatmulConfig, RuntimeConfig},
    launch::tma::tma_operand_typed,
};

// A separate integration-test binary keeps the process-wide setting independent
// of the main suite's defaults. Run with any supported CubeCL test backend.
#[test]
fn auto_preserves_f32_operands_and_tma_descriptor_types() {
    CubekConfig::set(CubekConfig {
        matmul: MatmulConfig { allow_tf32: false },
    });
    let client = cubecl::test_device().client();
    let ty = f32::elem_type_native();
    // Exact in F32 but rounds to 1.0 in TF32, exposing accidental narrowing.
    let value = 1.0f32 + 1.0 / 4096.0;
    let binding = |value| TensorBinding {
        handle: client
            .create_from_slice(bytemuck::cast_slice(&[value; 32 * 32]))
            .binding(),
        shape: shape![32, 32],
        strides: strides![32, 1],
        tiling: Tiling::UNTILED,
    };
    let lhs = binding(value);
    let rhs = binding(1.0f32);
    let output = client.empty(32 * 32 * 4);
    let out = TensorBinding {
        handle: output.clone().binding(),
        shape: shape![32, 32],
        strides: strides![32, 1],
        tiling: Tiling::UNTILED,
    };
    let mut elems = MatmulElems::from_single_dtype(ty);
    #[cfg(feature = "multi-level")]
    {
        use cubek_matmul::{
            definition::MatmulSetupError,
            multi_level::{Strategy as MultiLevel, definition::adjust_dtypes},
        };
        adjust_dtypes(&client, &mut elems, true);
        assert_eq!(elems.lhs_register, ty);
        assert_eq!(elems.rhs_register, ty);
        let offers_f32 = client
            .properties()
            .features
            .matmul
            .cmma
            .iter()
            .any(|cfg| cfg.a_type == ty && cfg.b_type == ty && cfg.cd_type == ty);
        if !offers_f32 {
            let result = launch_ref(
                &MultiLevel::SimpleCyclicCmma(Default::default()).into(),
                &client,
                InputBinding::Normal(lhs.clone(), ty),
                InputBinding::Normal(rhs.clone(), ty),
                out.clone(),
                &mut elems,
            );
            assert!(
                matches!(result, Err(MatmulSetupError::Unavailable(_))),
                "an accelerator without F32 instructions must decline: {result:?}"
            );
        }
    }
    launch_ref(
        &Strategy::Auto,
        &client,
        InputBinding::Normal(lhs.clone(), ty),
        InputBinding::Normal(rhs, ty),
        out,
        &mut elems,
    )
    .unwrap();
    let bytes = client.read_one(output).unwrap();
    let actual: &[f32] = bytemuck::cast_slice(&bytes);
    assert!(actual.iter().all(|&v| v == 32.0 * value));
    assert_eq!(elems.lhs_register, ty);
    assert_eq!(elems.rhs_register, ty);

    // Constructing descriptors needs no TMA-capable hardware. Check both layouts
    // and both resolved types: the helper must preserve the selector's decision.
    for elem in [ty, tf32::elem_type_native()] {
        let (arg, _) = tma_operand_typed(
            lhs.clone(),
            1,
            MatrixLayout::RowMajor,
            (32, 32),
            elem,
            TensorMapSwizzle::None,
        );
        assert_eq!(arg.metadata.elem_ty, elem);
        #[cfg(feature = "tiled")]
        {
            let mut tiled = lhs.clone();
            tiled.shape = shape![1, 1, 32, 32];
            tiled.strides = strides![1024, 1024, 32, 1];
            tiled.tiling = Tiling::new(&[2, 2]).unwrap();
            let arg = cubek_std::launch::tma::tma_operand_tiled_typed(
                tiled,
                (32, 32),
                cubek_tile::launch::TmaLanding::Tiles,
                elem,
                TensorMapSwizzle::None,
            );
            assert_eq!(arg.metadata.elem_ty, elem);
        }
    }
}
