use crate::resample::run_test;
use cubecl::{Runtime, TestRuntime};
use cubek_resample::definition::{
    BoundaryMode, Kernel, NormalizationMode, Placement, PlacementArgs, Resample, ResampleArgs,
    ResampleAxis, ResampleAxisArgs, Semiring, WindowArgs,
};

#[test]
fn resample_1d_identity_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];

    let resample_args = ResampleArgs::default().with_resample_axis_args(ResampleAxisArgs::new(
        WindowArgs::new(1),
        PlacementArgs::identity(),
    ));

    let resample_axis = ResampleAxis::new(0, Kernel::one(), Placement::Windowed);
    let config = Resample::new(
        Semiring::Linear,
        BoundaryMode::Clamp,
        NormalizationMode::None,
    )
    .with_axis(resample_axis);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        resample_args.to_launch(),
        config,
    );
}

#[test]
fn resample_1d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![8];
    let expected_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];

    let resample_args = ResampleArgs::default().with_resample_axis_args(ResampleAxisArgs::new(
        WindowArgs::new(1),
        PlacementArgs::continuous(0.5, 0.0),
    ));

    let resample_axis = ResampleAxis::new(0, Kernel::one(), Placement::Continuous);
    let config = Resample::new(
        Semiring::Linear,
        BoundaryMode::Clamp,
        NormalizationMode::None,
    )
    .with_axis(resample_axis);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        resample_args.to_launch(),
        config,
    );
}

#[test]
fn resample_2d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![2, 2, 2];
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let output_shape = vec![1, 2, 1];
    let expected_output = vec![1.0, 3.0];

    let resample_args = ResampleArgs::default()
        .with_resample_axis_args(ResampleAxisArgs::new(
            WindowArgs::new(1),
            PlacementArgs::continuous(0.5, 0.0),
        ))
        .with_resample_axis_args(ResampleAxisArgs::new(
            WindowArgs::new(1),
            PlacementArgs::continuous(0.5, 0.0),
        ));

    let kernel = Kernel::one();
    let resample_axis0 = ResampleAxis::new(0, kernel.clone(), Placement::Continuous);
    let resample_axis2 = ResampleAxis::new(2, kernel, Placement::Continuous);
    let config = Resample::new(
        Semiring::Linear,
        BoundaryMode::Clamp,
        NormalizationMode::None,
    )
    .with_axis(resample_axis0)
    .with_axis(resample_axis2);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        resample_args.to_launch(),
        config,
    );
}

#[test]
fn resample_nhwc_2d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![1, 2, 2, 1];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![1, 4, 2, 1];
    let expected_output = vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0];

    let resample_args = ResampleArgs::default()
        .with_resample_axis_args(ResampleAxisArgs::new(
            WindowArgs::new(1),
            PlacementArgs::continuous(0.5, 0.0),
        ))
        .with_resample_axis_args(ResampleAxisArgs::new(
            WindowArgs::new(1),
            PlacementArgs::continuous(0.5, 0.0),
        ));

    let kernel = Kernel::one();
    let resample_axis0 = ResampleAxis::new(0, kernel.clone(), Placement::Continuous);
    let resample_axis1 = ResampleAxis::new(1, kernel, Placement::Continuous);
    let config = Resample::new(
        Semiring::Linear,
        BoundaryMode::Clamp,
        NormalizationMode::None,
    )
    .with_axis(resample_axis0)
    .with_axis(resample_axis1);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        resample_args.to_launch(),
        config,
    );
}
