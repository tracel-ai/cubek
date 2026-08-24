use cubek_test_utils::CatalogEntry;

use crate::definition::{
    InterpolateBackwardProblem, InterpolateForwardProblem, InterpolateMode, InterpolateOptions,
    InterpolateProblem, NearestMode,
};

/// The same problems with every spatial extent divided, for devices where the full set does not
/// finish in a sitting.
///
/// Only the spatial axes shrink. Batch and channel counts carry the regimes the catalogue exists
/// to separate (RGB against a plane-filling channel count), and the tile geometry is derived from
/// channels and lanes rather than from height and width, so dividing those would compare a
/// different shape rather than the same one faster.
pub fn problems_scaled(divisor: usize) -> Vec<CatalogEntry<InterpolateProblem>> {
    assert!(divisor > 0, "problems_scaled: the divisor is a denominator");
    let shrink = |extent: usize| (extent / divisor).max(1);

    problems()
        .into_iter()
        .map(|entry| {
            let problem = match entry.value {
                InterpolateProblem::Forward(prob) => {
                    InterpolateProblem::Forward(InterpolateForwardProblem {
                        input_height: shrink(prob.input_height),
                        input_width: shrink(prob.input_width),
                        output_height: shrink(prob.output_height),
                        output_width: shrink(prob.output_width),
                        ..prob
                    })
                }
                InterpolateProblem::Backward(prob) => {
                    let [b, h, w, c] = prob.out_grad_shape;
                    InterpolateProblem::Backward(InterpolateBackwardProblem {
                        input_size: [shrink(prob.input_size[0]), shrink(prob.input_size[1])],
                        out_grad_shape: [b, shrink(h), shrink(w), c],
                        ..prob
                    })
                }
            };
            CatalogEntry::new(entry.id, describe(&problem), problem)
        })
        .collect()
}

/// Name a problem from its shapes, so a rescaled entry never keeps the label of the size it had.
fn describe(problem: &InterpolateProblem) -> String {
    match problem {
        InterpolateProblem::Forward(prob) => {
            let direction = match prob.output_height >= prob.input_height {
                true => "upsample",
                false => "downsample",
            };
            format!(
                "{:?} {direction} (b={} h={} w={} c={} -> {}x{})",
                prob.options.mode,
                prob.batch,
                prob.input_height,
                prob.input_width,
                prob.channels,
                prob.output_height,
                prob.output_width,
            )
        }
        InterpolateProblem::Backward(prob) => {
            let [b, h, w, c] = prob.out_grad_shape;
            format!(
                "{:?} backward (b={b} h={} w={} c={c} -> {h}x{w})",
                prob.options.mode, prob.input_size[0], prob.input_size[1],
            )
        }
    }
}

pub fn problems() -> Vec<CatalogEntry<InterpolateProblem>> {
    vec![
        // Nearest
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Nearest upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[4096, 4096],
                InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            )),
        ),
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Nearest upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[4, 512, 512, 16].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            )),
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Nearest downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            )),
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Nearest downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[8, 2048, 1024, 2].into(),
                &[512, 512],
                InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            )),
        ),
        // Bilinear
        CatalogEntry::new(
            "BILINEAR_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Bilinear upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[4096, 4096],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )),
        ),
        CatalogEntry::new(
            "BILINEAR_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Bilinear upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[4, 512, 512, 16].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )),
        ),
        CatalogEntry::new(
            "BILINEAR_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Bilinear downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )),
        ),
        CatalogEntry::new(
            "BILINEAR_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Bilinear downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[8, 2048, 1024, 2].into(),
                &[512, 512],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )),
        ),
        // Bicubic
        CatalogEntry::new(
            "BICUBIC_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Bicubic upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[4096, 4096],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            )),
        ),
        CatalogEntry::new(
            "BICUBIC_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Bicubic upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[4, 512, 512, 16].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            )),
        ),
        CatalogEntry::new(
            "BICUBIC_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Bicubic downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            )),
        ),
        CatalogEntry::new(
            "BICUBIC_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Bicubic downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[8, 2048, 1024, 2].into(),
                &[512, 512],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            )),
        ),
        // Lanczos3
        CatalogEntry::new(
            "LANCZOS3_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Lanczos3 upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[4096, 4096],
                InterpolateOptions::new(InterpolateMode::Lanczos3),
            )),
        ),
        CatalogEntry::new(
            "LANCZOS3_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Lanczos3 upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[4, 512, 512, 16].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Lanczos3),
            )),
        ),
        CatalogEntry::new(
            "LANCZOS3_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Lanczos3 downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[1, 2048, 2048, 3].into(),
                &[1024, 1024],
                InterpolateOptions::new(InterpolateMode::Lanczos3),
            )),
        ),
        CatalogEntry::new(
            "LANCZOS3_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Lanczos3 downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem::from_input_output_shapes(
                &[8, 2048, 1024, 2].into(),
                &[512, 512],
                InterpolateOptions::new(InterpolateMode::Lanczos3),
            )),
        ),
        // Nearest backward
        CatalogEntry::new(
            "NEAREST_BACKWARD_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Nearest backward upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 2048],
                out_grad_shape: [1, 4096, 4096, 3],
                options: InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Nearest backward upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [512, 512],
                out_grad_shape: [4, 1024, 1024, 16],
                options: InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Nearest backward downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 2048],
                out_grad_shape: [1, 1024, 1024, 3],
                options: InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Nearest backward downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 1024],
                out_grad_shape: [8, 512, 512, 2],
                options: InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
            }),
        ),
    ]
}
