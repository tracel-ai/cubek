use crate::registry::CatalogEntry;
use cubek::interpolate::definition::{
    InterpolateBackwardProblem, InterpolateForwardProblem, InterpolateMode, InterpolateOptions,
    InterpolateProblem,
};

pub fn problems() -> Vec<CatalogEntry<InterpolateProblem>> {
    vec![
        // Nearest
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Nearest upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [4096, 4096],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Nearest upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [4, 512, 512, 16],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Nearest downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Nearest downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [8, 2048, 1024, 2],
                output_size: [512, 512],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        // Bilinear
        CatalogEntry::new(
            "BILINEAR_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Bilinear upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [4096, 4096],
                options: InterpolateOptions::new(InterpolateMode::Bilinear),
            }),
        ),
        CatalogEntry::new(
            "BILINEAR_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Bilinear upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [4, 512, 512, 16],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Bilinear),
            }),
        ),
        CatalogEntry::new(
            "BILINEAR_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Bilinear downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Bilinear),
            }),
        ),
        CatalogEntry::new(
            "BILINEAR_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Bilinear downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [8, 2048, 1024, 2],
                output_size: [512, 512],
                options: InterpolateOptions::new(InterpolateMode::Bilinear),
            }),
        ),
        // Bicubic
        CatalogEntry::new(
            "BICUBIC_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Bicubic upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [4096, 4096],
                options: InterpolateOptions::new(InterpolateMode::Bicubic),
            }),
        ),
        CatalogEntry::new(
            "BICUBIC_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Bicubic upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [4, 512, 512, 16],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Bicubic),
            }),
        ),
        CatalogEntry::new(
            "BICUBIC_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Bicubic downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Bicubic),
            }),
        ),
        CatalogEntry::new(
            "BICUBIC_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Bicubic downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [8, 2048, 1024, 2],
                output_size: [512, 512],
                options: InterpolateOptions::new(InterpolateMode::Bicubic),
            }),
        ),
        // Lanczos3
        CatalogEntry::new(
            "LANCZOS3_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Lanczos3 upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [4096, 4096],
                options: InterpolateOptions::new(InterpolateMode::Lanczos3),
            }),
        ),
        CatalogEntry::new(
            "LANCZOS3_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Lanczos3 upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [4, 512, 512, 16],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Lanczos3),
            }),
        ),
        CatalogEntry::new(
            "LANCZOS3_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Lanczos3 downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [1024, 1024],
                options: InterpolateOptions::new(InterpolateMode::Lanczos3),
            }),
        ),
        CatalogEntry::new(
            "LANCZOS3_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Lanczos3 downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Forward(InterpolateForwardProblem {
                input_shape: [8, 2048, 1024, 2],
                output_size: [512, 512],
                options: InterpolateOptions::new(InterpolateMode::Lanczos3),
            }),
        ),
        // Nearest backward
        CatalogEntry::new(
            "NEAREST_BACKWARD_UPSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_4096X4096",
            "Nearest backward upsample (b=1 h=2048 w=2048 c=3 -> 4096x4096)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 2048],
                out_grad_shape: [1, 4096, 4096, 3],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_UPSAMPLE_4_BATCH_16_CHANNELS_512X512_TO_1024X1024",
            "Nearest backward upsample (b=4 h=512 w=512 c=16 -> 1024x1024)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [512, 512],
                out_grad_shape: [4, 1024, 1024, 16],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_DOWNSAMPLE_1_BATCH_3_CHANNELS_2048X2048_TO_1024X1024",
            "Nearest backward downsample (b=1 h=2048 w=2048 c=3 -> 1024x1024)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 2048],
                out_grad_shape: [1, 1024, 1024, 3],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
        CatalogEntry::new(
            "NEAREST_BACKWARD_DOWNSAMPLE_8_BATCH_2_CHANNELS_2048X1024_TO_512X512",
            "Nearest backward downsample (b=8 h=2048 w=1024 c=2 -> 512x512)",
            InterpolateProblem::Backward(InterpolateBackwardProblem {
                input_size: [2048, 1024],
                out_grad_shape: [8, 512, 512, 2],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            }),
        ),
    ]
}
