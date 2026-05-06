use crate::registry::CatalogEntry;
use cubek::interpolate::definition::{InterpolateMode, InterpolateOptions, InterpolateProblem};

pub fn problems() -> Vec<CatalogEntry<InterpolateProblem>> {
    vec![
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_2X_64X64",
            "Nearest upsample (b=1 h=64 w=64 c=3 -> 128x128)",
            InterpolateProblem {
                input_shape: [1, 64, 64, 3],
                output_size: [128, 128],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
        CatalogEntry::new(
            "NEAREST_UPSAMPLE_4X_512X512",
            "Nearest upsample (b=1 h=512 w=512 c=3 -> 2048x2048)",
            InterpolateProblem {
                input_shape: [1, 512, 512, 3],
                output_size: [2048, 2048],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_2X_256X256",
            "Nearest downsample (b=1 h=256 w=256 c=3 -> 128x128)",
            InterpolateProblem {
                input_shape: [1, 256, 256, 3],
                output_size: [128, 128],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
        CatalogEntry::new(
            "NEAREST_DOWNSAMPLE_4X_2048X2048",
            "Nearest downsample (b=1 h=2048 w=2048 c=3 -> 512x512)",
            InterpolateProblem {
                input_shape: [1, 2048, 2048, 3],
                output_size: [512, 512],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
        CatalogEntry::new(
            "NEAREST_BATCH8_UPSAMPLE_2X_64X64",
            "Nearest upsample (b=8 h=64 w=64 c=3 -> 128x128)",
            InterpolateProblem {
                input_shape: [8, 64, 64, 3],
                output_size: [128, 128],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
        CatalogEntry::new(
            "NEAREST_CHANNELS64_UPSAMPLE_2X_64X64",
            "Nearest upsample (b=1 h=64 w=64 c=64 -> 128x128)",
            InterpolateProblem {
                input_shape: [1, 64, 64, 64],
                output_size: [128, 128],
                options: InterpolateOptions::new(InterpolateMode::Nearest),
            },
        ),
    ]
}
