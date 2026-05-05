use crate::registry::ItemDescriptor;
use cubek::interpolate::definition::{InterpolateMode, InterpolateOptions, InterpolateProblem};

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const NEAREST_UPSAMPLE_4X_64X64: &str = "NEAREST_UPSAMPLE_4X_64X64";
pub const NEAREST_UPSAMPLE_4X_32X32: &str = "NEAREST_UPSAMPLE_4X_32X32";
pub const NEAREST_DOWNSAMPLE_2X_256X256: &str = "NEAREST_DOWNSAMPLE_2X_256X256";
pub const NEAREST_BATCH8_UPSAMPLE_2X_64X64: &str = "NEAREST_BATCH8_UPSAMPLE_2X_64X64";
pub const NEAREST_CHANNELS64_UPSAMPLE_2X_32X32: &str = "NEAREST_CHANNELS64_UPSAMPLE_2X_32X32";

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: NEAREST_UPSAMPLE_4X_64X64.to_string(),
            label: "Nearest upsample (b=1 h=64 w=64 c=3 -> 128x128)".to_string(),
        },
        ItemDescriptor {
            id: NEAREST_UPSAMPLE_4X_32X32.to_string(),
            label: "Nearest upsample (b=1 h=32 w=32 c=3 -> 128x128)".to_string(),
        },
        ItemDescriptor {
            id: NEAREST_DOWNSAMPLE_2X_256X256.to_string(),
            label: "Nearest downsample (b=1 h=256 w=256 c=3 -> 128x128)".to_string(),
        },
        ItemDescriptor {
            id: NEAREST_BATCH8_UPSAMPLE_2X_64X64.to_string(),
            label: "Nearest upsample (b=8 h=64 w=64 c=3 -> 128x128)".to_string(),
        },
        ItemDescriptor {
            id: NEAREST_CHANNELS64_UPSAMPLE_2X_32X32.to_string(),
            label: "Nearest upsample (b=1 h=32 w=32 c=64 -> 64x64)".to_string(),
        },
    ]
}

pub(crate) fn problem_for(id: &str) -> Option<InterpolateProblem> {
    Some(match id {
        "NEAREST_UPSAMPLE_4X_64X64" => InterpolateProblem {
            input_shape: [1, 64, 64, 3],
            output_size: [128, 128],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        },
        "NEAREST_UPSAMPLE_4X_32X32" => InterpolateProblem {
            input_shape: [1, 32, 32, 3],
            output_size: [128, 128],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        },
        "NEAREST_DOWNSAMPLE_2X_256X256" => InterpolateProblem {
            input_shape: [1, 256, 256, 3],
            output_size: [128, 128],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        },
        "NEAREST_BATCH8_UPSAMPLE_2X_64X64" => InterpolateProblem {
            input_shape: [8, 64, 64, 3],
            output_size: [128, 128],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        },
        "NEAREST_CHANNELS64_UPSAMPLE_2X_32X32" => InterpolateProblem {
            input_shape: [1, 32, 32, 64],
            output_size: [64, 64],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        },
        _ => return None,
    })
}
