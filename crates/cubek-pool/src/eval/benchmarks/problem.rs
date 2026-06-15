use crate::definition::{
    AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolBackward, PoolBackwardProblem,
    PoolForward, PoolForwardProblem, PoolProblem,
};
use cubecl::zspace::Shape;
use cubek_test_utils::CatalogEntry;

pub fn problems() -> Vec<CatalogEntry<PoolProblem>> {
    vec![
        // Max Pooling
        // Forward: ResNet-style initial layer (High spatial resolution)
        CatalogEntry::new(
            "MAX_POOL2D_FWD_RESNET_INIT",
            "max_pool2d_forward_resnet_init",
            PoolProblem::Forward(PoolForward::D2(PoolForwardProblem {
                input_shape: Shape::from(vec![32, 224, 224, 64]),
                with_indices: true,
                mode: MaxPoolOptions::new([3, 3], [2, 2], [1, 1], [1, 1], false).into(),
            })),
        ),
        // Backward: ResNet-style initial layer (Gradient scattering)
        CatalogEntry::new(
            "MAX_POOL2D_BWD_RESNET_INIT",
            "max_pool2d_backward_resnet_init",
            PoolProblem::Backward(PoolBackward::D2(PoolBackwardProblem {
                input_size: [224, 224],
                out_grad_shape: Shape::from(vec![32, 112, 112, 64]),
                with_indices: true,
                mode: MaxPoolOptions::new([3, 3], [2, 2], [1, 1], [1, 1], false).into(),
            })),
        ),
        // Forward: Deep bottleneck layer (High channel count)
        CatalogEntry::new(
            "MAX_POOL2D_FWD_DEEP",
            "max_pool2d_forward_deep",
            PoolProblem::Forward(PoolForward::D2(PoolForwardProblem {
                input_shape: Shape::from(vec![64, 28, 28, 512]),
                with_indices: true,
                mode: MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], false).into(),
            })),
        ),
        // Backward: Deep bottleneck layer
        CatalogEntry::new(
            "MAX_POOL2D_BWD_DEEP",
            "max_pool2d_backward_deep",
            PoolProblem::Backward(PoolBackward::D2(PoolBackwardProblem {
                input_size: [28, 28],
                out_grad_shape: Shape::from(vec![64, 14, 14, 512]),
                with_indices: true,
                mode: MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], false).into(),
            })),
        ),
        // Average Pooling
        // Forward: General high-throughput scenario
        CatalogEntry::new(
            "AVG_POOL2D_FWD_THROUGHPUT",
            "avg_pool2d_forward_throughput",
            PoolProblem::Forward(PoolForward::D2(PoolForwardProblem {
                input_shape: Shape::from(vec![128, 32, 32, 256]),
                with_indices: false,
                mode: AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, true).into(),
            })),
        ),
        // Backward: General high-throughput scenario
        CatalogEntry::new(
            "AVG_POOL2D_BWD_THROUGHPUT",
            "avg_pool2d_backward_throughput",
            PoolProblem::Backward(PoolBackward::D2(PoolBackwardProblem {
                input_size: [32, 32],
                out_grad_shape: Shape::from(vec![128, 16, 16, 256]),
                with_indices: false,
                mode: AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, true).into(),
            })),
        ),
        // Adaptive Average Pooling
        // Forward: Global Average Pooling (Reducing spatial to 1x1)
        CatalogEntry::new(
            "ADAPTIVE_AVG_POOL2D_FWD_GLOBAL",
            "adaptive_avg_pool2d_fwd_global",
            PoolProblem::Forward(PoolForward::D2(PoolForwardProblem {
                input_shape: Shape::from(vec![16, 224, 224, 128]),
                with_indices: false,
                mode: AdaptiveAvgPoolOptions::new([1, 1]).into(),
            })),
        ),
        // Backward: Global Average Pooling (Broadcasting gradients)
        CatalogEntry::new(
            "ADAPTIVE_AVG_POOL2D_BWD_GLOBAL",
            "adaptive_avg_pool2d_bwd_global",
            PoolProblem::Backward(PoolBackward::D2(PoolBackwardProblem {
                input_size: [224, 224],
                out_grad_shape: Shape::from(vec![16, 1, 1, 128]),
                with_indices: false,
                mode: AdaptiveAvgPoolOptions::new([1, 1]).into(),
            })),
        ),
        // Forward: Fixed feature map reduction (e.g., 14x14 -> 7x7)
        CatalogEntry::new(
            "ADAPTIVE_AVG_POOL2D_FWD_REDUCE",
            "adaptive_avg_pool2d_fwd_reduce",
            PoolProblem::Forward(PoolForward::D2(PoolForwardProblem {
                input_shape: Shape::from(vec![64, 14, 14, 1024]),
                with_indices: false,
                mode: AdaptiveAvgPoolOptions::new([7, 7]).into(),
            })),
        ),
        // Backward: Fixed feature map reduction
        CatalogEntry::new(
            "ADAPTIVE_AVG_POOL2D_BWD_REDUCE",
            "adaptive_avg_pool2d_bwd_reduce",
            PoolProblem::Backward(PoolBackward::D2(PoolBackwardProblem {
                input_size: [14, 14],
                out_grad_shape: Shape::from(vec![64, 7, 7, 1024]),
                with_indices: false,
                mode: AdaptiveAvgPoolOptions::new([7, 7]).into(),
            })),
        ),
    ]
}
