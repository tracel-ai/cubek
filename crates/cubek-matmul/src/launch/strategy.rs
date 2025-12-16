use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{
    components::{
        global::read::{
            async_full_cyclic, async_full_strided, async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading, sync_full_strided,
            sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
        tile::{cmma::CmmaMatmul, io::Filled, mma::MmaMatmul},
    },
    routines::{
        BlueprintStrategy,
        double_buffering::{
            AsyncCyclicDoubleBufferingAlgorithm, AsyncStridedDoubleBufferingAlgorithm,
            CyclicDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm,
            TilewiseDoubleBufferingAlgorithm, TmaDoubleBufferingAlgorithm,
        },
        double_unit::DoubleUnitAlgorithm,
        ordered_double_buffering::OrderedDoubleBufferingAlgorithm,
        simple::{SimpleAlgorithm, SimpleTmaAlgorithm},
        simple_unit::SimpleUnitAlgorithm,
        specialized::SpecializedAlgorithm,
        vecmat::{DoubleVecMatAlgorithm, SimpleVecMatAlgorithm},
    },
};

type Cmma = CmmaMatmul<Filled>;
type Mma = MmaMatmul;

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    SimpleCyclicCmma(BlueprintStrategy<SimpleAlgorithm<Cmma>>),
    SimpleCyclicMma(BlueprintStrategy<SimpleAlgorithm<Mma>>),
    SimpleStridedCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleStridedMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleTilewiseCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTilewiseMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncStridedCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncStridedMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncCyclicCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncCyclicMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTmaCmma(BlueprintStrategy<SimpleTmaAlgorithm<Cmma>>),
    SimpleTmaMma(BlueprintStrategy<SimpleTmaAlgorithm<Mma>>),
    DoubleCyclicCmma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleCyclicMma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleTilewiseCmma(BlueprintStrategy<TilewiseDoubleBufferingAlgorithm<Cmma>>),
    DoubleTilewiseMma(BlueprintStrategy<TilewiseDoubleBufferingAlgorithm<Mma>>),
    DoubleHybridCmma(BlueprintStrategy<HybridDoubleBufferingAlgorithm<Cmma>>),
    DoubleHybridMma(BlueprintStrategy<HybridDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncCyclicCmma(BlueprintStrategy<AsyncCyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncCyclicMma(BlueprintStrategy<AsyncCyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncStridedCmma(BlueprintStrategy<AsyncStridedDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncStridedMma(BlueprintStrategy<AsyncStridedDoubleBufferingAlgorithm<Mma>>),
    DoubleTmaCmma(BlueprintStrategy<TmaDoubleBufferingAlgorithm<Cmma>>),
    DoubleTmaMma(BlueprintStrategy<TmaDoubleBufferingAlgorithm<Mma>>),
    SpecializedCyclicCmma(
        BlueprintStrategy<
            SpecializedAlgorithm<Cmma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedCyclicMma(
        BlueprintStrategy<
            SpecializedAlgorithm<Mma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedStridedCmma(
        BlueprintStrategy<SpecializedAlgorithm<Cmma, AsyncPartialStridedLoading>>,
    ),
    SpecializedStridedMma(BlueprintStrategy<SpecializedAlgorithm<Mma, AsyncPartialStridedLoading>>),
    SpecializedTmaCmma(BlueprintStrategy<SpecializedAlgorithm<Cmma>>),
    SpecializedTmaMma(BlueprintStrategy<SpecializedAlgorithm<Mma>>),
    OrderedDoubleCmma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm<Cmma>>),
    OrderedDoubleMma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm<Mma>>),
    SimpleUnit(BlueprintStrategy<SimpleUnitAlgorithm>),
    DoubleUnit(BlueprintStrategy<DoubleUnitAlgorithm>),
    SimpleVecMat(BlueprintStrategy<SimpleVecMatAlgorithm>),
    DoubleVecMat(BlueprintStrategy<DoubleVecMatAlgorithm>),
    Naive,
    #[default]
    Auto,
}

// #[derive(Debug, Clone, Default)]
// /// The matmul algorithm to launch
// ///
// /// Most strategies have a selection input that can be overwritten or inferred from minimal information
// /// Some strategies must have a specified loading strategy
// pub enum Strategy {
//     Simple {
//         read_strategy: ReadingStrategy,
//         selection: BlueprintStrategy<SimpleAlgorithm>,
//         tile_kind: AcceleratedTileKind,
//     },
//     DoubleBuffering {
//         read_strategy: PartialReadingStrategy,
//         selection: BlueprintStrategy<CyclicDoubleBufferingAlgorithm>,
//         tile_kind: AcceleratedTileKind,
//     },
//     Specialized {
//         read_strategy: AsyncPartialReadingStrategy,
//         selection: BlueprintStrategy<SpecializedAlgorithm>,
//         tile_kind: AcceleratedTileKind,
//     },
//     SimpleUnit(BlueprintStrategy<SimpleUnitAlgorithm>),
//     DoubleUnit(BlueprintStrategy<DoubleUnitAlgorithm>),
//     SimpleVecMat(BlueprintStrategy<SimpleVecMatAlgorithm>),
//     DoubleVecMat(BlueprintStrategy<DoubleVecMatAlgorithm>),
//     OrderedDoubleBuffering {
//         selection: BlueprintStrategy<OrderedDoubleBufferingAlgorithm>,
//         tile_kind: AcceleratedTileKind,
//     },
//     Naive,
//     #[default]
//     /// Tries using a Simple matmul, then a SimpleUnit if the former failed
//     Auto,
// }

// impl Display for Strategy {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Strategy::Simple {
//                 read_strategy,
//                 selection,
//                 tile_kind,
//             } => {
//                 f.write_fmt(format_args!("matmul_simple_{read_strategy}_{tile_kind}"))?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(args) => {
//                         if args.multi_rows {
//                             f.write_str("_multirows")?;
//                         }
//                     }
//                 };
//             }
//             Strategy::DoubleBuffering {
//                 read_strategy,
//                 selection,
//                 tile_kind,
//             } => {
//                 f.write_fmt(format_args!(
//                     "matmul_double_buffering_{read_strategy}_{tile_kind}"
//                 ))?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(args) => {
//                         if args.specialized {
//                             f.write_str("_specialized")?;
//                         }
//                     }
//                 };
//             }
//             Strategy::Specialized {
//                 read_strategy,
//                 selection,
//                 tile_kind,
//             } => {
//                 f.write_fmt(format_args!(
//                     "matmul_specialized_{read_strategy}_{tile_kind}"
//                 ))?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(_) => {}
//                 };
//             }
//             Strategy::SimpleUnit(selection) => {
//                 f.write_fmt(format_args!("matmul_simple_unit"))?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(args) => {
//                         f.write_fmt(format_args!("_{}", args.tile_size))?;
//                     }
//                 };
//             }
//             Strategy::DoubleUnit(selection) => {
//                 f.write_str("matmul_double_buffering_unit")?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(args) => {
//                         f.write_fmt(format_args!("_{}", args.tile_size))?;
//                     }
//                 };
//             }
//             Strategy::SimpleVecMat(selection) => {
//                 f.write_str("vecmat_simple")?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(_) => {}
//                 };
//             }
//             Strategy::DoubleVecMat(selection) => {
//                 f.write_str("vecmat_double_buffering")?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(_) => {}
//                 };
//             }
//             Strategy::OrderedDoubleBuffering {
//                 selection,
//                 tile_kind,
//             } => {
//                 f.write_fmt(format_args!("matmul_double_buffering_ordered_{tile_kind}"))?;

//                 match selection {
//                     BlueprintStrategy::Forced(_) => f.write_str("_forced_selection")?,
//                     BlueprintStrategy::Inferred(args) => {
//                         if let Some(k) = args.partition_k {
//                             f.write_fmt(format_args!("_partition_k{}", k))?;
//                         }
//                         if let Some(r) = args.row_count {
//                             f.write_fmt(format_args!("_row_count{}", r))?;
//                         }
//                         if let Some(r) = args.rows_per_plane {
//                             f.write_fmt(format_args!("_row_per_plane{}", r))?;
//                         }
//                     }
//                 };
//             }
//             Strategy::Naive => f.write_str("matmul_naive")?,
//             Strategy::Auto => f.write_str("matmul_auto")?,
//         };

//         Ok(())
//     }
// }

#[derive(Debug, Clone, Copy)]
/// Which reader to use in simple algorithms
pub enum ReadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
    AsyncCyclic,
    AsyncStrided,
    Tma,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in double buffering algorithms
pub enum PartialReadingStrategy {
    Cyclic,
    Tilewise,
    Hybrid,
    Tma,
    AsyncCyclic,
    AsyncStrided,
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in specialized algorithms
pub enum AsyncPartialReadingStrategy {
    Cyclic,
    Strided,
    Tma,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

// Display implementations are used to combine and save names when autotuning.

impl Display for AcceleratedTileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceleratedTileKind::Cmma => f.write_str("cmma"),
            AcceleratedTileKind::Mma => f.write_str("mma"),
        }
    }
}

impl Display for ReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadingStrategy::Cyclic => f.write_str("cyclic"),
            ReadingStrategy::Strided => f.write_str("strided"),
            ReadingStrategy::Tilewise => f.write_str("tilewise"),
            ReadingStrategy::AsyncCyclic => f.write_str("async_cyclic"),
            ReadingStrategy::AsyncStrided => f.write_str("async_strided"),
            ReadingStrategy::Tma => f.write_str("tma"),
        }
    }
}

impl Display for PartialReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PartialReadingStrategy::Cyclic => f.write_str("cyclic"),
            PartialReadingStrategy::Tilewise => f.write_str("tilewise"),
            PartialReadingStrategy::Hybrid => f.write_str("hybrid"),
            PartialReadingStrategy::Tma => f.write_str("tma"),
            PartialReadingStrategy::AsyncCyclic => f.write_str("async_cyclic"),
            PartialReadingStrategy::AsyncStrided => f.write_str("async_strided"),
        }
    }
}

impl Display for AsyncPartialReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsyncPartialReadingStrategy::Cyclic => f.write_str("cyclic"),
            AsyncPartialReadingStrategy::Strided => f.write_str("strided"),
            AsyncPartialReadingStrategy::Tma => f.write_str("tma"),
        }
    }
}
