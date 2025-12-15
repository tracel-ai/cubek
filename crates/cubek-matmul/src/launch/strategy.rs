use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::routines::{
    Selection, double_buffering::DoubleBufferingArgs, double_unit::DoubleUnitSelectionArgs,
    ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
    simple_unit::SimpleUnitSelectionArgs,
};

#[derive(Debug, Clone, Default)]
/// The matmul algorithm to launch
///
/// Most strategies have a selection input that can be overwritten or inferred from minimal information
/// Some strategies must have a specified loading strategy
pub enum Strategy {
    Simple {
        read_strategy: ReadingStrategy,
        selection: Selection<SimpleArgs>,
        tile_kind: AcceleratedTileKind,
    },
    DoubleBuffering {
        read_strategy: PartialReadingStrategy,
        selection: Selection<DoubleBufferingArgs>,
        tile_kind: AcceleratedTileKind,
    },
    Specialized {
        read_strategy: AsyncPartialReadingStrategy,
        selection: Selection<()>,
        tile_kind: AcceleratedTileKind,
    },
    SimpleUnit(Selection<SimpleUnitSelectionArgs>),
    DoubleUnit(Selection<DoubleUnitSelectionArgs>),
    SimpleVecMat(Selection<()>),
    DoubleVecMat(Selection<()>),
    OrderedDoubleBuffering {
        selection: Selection<OrderedSelectionArgs>,
        tile_kind: AcceleratedTileKind,
    },
    Naive,
    #[default]
    /// Tries using a Simple matmul, then a SimpleUnit if the former failed
    Auto,
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Simple {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!("matmul_simple_{read_strategy}_{tile_kind}"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if args.multi_rows {
                            f.write_str("_multirows")?;
                        }
                    }
                };
            }
            Strategy::DoubleBuffering {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!(
                    "matmul_double_buffering_{read_strategy}_{tile_kind}"
                ))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if args.specialized {
                            f.write_str("_specialized")?;
                        }
                    }
                };
            }
            Strategy::Specialized {
                read_strategy,
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!(
                    "matmul_specialized_{read_strategy}_{tile_kind}"
                ))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::SimpleUnit(selection) => {
                f.write_fmt(format_args!("matmul_simple_unit"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        f.write_fmt(format_args!("_{}", args.tile_size))?;
                    }
                };
            }
            Strategy::DoubleUnit(selection) => {
                f.write_str("matmul_double_buffering_unit")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        f.write_fmt(format_args!("_{}", args.tile_size))?;
                    }
                };
            }
            Strategy::SimpleVecMat(selection) => {
                f.write_str("vecmat_simple")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::DoubleVecMat(selection) => {
                f.write_str("vecmat_double_buffering")?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(_) => {}
                };
            }
            Strategy::OrderedDoubleBuffering {
                selection,
                tile_kind,
            } => {
                f.write_fmt(format_args!("matmul_double_buffering_ordered_{tile_kind}"))?;

                match selection {
                    Selection::Forced(_) => f.write_str("_forced_selection")?,
                    Selection::Inferred(args) => {
                        if let Some(k) = args.partition_k {
                            f.write_fmt(format_args!("_partition_k{}", k))?;
                        }
                        if let Some(r) = args.row_count {
                            f.write_fmt(format_args!("_row_count{}", r))?;
                        }
                        if let Some(r) = args.rows_per_plane {
                            f.write_fmt(format_args!("_row_per_plane{}", r))?;
                        }
                    }
                };
            }
            Strategy::Naive => f.write_str("matmul_naive")?,
            Strategy::Auto => f.write_str("matmul_auto")?,
        };

        Ok(())
    }
}

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
