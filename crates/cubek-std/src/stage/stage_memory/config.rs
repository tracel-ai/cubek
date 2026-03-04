use std::{fmt::Debug, hash::Hash};

use cubecl::ir::StorageType;

use crate::{MatrixLayout, stage::stage_memory::swizzle::SwizzleMode};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct StageMemoryConfig {
    // Planes that read or write this stage memory
    pub num_planes: u32,
    pub elements_per_tile_along_row: u32,
    pub elements_per_tile_along_col: u32,
    pub tiles_per_partition_along_row: u32,
    pub tiles_per_partition_along_col: u32,
    pub partitions_per_stage_along_row: u32,
    pub partitions_per_stage_along_col: u32,
    pub line_size: u32,
    pub matrix_layout: MatrixLayout,
    pub swizzle: SwizzleMode,
    pub num_stages: u32,
    pub dtype: StorageType,
}

impl StageMemoryConfig {
    pub fn tiles_per_stage_along_row(&self) -> u32 {
        self.tiles_per_partition_along_row * self.partitions_per_stage_along_row
    }

    pub fn tiles_per_stage_along_col(&self) -> u32 {
        self.tiles_per_partition_along_col * self.partitions_per_stage_along_col
    }

    pub fn elements_per_stage_along_row(&self) -> u32 {
        self.tiles_per_stage_along_row() * self.elements_per_tile_along_row
    }

    pub fn elements_per_stage_along_col(&self) -> u32 {
        self.tiles_per_stage_along_col() * self.elements_per_tile_along_col
    }

    pub fn elements_per_tile(&self) -> u32 {
        self.elements_per_tile_along_row * self.elements_per_tile_along_col
    }

    pub fn elements_per_stage(&self) -> u32 {
        self.elements_per_stage_along_row() * self.elements_per_stage_along_col()
    }

    pub fn tiles_per_stage(&self) -> u32 {
        self.tiles_per_stage_along_row() * self.tiles_per_stage_along_col()
    }

    pub fn elements_per_tile_along_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_per_tile_along_col,
            MatrixLayout::ColMajor => self.elements_per_tile_along_row,
        }
    }

    pub fn elements_per_stage_along_contiguous_dim(&self) -> u32 {
        match self.matrix_layout {
            MatrixLayout::RowMajor => self.elements_per_stage_along_col(),
            MatrixLayout::ColMajor => self.elements_per_stage_along_row(),
        }
    }
}
