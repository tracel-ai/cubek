use crate::components::{global::GlobalConfig, stage::StageConfig, tile::TileConfig};

impl GlobalConfig for () {
    type StageConfig = ();

    fn stage_config(&self) -> Self::StageConfig {
        todo!()
    }

    fn lhs_reader_config(&self) -> crate::components::global::GlobalReaderConfig {
        todo!()
    }

    fn rhs_reader_config(&self) -> crate::components::global::GlobalReaderConfig {
        todo!()
    }

    fn writer_config(&self) -> crate::components::global::GlobalWriterConfig {
        todo!()
    }

    fn cube_dim(&self) -> cubecl::CubeDim {
        todo!()
    }

    fn global_line_sizes(&self) -> crate::definition::MatmulLineSizes {
        todo!()
    }

    fn must_sync_plane_after_execution(&self) -> bool {
        todo!()
    }
}

impl StageConfig for () {
    type TileConfig = ();

    fn elements_in_stage_m(&self) -> u32 {
        todo!()
    }

    fn elements_in_stage_n(&self) -> u32 {
        todo!()
    }

    fn elements_in_stage_k(&self) -> u32 {
        todo!()
    }

    fn elements_in_tile_k(&self) -> u32 {
        todo!()
    }

    fn tiles_in_partition_mn(&self) -> u32 {
        todo!()
    }

    fn num_main_flow_planes(&self) -> u32 {
        todo!()
    }

    fn plane_dim(&self) -> u32 {
        todo!()
    }

    fn plane_role_config(&self) -> crate::components::global::PlaneRoleConfig {
        todo!()
    }

    fn lhs_smem_config(&self) -> crate::components::stage::StageMemoryConfig {
        todo!()
    }

    fn rhs_smem_config(&self) -> crate::components::stage::StageMemoryConfig {
        todo!()
    }

    fn out_smem_config(&self) -> crate::components::stage::StageMemoryConfig {
        todo!()
    }
}

impl TileConfig for () {
    fn plane_dim(&self) -> u32 {
        todo!()
    }

    fn elements_in_tile_m(&self) -> u32 {
        todo!()
    }

    fn elements_in_tile_n(&self) -> u32 {
        todo!()
    }

    fn elements_in_tile_k(&self) -> u32 {
        todo!()
    }

    fn swizzle_mode(
        &self,
        ident: crate::definition::StageIdent,
    ) -> crate::components::stage::SwizzleMode {
        todo!()
    }
}
