pub mod tiling_scheme_ops {
    use cubek_attention::launch::{AttentionDims, AttentionTilingScheme};

    pub fn elements_in_stage_seq_q(tiling_scheme: &AttentionTilingScheme) -> usize {
        tiling_scheme.stage_size.seq_q as usize * elements_in_partition_seq_q(tiling_scheme)
    }

    pub fn elements_in_partition_seq_q(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.seq_q * tiling_scheme.partition_size.seq_q) as usize
    }

    pub fn elements_in_partition_head_dim(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.head_dim * tiling_scheme.partition_size.head_dim) as usize
    }

    pub fn elements_in_partition_seq_kv(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.seq_kv * tiling_scheme.partition_size.seq_kv) as usize
    }

    pub fn elements_in_partition_val_dim(tiling_scheme: &AttentionTilingScheme) -> usize {
        (tiling_scheme.tile_size.val_dim * tiling_scheme.partition_size.val_dim) as usize
    }

    #[allow(unused)]
    pub fn print_dims_vs_scheme(dims: &AttentionDims, tiling_scheme: &AttentionTilingScheme) {
        println!(
            "seq_q: problem {:?} vs scheme {:?}",
            dims.seq_q,
            elements_in_stage_seq_q(tiling_scheme),
        );
        println!(
            "seq_kv: problem {:?} vs scheme {:?}",
            dims.seq_kv,
            elements_in_partition_seq_kv(tiling_scheme)
        );
        println!(
            "head_dim: problem {:?} vs scheme {:?}",
            dims.head_dim,
            elements_in_partition_head_dim(tiling_scheme)
        );
        println!(
            "val_dim: problem {:?} vs scheme {:?}",
            dims.val_dim,
            elements_in_partition_val_dim(tiling_scheme)
        );
    }
}
