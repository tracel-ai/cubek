mod partition_buffering;
mod swizzle;

#[macro_export]
macro_rules! testgen_convolution_advanced {
    ($algorithm: expr, $dtypes: expr, $tiling_scheme_builder: expr) => {
        use cubek_matmul::definition::{TilingBlueprint, TilingBlueprintBuilder};

        $crate::testgen_convolution_swizzle!(
            $algorithm,
            $dtypes,
            $tiling_scheme_builder.build().unwrap()
        );
    };
}
