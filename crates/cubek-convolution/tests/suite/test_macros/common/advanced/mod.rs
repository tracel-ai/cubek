mod partition_buffering;
mod swizzle;

#[macro_export]
macro_rules! testgen_convolution_advanced {
    ($algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use cubek_matmul::definition::{TilingBlueprint, TilingBlueprintBuilder};

        mod _advanced {
            use super::*;
            use cubecl::Runtime;

            pub fn get_selection_builder() -> TilingBlueprintBuilder {
                let tiling_scheme = $tiling_scheme_builder.build().unwrap();
                let client = cubecl::TestRuntime::client(&Default::default());
                let plane_dim = client.properties().hardware.plane_size_max;
                TilingBlueprint::builder(tiling_scheme, plane_dim)
            }
        }

        $crate::testgen_convolution_swizzle!(
            $algorithm,
            $precision,
            _advanced::get_selection_builder()
        );
    };
}
