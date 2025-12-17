mod problem_size;

#[macro_export]
macro_rules! testgen_convolution_problem {
    ($algorithm: ty, $precision: ty, $selection_builder: expr) => {
        mod _problem_generated {
            use super::*;
            use cubek_matmul::definition::TilingBlueprint;

            pub fn get_selection() -> TilingBlueprint {
                $selection_builder.build()
            }
        }

        $crate::testgen_convolution_problem_size!(
            $algorithm,
            $precision,
            _problem_generated::get_selection()
        );
    };
}
