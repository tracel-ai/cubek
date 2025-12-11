mod matmul_unit {
    use crate::suite::layered::matmul_test_launcher::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    include!("algorithm.rs");
}
