mod matmul_unit {
    use crate::suite::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    include!("algorithm.rs");
}
