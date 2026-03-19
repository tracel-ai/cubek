mod matmul_unit {
    use crate::suite::launcher::InputRepresentation;

    fn input_representation() -> InputRepresentation {
        InputRepresentation::Normal
    }

    include!("algorithm.rs");
}
