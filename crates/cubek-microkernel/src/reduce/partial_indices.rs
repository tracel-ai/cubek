use crate::reduce::ReduceCarrier;

/// Whether partials of `carrier` are rightly handed indices beside them (`handed`): exactly where it
/// keeps any.
pub(crate) const fn takes_partial_indices(carrier: ReduceCarrier, handed: bool) -> bool {
    handed == carrier.carries_indices()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_carrier_of_indices_takes_them() {
        assert!(takes_partial_indices(ReduceCarrier::Indexed, true));
    }

    #[test]
    fn a_carrier_of_indices_refuses_to_do_without() {
        assert!(!takes_partial_indices(ReduceCarrier::Indexed, false));
    }

    #[test]
    fn a_carrier_of_values_refuses_indices() {
        assert!(!takes_partial_indices(ReduceCarrier::Values, true));
    }
}
