use crate::reduce::ReduceCarrier;

/// A check that `microkernel` is handed its partials' indices (`handed`) exactly where `carrier`
/// keeps any.
///
/// # Panics
///
/// Where it is not.
pub(crate) fn assert_partial_indices(microkernel: &str, carrier: ReduceCarrier, handed: bool) {
    assert!(
        handed == carrier.carries_indices(),
        "{microkernel}: {carrier:?} takes its partials' indices exactly where it keeps any"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A carrier that keeps indices takes them.
    #[test]
    fn a_carrier_of_indices_takes_them() {
        assert_partial_indices("test", ReduceCarrier::Indexed, true);
    }

    /// Without them, it has nowhere to store them.
    #[test]
    #[should_panic(expected = "exactly where it keeps any")]
    fn a_carrier_of_indices_panics_without_them() {
        assert_partial_indices("test", ReduceCarrier::Indexed, false);
    }

    /// A carrier of values alone has none to store.
    #[test]
    #[should_panic(expected = "exactly where it keeps any")]
    fn a_carrier_of_values_panics_with_them() {
        assert_partial_indices("test", ReduceCarrier::Values, true);
    }
}
