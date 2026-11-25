use cubecl::tile::launch::Decline;

/// What a launch its routine claims may be refused, where a condition of the routine should have
/// ruled that out.
pub trait Breakable {
    /// A decline made [`Broken`](Decline::Broken), as `condition` should be true, and logged as
    /// the bug it is.
    fn or_broken(self, condition: &'static str) -> Self;
}

impl<T> Breakable for Result<T, Decline> {
    fn or_broken(self, condition: &'static str) -> Self {
        self.map_err(|cause| {
            let broken = Decline::broken_by(condition, cause);
            log::error!("{broken}");
            broken
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONDITION: &str = "the cube holds units";

    #[test]
    fn a_decline_becomes_broken() {
        let declined: Result<(), Decline> = Err(Decline::NoUnits);

        let broken = declined.or_broken(CONDITION);

        assert_eq!(broken, Err(Decline::broken_by(CONDITION, Decline::NoUnits)));
    }

    #[test]
    fn what_was_served_is_kept() {
        let served: Result<u32, Decline> = Ok(7);

        assert_eq!(served.or_broken(CONDITION), Ok(7));
    }
}
