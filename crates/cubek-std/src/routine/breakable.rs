use cubecl::tile::launch::Decline;

/// What a launch its routine claims may be refused, where a condition of the routine should have
/// ruled that out.
pub trait Breakable {
    /// What holds once it is not broken.
    type Held;

    /// A refusal made [`Broken`](Decline::Broken), as `condition` should have held, and logged as
    /// the bug it is.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`].
    fn or_broken(self, condition: &'static str) -> Result<Self::Held, Decline>;
}

/// Whether the condition holds.
impl Breakable for bool {
    type Held = ();

    fn or_broken(self, condition: &'static str) -> Result<(), Decline> {
        if self {
            return Ok(());
        }
        let broken = Decline::Broken {
            condition,
            cause: None,
        };
        log::error!("{broken}");
        Err(broken)
    }
}

impl<T> Breakable for Result<T, Decline> {
    type Held = T;

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
    fn a_condition_that_does_not_hold_is_broken() {
        let broken = Decline::Broken {
            condition: CONDITION,
            cause: None,
        };

        assert_eq!(false.or_broken(CONDITION), Err(broken));
    }

    #[test]
    fn a_condition_that_holds_is_not_broken() {
        assert_eq!(true.or_broken(CONDITION), Ok(()));
    }

    #[test]
    fn what_was_served_is_kept() {
        let served: Result<u32, Decline> = Ok(7);

        assert_eq!(served.or_broken(CONDITION), Ok(7));
    }
}
