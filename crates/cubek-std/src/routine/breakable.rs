use cubecl::tile::launch::Decline;

/// A result a launch reaches where its routine's `prepare` claimed it cannot fail: a refusal there
/// is a [`Broken`](Decline::Broken) its routine owns, never one the device does.
///
/// # Guarantees
///
/// Every [`Broken`](Decline::Broken) it makes is logged as an error once, where it is made.
pub trait Breakable {
    /// The result kept where it is `Ok`, and where it is `Err`, a [`Broken`](Decline::Broken)
    /// naming `condition` with the refusal as its cause, or the refusal itself where it is already
    /// one, as the condition that failed first.
    fn or_broken(self, condition: &'static str) -> Self;

    /// A [`Broken`](Decline::Broken) naming `condition`, met with no other refusal to show it.
    fn broken(condition: &'static str) -> Self;
}

impl<T> Breakable for Result<T, Decline> {
    fn or_broken(self, condition: &'static str) -> Self {
        self.map_err(|cause| match cause.is_broken() {
            true => cause,
            false => logged(Decline::broken_by(condition, cause)),
        })
    }

    fn broken(condition: &'static str) -> Self {
        Err(logged(Decline::Broken {
            condition,
            cause: None,
        }))
    }
}

/// `broken`, logged as the bug it is.
fn logged(broken: Decline) -> Decline {
    log::error!("{broken}");
    broken
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

    #[test]
    fn a_broken_cause_is_kept_as_the_first_condition() {
        let first: Result<(), Decline> = Breakable::broken(CONDITION);

        assert_eq!(first.clone().or_broken("a later condition"), first);
    }

    #[test]
    fn a_condition_alone_is_broken_with_no_cause() {
        let broken: Result<(), Decline> = Breakable::broken(CONDITION);

        assert_eq!(
            broken,
            Err(Decline::Broken {
                condition: CONDITION,
                cause: None,
            })
        );
    }
}
