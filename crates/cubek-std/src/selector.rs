//! The shape of a selector's sizing loop: grow over an ascending list of candidates and keep the
//! one that fits.

/// The largest candidate that fits, over an **ascending** list.
///
/// Growing from the floor makes the ceiling the device's: a loop that starts at a constant and
/// shrinks until legal stops at its first try on a roomier device. Every candidate is tested, not
/// only those before the first refusal, since a divisibility test is not monotone.
pub fn largest_that_fits<T: Copy>(
    ascending: impl IntoIterator<Item = T>,
    fits: impl Fn(T) -> bool,
) -> Option<T> {
    ascending.into_iter().filter(|&c| fits(c)).last()
}

/// The **smallest** candidate that fits, for a stage whose cost is its own residency rather than
/// a reuse a larger one buys. Having both directions makes a call site say which it grows in.
pub fn smallest_that_fits<T: Copy>(
    ascending: impl IntoIterator<Item = T>,
    fits: impl Fn(T) -> bool,
) -> Option<T> {
    ascending.into_iter().find(|&c| fits(c))
}

/// The largest divisor of `whole` not exceeding `cap`, at least 1.
pub fn divisor_at_most(whole: usize, cap: usize) -> usize {
    let cap = cap.clamp(1, whole.max(1));
    largest_that_fits(1..=cap, |d| whole.is_multiple_of(d)).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_roomier_budget_buys_a_larger_candidate() {
        let blocks = [16usize, 32, 64, 128, 256];
        let fits = |budget: usize| move |b: usize| b * 512 <= budget;
        assert_eq!(largest_that_fits(blocks, fits(32 * 1024)), Some(64));
        assert_eq!(largest_that_fits(blocks, fits(228 * 1024)), Some(256));
        assert_eq!(smallest_that_fits(blocks, |b| b >= 20), Some(32));
    }

    #[test]
    fn a_floor_that_does_not_fit_is_an_answer() {
        assert_eq!(largest_that_fits([16usize, 32], |_| false), None);
    }

    #[test]
    fn a_divisor_never_leaves_an_overhang() {
        assert_eq!(divisor_at_most(24, 5), 4);
        assert_eq!(divisor_at_most(24, 24), 24);
        assert_eq!(divisor_at_most(7, 5), 1);
        assert_eq!(divisor_at_most(0, 4), 1);
    }
}
