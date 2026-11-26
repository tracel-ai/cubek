use std::collections::VecDeque;

use cubecl::tile::launch::Decline;
use cubecl_environment::sync::Mutex;

use crate::Hardware;
use crate::routine::{Grid, Routine};

/// What [`prepare`](Routine::prepare) answers: the launch, or why the routine declines it.
type Prepared<R> = Result<(<R as Routine>::Blueprint, Grid), Decline>;

/// The launches one routine prepared at one setting on one device, per problem: what a caller
/// that launches the same problem again and again reads instead of preparing it again.
///
/// [`prepare`](Routine::prepare) is pure, so a launch prepared once is the launch every later call
/// of the same problem would prepare. Preparing walks the microkernel's candidates and resolves
/// each, which costs more host time than the launch of a small product; a tuned winner runs the
/// same few problems for the rest of the process.
///
/// It keeps the [`CAPACITY`](Self::CAPACITY) problems prepared last, so a caller cycling through
/// many shapes costs a comparison per kept problem and no more than preparing each.
pub struct PreparedLaunches<R: Routine> {
    /// The problems prepared last, oldest first, each beside what it prepared: looked up by its
    /// problem (a Keyed Usage, by comparison, a problem being neither hashed nor ordered), and the
    /// oldest dropped first once full (an Arrival Usage).
    recent: Mutex<VecDeque<(R::Problem, Prepared<R>)>>,
}

impl<R: Routine> PreparedLaunches<R>
where
    R::Problem: PartialEq,
{
    /// The most problems kept: a model's handful of shapes per winner.
    pub const CAPACITY: usize = 16;

    /// The launch `R` prepares for `problem` at `settings` on `hw`, or why it declines it: the one
    /// kept for `problem`, or prepared now and kept.
    ///
    /// # Assumptions
    ///
    /// Every call hands the same `settings` and `hw`.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run `problem` there, as [`prepare`](Routine::prepare) says.
    pub fn prepare(
        &self,
        hw: &Hardware,
        problem: &R::Problem,
        settings: &R::Settings,
    ) -> Prepared<R> {
        let mut recent = self.recent.lock();
        if let Some((_, prepared)) = recent.iter().find(|(kept, _)| kept == problem) {
            return prepared.clone();
        }
        let prepared = R::prepare(hw, problem, settings);
        if recent.len() == Self::CAPACITY {
            recent.pop_front();
        }
        recent.push_back((problem.clone(), prepared.clone()));
        prepared
    }
}

impl<R: Routine> Default for PreparedLaunches<R> {
    fn default() -> Self {
        Self {
            recent: Mutex::new(VecDeque::new()),
        }
    }
}

impl<R: Routine> core::fmt::Debug for PreparedLaunches<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedLaunches")
            .field("recent", &self.recent.lock().len())
            .finish()
    }
}
