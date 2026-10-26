use cubecl::tile::launch::Refusal;
use cubecl_environment::sync::Mutex;

use crate::Hardware;
use crate::routine::{Grid, Routine};

/// The launches one routine prepared at one setting on one device, per problem: what a caller
/// that launches the same problem again and again reads instead of preparing it again.
///
/// [`prepare`](Routine::prepare) is pure, so a launch prepared once is the launch every later call
/// of the same problem would prepare. Preparing walks the microkernel's candidates and resolves
/// each, which costs more host time than the launch of a small product; a tuned winner runs the
/// same few problems for the rest of the process.
///
/// It holds the [`CAPACITY`](Self::CAPACITY) problems prepared last, so a caller cycling through
/// many shapes costs a comparison per held problem and no more than preparing each.
pub struct PreparedLaunches<R: Routine> {
    held: Mutex<Vec<(R::Problem, Result<(R::Blueprint, Grid), Refusal>)>>,
}

impl<R: Routine> PreparedLaunches<R>
where
    R::Problem: PartialEq,
{
    /// The most problems held: a model's handful of shapes per winner.
    pub const CAPACITY: usize = 16;

    /// The launch `routine` prepares for `problem` at `settings` on `hw`, or why it declines it:
    /// the one held for `problem`, or prepared now and held. The caller hands the same routine,
    /// setting and device every time.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run `problem` there, as [`prepare`](Routine::prepare) says.
    pub fn prepare(
        &self,
        routine: &R,
        hw: &Hardware,
        problem: &R::Problem,
        settings: &R::Settings,
    ) -> Result<(R::Blueprint, Grid), Refusal> {
        let mut held = self.held.lock();
        if let Some((_, prepared)) = held.iter().find(|(held, _)| held == problem) {
            return prepared.clone();
        }
        let prepared = routine.prepare(hw, problem, settings);
        if held.len() == Self::CAPACITY {
            drop(held.remove(0));
        }
        held.push((problem.clone(), prepared.clone()));
        prepared
    }
}

impl<R: Routine> Default for PreparedLaunches<R> {
    fn default() -> Self {
        Self {
            held: Mutex::new(Vec::new()),
        }
    }
}

impl<R: Routine> core::fmt::Debug for PreparedLaunches<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedLaunches")
            .field("held", &self.held.lock().len())
            .finish()
    }
}
