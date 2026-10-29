use super::{block_scaled, few_rows, many_rows};
use crate::kernels::plan::{MatmulCall, MatmulPlan};
use crate::kernels::select::problem::MatmulProblem;
use crate::kernels::tune::MatmulRoutineSettings;
use crate::kernels::tune::key::MatmulTuneKey;
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tune::{AutotuneOutput, LocalTuner, TunableSet, local_tuner};
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, CandidateIdentity, GroupPriority};
use cubek_std::plan::Fastest;
use std::sync::Arc;

/// The autotunes a product races on, split by the rows a routine that does not stage would stream:
/// each holds the candidates for a float product and for a packed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulAutotune {
    /// A handful of rows, each read once: the routines that do not stage, on packed inputs at the
    /// settings worth trying on them, reading them as stored.
    FewRows,
    /// Everything past the few-rows bound: the staged routines, and on packed inputs the packed
    /// staged routine reading them as stored and the float routines on them decoded first.
    ManyRows,
    /// Two packed inputs multiplied on a GPU's block-scaled tensor cores, at each of
    /// [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores)'s
    /// settings: what that pin runs, and nothing else.
    BlockScaled,
}

/// The most rows a float product goes to the few-rows autotune with: the rows every float few-rows
/// candidate holds whole, each one more accumulator line per cell of its register block (the plane
/// shuffle's `MAX_ROWS` is held to it by a test). Past it, a body that reuses each column's values
/// across rows is the better shape anyway.
pub const FLOAT_FEW_ROWS: usize = 8;

/// Whether a float product of `rows` rows goes to the few-rows autotune: from one row to
/// [`FLOAT_FEW_ROWS`]. A float product of no rows goes to the many-rows autotune, which has a
/// candidate for it.
pub fn float_few_rows(rows: usize) -> bool {
    (1..=FLOAT_FEW_ROWS).contains(&rows)
}

/// The most rows a packed product goes to the few-rows autotune with. Past it the staged routines
/// win on arithmetic, and measuring a routine that does not stage at prefill widths would cost
/// minutes of first-touch tuning for a candidate that cannot win there.
pub const QUANTIZED_FEW_ROWS: usize = 32;

/// Whether a packed product of `rows` rows goes to the few-rows autotune: up to
/// [`QUANTIZED_FEW_ROWS`], no rows included, the packed candidates' own admission answering every
/// width below the bound.
pub fn quantized_few_rows(rows: usize) -> bool {
    rows <= QUANTIZED_FEW_ROWS
}

/// The priority each autotune's routines ship at: a caller's to set, since which group a device
/// measures first is a deployment's call, and fixed for the process once an autotune is first
/// built. The block-scaled autotune races one kernel's settings in a group of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulAutotuneGroups {
    /// The few-rows autotune's.
    pub few_rows: GroupPriority,
    /// The many-rows autotune's.
    pub many_rows: GroupPriority,
}

impl Default for MatmulAutotuneGroups {
    fn default() -> Self {
        Self {
            few_rows: GroupPriority::DEFAULT,
            many_rows: GroupPriority::DEFAULT,
        }
    }
}

/// What a candidate of every autotune is identified by: a routine by its setting, which something
/// other than the tuner can launch again ([`MatmulAutotune::fastest_routine`]), every other by its
/// name.
pub type RoutineIdentity = CandidateIdentity<MatmulRoutineSettings>;

/// Any autotune's candidates.
pub type MatmulTunables = TunableSet<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// An autotune being built.
pub(super) type CandidateList = Autotune<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// What an autotune's candidate hands back: the product, and the client it lives on, which the
/// autotune checks read two candidates' products through.
#[derive(Clone)]
pub struct MatmulProduct {
    pub client: Client,
    pub handle: TensorHandle,
}

impl AutotuneOutput for MatmulProduct {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        crate::kernels::tune::assert_equivalent(&self.client, self.handle.clone(), other.handle);
    }
}

static FEW_ROWS_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("few-rows");
static MANY_ROWS_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("many-rows");
static BLOCK_SCALED_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("block-scaled");

impl MatmulAutotune {
    /// Which autotune a float product goes to, by [`float_few_rows`], so no width reaches an
    /// autotune whose every candidate declines it. Packed inputs decoded first make a float product
    /// here.
    pub fn new(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(float_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// Which autotune a product reading packed inputs as stored goes to, by the rows alone
    /// ([`quantized_few_rows`]): the few-rows autotune's candidates decline what they do not close,
    /// and its floor, the packed inputs decoded, answers there.
    pub fn packed(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(quantized_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// What [`PlanTree`](cubek_std::plan::PlanTree) calls this autotune.
    pub fn name(self) -> &'static str {
        match self {
            Self::FewRows => "few-rows",
            Self::ManyRows => "many-rows",
            Self::BlockScaled => "block-scaled",
        }
    }

    /// This autotune as a plan: its fastest candidate for the call's key, its groups at `groups`.
    pub fn plan(self, groups: MatmulAutotuneGroups) -> MatmulPlan {
        Box::new(Fastest::new(self.name(), move |call: &MatmulCall| {
            self.launch(call, groups)
        }))
    }

    /// `call` through this autotune's fastest candidate, measuring first where no winner is settled
    /// for its key.
    pub fn launch(self, call: &MatmulCall, groups: MatmulAutotuneGroups) -> TensorHandle {
        let (tuner, tunables) = self.tunables(call, groups);
        tuner
            .execute(&call.tune_id, &call.client, tunables, call.clone())
            .handle
    }

    /// The routine, at its setting, this autotune settled as the fastest for `call`, and the key it
    /// was settled under: `None` until the tuner has settled one for the key, and where the winner
    /// is no routine. What a caller that launches the routine itself, into a sink of its own, asks.
    pub fn fastest_routine(
        self,
        call: &MatmulCall,
        groups: MatmulAutotuneGroups,
    ) -> Option<(MatmulRoutineSettings, MatmulTuneKey)> {
        let (tuner, tunables) = self.tunables(call, groups);
        let key = tunables.generate_key(call);
        let settings = tuner
            .fastest_identity(&call.tune_id, &tunables, &key)?
            .routine()?;
        Some((settings, key))
    }

    /// This autotune's tuner and its candidates on `call`'s device, built on first touch.
    ///
    /// The one place each set is built: a tuner caches a set per initializer type, so a second
    /// `init` site would build a second set.
    fn tunables(
        self,
        call: &MatmulCall,
        groups: MatmulAutotuneGroups,
    ) -> (
        &'static LocalTuner<MatmulTuneKey, CubeTuneId>,
        Arc<MatmulTunables>,
    ) {
        let hw = call.hw.clone();
        let (tuner, candidates): (_, fn(Hardware, MatmulAutotuneGroups) -> MatmulTunables) =
            match self {
                Self::FewRows => (&FEW_ROWS_TUNER, few_rows::candidates),
                Self::ManyRows => (&MANY_ROWS_TUNER, many_rows::candidates),
                Self::BlockScaled => (&BLOCK_SCALED_TUNER, block_scaled::candidates),
            };
        let tunables = tuner.init(&call.tune_id, move || candidates(hw.clone(), groups));
        (tuner, tunables)
    }
}

/// An autotune measured on representative inputs, its candidates keyed as the call keys them.
pub(super) fn candidate_list() -> CandidateList {
    CandidateList::measured_on(MatmulCall::key, "as-keyed", MatmulCall::representative)
}
