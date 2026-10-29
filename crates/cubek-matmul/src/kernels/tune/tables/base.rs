use super::{few_rows, many_rows, nvfp4};
use crate::kernels::plan::{Decoded, MatmulCall, MatmulPlan};
use crate::kernels::select::problem::OperandKind;
use crate::kernels::select::selector::MatmulTable;
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::{MatmulBatching, MatmulTuneKey};
use crate::kernels::tune::{ListedFor, Product};
use crate::kernels::tune::{
    MatmulPackedCandidate, MatmulRoutineCandidate, MatmulRoutineSettings, PackedContraction,
};
use crate::kernels::{MatmulBlueprint, MatmulRoutine, UnitRegistersMatmulRoutine};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::{MemoryAccess, compute_throughput_key};
use cubecl::tune::{AutotuneOutput, LocalTuner, Tunable, TunableSet, Work, local_tuner};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::Instructions;
use cubek_std::autotune::{
    Autotune, CandidateIdentity, Ceiling, GroupPriority, Roofline, TuneCandidate,
};
use cubek_std::plan::{Fastest, Step};
use cubek_std::routine::Routine;
use std::sync::Arc;

/// The priority each table's routines ship at: a caller's to set, since which group a device
/// measures first is a deployment's call, and fixed for the process once a table is first built.
/// The NVFP4 table races one kernel's settings in a group of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulTableGroups {
    /// The few-rows table's.
    pub few_rows: GroupPriority,
    /// The many-rows table's.
    pub many_rows: GroupPriority,
}

impl Default for MatmulTableGroups {
    fn default() -> Self {
        Self {
            few_rows: GroupPriority::DEFAULT,
            many_rows: GroupPriority::DEFAULT,
        }
    }
}

/// What a candidate of every table is identified by: a routine by its setting, which something
/// other than the tuner can launch again ([`MatmulTable::fastest_routine`]), every other by its
/// name.
pub type RoutineIdentity = CandidateIdentity<MatmulRoutineSettings>;

/// Any table's candidates.
pub type MatmulTunables = TunableSet<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// A table being built.
pub(super) type CandidateTable =
    Autotune<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// What a table's candidate hands back: the product, and the client it lives on, which the
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
static NVFP4_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("nvfp4");

impl MatmulTable {
    /// What [`PlanTree`](cubek_std::plan::PlanTree) calls this table.
    pub fn name(self) -> &'static str {
        match self {
            Self::FewRows => "few-rows",
            Self::ManyRows => "many-rows",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// This table as a plan: its fastest candidate for the call's key, its groups at `groups`.
    pub fn plan(self, groups: MatmulTableGroups) -> MatmulPlan {
        Box::new(Fastest::new(self.name(), move |call: &MatmulCall| {
            self.launch(call, groups)
        }))
    }

    /// `call` through this table's fastest candidate, measuring first where no winner is settled
    /// for its key.
    pub fn launch(self, call: &MatmulCall, groups: MatmulTableGroups) -> TensorHandle {
        let (tuner, tunables) = self.tunables(call, groups);
        tuner
            .execute(&call.tune_id, &call.client, tunables, call.clone())
            .handle
    }

    /// The routine, at its setting, this table settled as the fastest for `call`, and the key it
    /// was settled under: `None` until the tuner has settled one for the key, and where the winner
    /// is no routine. What a caller that launches the routine itself, into a sink of its own, asks.
    pub fn fastest_routine(
        self,
        call: &MatmulCall,
        groups: MatmulTableGroups,
    ) -> Option<(MatmulRoutineSettings, MatmulTuneKey)> {
        let (tuner, tunables) = self.tunables(call, groups);
        let key = tunables.generate_key(call);
        let settings = tuner
            .fastest_identity(&call.tune_id, &tunables, &key)?
            .routine()?;
        Some((settings, key))
    }

    /// This table's tuner and its candidates on `call`'s device, built on first touch.
    ///
    /// The one place each set is built: a tuner caches a set per initializer type, so a second
    /// `init` site would build a second set.
    fn tunables(
        self,
        call: &MatmulCall,
        groups: MatmulTableGroups,
    ) -> (
        &'static LocalTuner<MatmulTuneKey, CubeTuneId>,
        Arc<MatmulTunables>,
    ) {
        let hw = call.hw.clone();
        let (tuner, candidates): (_, fn(Hardware, MatmulTableGroups) -> MatmulTunables) = match self
        {
            Self::FewRows => (&FEW_ROWS_TUNER, few_rows::table),
            Self::ManyRows => (&MANY_ROWS_TUNER, many_rows::table),
            Self::Nvfp4 => (&NVFP4_TUNER, nvfp4::table),
        };
        let tunables = tuner.init(&call.tune_id, move || candidates(hw.clone(), groups));
        (tuner, tunables)
    }
}

/// A table measured on representative inputs, its candidates keyed as the call keys them.
pub(super) fn measured_table() -> CandidateTable {
    CandidateTable::measured_on(MatmulCall::key, "as-keyed", MatmulCall::representative)
}

/// A float routine at `candidate`'s setting, or the reason it did not run.
///
/// # Errors
///
/// A packed input, which the routine's float settings are not listed for, and the routine's own
/// decline.
pub(super) fn microkernel_candidate<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    candidate: &MatmulRoutineCandidate<R>,
    call: MatmulCall,
) -> Result<MatmulProduct, String>
where
    R: MatmulRoutine<M, SM> + Routine<Microkernel = M, Blueprint = MatmulBlueprint<M, SM>> + Copy,
{
    ListedFor::Float
        .admits(Product::of(call_kinds(&call)))
        .map_err(|refusal| format!("{refusal:?}"))?;
    let settings = candidate.strategy();
    let launched = match (&call.routes, call.destination) {
        (Some(routes), _) => routes.launch(&call.hw, &candidate.routine(), settings),
        (None, Some(destination)) => candidate
            .launch_into(
                &call.client,
                call.lhs.tensor,
                call.rhs.tensor,
                PlainTensor::of_handle(destination.clone()),
            )
            .map(|()| destination),
        (None, None) => candidate.launch(&call.client, call.lhs.tensor, call.rhs.tensor),
    };
    launched
        .map(|handle| MatmulProduct {
            client: call.client,
            handle,
        })
        .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
}

/// A routine at `candidate`'s packed setting, reading the packed inputs as stored, written in the
/// call's element, or the reason it did not run.
///
/// # Errors
///
/// Float inputs, which the packed settings are not listed for, and the routine's own decline.
pub(super) fn packed_candidate<R, M, SM>(
    candidate: &MatmulPackedCandidate<R>,
    call: MatmulCall,
) -> Result<MatmulProduct, String>
where
    M: MatmulMicrokernel,
    R: MatmulRoutine<M, SM> + Copy,
    R::Settings: PackedContraction,
{
    ListedFor::Packed
        .admits(Product::of(call_kinds(&call)))
        .map_err(|refusal| format!("{refusal:?}"))?;
    let settings = candidate.settings();
    let launched = match (&call.routes, call.destination) {
        (Some(routes), _) => routes.launch(&call.hw, &candidate.routine(), settings),
        (None, Some(destination)) => candidate
            .launch_into(
                &call.client,
                call.lhs.tensor,
                call.rhs.tensor,
                PlainTensor::of_handle(destination.clone()),
            )
            .map(|()| destination),
        (None, None) => {
            candidate.launch_written_in(&call.client, call.lhs.tensor, call.rhs.tensor, call.out)
        }
    };
    launched
        .map(|handle| MatmulProduct {
            client: call.client,
            handle,
        })
        .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
}

/// [`packed_candidate`] with a packed lhs decoded first, the rhs still read as stored: a packed
/// activation beside a weight no routine reads it beside (an MXFP8 activation's blocks of 32 beside
/// an NVFP4 weight's 16), or beside one a routine reads it beside slower than the weight alone,
/// contracted as fast as the weight alone allows. An activation is a few rows against the weight's
/// many, so decoding it costs little of a product that streams the weight; decoding the weight
/// instead, as the [`decoded_floor`] does, rewrites the weight on every call.
///
/// # Errors
///
/// A float lhs, which the candidate as stored serves; and every error of [`packed_candidate`].
pub(super) fn activation_decoded_candidate<R, M, SM>(
    candidate: &MatmulPackedCandidate<R>,
    call: MatmulCall,
) -> Result<MatmulProduct, String>
where
    M: MatmulMicrokernel,
    R: MatmulRoutine<M, SM> + Copy,
    R::Settings: PackedContraction,
{
    if matches!(call.lhs.tensor, StoredTensor::Plain(_)) {
        return Err("the lhs is float: the candidate reading it as stored serves it".into());
    }
    let lhs = Decoded
        .input(&call, &call.lhs)
        .map_err(|why| why.to_string())?;
    packed_candidate(candidate, MatmulCall { lhs, ..call })
}

/// What an [`activation_decoded_candidate`] is identified by: its setting, named apart from the
/// same setting reading the lhs as stored, which a caller may launch again as a routine; this one
/// it may not, since its lhs is decoded first.
pub(super) fn activation_decoded_identity<R: Routine>(
    candidate: &MatmulPackedCandidate<R>,
) -> RoutineIdentity
where
    R::Settings: core::fmt::Display + PackedContraction,
{
    CandidateIdentity::Named(format!("{} activation decoded", candidate.strategy()))
}

/// Both tables' floor: the unit routine at its default, every packed input decoded first
/// ([`Decoded`]). It asks nothing of the device, so it serves what no other candidate binds: an
/// input laid out no body can describe, or two packed inputs whose scales group `k` apart, which
/// every routine reading packed inputs as stored declines. A float input passes through as it is.
/// It is identified, so where it wins over float inputs a caller can launch it again.
pub(super) fn decoded_floor(
    hw: &Hardware,
) -> Tunable<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity> {
    let lead = MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, hw).swap_remove(0);
    Tunable::identified(RoutineIdentity::new(&lead), move |call: MatmulCall| {
        let decoded = Decoded.apply(&call).map_err(|why| why.to_string())?;
        microkernel_candidate(&lead, decoded)
    })
}

/// What a product costs, for the roofline a round stops at: the contraction's operations and the
/// bytes of the three operands. A packed input counts its packed bytes, what a kernel streaming it
/// at its stored width reads, so the floor every candidate is measured against is the same. A stack
/// read through routes counts the slabs its batches can reach, one a batch at most.
fn work(key: &MatmulTuneKey, call: &MatmulCall) -> Work {
    let shape = key.shape();
    let bytes = |tensor: &StoredTensor| -> usize {
        tensor
            .buffers()
            .iter()
            .map(|buffer| buffer.size as usize)
            .sum()
    };
    let output = shape.m * shape.n * shape.batch * shape.elem.size();
    let stack = bytes(&call.rhs.tensor);
    let rhs = match shape.rhs.batching {
        MatmulBatching::Slabs { slabs } => stack / slabs * shape.batch.min(slabs),
        MatmulBatching::Shared | MatmulBatching::Batched => stack,
    };
    Work {
        compute_ops: 2 * shape.m * shape.n * shape.k * shape.batch,
        bytes: bytes(&call.lhs.tensor) + rhs + output,
    }
}

/// The few-rows table's roofline: the bus, read. A routine that does not stage streams its rhs
/// past a handful of lhs rows and writes a fraction of what it reads, so its ceiling is the read
/// bandwidth at the product's footprint.
pub(super) fn few_rows_roofline(key: &MatmulTuneKey, call: &MatmulCall) -> Roofline {
    Roofline {
        client: call.client.clone(),
        ceiling: Ceiling::Memory(MemoryAccess::Read),
        work: work(key, call),
    }
}

/// The many-rows table's roofline: both walls, and the key picks. Past the few-rows line the rhs
/// is reused across the rows, so the arithmetic starts to bind, but not on every key a prompt
/// makes: a product's flops per byte is roughly its rows, and the device's own balance sits
/// among them. So the table names the largest matrix instruction the device offers at the
/// product's element accumulating in `f32`, plain arithmetic where it offers none, and the read
/// bus, and argues the stop from whichever is higher for the key ([`Ceiling::Either`]).
pub(super) fn many_rows_roofline(key: &MatmulTuneKey, call: &MatmulCall) -> Roofline {
    let elem = key.shape().elem;
    let accumulate = ElemType::Float(FloatKind::F32);
    let instruction = call
        .hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .for_triple(elem, elem, accumulate)
        .largest()
        .map(|i| (i.m, i.n, i.k));
    Roofline {
        client: call.client.clone(),
        ceiling: Ceiling::Either {
            compute: compute_throughput_key(instruction, elem, accumulate),
            memory: MemoryAccess::Read,
        },
        work: work(key, call),
    }
}

/// Declines a routed product, for the candidates that read no routes.
///
/// # Errors
///
/// Where `call` states a routed product.
pub(super) fn reads_no_routes(call: &MatmulCall) -> Result<(), String> {
    match call.routes {
        Some(_) => Err("reads no routes; the rhs is a stack read through them".to_string()),
        None => Ok(()),
    }
}

/// What the two inputs of `call` are.
fn call_kinds(call: &MatmulCall) -> [OperandKind; 2] {
    [call.lhs.tensor.kind(), call.rhs.tensor.kind()]
}
