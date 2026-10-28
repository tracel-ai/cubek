use super::MatmulCall;
use crate::kernels::MatmulRoutine;
use crate::kernels::tensor::StoredTensor;
use core::marker::PhantomData;
use cubecl::std::tensor::TensorHandle;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::plan::{Declined, Plan, PlanTree};

/// One routine as a plan: at the first of its settings that serves the product, or at one setting.
///
/// A packed input is read as it is stored. A routine that does not read packed inputs declines
/// them while it prepares, and one at a setting is listed for float products only.
pub struct RoutinePlan<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM = ()> {
    routine: R,
    name: &'static str,
    settings: Option<R::Settings>,
    _microkernel: PhantomData<fn() -> (M, SM)>,
}

impl<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM> RoutinePlan<R, M, SM> {
    /// `routine`, named `name`, at the first of its settings that serves.
    pub fn first_served(routine: R, name: &'static str) -> Self {
        Self {
            routine,
            name,
            settings: None,
            _microkernel: PhantomData,
        }
    }

    /// `routine`, named `name`, at `settings` alone.
    pub fn at(routine: R, name: &'static str, settings: R::Settings) -> Self {
        Self {
            routine,
            name,
            settings: Some(settings),
            _microkernel: PhantomData,
        }
    }
}

impl<R: MatmulRoutine<M, SM>, M: MatmulMicrokernel, SM> Plan<MatmulCall> for RoutinePlan<R, M, SM> {
    type Output = TensorHandle;

    /// A float product is written in its inputs' element, a packed one in the call's.
    fn launch(&self, call: &MatmulCall) -> Result<TensorHandle, Declined> {
        let (hw, client) = (&call.hw, &call.client);
        let (lhs, rhs) = (call.lhs.tensor.clone(), call.rhs.tensor.clone());
        let float = matches!(
            (&lhs, &rhs),
            (StoredTensor::Plain(_), StoredTensor::Plain(_))
        );
        let launched = match (&self.settings, float) {
            (None, true) => self.routine.launch_first_served_on(hw, client, lhs, rhs),
            (None, false) => self
                .routine
                .launch_first_served_in_on(hw, client, lhs, rhs, call.out),
            (Some(settings), _) => self.routine.launch_on(hw, client, lhs, rhs, settings),
        };
        launched.map_err(|why| Declined::by(self, why))
    }

    fn tree(&self) -> PlanTree {
        PlanTree::Routine(self.name.to_string())
    }
}
