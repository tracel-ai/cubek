//! A tile-DSL routine reached by name, at one of the settings it lists, against the host's
//! product: a dense one ([`Product`]), or one an index or a packing makes
//! ([`RoutineProduct`]).

use core::fmt::{Debug, Display, Formatter};
use core::marker::PhantomData;

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::{CubeElement, Scalar};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_matmul::kernels::{
    MatmulRoutine,
    tensor::{PlainTensor, StoredTensor},
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, ScopeUnderTest, Written};

use crate::matmul::{Declines, F16, F32, I32, Product};

/// A product a routine launches by name: the tensors it hands the routine, and what the host says
/// the routine writes.
pub(crate) trait RoutineProduct: Debug + Clone {
    /// `routine` launched at `settings` on this product, and the output it wrote.
    ///
    /// # Errors
    ///
    /// Why the routine runs no such product at `settings` on `hw`.
    fn launch<R, M, SM>(
        &self,
        routine: &R,
        hw: &Hardware,
        client: &Client,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Refusal>
    where
        R: MatmulRoutine<M, SM>,
        M: MatmulMicrokernel;

    /// What the routine writes, computed on the host.
    fn reference(&self) -> HostData;

    /// The largest difference from the reference accepted.
    fn tolerance(&self) -> f32;
}

impl RoutineProduct for Product {
    fn launch<R, M, SM>(
        &self,
        routine: &R,
        hw: &Hardware,
        client: &Client,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Refusal>
    where
        R: MatmulRoutine<M, SM>,
        M: MatmulMicrokernel,
    {
        let (lhs, rhs, out) = self.tensors(client);
        routine.launch_into(
            hw,
            client,
            StoredTensor::Plain(PlainTensor::of_handle(lhs)),
            StoredTensor::Plain(PlainTensor::of_handle(rhs)),
            PlainTensor::of_handle(out.clone()),
            settings,
        )?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        Product::reference(self)
    }

    fn tolerance(&self) -> f32 {
        Product::tolerance(self, &cubecl::test_device().client())
    }
}

/// A routine at one of its settings, on one product.
pub(crate) struct AtSetting<S, P> {
    pub settings: S,
    pub product: P,
}

impl<S: Display, P: Debug> Debug for AtSetting<S, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} on {:?}", self.settings, self.product)
    }
}

/// A matmul routine, named, against the host's product.
pub(crate) struct ByRoutine<R, M, SM, P = Product> {
    routine: R,
    hw: Hardware,
    declines: Declines,
    kinds: PhantomData<(M, SM, P)>,
}

impl<R, M, SM, P> ByRoutine<R, M, SM, P>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
    P: RoutineProduct,
{
    pub(crate) fn new(routine: R, declines: Declines) -> Self {
        Self {
            routine,
            hw: Hardware::from_client(&cubecl::test_device().client()),
            declines,
            kinds: PhantomData,
        }
    }

    /// Every setting the routine lists on the test device, on each of `products`.
    pub(crate) fn at_every_setting(&self, products: &[P]) -> Vec<AtSetting<R::Settings, P>> {
        self.at(self.routine.settings(&self.hw), products)
    }

    /// The routine's lead, the first setting it lists on the test device, on each of `products`.
    pub(crate) fn at_its_lead(&self, products: &[P]) -> Vec<AtSetting<R::Settings, P>> {
        self.at(
            self.routine.settings(&self.hw).into_iter().take(1),
            products,
        )
    }

    /// Each of `settings` on each of `products`.
    pub(crate) fn at(
        &self,
        settings: impl IntoIterator<Item = R::Settings>,
        products: &[P],
    ) -> Vec<AtSetting<R::Settings, P>> {
        settings
            .into_iter()
            .flat_map(|settings| {
                products.iter().map(move |product| AtSetting {
                    settings: settings.clone(),
                    product: product.clone(),
                })
            })
            .collect()
    }

    /// The device the routine is listed and launched for.
    pub(crate) fn hw(&self) -> &Hardware {
        &self.hw
    }
}

impl<R, M, SM, P> ScopeUnderTest for ByRoutine<R, M, SM, P>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
    P: RoutineProduct,
{
    type Case = AtSetting<R::Settings, P>;

    /// # Panics
    ///
    /// Where a routine that declines nothing refuses a product.
    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launched = case
            .product
            .launch(&self.routine, &self.hw, client, &case.settings);
        match (launched, self.declines) {
            (Ok(out), _) => Ok(Product::written(out)),
            (Err(why), Declines::AsItStates) => Err(why),
            (Err(why), Declines::Never) => panic!("{case:?} refused: {why:?}"),
        }
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![case.product.reference()]
    }

    fn tolerance(&self, case: &Self::Case) -> f32 {
        case.product.tolerance()
    }
}

/// The products the routines are held to, by how many rows they hold.
impl Product {
    /// Products of many rows, for the routines that tile them: square in both elements, a
    /// rectangle, batches, an lhs laid along `m`, an rhs laid along `k` at a depth no line wider
    /// than eight divides, and extents no tile divides.
    pub(crate) fn many_rows() -> Vec<Self> {
        vec![
            Self::of(F32).m(256).n(256).k(256),
            Self::of(F16).m(256).n(256).k(256),
            Self::of(F32).m(64).n(128).k(32),
            Self::of(F32)
                .m(64)
                .n(64)
                .k(64)
                .batched(shape![3], shape![3]),
            Self::of(F32).m(256).n(128).k(64).lhs_along_m(),
            Self::of(F16).m(256).n(128).k(64).lhs_along_m(),
            Self::of(F32).m(200).n(48).k(56).rhs_along_k(),
            Self::of(F32).m(37).n(41).k(53),
        ]
    }

    /// Products of at most eight rows, a decode step's, for the routines that hold their rows
    /// whole: an rhs laid either way, a depth no line wider than one divides, and a model's
    /// projection in `f16`.
    pub(crate) fn few_rows() -> Vec<Self> {
        vec![
            Self::of(F32).m(4).n(24).k(40),
            Self::of(F32).m(4).n(24).k(40).rhs_along_k(),
            Self::of(F32).m(3).n(40).k(41).rhs_along_k(),
            Self::of(F16).m(1).n(256).k(256).rhs_along_k(),
            Self::of(F16).m(8).n(96).k(128),
        ]
    }

    /// `m` rows of an `i32` product, its rhs laid either way: summed exactly, in `i32`.
    pub(crate) fn integers(m: usize) -> Vec<Self> {
        let product = Self::of(I32).m(m).n(24).k(40);
        vec![product.clone(), product.rhs_along_k()]
    }
}

/// `values` uploaded as a dense row-major tensor of `shape`.
pub(crate) fn upload<E: CubeElement + Scalar>(
    client: &Client,
    shape: &[usize],
    values: &[E],
) -> PlainTensor {
    let handle = TensorHandle::new_contiguous(
        Shape::from(shape.to_vec()),
        client.create(Bytes::from_elems(values.to_vec())),
        E::elem_type_native(),
    );
    PlainTensor::of_handle(handle)
}
