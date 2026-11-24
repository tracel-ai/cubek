//! The microkernels every kernel family is built from. What a microkernel and its scope are, and
//! the [`Microkernel`](cubek_std::microkernel::Microkernel) trait and estimate every one shares,
//! are cubek-std's ([`microkernel`](cubek_std::microkernel)). Beyond a smaller scope's, a
//! microkernel here may be built from microkernels of another problem: an attention scores its
//! keys with a matmul microkernel.
//!
//! Nothing here is launched. A kernel places microkernels over a grid, binds its tensors and
//! launches; the kernels are their family's crate's (`cubek-matmul`'s, `cubek-attention`'s,
//! `cubek-rms-norm`'s and `cubek-swiglu`'s `kernels`).
//!
//! A reduction's microkernels, by the part of the row each takes: a unit's
//! ([`UnitDirectReduceMicrokernel`](reduce::unit::UnitDirectReduceMicrokernel)), a plane's
//! ([`PlaneShuffleReduceMicrokernel`](reduce::plane::PlaneShuffleReduceMicrokernel), built from
//! [`PlaneLineReduceMicrokernel`](reduce::plane::PlaneLineReduceMicrokernel) and
//! [`PlaneColumnsReduceMicrokernel`](reduce::plane::PlaneColumnsReduceMicrokernel)), a cube's
//! ([`CubeStagedReduceMicrokernel`](reduce::cube::CubeStagedReduceMicrokernel), built from
//! [`CubeLineReduceMicrokernel`](reduce::cube::CubeLineReduceMicrokernel),
//! [`CubeColumnsReduceMicrokernel`](reduce::cube::CubeColumnsReduceMicrokernel) and
//! [`CubeStridedReduceMicrokernel`](reduce::cube::CubeStridedReduceMicrokernel)) and a hypercube's
//! ([`HypercubeLastArrivalReduceMicrokernel`](reduce::hypercube::HypercubeLastArrivalReduceMicrokernel),
//! built from [`CubePartialReduceMicrokernel`](reduce::cube::CubePartialReduceMicrokernel)). They
//! share the units' reads
//! ([`UnitStridedReduceMicrokernel`](reduce::read::UnitStridedReduceMicrokernel),
//! [`UnitWalkReduceMicrokernel`](reduce::read::UnitWalkReduceMicrokernel),
//! [`UnitPartialsReduceMicrokernel`](reduce::read::UnitPartialsReduceMicrokernel)) and writes
//! ([`UnitCarrierWriteMicrokernel`](reduce::write::UnitCarrierWriteMicrokernel),
//! [`UnitLinesWriteMicrokernel`](reduce::write::UnitLinesWriteMicrokernel),
//! [`UnitRegionWriteMicrokernel`](reduce::write::UnitRegionWriteMicrokernel),
//! [`UnitGroupColumnWriteMicrokernel`](reduce::write::UnitGroupColumnWriteMicrokernel),
//! [`UnitPartialStoreMicrokernel`](reduce::write::UnitPartialStoreMicrokernel)), and accumulate into the
//! carrier a [`ReduceCarrierFamily`](reduce::ReduceCarrierFamily) names. A zeroing's is
//! [`UnitDirectZeroMicrokernel`](zero::unit::UnitDirectZeroMicrokernel).
//!
//! A folder per problem, behind its family's feature: what the problem is in its `base`, beside a
//! folder per scope. What every family shares, as the ring a staged walk holds its stages in, is
//! cubek-std's; what the producers of a product's lhs share, how they store their rows, is
//! `row_store`.

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "attention-backward")]
pub mod attention_backward;
#[cfg(feature = "interpolate")]
pub mod interpolate;
#[cfg(feature = "matmul")]
pub mod matmul;
#[cfg(feature = "pool")]
pub mod pool;
#[cfg(feature = "random")]
pub mod random;
#[cfg(feature = "reduce")]
pub mod reduce;
#[cfg(feature = "rms-norm")]
pub mod rms_norm;
#[cfg(feature = "row-store")]
pub mod row_store;
#[cfg(feature = "attention")]
pub mod split_keys_merge;
#[cfg(feature = "swiglu")]
pub mod swiglu;
#[cfg(feature = "zero")]
pub mod zero;
