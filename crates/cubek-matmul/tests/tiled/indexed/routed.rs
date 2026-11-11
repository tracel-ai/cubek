//! A routed product: every token's row against each of the slabs its routes name, as a mixture of
//! experts runs its experts. A route past the stack answers zeros.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::indexed::routed::{RoutedMatmulTensors, launch_tuned};
use cubek_matmul::kernels::indexed::stack::SlabAxis;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine};
use cubek_std::Hardware;
use cubek_std::autotune::GroupPriority;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

use crate::matmul::Declines;
use crate::tiled::indexed::{Routed, Stack, values};
use crate::tiled::packed::{PackedFormat, PackedSlabs};
use crate::tiled::{ByRoutine, upload};

#[test]
fn every_plane_shuffle_setting_reads_each_route_s_slab() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check(shuffle.at_every_setting(&[Routed::MOE]));
}

#[test]
fn every_unit_registers_setting_reads_each_route_s_slab() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::AsItStates);
    registers.check(registers.at_every_setting(&[Routed::MOE]));
}

/// Every packed setting of the plane shuffle, against a stack packed in each format, its `n` a
/// multiple of the column tile and not.
#[test]
fn every_packed_plane_shuffle_setting_reads_each_route_s_packed_slab() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let products: Vec<Routed> = [PackedFormat::Q8, PackedFormat::Q4, PackedFormat::Nvfp4]
        .into_iter()
        .flat_map(|format| [Routed::MOE.packed(format), Routed::NARROW.packed(format)])
        .collect();
    shuffle.check(shuffle.at(PlaneShuffleMatmulRoutine.packed_settings(), &products));
}

/// The routed autotune, against a stack laid either way it takes: a col float stack, its buffer
/// `[slabs · n, k]` behind the `[k, slabs · n]` view, and a stack packed in each format.
#[test]
fn the_routed_autotune_reads_each_route_s_slab() {
    TunedRouted.check(
        [
            Stack::Float,
            Stack::Packed(PackedFormat::Q8),
            Stack::Packed(PackedFormat::Q4),
            Stack::Packed(PackedFormat::Nvfp4),
            Stack::Packed(PackedFormat::Mxfp4),
        ]
        .map(|stack| Routed {
            stack,
            ..Routed::MOE
        }),
    );
}

/// The routed product through the routed autotune, which picks the routine and its setting.
struct TunedRouted;

impl ScopeUnderTest for TunedRouted {
    type Case = Routed;

    /// A cube larger than the device holds is a skip, the CPU runtime's few units; so is a packed
    /// stack on a device with no matrix instruction, which no floor stages for.
    ///
    /// # Panics
    ///
    /// On any other refusal: the autotune always reaches a floor that answers.
    fn launch(&self, client: &Client, routed: &Routed) -> Result<Vec<Written>, Refusal> {
        let Routed {
            tokens,
            picks,
            slabs,
            k,
            n,
            stack,
        } = *routed;
        let stored = match routed.packed_slabs() {
            Some(spec) => StoredTensor::Packed(PackedSlabs::new(client, spec).matrix()),
            None => {
                let values = values(slabs * k * n, 2);
                let col: Vec<f32> = (0..slabs * n)
                    .flat_map(|row| (0..k).map(move |kk| (row, kk)))
                    .map(|(row, kk)| values[((row / n) * k + kk) * n + row % n])
                    .collect();
                StoredTensor::Plain(PlainTensor::of_handle(TensorHandle::new(
                    client.create(Bytes::from_elems(col)),
                    [k, slabs * n],
                    [1, k],
                    f32::elem_type_native(),
                )))
            }
        };
        let tensors = RoutedMatmulTensors::new(
            client,
            routed.activation(client).0,
            stored,
            upload(client, &[tokens, picks], &routed.routes()),
            slabs,
            SlabAxis::Output,
        )
        .expect("a stack of whole slabs derives a routed launch");
        let hw = Hardware::from_client(client);
        let no_matrix_instruction = hw.properties.features.matmul.cmma.iter().next().is_none();
        match launch_tuned(&cubecl::test_device(), tensors, GroupPriority::DEFAULT) {
            Ok(out) => Ok(vec![Written {
                name: "out",
                tensor: out,
                host: HostDataType::F32,
            }]),
            Err(why @ Refusal::CubePastDevice { .. }) => Err(why),
            Err(why) if stack != Stack::Float && no_matrix_instruction => Err(why),
            Err(why) => panic!("{routed:?} refused: {why:?}"),
        }
    }

    fn reference(&self, routed: &Routed) -> Vec<HostData> {
        let shape = vec![routed.tokens, routed.picks, routed.n];
        vec![HostData::row_major(
            HostDataVec::F32(routed.expected()),
            shape.into(),
        )]
    }

    fn tolerance(&self, routed: &Routed) -> f32 {
        routed.tolerance()
    }
}
