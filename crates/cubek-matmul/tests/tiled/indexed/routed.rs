//! A routed product: every token's row against each of the slabs its routes name, as a mixture of
//! experts runs its experts. A route past the stack answers zeros.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubek_matmul::kernels::indexed::m_grouped::MGroupedSettings;
use cubek_matmul::kernels::indexed::routed::{RoutedMatmulCandidates, RoutedMatmulTensors};
use cubek_matmul::kernels::indexed::stack::SlabDimension;
use cubek_matmul::kernels::tensor::{PlainTensor, StoredTensor};
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

use crate::matmul::{Declines, skip_unless_matrix_instruction};
use crate::tiled::indexed::{Routed, Stack, values};
use crate::tiled::packed::PackedFormat;
use crate::tiled::{ByRoutine, upload};

#[test]
fn every_plane_shuffle_setting_reads_each_route_s_slab() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check_some(shuffle.at_every_setting(&[Routed::MOE]));
}

#[test]
fn every_unit_registers_setting_reads_each_route_s_slab() {
    let registers = ByRoutine::new(UnitRegistersMatmulRoutine, Declines::AsItStates);
    registers.check_some(registers.at_every_setting(&[Routed::MOE]));
}

/// Every packed setting of the plane shuffle, against a stack packed in each format, its `n` a
/// multiple of the column tile and not.
#[test]
fn every_packed_plane_shuffle_setting_reads_each_route_s_packed_slab() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let products: Vec<Routed> = [
        PackedFormat::Q8,
        PackedFormat::Q4,
        PackedFormat::Nvfp4,
        PackedFormat::Mxfp4,
    ]
    .into_iter()
    .flat_map(|format| [Routed::MOE.packed(format), Routed::NARROW.packed(format)])
    .collect();
    shuffle.check_some(shuffle.at(PlaneShuffleMatmulRoutine::packed_settings(), &products));
}

/// The routed products of a stack laid in each tile order: a slab of several strips, its `n` a
/// multiple of the column tile and not, and a slab of one strip.
fn in_tile_order() -> Vec<Routed> {
    Stack::IN_TILES
        .into_iter()
        .flat_map(|stack| {
            [Routed::MOE, Routed::NARROW, Routed::ONE_STRIP].map(|routed| routed.stored(stack))
        })
        .collect()
}

/// Every packed setting of the plane shuffle against a stack in tile order: each route's slab a
/// fixed run of whole strips, the table moving the window by it.
#[test]
fn every_packed_plane_shuffle_setting_reads_each_route_s_slab_of_a_stack_in_tile_order() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    shuffle.check_some(shuffle.at(
        PlaneShuffleMatmulRoutine::packed_settings(),
        &in_tile_order(),
    ));
}

/// Every packed setting of the cube staged routine against a stack in tile order: skipped on a
/// device with no matrix instruction, which it stages a packed input for.
#[test]
fn every_packed_cube_staged_setting_reads_each_route_s_slab_of_a_stack_in_tile_order() {
    if skip_unless_matrix_instruction() {
        return;
    }
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let settings = MGroupedSettings::cube_staged_packed(staged.hw())
        .into_iter()
        .map(|settings| settings.routine);
    staged.check_some(staged.at(settings, &in_tile_order()));
}

/// The routed floors, against a stack laid either way they take: a col float stack, its buffer
/// `[slabs · n, k]` behind the `[k, slabs · n]` view, and a stack packed in each format.
#[test]
fn the_routed_floors_read_each_route_s_slab() {
    FlooredRouted.check(
        [
            Stack::Float,
            Stack::Packed(PackedFormat::Q8),
            Stack::Packed(PackedFormat::Q4),
            Stack::Packed(PackedFormat::Nvfp4),
            Stack::Packed(PackedFormat::Mxfp4),
            Stack::IN_TILES[0],
            Stack::IN_TILES[1],
        ]
        .map(|stack| Routed {
            stack,
            ..Routed::MOE
        }),
    );
}

/// An NVFP4 stack whose `k` is 2.5 words of scales, which only a rhs read a scale block at a time
/// closes: a floor still serves it.
#[test]
fn the_routed_floors_read_an_nvfp4_stack_whose_k_is_no_whole_number_of_words() {
    FlooredRouted.check([Routed {
        k: 160,
        n: 96,
        ..Routed::MOE
    }
    .packed(PackedFormat::Nvfp4)]);
}

/// `routed`'s float stack as its `[k, slabs · n]` view, its buffer `[slabs · n, k]`.
fn col_stack(client: &Client, routed: &Routed) -> PlainTensor {
    let Routed { slabs, k, n, .. } = *routed;
    let values = values(slabs * k * n, 2);
    let col: Vec<f32> = (0..slabs * n)
        .flat_map(|row| (0..k).map(move |kk| (row, kk)))
        .map(|(row, kk)| values[((row / n) * k + kk) * n + row % n])
        .collect();
    PlainTensor::of_handle(TensorHandle::new(
        client.create(Bytes::from_elems(col)),
        [k, slabs * n],
        [1, k],
        f32::elem_type_native(),
    ))
}

/// The routed product through the first of its floors that serves it
/// ([`RoutedMatmulCandidates::floors`]), what a caller racing the routed candidates falls back on.
struct FlooredRouted;

impl ScopeUnderTest for FlooredRouted {
    type Case = Routed;

    /// A cube larger than the device holds is a skip, the CPU runtime's few units; so is a packed
    /// stack on a device with no matrix instruction, which no floor stages for.
    ///
    /// # Panics
    ///
    /// On any other decline: a product some floor prepares is served by one.
    fn launch(&self, client: &Client, routed: &Routed) -> Result<Vec<Written>, Decline> {
        let Routed {
            tokens,
            picks,
            slabs,
            k,
            n,
            stack,
        } = *routed;
        let stored = match routed.packed_matrix(client) {
            Some(matrix) => StoredTensor::Packed(matrix),
            None => StoredTensor::Plain(col_stack(client, routed)),
        };
        let tensors = RoutedMatmulTensors::new(
            client,
            routed.activation(client).0,
            stored,
            upload(client, &[tokens, picks], &routed.routes()),
            slabs,
            SlabDimension::Output,
        )
        .expect("a stack of whole slabs derives a routed launch");
        let hw = Hardware::from_client(client);
        let no_matrix_instruction = hw.properties.features.matmul.cmma.iter().next().is_none();
        match tensors.served_on_floor(&hw) {
            Ok(()) => {}
            Err(why @ Decline::CubePastDevice { .. }) => return Err(why),
            Err(why) if stack != Stack::Float && no_matrix_instruction => return Err(why),
            Err(why) => panic!("{routed:?} refused: {why:?}"),
        }
        let floors = RoutedMatmulCandidates::floors();
        let plane_shuffle = floors
            .plane_shuffle
            .iter()
            .map(|floor| floor.launch(&tensors));
        let cube_staged = floors
            .cube_staged
            .iter()
            .map(|floor| floor.launch(&tensors));
        let unit_registers = floors
            .unit_registers
            .iter()
            .map(|floor| floor.launch(&tensors));
        let out = plane_shuffle
            .chain(cube_staged)
            .chain(unit_registers)
            .find_map(Result::ok)
            .unwrap_or_else(|| panic!("{routed:?}: a floor prepares it and none launches it"));
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
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
