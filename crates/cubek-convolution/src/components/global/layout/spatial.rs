use cubecl::std::tensor::{
    layout::{
        Coordinates, Coords1d, Layout, LayoutExpand,
        as_dyn::{IntoDyn, IntoDynExpand},
    },
    r#virtual::VirtualTensor,
};
use cubecl::{
    prelude::*,
    std::tensor::launch::{BufferArg, ViewLayoutLaunchArg},
};
use enumset::{EnumSet, EnumSetType};

use crate::components::Dimensionality;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct NhwcCoords {
    pub batch: u32,
    pub spatial: Sequence<i32>,
    pub channel: u32,
}

#[cube]
impl IntoDyn for NhwcCoords {
    fn into_dyn(self) -> Sequence<i32> {
        let mut seq = Sequence::new();
        seq.push(self.batch as i32);
        for x in self.spatial {
            seq.push(x);
        }
        seq.push(self.channel as i32);
        seq
    }
}

type NhwcTuple = (u32, Sequence<i32>, u32);

#[cube]
impl NhwcCoords {
    pub fn new(batch: u32, spatial: Sequence<i32>, channel: u32) -> Self {
        NhwcCoords {
            batch,
            spatial,
            channel,
        }
    }

    fn into_tuple(self) -> NhwcTuple {
        (self.batch, self.spatial, self.channel)
    }

    fn from_tuple(tuple: NhwcTuple) -> Self {
        NhwcCoords::new(tuple.0, tuple.1, tuple.2)
    }
}

#[cube]
impl Coordinates for NhwcCoords {
    fn add(this: Self, other: Self) -> Self {
        let tuple = NhwcTuple::add(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn sub(this: Self, other: Self) -> Self {
        let tuple = NhwcTuple::sub(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn min(this: Self, other: Self) -> Self {
        let tuple = <NhwcTuple as Coordinates>::min(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn max(this: Self, other: Self) -> Self {
        let tuple = <NhwcTuple as Coordinates>::max(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn is_in_bounds(pos: &Self, bounds: &Self) -> bool {
        NhwcTuple::is_in_bounds(&pos.clone().into_tuple(), &bounds.clone().into_tuple())
    }

    fn from_int(this: &Self, #[comptime] value: i64) -> Self {
        let tuple = NhwcTuple::from_int(&this.clone().into_tuple(), value);
        NhwcCoords::from_tuple(tuple)
    }
}

#[derive(EnumSetType, Debug, Hash)]
pub enum NhwcCheck {
    Batch,
    Spatial,
    Channel,
}

/// Layout for a spatial (i.e. NHWC) tensor. Bounds check only applies to spatial dimensions, not
/// channel or batch (because these are implicitly checked in the layouts used with spatial tensors).
#[derive(CubeType, Clone)]
pub struct NhwcLayout {
    /// Stride for N
    pub stride_batch: usize,
    /// Strides for DHW
    pub strides_spatial: Sequence<usize>,
    /// Stride for C
    pub stride_channel: usize,

    /// Shape of N
    pub shape_batch: u32,
    /// Shape of DHW
    pub shapes_spatial: Sequence<u32>,
    /// Shape of C
    pub shape_channel: u32,

    #[cube(comptime)]
    pub vector_size: VectorSize,
    #[cube(comptime)]
    pub checks: EnumSet<NhwcCheck>,
}

#[cube]
impl NhwcLayout {
    pub fn new<E: Numeric, N: Size, IO: Clone>(
        tensor: VirtualTensor<E, N, IO>,
        #[comptime] dim: Dimensionality,
        #[comptime] checks: EnumSet<NhwcCheck>,
    ) -> Self {
        let spatial_dims = dim.num_dims();
        let mut strides_spatial = Sequence::new();
        let mut shapes_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
            shapes_spatial.push(tensor.shape(i + 1) as u32);
        }

        let stride_batch = tensor.stride(0);
        let stride_channel = tensor.stride(spatial_dims + 1);

        let shape_batch = tensor.shape(0) as u32;
        let shape_channel = tensor.shape(spatial_dims + 1) as u32;

        NhwcLayout {
            stride_batch,
            strides_spatial,
            stride_channel,
            shape_batch,
            shapes_spatial,
            shape_channel,
            vector_size: tensor.vector_size(),
            checks,
        }
    }
}

#[cube]
impl Layout for NhwcLayout {
    type Coordinates = NhwcCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let NhwcCoords {
            batch,
            spatial,
            channel,
        } = pos;

        let spatial_dims = self.shapes_spatial.len();
        let mut read_pos =
            batch as usize * self.stride_batch + channel as usize * self.stride_channel;

        #[unroll]
        for i in 0..spatial_dims {
            read_pos += spatial[i] as usize * self.strides_spatial[i];
        }

        read_pos / self.vector_size
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos.clone()), self.is_in_bounds(pos))
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut in_bounds = true.runtime();
        if self.checks.comptime().contains(NhwcCheck::Batch) {
            in_bounds &= pos.batch < self.shape_batch;
        }
        if self.checks.comptime().contains(NhwcCheck::Spatial) {
            let spatial_dims = self.shapes_spatial.len();

            #[unroll]
            for i in 0..spatial_dims {
                let pos = pos.spatial[i];
                in_bounds &= pos >= 0 && (pos as u32) < self.shapes_spatial[i];
            }
        }
        if self.checks.comptime().contains(NhwcCheck::Channel) {
            in_bounds &= pos.channel < self.shape_channel;
        }

        in_bounds
    }

    fn shape(&self) -> Self::Coordinates {
        NhwcCoords {
            batch: self.shape_batch,
            spatial: cast_seq(self.shapes_spatial.clone()),
            channel: self.shape_channel,
        }
    }
}

#[cube]
pub(crate) fn cast_seq<From: CubePrimitive, To: CubePrimitive>(
    seq: Sequence<From>,
) -> Sequence<To> {
    let num_elems = seq.len();
    let mut out_seq = Sequence::new();
    #[unroll]
    for i in 0..num_elems {
        let elem = To::cast_from(seq[i]);
        out_seq.push(elem);
    }
    out_seq
}

pub struct NhwcLayoutLaunch {
    checks: EnumSet<NhwcCheck>,
}

impl NhwcLayoutLaunch {
    pub fn checked(checks: EnumSet<NhwcCheck>) -> Self {
        Self { checks }
    }

    pub fn unchecked() -> Self {
        Self {
            checks: EnumSet::empty(),
        }
    }
}

#[derive_cube_comptime]
pub struct NhwcLayoutCompilationArg {
    pub spatial_rank: usize,
    pub checks: EnumSet<NhwcCheck>,
}

impl ViewLayoutLaunchArg for NhwcLayout {
    type RuntimeArg<R: Runtime> = NhwcLayoutLaunch;
    type CompilationArg = NhwcLayoutCompilationArg;

    fn register<R: Runtime, B: BufferArg>(
        arg: Self::RuntimeArg<R>,
        buffer: &B,
        _: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let shape = buffer.shape();
        let strides = buffer.strides();

        let rank = shape.len();
        let dim_c = rank - 1;

        let stride_batch = strides[0];
        let strides_spatial = strides[1..dim_c].iter().copied().collect();
        let stride_channel = strides[dim_c];

        let shape_batch = shape[0] as u32;
        let shapes_spatial = shape[1..dim_c].iter().map(|s| *s as u32).collect();
        let shape_channel = shape[dim_c] as u32;

        <usize as LaunchArg>::register(stride_batch, launcher);
        <Sequence<usize> as LaunchArg>::register(strides_spatial, launcher);
        <usize as LaunchArg>::register(stride_channel, launcher);
        <u32 as LaunchArg>::register(shape_batch, launcher);
        <Sequence<u32> as LaunchArg>::register(shapes_spatial, launcher);
        <u32 as LaunchArg>::register(shape_channel, launcher);

        NhwcLayoutCompilationArg {
            spatial_rank: buffer.shape().len() - 2,
            checks: arg.checks,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let strides_comp_arg = (0..arg.spatial_rank).map(|_| ()).collect();
        let shape_comp_arg = (0..arg.spatial_rank).map(|_| ()).collect();
        NhwcLayoutExpand {
            stride_batch: <usize as LaunchArg>::expand(&(), builder),
            strides_spatial: <Sequence<usize> as LaunchArg>::expand(&strides_comp_arg, builder),
            stride_channel: <usize as LaunchArg>::expand(&(), builder),
            shape_batch: <u32 as LaunchArg>::expand(&(), builder),
            shapes_spatial: <Sequence<u32> as LaunchArg>::expand(&shape_comp_arg, builder),
            shape_channel: <u32 as LaunchArg>::expand(&(), builder),
            vector_size: ty.vector_size(),
            checks: arg.checks,
        }
    }
}
