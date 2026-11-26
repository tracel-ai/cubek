use cubecl::std::tensor::{
    TensorHandle,
    layout::tiled_view::{TileSpec, TiledViewLaunch, TiledViewLayout},
};
use cubecl::tile::{
    Dimension, Launcher, Space, TileTensorLaunch,
    launch::{Arg, Labelled},
};
use cubecl::{client::Client, prelude::TensorArg, zspace::Tiling};

use crate::TileInputBuilder;

/// A tile-shaped test input: the device buffer plus the logical [`Space`] it's
/// viewed in. The sub-tile sizes live in the buffer's trailing dims, so the view
/// reads them from there.
pub struct TileInput {
    pub(super) handle: TensorHandle,
    space: Space,
    /// The dimensions the tile spans, in buffer order.
    dimensions: Vec<Dimension>,
    levels: usize,
}

impl TileInput {
    /// Start building a tile over `space`. Stack tiling levels coarse→fine with
    /// [`split`](TileInputBuilder::split) (by count) or
    /// [`tile`](TileInputBuilder::tile) (by element edge): chain them for
    /// recursion, or [`untiled`](TileInputBuilder::untiled) for none, then a data
    /// finalizer ([`arange`](TileInputBuilder::arange) /
    /// [`zeros`](TileInputBuilder::zeros)).
    pub fn builder(client: &Client, space: Space) -> TileInputBuilder {
        TileInputBuilder {
            client: client.clone(),
            space,
            levels: None,
        }
    }

    /// Launch arg for this tile's view: the buffer seen in its logical space.
    /// Every logical axis is tiled (`num_tiled = space.rank()`), recursively for
    /// `levels` nested tile levels.
    pub fn view(&self) -> TiledViewLaunch {
        TiledViewLaunch::new_tensor::<TiledViewLayout>(
            self.handle.clone().binding().into_tensor_arg(),
            TileSpec {
                start_axis: 0,
                num_tiled: self.space.rank(),
                levels: self.levels,
            },
        )
    }

    /// Launch arg for this tile's view vectorized with line size `vector_size`
    /// along the innermost (contiguous) axis. Same buffer; only the metadata is
    /// reinterpreted in line units (innermost shape ÷ `S`, every other stride
    /// ÷ `S`), so a kernel reading `Vector<E, S>` lands on contiguous lines.
    /// `vector_size == 1` is exactly [`view`](Self::view).
    pub fn view_vectorized(&self, vector_size: usize) -> TiledViewLaunch {
        let shape = self.handle.shape();
        let strides = self.handle.strides();
        let inner = shape.len() - 1;
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s / vector_size } else { s })
            .collect();
        let new_strides: Vec<usize> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s } else { s / vector_size })
            .collect();
        let lined = TensorHandle::new(
            self.handle.handle.clone(),
            new_shape,
            new_strides,
            self.handle.dtype,
        );
        TiledViewLaunch::new_tensor::<TiledViewLayout>(
            lined.binding().into_tensor_arg(),
            TileSpec {
                start_axis: 0,
                num_tiled: self.space.rank(),
                levels: self.levels,
            },
        )
    }

    /// Launch arg for this tile's raw global buffer: a plain [`TensorArg`] over the
    /// `[grid…, tile…]` buffer, optionally re-lined by `vector_size` along the inner
    /// axis (so a kernel reading raw `Vector<E, S>` slices lands on contiguous lines).
    /// `vector_size == 1` is the plain buffer. For a `TileTensor` pass the plain buffer: the
    /// argument states the width it is bound at and the metadata stays scalar-unit.
    pub fn tensor_arg(&self, vector_size: usize) -> TensorArg {
        if vector_size <= 1 {
            return self.handle.clone().binding().into_tensor_arg();
        }
        let shape = self.handle.shape();
        let strides = self.handle.strides();
        let inner = shape.len() - 1;
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s / vector_size } else { s })
            .collect();
        let new_strides: Vec<usize> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s } else { s / vector_size })
            .collect();
        TensorHandle::new(
            self.handle.handle.clone(),
            new_shape,
            new_strides,
            self.handle.dtype,
        )
        .binding()
        .into_tensor_arg()
    }

    /// The tile as one operand of `launcher`, served at the element it holds, bound `width` wide.
    /// The launch derives its spec: the dimensions its tiles reach past are masked, and a stored
    /// tiling is read as the buffer lays it out.
    ///
    /// # Panics
    ///
    /// Where the launch refuses the binding.
    pub fn arg(&self, launcher: &Launcher, width: usize) -> TileTensorLaunch {
        self.bind(launcher)
            .vectorize(width)
            .build()
            .unwrap_or_else(|decline| panic!("TileInput::arg: {decline}"))
            .arg()
    }

    /// [`arg`](Self::arg) before it is built: the launch's binding of this tile, labelled with
    /// its dimensions, for a caller that states more of it (a delivery, a boundary).
    pub fn bind<'a>(&'a self, launcher: &'a Launcher) -> Arg<'a, Labelled> {
        let mut binding = self.handle.clone().binding();
        binding.tiling = self.tiling();
        launcher
            .arg(binding, self.handle.dtype)
            .dimensions(&self.dimensions)
    }

    /// How the buffer stores each dimension: once untiled, else once per level and once for the
    /// grid.
    fn tiling(&self) -> Tiling {
        match self.levels {
            0 => Tiling::UNTILED,
            levels => Tiling::new(&vec![levels + 1; self.space.rank()])
                .expect("TileInput: a tiling the buffer's dims hold"),
        }
    }

    /// The semantic space the tile lives in.
    pub fn space(&self) -> Space {
        self.space.clone()
    }

    /// The device handle, for reading an output back.
    pub fn handle(&self) -> TensorHandle {
        self.handle.clone()
    }
}

impl TileInput {
    /// The tile `handle` holds over `space`, stored `levels` levels deep.
    pub(super) fn new(handle: TensorHandle, space: Space, levels: usize) -> Self {
        let dimensions = (0..space.rank()).map(|i| space.dimension_at(i)).collect();
        TileInput {
            handle,
            space,
            dimensions,
            levels,
        }
    }
}
