use cubecl::std::tensor::TensorHandle;
use cubecl::tile::{Dimension, TileSpec as CubekTileSpec, TileTensorLaunch, kind::Field};
use cubecl::{prelude::TensorBinding, quant::scheme::QuantScheme};

use crate::TileInput;

/// A packed-quantized input born whole: a quantized tensor is one thing (data, scales,
/// scheme), so the [`quantized builder`](crate::QuantizedTileInputBuilder) mints the pair together.
pub struct QuantizedTileInput {
    tile: TileInput,
    scales: TensorHandle,
    scheme: QuantScheme,
    /// An index format's table, present exactly when the scheme's value is one
    /// ([`lookup_arange`](QuantizedTileInputBuilder::lookup_arange) mints it).
    table: Option<TensorHandle>,
}

impl QuantizedTileInput {
    /// Binding for the innermost level's scales tensor.
    pub fn scales_binding(&self) -> TensorBinding {
        self.scales.clone().binding()
    }

    /// The field each stored value occupies.
    pub fn field(&self) -> Field {
        Field::from(self.scheme.value)
    }

    /// The packed values as an operand, bound `width` words wide: `u32` words of
    /// [`field`](Self::field)s, addressed as `spec` says (declared in values), under none of their
    /// scales.
    pub fn values_arg(&self, width: usize, spec: CubekTileSpec) -> TileTensorLaunch {
        TileTensorLaunch::new(
            self.tile.tensor_arg(1),
            self.tile.handle().dtype,
            width,
            spec.packed(self.field()),
        )
    }

    /// The innermost level's scales as an operand, addressed as `spec` says: one per block of
    /// the scheme's grid, row-major.
    pub fn scales_arg(&self, width: usize, spec: CubekTileSpec) -> TileTensorLaunch {
        TileTensorLaunch::new(
            self.scales_binding().into_tensor_arg(),
            self.scales.dtype,
            width,
            spec,
        )
    }

    /// A lookup scheme's table as an operand over its one dimension, the positions.
    ///
    /// # Assumptions
    ///
    /// - The scheme is a lookup one, which indexes a table.
    ///
    /// # Panics
    ///
    /// Where its Assumption is broken.
    pub fn table_arg(&self, width: usize, dimension: Dimension) -> TileTensorLaunch {
        let table = self
            .table
            .as_ref()
            .expect("QuantizedTileInput::table_arg: this scheme indexes no table");
        TileTensorLaunch::new(
            table.clone().binding().into_tensor_arg(),
            table.dtype,
            width,
            CubekTileSpec::direct(&[dimension]),
        )
    }
}

impl QuantizedTileInput {
    /// The input whose packed values `tile` holds, scaled by `scales` and decoded under `scheme`
    /// through `table` where it is a lookup one.
    pub(super) fn new(
        tile: TileInput,
        scales: TensorHandle,
        scheme: QuantScheme,
        table: Option<TensorHandle>,
    ) -> Self {
        Self {
            tile,
            scales,
            scheme,
            table,
        }
    }
}
