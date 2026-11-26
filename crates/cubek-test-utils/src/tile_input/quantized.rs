use cubecl::std::tensor::TensorHandle;
use cubecl::tile::{Dimension, TileSpec as CubekTileSpec, TileTensorLaunch, kind::Field};
use cubecl::{prelude::TensorBinding, quant::scheme::QuantScheme};

use crate::TileInput;

/// A packed-quantized input born whole: a quantized tensor is one thing (data, scales,
/// scheme), so the [`quantized builder`](crate::QuantizedTileInputBuilder) mints the pair together,
/// plus the exact numbers behind both for host references.
pub struct QuantizedTileInput {
    pub tile: TileInput,
    pub(super) scales: TensorHandle,
    pub(super) scheme: QuantScheme,
    /// A lookup scheme's table, present exactly under `QuantMode::Lookup`
    /// ([`lookup_arange`](QuantizedTileInputBuilder::lookup_arange) mints it).
    pub(super) table: Option<TensorHandle>,
    /// The quant values, row-major in the logical shape. Table indices under a lookup scheme.
    pub q: Vec<i32>,
    /// One scale per block, row-major over the scheme's block grid.
    pub scale_values: Vec<f32>,
    /// The lookup table's entries, empty for a non-lookup scheme.
    pub table_values: Vec<f32>,
}

impl QuantizedTileInput {
    /// Binding for the innermost level's scales tensor.
    pub fn scales_binding(&self) -> TensorBinding {
        self.scales.clone().binding()
    }

    /// The field each stored value occupies: the scheme's value, or under a lookup scheme an
    /// index as wide, which names a table entry rather than a number.
    pub fn field(&self) -> Field {
        match self.scheme.mode {
            cubecl::quant::scheme::QuantMode::Lookup => Field::Index {
                bits: self.scheme.size_bits_value(),
            },
            _ => Field::Quant(self.scheme.value),
        }
    }

    /// The packed values as an operand, bound `width` words wide: `u32` words of
    /// [`field`](Self::field)s, addressed as `spec` says (declared in values), under none of their
    /// scales.
    pub fn values_arg(&self, width: usize, spec: CubekTileSpec) -> TileTensorLaunch {
        TileTensorLaunch::new(
            self.tile.tensor_arg(1),
            self.tile.handle.dtype,
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
    /// # Panics
    ///
    /// Unless the scheme is a lookup one.
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
