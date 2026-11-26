use cubecl::tile::Space;
use cubecl::{
    bytes::Bytes,
    client::Client,
    quant::scheme::{QuantScheme, QuantValue},
    zspace::Shape,
};
use cubecl::{frontend::Scalar, std::tensor::TensorHandle};

use crate::{QuantizedTileInput, TestInput, TileInput};

/// The quantized continuation of a [`TileInputBuilder`](crate::TileInputBuilder): the container is fixed
/// (packed-u32, values-unit declaration, the scheme's block grid for scales); a content
/// finalizer fills it and mints the values tile and its scales together: a quantized
/// tensor is one thing (data, scales, scheme).
pub struct QuantizedTileInputBuilder {
    client: Client,
    space: Space,
    scheme: QuantScheme,
}

impl QuantizedTileInputBuilder {
    /// `arange`, wrapped into the scheme's representable range: values run
    /// `lo, lo+1, …, hi, lo, …`, so every representable value (and its sign extension)
    /// appears. Scales are a distinct-per-block ramp (`0.05 · (block + 1)`) on the scheme's
    /// block grid.
    ///
    /// The walk is over whole numbers packed as two's complement, which a test reads as the
    /// magnitude each field stands for: under a minifloat the fields would be codes, so a test
    /// folding them with the scales would compare the kernel against the integer grid the format
    /// exists to avoid.
    ///
    /// # Assumptions
    ///
    /// - The scheme's value is an integer type, whose codes are its magnitudes.
    ///
    /// # Panics
    ///
    /// Where its Assumption is broken.
    pub fn arange(self) -> QuantizedTileInput {
        assert!(
            integer_valued(self.scheme.value),
            "QuantizedTileInputBuilder::arange: {:?} is a minifloat, whose fields are codes rather \
             than magnitudes; state the codes with `pack_q_values` and fold the format's grid into \
             the test's own reference",
            self.scheme.value
        );

        let rank = self.space.rank();
        let shape: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.dimension_at(i)))
            .collect();

        let (lo, hi) = self.scheme.value.range();
        let (lo, hi) = (lo as i32, hi as i32);
        let span = hi - lo + 1;
        let q: Vec<i32> = (0..shape.iter().product())
            .map(|i| lo + (i as i32 % span))
            .collect();
        let words = crate::stubs::quant::pack_q_values(&q, &self.scheme);
        let handle = self.client.create(Bytes::from_elems(words));

        let block = crate::stubs::quant::block_dims(&self.scheme, &shape);
        let grid = crate::quant_layout::scales_grid(&shape, &block);
        let scale_values: Vec<f32> = (0..grid.iter().product())
            .map(|g| 0.05 * (g + 1) as f32)
            .collect();
        let scales = TestInput::builder(self.client, Shape::from(grid))
            .custom(scale_values)
            .generate_without_host_data();

        QuantizedTileInput::new(
            TileInput::new(
                TensorHandle::new_contiguous(shape, handle, u32::elem_type_native()),
                self.space,
                0,
            ),
            scales,
            self.scheme,
            None,
        )
    }

    /// [`arange`](Self::arange) for a lookup scheme: the packed fields walk the table indices
    /// **descending** (`2^bits - 1, …, 1, 0, 2^bits - 1, …`), so every entry is read *and* the
    /// field is never the identity of its position: a decode that indexed the table by position
    /// instead of by the unpacked field would pass an ascending walk. The scales are the same
    /// distinct-per-block ramp, and `table` uploads beside them.
    ///
    /// # Assumptions
    ///
    /// - The scheme's mode is lookup, and `table` holds exactly `2^bits` entries.
    ///
    /// # Panics
    ///
    /// Where its Assumption is broken.
    pub fn lookup_arange(self, table: &[f32]) -> QuantizedTileInput {
        assert!(
            matches!(self.scheme.mode, cubecl::quant::scheme::QuantMode::Lookup),
            "lookup_arange: the scheme's mode is {:?}, not Lookup",
            self.scheme.mode
        );
        let span = 1usize << self.scheme.size_bits_value();
        assert_eq!(
            table.len(),
            span,
            "lookup_arange: a {}-bit scheme indexes {span} entries",
            self.scheme.size_bits_value()
        );
        let rank = self.space.rank();
        let shape: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.dimension_at(i)))
            .collect();

        let q: Vec<i32> = (0..shape.iter().product())
            .map(|i| (span - 1 - (i % span)) as i32)
            .collect();
        let words = crate::stubs::quant::pack_q_values(&q, &self.scheme);
        let handle = self.client.create(Bytes::from_elems(words));

        let block = crate::stubs::quant::block_dims(&self.scheme, &shape);
        let grid = crate::quant_layout::scales_grid(&shape, &block);
        let scale_values: Vec<f32> = (0..grid.iter().product())
            .map(|g| 0.05 * (g + 1) as f32)
            .collect();
        let scales = TestInput::builder(self.client.clone(), Shape::from(grid))
            .custom(scale_values)
            .generate_without_host_data();
        let table_handle = TestInput::builder(self.client, Shape::from(vec![span]))
            .custom(table.to_vec())
            .generate_without_host_data();

        QuantizedTileInput::new(
            TileInput::new(
                TensorHandle::new_contiguous(shape, handle, u32::elem_type_native()),
                self.space,
                0,
            ),
            scales,
            self.scheme,
            Some(table_handle),
        )
    }
}

impl QuantizedTileInputBuilder {
    /// The quantized builder of a tile over `space` on `client`, in `scheme`.
    pub(super) fn new(client: Client, space: Space, scheme: QuantScheme) -> Self {
        Self {
            client,
            space,
            scheme,
        }
    }
}

/// Whether the format's codes are the integers they stand for, which is what makes an integer
/// walk over the fields a walk over the values. Mirrors `rounds_to_integers` in the quant stub.
fn integer_valued(value: QuantValue) -> bool {
    match value {
        QuantValue::Q8F
        | QuantValue::Q8S
        | QuantValue::Q4F
        | QuantValue::Q4S
        | QuantValue::Q2F
        | QuantValue::Q2S => true,
        QuantValue::E5M2 | QuantValue::E4M3 | QuantValue::E2M1 => false,
    }
}
