use crate::forward::definition::AttentionSetupError;
use cubecl::{
    client::Client,
    ir::{AddressType, ElemType, FloatKind},
    prelude::Scalar,
};

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionProblem {
    pub dims: AttentionDims,

    /// Whether a mask is supplied (shape is always [batch, seq_q, heads, seq_kv])
    pub masked: bool,

    pub global_dtypes: AttentionGlobalTypes,

    pub options: AttentionOptions,

    /// Address type, defined by the max of all input handles' `required_address_type`
    pub address_type: AddressType,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum AttentionIdent {
    Query,
    Key,
    Softmax,
    Value,
    Mask,
    Out,
}

#[derive(Clone, Debug, Default)]
pub struct AttentionOptions {
    pub causal: bool,
    pub accumulator_precision: AccumulatorPrecision,
}

impl AttentionProblem {
    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        self.dims.shape(ident)
    }
}

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionDims {
    /// Batch size
    pub batch: usize,
    /// Number of query (attention) heads
    pub num_heads: usize,
    /// Number of key/value heads, for grouped-query attention.
    ///
    /// Equals `num_heads` for MHA. When smaller, `num_heads` query heads share
    /// each key/value head: the ratio `num_heads / num_kv_heads` is what the
    /// kernel indexes K and V with, so a KV tensor stays in its packed form
    /// instead of being expanded to `num_heads` before the launch. That packing
    /// is the point — it is the whole reason GQA exists — and the ratio is part
    /// of the trained model, not a kernel detail.
    pub num_kv_heads: usize,

    /// Query sequence length
    pub seq_q: usize,
    /// Key/Value sequence length
    pub seq_kv: usize,
    /// Dimension of each head (d)
    pub head_dim: usize,
    /// Dimension of each value vector.  
    /// Usually equal to `head_dim`, but may differ in some variants
    pub val_dim: usize,
}

impl AttentionDims {
    /// Number of query heads that share one key/value head, i.e. the GQA ratio.
    ///
    /// Integer division: [`validate`](Self::validate) rejects a problem whose
    /// heads do not divide evenly, so a truncated ratio cannot reach a launch.
    pub fn kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// The key/value head that query head `q_head` attends through.
    ///
    /// Contiguous and non-interleaved: query heads `[g * k, (g + 1) * k)` all
    /// read KV head `k` for a group of `k`. The mapping is a property of the
    /// model's head ordering, not of this kernel — a checkpoint that interleaves
    /// its KV heads needs the same remap applied to the weights, and is not
    /// served by this formula.
    pub fn kv_head(&self, q_head: usize) -> usize {
        q_head / self.kv_group()
    }

    /// Reject head counts the GQA mapping cannot express.
    ///
    /// Called by every routine's blueprint validation, next to the divisibility
    /// checks it already performs, so a bad head ratio is a setup error rather
    /// than a kernel that silently reads the wrong K/V rows.
    pub fn validate(&self) -> Result<(), AttentionSetupError> {
        if self.num_kv_heads == 0 || self.num_heads == 0 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
                "Head counts must be non-zero, got num_heads = {}, num_kv_heads = {}",
                self.num_heads, self.num_kv_heads
            ))));
        }

        if self.num_heads % self.num_kv_heads != 0 {
            return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
                "num_heads ({}) must be a multiple of num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            ))));
        }

        Ok(())
    }

    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        match ident {
            AttentionIdent::Query => [self.batch, self.num_heads, self.seq_q, self.head_dim],
            AttentionIdent::Key => [self.batch, self.num_kv_heads, self.seq_kv, self.head_dim],
            AttentionIdent::Value => [self.batch, self.num_kv_heads, self.seq_kv, self.val_dim],
            // The mask is indexed by query head — it is a predicate on a
            // `(query, key)` pair, not a property of the KV cache. A mask
            // tensor whose head extent is 1 broadcasts over all query heads.
            AttentionIdent::Mask => [self.batch, self.num_heads, self.seq_q, self.seq_kv],
            AttentionIdent::Out => [self.batch, self.num_heads, self.seq_q, self.val_dim],
            AttentionIdent::Softmax => unreachable!("Not a materialized tensor"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AttentionGlobalTypes {
    pub query: ElemType,
    pub key: ElemType,
    pub value: ElemType,
    pub mask: ElemType,
    pub out: ElemType,
}

impl AttentionGlobalTypes {
    pub fn from_single_float_dtype(
        float_dtype: ElemType,
        mask_dtype: ElemType,
    ) -> AttentionGlobalTypes {
        Self {
            query: float_dtype,
            key: float_dtype,
            value: float_dtype,
            mask: mask_dtype,
            out: float_dtype,
        }
    }

    pub fn mask_dtype(client: &Client) -> ElemType {
        let props = client.properties();
        let u8_ty = u8::elem_type_native();
        let u32_ty = u32::elem_type_native();

        if props.supports_type(u8_ty) {
            u8_ty
        } else if props.supports_type(u32_ty) {
            u32_ty
        } else {
            panic!("Client does not support u8 or u32 native types");
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum AccumulatorPrecision {
    Strict(ElemType),
    // Let algorithm decide
    Loose,
}

impl AccumulatorPrecision {
    pub fn default_accumulator_type() -> ElemType {
        ElemType::Float(FloatKind::F32)
    }
}

impl Default for AccumulatorPrecision {
    fn default() -> Self {
        Self::Strict(Self::default_accumulator_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(num_heads: usize, num_kv_heads: usize) -> AttentionDims {
        AttentionDims {
            batch: 1,
            num_heads,
            num_kv_heads,
            seq_q: 8,
            seq_kv: 8,
            head_dim: 64,
            val_dim: 64,
        }
    }

    #[test]
    fn query_heads_map_onto_kv_heads_in_contiguous_groups() {
        // LFM2: 32 query heads over 8 KV heads, four to a group.
        let dims = dims(32, 8);

        assert_eq!(dims.kv_group(), 4);
        for q_head in 0..32 {
            assert_eq!(dims.kv_head(q_head), q_head / 4);
        }
        for kv_head in 0..8 {
            assert_eq!(dims.kv_head(kv_head * 4), kv_head);
            assert_eq!(dims.kv_head(kv_head * 4 + 3), kv_head);
        }
    }

    #[test]
    fn mha_maps_every_query_head_onto_itself() {
        let dims = dims(12, 12);

        assert_eq!(dims.kv_group(), 1);
        assert!((0..12).all(|h| dims.kv_head(h) == h));
    }

    #[test]
    fn key_and_value_are_packed_while_the_mask_is_not() {
        let dims = dims(32, 8);

        assert_eq!(dims.shape(AttentionIdent::Key), [1, 8, 8, 64]);
        assert_eq!(dims.shape(AttentionIdent::Value), [1, 8, 8, 64]);
        // A mask is a predicate on a (query, key) pair: one per query head.
        assert_eq!(dims.shape(AttentionIdent::Query), [1, 32, 8, 64]);
        assert_eq!(dims.shape(AttentionIdent::Out), [1, 32, 8, 64]);
        assert_eq!(dims.shape(AttentionIdent::Mask), [1, 32, 8, 8]);
    }

    #[test]
    fn a_head_count_that_does_not_divide_is_rejected() {
        // 5 query heads over 2 KV heads leaves a group of 2.5; a truncated ratio
        // would silently point the last query head at KV head 2 of 2.
        assert!(dims(5, 2).validate().is_err());
        assert!(dims(32, 8).validate().is_ok());
        assert!(dims(32, 32).validate().is_ok());
    }

    #[test]
    fn a_zero_head_count_is_rejected_before_the_ratio_is_taken() {
        // `kv_group` divides, so a zero KV head count must not reach it.
        assert!(dims(32, 0).validate().is_err());
        assert!(dims(0, 0).validate().is_err());
    }
}
