use cubecl::ir::{ElemType, FloatKind, StorageType};

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionDefinition {
    pub dims: AttentionDims,

    /// Whether a mask is supplied (shape is always [batch, seq_q, heads, seq_kv])
    pub masked: bool,

    pub global_dtypes: AttentionGlobalTypes,

    pub options: AttentionOptions,
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

impl AttentionDefinition {
    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        self.dims.shape(ident)
    }
}

#[derive(Clone, Debug)]
/// Description of an attention problem to solve, regardless of actual data
pub struct AttentionDims {
    /// Batch size
    pub batch: usize,
    /// Number of attention heads
    pub num_heads: usize,

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
    pub fn shape(&self, ident: AttentionIdent) -> [usize; 4] {
        match ident {
            AttentionIdent::Query => [self.batch, self.num_heads, self.seq_q, self.head_dim],
            AttentionIdent::Key => [self.batch, self.num_heads, self.seq_kv, self.head_dim],
            AttentionIdent::Value => [self.batch, self.num_heads, self.seq_kv, self.val_dim],
            AttentionIdent::Mask => [self.batch, self.num_heads, self.seq_q, self.seq_kv],
            AttentionIdent::Out => [self.batch, self.num_heads, self.seq_q, self.val_dim],
            AttentionIdent::Softmax => unreachable!("Not a materialized tensor"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AttentionGlobalTypes {
    pub query: StorageType,
    pub key: StorageType,
    pub value: StorageType,
    pub mask: StorageType,
    pub out: StorageType,
}

impl AttentionGlobalTypes {
    pub fn from_single_dtype(dtype: StorageType) -> AttentionGlobalTypes {
        Self {
            query: dtype,
            key: dtype,
            value: dtype,
            mask: StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U8)),
            out: dtype,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum AccumulatorPrecision {
    Strict(StorageType),
    // Let algorithm decide
    Loose,
}

impl AccumulatorPrecision {
    pub fn default_accumulator_type() -> StorageType {
        StorageType::Scalar(ElemType::Float(FloatKind::F32))
    }
}

impl Default for AccumulatorPrecision {
    fn default() -> Self {
        Self::Strict(Self::default_accumulator_type())
    }
}
