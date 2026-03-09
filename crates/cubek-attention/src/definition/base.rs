use cubecl::{
    Runtime,
    client::ComputeClient,
    frontend::CubePrimitive,
    ir::{AddressType, ElemType, FloatKind, StorageType, Type},
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
    pub fn from_single_float_dtype(
        float_dtype: Type,
        mask_dtype: StorageType,
    ) -> AttentionGlobalTypes {
        let float_dtype = float_dtype.storage_type();
        Self {
            query: float_dtype,
            key: float_dtype,
            value: float_dtype,
            mask: mask_dtype,
            out: float_dtype,
        }
    }

    pub fn mask_dtype<R: Runtime>(client: &ComputeClient<R>) -> StorageType {
        let props = client.properties();
        let u8_ty = u8::as_type_native_unchecked().storage_type();
        let u32_ty = u32::as_type_native_unchecked().storage_type();

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
