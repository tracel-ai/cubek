# CubeCL Attention Kernels Engine

## What an attention is given

Every attention goes through one entry, `AttentionTensors::attention(client, queries, kv, scoring)`.
`AttentionTensors::new`, `paged` and `packed` are that call on one kind of source.

**Queries** (`AttentionQueries`), one of:

- `Dense`: every sequence's `q` queries, `[b, h, q, d]`.
- `Packed`: the sequences' queries one after another, `[Σ q_b, h, d]`, sequence `b`'s queries
  `ends[b - 1] .. ends[b]` along the first dim (from `0` for the first), of the `u32` cumulative
  counts `ends` `[b]`, and `longest`, the most queries any sequence has, which the launch's grid
  covers.

A decode step is either kind with one query per sequence.

**Lengths**, `u32` `[b]` on the device (or `[1]` for every sequence), travel with the queries.
Sequence `b` attends its first `lengths[b]` keys, and its `q_b` queries are the last `q_b` of them:
they sit at positions `lengths[b] - q_b .. lengths[b]`, causal from there. A caller holding the
cached length `c_b` passes `c_b + q_b`.

**Keys and values** (`KvSource`), three choices that combine:

| Choice | Field | Options |
|---|---|---|
| What is stored | `cache` | float keys and values, or a packed cache (codes, norms, codebooks) |
| Where a position lies | `pages` | each sequence's own positions `[b, kv, s, ·]`, or pools of pages `[pages, kv, page_size, ·]` named by a `u32` table `[b, pages_per_sequence]` |
| Which positions a sequence reads | `blocks` | every attended one, or blocks of `size` consecutive positions, a `u32` list `[b, n]` of block numbers: sequence `b`'s keys are its listed blocks laid end to end, and its length counts them |

A source only says how a tile of positions `[p, p + T)` of sequence `b` and key head `h` is
loaded. The attention's answer doesn't depend on where pages break or which list is read, so neither
is an axis of the attention's space.

**Scoring** (`AttentionScoring`): which keys a query sees, and the scale its scores take.

## What the routines read

| Queries | Cache | Pages | Blocks | |
|---|---|---|---|---|
| `Dense` | float | either | none | read |
| `Dense` | float | either | a list | read as pages of one block each, at `Attended` visibility (else `INDEXED_VISIBILITY_UNREAD`), a block within one page (else `KEY_BLOCK_SPLITS_A_PAGE`) |
| `Dense` | packed | none | none | read |
| `Packed` | float | either | none | read by the cube routines; the unit and hypercube routines decline (`PACKED_QUERIES_UNREAD`) |
| any | packed | any | a list | declined: `INDEXED_KEYS_UNREAD` |
| any | packed | pages | none | declined: `PAGED_PACKED_UNREAD` |

Packed queries are one launch for every sequence. Each sequence's run of the queries and of the
output is placed by its entry of `ends` (the tile DSL's `Indexing::Run` along the queries, keyed by
the batch) at the sequence's own cube level, and the cube level below cuts that run in boxes. The
grid covers `sequences × longest`: a cube past a sequence's run attends no key.

A listed source is read as an ordinary paged one (`KvSource::read_as_block_pages`): each buffer is
viewed as a pool of pages of one block, the key head a stride of the view, and a small kernel
composes the list with the cache's page table (a cache laid out as sequences is each sequence's one
page). The routines then read the blocks through the page table, each block's positions contiguous.
