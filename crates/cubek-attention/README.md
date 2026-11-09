# CubeCL Attention Kernels Engine

## What an attention is given

Every attention goes through one entry, `AttentionTensors::attention(client, queries, kv, scoring)`.
`AttentionTensors::new`, `paged` and `packed` are that call on one kind of source.

**Queries** (`AttentionQueries`), one of:

- `Dense`: every user's `q` queries, `[b, h, q, d]`.
- `Packed`: the users' queries one after another, `[Σ q_b, h, d]`, user `b`'s queries
  `ends[b - 1] .. ends[b]` along the first dim (from `0` for the first), of the `u32` cumulative
  counts `ends` `[b]`, and `longest`, the most queries any user has, which the launch's grid
  covers.

A decode step is either kind with one query per user.

**Lengths**, `u32` `[b]` on the device (or `[1]` for every user), travel with the queries. User `b`
attends its first `lengths[b]` keys, and its `q_b` queries are the last `q_b` of them: they sit at
positions `lengths[b] - q_b .. lengths[b]`, causal from there. A caller holding the cached length
`c_b` passes `c_b + q_b`.

**Keys and values** (`KvSource`), three choices that combine:

| Choice | Field | Options |
|---|---|---|
| What is stored | `cache` | float keys and values, or a packed cache (codes, norms, codebooks) |
| Where a position lies | `pages` | each user's own positions `[b, kv, s, ·]`, or pools of pages `[pages, kv, page_size, ·]` named by a `u32` table `[b, pages_per_sequence]` |
| Which positions a user reads | `indices` | every attended one, or the `u32` list `[b, n]` |

A source only says how a tile of positions `[p, p + T)` of user `b` and key head `h` is loaded. The
attention's answer doesn't depend on where pages break or which list is read, so neither is an
axis of the attention's space.

**Scoring** (`AttentionScoring`): which keys a query sees, and the scale its scores take.

## What the routines read

| Queries | Cache | Pages | Indices | |
|---|---|---|---|---|
| `Dense` | float | either | none | read |
| `Dense` | packed | none | none | read |
| `Packed` | float | either | none | read by the cube routines; the unit and hypercube routines decline (`PACKED_QUERIES_UNREAD`) |
| any | any | any | a list | declined: `INDEXED_KEYS_UNREAD` |
| any | packed | pages | any | declined: `PAGED_PACKED_UNREAD` |

Packed queries are one launch for every user. Each user's run of the queries and of the output is
placed by its entry of `ends` (the tile DSL's `Indexing::Run` along the queries, keyed by the batch)
at the user's own cube level, and the cube level below cuts that run in boxes. The grid covers
`users × longest`: a cube past a user's run attends no key.
