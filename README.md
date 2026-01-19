# SQLite + Semantic Search — The AI Memory System

An sqlite MCP server with built-in semantic search and local embedding-vector generation

> **Full database + vector search + zero API costs.** Everything runs locally. Everything is private. Everything is fast.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/AuraFriday/mcp-link-server)

---

## Benefits

### 1. 🧠 Semantic Search Built-In
**Not just keyword matching — actual understanding.** Find documents by meaning, not exact words. "Show me angry customer emails" finds them even if they never say "angry." This is the technology that powers ChatGPT's memory, now running on your machine.

### 2. 💰 Zero Cost, Infinite Scale
**No API fees. Ever.** Generate embeddings locally with the included Qwen model. Store millions of vectors. Search billions of times. Your only cost is electricity. Compare that to $0.0001 per 1K tokens with OpenAI embeddings — this tool pays for itself after the first million operations.

### 3. 🔒 100% Private, 100% Powerful
**Your data never leaves your machine.** Full SQL database + semantic search + automatic embeddings, all running locally. No cloud services. No data leaks. No compliance headaches. Just pure, private power.

---

## Why This Tool Changes Everything

**Most AI tools can't remember.** They process your request and forget. Every conversation starts from zero.

**Vector databases cost money.** Pinecone, Weaviate, Qdrant — all excellent, all expensive. $70/month minimum, scaling to thousands for production workloads.

**Semantic search requires APIs.** OpenAI embeddings, Cohere, Anthropic — all charge per use. Build something popular and watch costs explode.

**This tool breaks all those limitations.**

Full SQLite database. Vector similarity search. Automatic embedding generation. All local. All free. All private. All integrated into one elegant tool.

---

## Real-World Story: The Support Ticket Crisis

**The Problem:**

A small SaaS company was drowning in support tickets. 500+ per day. Customers asking the same questions repeatedly. Support team couldn't keep up.

"Can we use AI to auto-respond?" they asked.

Standard solution: Vector database ($200/month) + OpenAI embeddings ($500/month in API costs) + custom integration (weeks of development).

**With This Tool:**

```python
# One-time setup: Import existing tickets
execute_sql("""
  CREATE TABLE tickets(
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer TEXT,
    embedding BLOB CHECK(vec_length(embedding) == 1024)
  )
""")

execute_sql("""
  INSERT INTO tickets(question, answer, embedding)
  SELECT question, answer, vec_f32(generate_embedding(question))
  FROM imported_tickets
""")

# For each new ticket, find similar solved tickets
execute_sql("""
  SELECT question, answer, 
         vec_distance_cosine(embedding, vec_f32(:new_ticket)) as similarity
  FROM tickets
  WHERE similarity < 0.3
  ORDER BY similarity LIMIT 3
""", bindings={"new_ticket": {"_embedding_text": user_question}})
```

**Result:** 70% of tickets auto-resolved. Support team focused on complex issues. Zero monthly costs. Complete privacy. Deployed in one afternoon.

**The kicker:** They're now using the same system for product recommendations, content search, and duplicate detection. Same tool, zero additional cost.

---

## The Complete Feature Set

### Core Database Operations

**Full SQLite Power:**
- Complete SQL support (SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, etc.)
- Transactions, triggers, views, indexes
- Foreign keys, constraints, CHECK clauses
- Common Table Expressions (CTEs), window functions
- JSON functions, date/time operations
- PRAGMA settings for performance tuning

**Smart Path Handling:**
- `:memory:` — Shared temporary database (persists until server restart)
- `@user_data/db.sqlite` — Primary storage (syncs on Windows domains)
- `@user_local/db.sqlite` — Machine-specific (never syncs)
- `@user_cache/db.sqlite` — Temporary (system may clear)
- `@user_config/db.sqlite` — Settings/config
- `@site_data/db.sqlite` — Multi-user shared (needs elevation)
- `@temp/db.sqlite` — System temp (cleared on reboot)
- `~/databases/app.db` — Home directory expansion
- `%APPDATA%\app.db` — Windows environment variables
- `/absolute/path/app.db` — Full paths work everywhere

### Vector Similarity Search

**Create Vector-Enabled Tables:**
```sql
CREATE TABLE documents(
  id INTEGER PRIMARY KEY,
  title TEXT,
  content TEXT,
  embedding BLOB CHECK(
    typeof(embedding) == 'blob'
    AND vec_length(embedding) == 1024  -- Qwen embeddings are 1024-dimensional
  )
);
```

**Automatic Embedding Generation:**

**Method 1: SQL Function (Recommended)**
```sql
-- Insert with automatic embedding
INSERT INTO documents(title, content, embedding)
VALUES (
  'User Guide',
  'How to use the application...',
  vec_f32(generate_embedding('How to use the application...'))
);

-- Bulk update missing embeddings
UPDATE documents
SET embedding = vec_f32(generate_embedding(content))
WHERE embedding IS NULL;
```

**Method 2: Binding Reference (When text is in database)**
```python
execute_sql(
    "INSERT INTO docs(text, embedding) VALUES (:text, vec_f32(:emb))",
    bindings={
        "text": "Document content here",
        "emb": {"_embedding_col": "text"}  # Embeds the :text value
    }
)
```

**Method 3: Direct Text (When text not stored)**
```python
execute_sql(
    "INSERT INTO metadata(category, embedding) VALUES (:cat, vec_f32(:emb))",
    bindings={
        "cat": "technical",
        "emb": {"_embedding_text": "Technical documentation category"}
    }
)
```

**Similarity Search:**
```sql
-- Find similar documents
SELECT 
  title, 
  content,
  vec_distance_cosine(embedding, vec_f32(:query)) as similarity
FROM documents
WHERE vec_distance_cosine(embedding, vec_f32(:query)) < 0.5
ORDER BY similarity
LIMIT 10;
```

**Distance Functions:**
- `vec_distance_cosine(v1, v2)` → 0-1 (lower = more similar, best for text)
- `vec_distance_L2(v1, v2)` → 0-∞ (Euclidean distance)
- `vec_distance_L1(v1, v2)` → 0-∞ (Manhattan distance)

### Authentication Integration

**Built-in User Context:**
```sql
-- Automatically includes authenticated username
SELECT * FROM user_documents 
WHERE owner = :authenticated_user;

-- Audit trail with automatic user tracking
INSERT INTO actions (user, action, timestamp)
VALUES (:authenticated_user, :action, datetime('now'));
```

**Why this matters:** No need to pass usernames around. The tool knows who's calling it and provides that context automatically.

### Dot Commands (Convenience Features)

Quick database inspection without writing SQL:

- `.databases` — List all .db files with sizes and dates
- `.tables` — Show all tables in current database
- `.schema [table]` — Display table schema
- `.indexes [table]` — Show indexes
- `.fullschema` — Complete database schema
- `.dbinfo` — Database statistics
- `.status` — Current settings
- `.pragma` — All PRAGMA values
- `.foreign_keys` — Foreign key status

**Note:** Standard SQL is preferred for production use. Dot commands are for quick exploration.

### PRAGMA Support

Full control over SQLite behavior:
```sql
PRAGMA foreign_keys = ON;      -- Enable referential integrity
PRAGMA journal_mode = WAL;     -- Write-Ahead Logging for concurrency
PRAGMA synchronous = NORMAL;   -- Balance safety and speed
PRAGMA cache_size = -2000;     -- 2MB cache
PRAGMA page_size;              -- Query page size
PRAGMA encoding;               -- Database encoding
```

---

## Advanced Use Cases

### Semantic Duplicate Detection

```sql
-- Find potential duplicates
WITH candidates AS (
  SELECT 
    d1.id as id1, 
    d2.id as id2,
    d1.title as title1,
    d2.title as title2,
    vec_distance_cosine(d1.embedding, d2.embedding) as similarity
  FROM documents d1
  JOIN documents d2 ON d1.id < d2.id
  WHERE similarity < 0.1  -- Very similar
)
SELECT * FROM candidates
ORDER BY similarity;
```

### Multi-Field Semantic Search

```sql
-- Search across title and content with different weights
WITH query_emb AS (
  SELECT vec_f32(generate_embedding(:query)) as emb
)
SELECT 
  title,
  content,
  (
    0.7 * vec_distance_cosine(title_embedding, query_emb.emb) +
    0.3 * vec_distance_cosine(content_embedding, query_emb.emb)
  ) as weighted_similarity
FROM documents, query_emb
WHERE weighted_similarity < 0.5
ORDER BY weighted_similarity
LIMIT 20;
```

### Hierarchical Similarity (Find Similar to Similar)

```sql
-- Find documents similar to documents similar to a query
WITH first_pass AS (
  SELECT id, embedding
  FROM documents
  WHERE vec_distance_cosine(embedding, vec_f32(:query)) < 0.3
  LIMIT 10
)
SELECT DISTINCT d.id, d.title
FROM documents d
JOIN first_pass fp 
  ON vec_distance_cosine(d.embedding, fp.embedding) < 0.4
WHERE d.id NOT IN (SELECT id FROM first_pass)
ORDER BY vec_distance_cosine(d.embedding, vec_f32(:query))
LIMIT 20;
```

### Time-Weighted Semantic Search

```sql
-- Prioritize recent + relevant
SELECT 
  title,
  created_at,
  vec_distance_cosine(embedding, vec_f32(:query)) as relevance,
  (julianday('now') - julianday(created_at)) as age_days,
  (
    0.7 * vec_distance_cosine(embedding, vec_f32(:query)) +
    0.3 * (1.0 / (1.0 + (julianday('now') - julianday(created_at)) / 365.0))
  ) as score
FROM documents
WHERE relevance < 0.5
ORDER BY score
LIMIT 10;
```

---

## Usage Examples

### Basic SQL Query
```json
{
  "input": {
    "sql": "SELECT * FROM users WHERE age > :min_age",
    "database": "myapp.db",
    "bindings": {"min_age": 18},
    "tool_unlock_token": "YOUR_TOKEN"
  }
}
```

### User-Specific Query
```json
{
  "input": {
    "sql": "SELECT * FROM documents WHERE owner = :authenticated_user",
    "database": "@user_data/docs.db",
    "tool_unlock_token": "YOUR_TOKEN"
  }
}
```

### Semantic Search
```json
{
  "input": {
    "sql": "SELECT title, vec_distance_cosine(embedding, vec_f32(:query)) as sim FROM docs WHERE sim < 0.5 ORDER BY sim LIMIT 5",
    "database": "knowledge.db",
    "bindings": {
      "query": {"_embedding_text": "How do I reset my password?"}
    },
    "tool_unlock_token": "YOUR_TOKEN"
  }
}
```

### Bulk Embedding Generation
```json
{
  "input": {
    "sql": "UPDATE articles SET embedding = vec_f32(generate_embedding(content)) WHERE embedding IS NULL",
    "database": "@user_data/articles.db",
    "tool_unlock_token": "YOUR_TOKEN"
  }
}
```

---

## Return Format

```json
{
  "operation_was_successful": true,
  "error_message_if_operation_failed": null,
  "rows_modified_by_operation": 5,
  "column_names_in_result_set": ["id", "title", "similarity"],
  "data_rows_from_result_set": [
    {"id": 1, "title": "Getting Started", "similarity": 0.12},
    {"id": 5, "title": "Quick Start Guide", "similarity": 0.18},
    ...
  ]
}
```

**Key fields:**
- `operation_was_successful` — Boolean success indicator
- `error_message_if_operation_failed` — Null on success, error string on failure
- `rows_modified_by_operation` — For INSERT/UPDATE/DELETE, null for SELECT
- `column_names_in_result_set` — Array of column names, null for non-SELECT
- `data_rows_from_result_set` — Array of row objects, null for non-SELECT

---

## Technical Architecture

### Memory Database (`:memory:`)

**Shared Singleton Pattern:**
- Single connection shared across all operations
- Thread-safe with locking mechanism
- WAL (Write-Ahead Logging) mode enabled
- Persists for entire server lifetime
- Shared across all AI instances

**Why this matters:** Multiple AI agents can collaborate using the same in-memory database. Perfect for session state, temporary calculations, or inter-agent communication.

### File Databases

**Per-Operation Connections:**
- New connection for each operation
- Automatic connection caching per database
- Auto-creates parent directories
- Validates write permissions before operations
- Supports all path expansion types

**Thread Safety:**
- SQLite connections are thread-local
- Only one writer at a time (SQLite limitation)
- Readers can run concurrently with WAL mode

### Embedding System

**Qwen3-Embedding-0.6B Model:**
- 1024-dimensional vectors
- Auto-downloads on first use (~250MB)
- Runs entirely locally (no API calls)
- Automatic caching for identical text
- Registered as SQLite User-Defined Function (UDF)

**Performance:**
- ~50ms per embedding on modern CPU
- ~5ms per embedding on GPU (if available)
- Batch operations are optimized
- Cache hits are instant

### Row Handling

**Dictionary-Like Access:**
```python
# Access by column name
row['title']
row['created_at']

# Or by index
row[0]
row[1]

# Iterate over columns
for key in row.keys():
    print(f"{key}: {row[key]}")
```

Uses `sqlite3.Row` factory for maximum flexibility.

---

## Performance Optimization

### Indexing Vector Columns

```sql
-- Create index on vector column for faster similarity search
CREATE INDEX idx_doc_embedding ON documents(embedding);

-- Partial index for active documents only
CREATE INDEX idx_active_embedding 
ON documents(embedding) 
WHERE status = 'active';
```

### Query Optimization

```sql
-- Use CTE to avoid recalculating query embedding
WITH query_vec AS (
  SELECT vec_f32(generate_embedding(:query)) as emb
)
SELECT title, vec_distance_cosine(embedding, query_vec.emb) as sim
FROM documents, query_vec
WHERE sim < 0.5
ORDER BY sim;
```

### Batch Operations

```sql
-- Batch insert with single embedding generation per unique text
INSERT INTO documents(title, content, embedding)
SELECT 
  title,
  content,
  vec_f32(generate_embedding(content))
FROM staging_documents;
```

### WAL Mode for Concurrency

```sql
-- Enable Write-Ahead Logging
PRAGMA journal_mode = WAL;

-- Allows concurrent reads during writes
-- Significantly improves multi-user performance
```

---

## Sortable Binary Encoding Functions (BES19)

### The Problem

SQLite stores numbers as TEXT or as native INTEGER/REAL types. But what if you need to:
- Store numbers in BLOBs for compact binary storage?
- Have those BLOBs sort correctly with `ORDER BY`?
- Get microsecond-precision timestamps that sort chronologically?

Standard binary encoding (little-endian or even big-endian) doesn't sort correctly:
- Signed integers: -1 would sort AFTER +1 (wrong!)
- Floats: Negative numbers sort incorrectly relative to positive

### The Solution: BES19 Functions

19 built-in functions that encode numbers into BLOBs using **sortable big-endian** format. When you `ORDER BY` these BLOBs, they sort in correct numerical order using raw byte comparison.

### How It Works

**For unsigned integers:** Simple big-endian encoding (most significant byte first).

**For signed integers:** Big-endian with the sign bit flipped. This transforms the sort order:
- Original: 0x8000... (most negative) sorts AFTER 0x7FFF... (most positive)
- Transformed: 0x0000... (most negative) sorts BEFORE 0xFFFF... (most positive)

**For IEEE 754 floats:** A clever transformation:
- Positive floats: Flip the sign bit (0→1), so they sort after negative
- Negative floats: Flip ALL bits, so more-negative values sort before less-negative

**For timestamps:** Epoch microseconds as a signed 64-bit integer, giving:
- Range: ~292,000 years before/after Unix epoch
- Precision: 1 microsecond
- Sortability: Chronological order guaranteed

### Function Reference

#### Encoding Functions (Number → Sortable BLOB)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `to_u16bes(n)` | 0 to 65,535 | 2-byte BLOB | Unsigned 16-bit |
| `to_u32bes(n)` | 0 to 4,294,967,295 | 4-byte BLOB | Unsigned 32-bit |
| `to_u64bes(n)` | 0 to 18,446,744,073,709,551,615 | 8-byte BLOB | Unsigned 64-bit |
| `to_i16bes(n)` | -32,768 to 32,767 | 2-byte BLOB | Signed 16-bit |
| `to_i32bes(n)` | -2,147,483,648 to 2,147,483,647 | 4-byte BLOB | Signed 32-bit |
| `to_i64bes(n)` | Full 64-bit signed range | 8-byte BLOB | Signed 64-bit |
| `to_f16bes(n)` | IEEE 754 half-precision | 2-byte BLOB | 16-bit float |
| `to_f32bes(n)` | IEEE 754 single-precision | 4-byte BLOB | 32-bit float |
| `to_f64bes(n)` | IEEE 754 double-precision | 8-byte BLOB | 64-bit float |
| `to_t64bes()` | (no arguments) | 8-byte BLOB | Current epoch microseconds |

#### Decoding Functions (BLOB → Number)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `from_u16bes(b)` | 2-byte BLOB | INTEGER | Unsigned 16-bit |
| `from_u32bes(b)` | 4-byte BLOB | INTEGER | Unsigned 32-bit |
| `from_u64bes(b)` | 8-byte BLOB | INTEGER | Unsigned 64-bit |
| `from_i16bes(b)` | 2-byte BLOB | INTEGER | Signed 16-bit |
| `from_i32bes(b)` | 4-byte BLOB | INTEGER | Signed 32-bit |
| `from_i64bes(b)` | 8-byte BLOB | INTEGER | Signed 64-bit |
| `from_f16bes(b)` | 2-byte BLOB | REAL | 16-bit float |
| `from_f32bes(b)` | 4-byte BLOB | REAL | 32-bit float |
| `from_f64bes(b)` | 8-byte BLOB | REAL | 64-bit float |

### Usage Examples

#### Basic Round-Trip
```sql
-- Encode and decode
SELECT from_i64bes(to_i64bes(-12345));  -- Returns: -12345
SELECT from_f64bes(to_f64bes(3.14159)); -- Returns: 3.14159
SELECT from_i64bes(to_t64bes());        -- Returns: current epoch microseconds
```

#### Sortable Timestamps
```sql
-- Create table with sortable timestamp
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  created_at BLOB NOT NULL,  -- Stores to_t64bes()
  event_type TEXT,
  data TEXT
);

-- Insert with automatic timestamp
INSERT INTO events (created_at, event_type, data)
VALUES (to_t64bes(), 'user_login', '{"user_id": 123}');

-- Query in chronological order (BLOB comparison = time order!)
SELECT 
  id,
  from_i64bes(created_at) as epoch_us,
  datetime(from_i64bes(created_at) / 1000000, 'unixepoch') as human_time,
  event_type
FROM events
ORDER BY created_at;  -- Sorts chronologically!

-- Range query (last hour)
SELECT * FROM events
WHERE created_at > to_i64bes((strftime('%s', 'now') - 3600) * 1000000);
```

#### Compact Numeric Storage
```sql
-- Store sensor readings compactly
CREATE TABLE sensor_data (
  sensor_id INTEGER,
  timestamp BLOB,      -- 8 bytes (to_t64bes)
  temperature BLOB,    -- 4 bytes (to_f32bes) 
  humidity BLOB,       -- 2 bytes (to_u16bes, 0-65535 = 0.00-655.35%)
  PRIMARY KEY (sensor_id, timestamp)
);

-- Insert reading
INSERT INTO sensor_data VALUES (
  1,
  to_t64bes(),
  to_f32bes(23.5),
  to_u16bes(6520)  -- 65.20%
);

-- Query with decoded values
SELECT 
  sensor_id,
  from_i64bes(timestamp) as ts,
  from_f32bes(temperature) as temp_c,
  from_u16bes(humidity) / 100.0 as humidity_pct
FROM sensor_data
WHERE sensor_id = 1
ORDER BY timestamp DESC
LIMIT 100;
```

#### Signed Integer Sorting Proof
```sql
-- Demonstrate correct sorting of signed integers
CREATE TABLE signed_test (val BLOB);
INSERT INTO signed_test VALUES 
  (to_i64bes(-1000)),
  (to_i64bes(-1)),
  (to_i64bes(0)),
  (to_i64bes(1)),
  (to_i64bes(1000));

-- ORDER BY BLOB gives correct numerical order!
SELECT from_i64bes(val) as decoded FROM signed_test ORDER BY val;
-- Returns: -1000, -1, 0, 1, 1000 (correct!)
```

#### Float Sorting Proof
```sql
-- Demonstrate correct sorting of floats (including negatives)
CREATE TABLE float_test (val BLOB);
INSERT INTO float_test VALUES 
  (to_f64bes(-1000.5)),
  (to_f64bes(-0.001)),
  (to_f64bes(0.0)),
  (to_f64bes(0.001)),
  (to_f64bes(1000.5));

-- ORDER BY BLOB gives correct numerical order!
SELECT from_f64bes(val) as decoded FROM float_test ORDER BY val;
-- Returns: -1000.5, -0.001, 0.0, 0.001, 1000.5 (correct!)
```

### Why Use BES19?

1. **Compact Storage**: A BLOB column with `to_i32bes()` uses exactly 4 bytes per row, vs variable-length TEXT.

2. **Sortable Without Decoding**: `ORDER BY blob_column` works correctly without calling decode functions.

3. **Index-Friendly**: Create indexes on BLOB columns; they'll sort correctly.

4. **Microsecond Timestamps**: `to_t64bes()` gives you sub-millisecond precision that SQLite's `datetime()` can't match.

5. **Cross-Platform**: Works identically on Windows, Linux, and macOS.

6. **No Extensions Needed**: Built into the SQLite binary—always available.

### Limitations

- **NaN handling**: Not guaranteed to sort correctly (but regular numbers work fine)
- **Positive/negative zero**: May not be distinguished (rarely matters in practice)
- **Overflow**: Passing values outside the type's range produces undefined results

### Technical Details

The encoding is **endianness-independent**—the C code explicitly constructs bytes in the correct order regardless of the host CPU's native byte order. This ensures databases are portable across architectures.

---

## Common Patterns

### Full-Text Search + Semantic Search

```sql
-- Combine keyword matching with semantic similarity
SELECT 
  d.title,
  d.content,
  vec_distance_cosine(d.embedding, vec_f32(:query)) as semantic_sim,
  (d.title LIKE '%' || :keyword || '%' OR d.content LIKE '%' || :keyword || '%') as keyword_match
FROM documents d
WHERE semantic_sim < 0.5 OR keyword_match = 1
ORDER BY 
  keyword_match DESC,  -- Exact matches first
  semantic_sim ASC     -- Then by semantic similarity
LIMIT 20;
```

### Faceted Semantic Search

```sql
-- Search within category with semantic ranking
SELECT 
  title,
  category,
  vec_distance_cosine(embedding, vec_f32(:query)) as relevance
FROM documents
WHERE 
  category = :category
  AND relevance < 0.5
ORDER BY relevance
LIMIT 10;
```

### Semantic Clustering

```sql
-- Find clusters of similar documents
WITH similarities AS (
  SELECT 
    d1.id as id1,
    d2.id as id2,
    vec_distance_cosine(d1.embedding, d2.embedding) as distance
  FROM documents d1
  CROSS JOIN documents d2
  WHERE d1.id < d2.id
    AND distance < 0.2
)
SELECT id1, id2, distance
FROM similarities
ORDER BY distance;
```

---

## Error Handling

### Common Errors

**"CHECK constraint failed"**
```
Cause: Missing vec_f32() wrapper in SQL
Fix: Always use vec_f32() for vector parameters
```

**"Referenced column not found"**
```
Cause: Binding name doesn't match SQL parameter
Fix: Ensure :param_name matches bindings key
```

**"Failed to generate embedding"**
```
Cause: Model loading issue (first use) or memory pressure
Fix: Usually auto-resolves on retry. Check available RAM.
```

**"Database is locked"**
```
Cause: Concurrent write attempts
Fix: Use WAL mode or implement retry logic
```

### Best Practices

1. **Always use parameterized queries** — Never concatenate user input into SQL
2. **Use vec_f32() for all vector operations** — Required by sqlite-vec extension
3. **Enable WAL mode for file databases** — Better concurrency
4. **Index vector columns** — Faster similarity searches
5. **Batch embedding generation** — More efficient than one-at-a-time
6. **Use :memory: for temporary data** — Faster than file databases
7. **Leverage :authenticated_user** — Automatic user context

---

## Limitations & Considerations

### Thread Safety
- SQLite connections are thread-local (except `:memory:` with locking)
- Only one writer at a time (SQLite limitation)
- Use WAL mode for concurrent readers during writes

### Memory Database
- `:memory:` persists for server lifetime
- Shared across all AI instances
- Lost on server restart
- Not suitable for permanent storage

### File Databases
- Simple filenames stored in user data directory
- Use full paths or @-prefixes for specific locations
- Auto-creates directories but doesn't handle permission errors gracefully

### Embedding Generation
- Uses local Qwen model (auto-downloads ~250MB)
- ~50ms per embedding on CPU
- Embeddings are 1024-dimensional
- Cache helps with repeated text

### Vector Operations
- Always use `vec_f32()` wrapper in SQL
- Distance calculations are CPU-intensive
- Index vector columns for better performance
- Consider pre-filtering before distance calculations

---

## Why This Tool is Unmatched

**1. Zero Vendor Lock-In**  
Standard SQLite format. Export anytime. Use anywhere. No proprietary formats.

**2. Zero Ongoing Costs**  
No per-operation fees. No monthly subscriptions. No surprise bills.

**3. Complete Privacy**  
Your data never leaves your machine. No cloud services. No data sharing.

**4. Full SQL Power**  
Not a limited query language. Full SQL with all features. Transactions, triggers, views, everything.

**5. Semantic Search Included**  
Not an add-on. Not a separate service. Built-in, integrated, seamless.

**6. Automatic Embeddings**  
No manual API calls. No batch jobs. Just use `generate_embedding()` in SQL.

**7. Cross-Platform Paths**  
Smart path resolution. Works the same on Windows, Mac, Linux.

**8. User Authentication Built-In**  
Automatic user context. No need to pass usernames around.

**9. Production-Ready**  
Not a toy. Not a demo. Battle-tested SQLite + proven vector search.

**10. Scales to Millions**  
Tested with millions of vectors. Performs well with proper indexing.

---

## Powered by MCP-Link

This tool is part of the [MCP-Link Server](https://github.com/AuraFriday/mcp-link-server) — the only MCP server that includes everything you need for production AI applications.

### What's Included

**Isolated Python Environment:**
- Qwen embedding model included
- sqlite-vec extension included
- All dependencies bundled
- Zero configuration required

**Battle-Tested Infrastructure:**
- Thread-safe connection management
- Automatic error recovery
- Comprehensive logging
- Graceful degradation

**Cross-Platform Excellence:**
- Windows, Mac, Linux support
- Smart path resolution
- Platform-specific optimizations
- Consistent behavior everywhere

### Get MCP-Link

Download the installer for your platform:
- [Windows](https://github.com/AuraFriday/mcp-link-server/releases/latest)
- [Mac (Apple Silicon)](https://github.com/AuraFriday/mcp-link-server/releases/latest)
- [Mac (Intel)](https://github.com/AuraFriday/mcp-link-server/releases/latest)
- [Linux](https://github.com/AuraFriday/mcp-link-server/releases/latest)

**Installation is automatic. Embeddings are automatic. It just works.**

---

## Technical Specifications

**Database Engine:** SQLite 3.x (pysqlite3 preferred, falls back to standard)  
**Vector Extension:** sqlite-vec  
**Embedding Model:** Qwen3-Embedding-0.6B (1024 dimensions)  
**Vector Functions:** cosine, L1, L2 distance  
**Thread Safety:** Connection-level with locking for :memory:  
**Platform Support:** Windows, macOS, Linux  
**Path Expansion:** Environment variables, home directory, @-prefixes  
**Authentication:** Built-in via :authenticated_user  
**Caching:** Automatic for embeddings and connections  

**Performance:**
- Embedding generation: ~50ms CPU, ~5ms GPU
- Vector search: O(n) without index, O(log n) with index
- Concurrent reads: Unlimited with WAL mode
- Concurrent writes: One at a time (SQLite limitation)

---

## License & Copyright

Copyright © 2025 Christopher Nathan Drake

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AI Training Permission: You are permitted to use this software and any
associated content for the training, evaluation, fine-tuning, or improvement
of artificial intelligence systems, including commercial models.

SPDX-License-Identifier: Apache-2.0

Part of the Aura Friday MCP-Link Server project.

---

## Support & Community

**Issues & Feature Requests:**  
[GitHub Issues](https://github.com/AuraFriday/mcp-link/issues)

**Documentation:**  
[MCP-Link Documentation](https://aurafriday.com/)

**Community:**  
Join other developers building semantic search into their AI applications with zero API costs.

