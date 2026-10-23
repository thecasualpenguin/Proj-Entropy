# Approved upstream manifest and extraction-index contracts (v1.0)

These contracts are the foundation for the bounded WPP packer described in `bounded-wpp-dataset.md`. They do **not** change a model, run encoding/extraction, reuse legacy encoded media, or make `extract_packet_dataset.py` consume either contract. That producer remains partition-agnostic; the bounded packer alone applies already-declared source-manifest partitions after native verification.

## Source manifest: `approved_upstream_source_manifest` v1.0

A published manifest is a new directory containing `source_manifest.csv` and `source_manifest.metadata.json`. Publication is staged then atomically renamed; an existing destination is rejected. Both source-manifest and extraction-index publication use a per-destination lock directory, so concurrent contract writers fail closed rather than overwrite one another; stale locks are not stolen and require operator cleanup. This is a cooperative-writer guarantee, not protection against an unrelated process that ignores the lock. The CSV header is exactly:

```text
video_id,source_sha256,source_relative_path,source_bytes,label,group_id,partition,prompt_id,variant,legacy_video_id
```

Every row denotes one **original** source file. `video_id` is exactly `sha256:<full source_sha256>`; it is never an opaque legacy identifier. Paths are safe POSIX-relative paths under metadata `source_root_ref`; absolute paths, `..`, and backslashes are rejected. Rows are lexically ordered by that path. Source IDs, original digests, and paths are unique—identical original bytes are an explicit canonical-ID collision, not silently deduplicated. `group_id`, `partition`, `prompt_id`, `variant`, and `legacy_video_id` are optional; if a partition is supplied it is `train`, `validation`, or `test`.

Metadata names the contract/version, dataset and source-root references, exact CSV SHA-256/count, path/ordering rules, provenance, and the important future guidance: processing must retain native original geometry/timing. No encoding profile is declared or implied by this source contract.

The generic core accepts explicit record mappings (`write_source_manifest`) or a caller-supplied discovery mapping (`records_from_discovery`). It deliberately makes no claim to infer arbitrary-tree labels.

## DeepAction v1 adapter

DeepAction's only accepted original layout is:

```text
<source root>/<label>/<prompt_id>/<variant>.mp4
```

It requires the seven approved labels, canonical decimal prompt IDs, and the legacy audit's exact schema. The adapter derives a relative source path from `original_source_path` relative to the actual root (therefore retaining `<label>/...`), joins identity using authoritative `original_source_sha256`, assigns `group_id=dataset:prompt:<prompt_id>` globally across labels, and maps historical `val` to `validation`. It checks source/audit path, label, prompt, partition, non-empty regular-file, inventory, symlinks, and global prompt-partition consistency. Legacy audit `video_id` can only appear in `legacy_video_id` provenance.

The generic source contract always rejects repeated original SHA-256 values. DeepAction has one **explicit, opt-in, dataset-only** exception: `--duplicate-policy quarantine-all`. It excludes **every occurrence** in every repeated audit SHA-256 group; it never keeps one representative and it does not relax generic-manifest uniqueness or infer a policy for any other dataset. Default `error` remains strict. The quarantine report is useful to inspect before building:

```sh
cd WPP_Classification_Model
python3 build_manifest_contract.py inspect-deepaction --duplicate-policy quarantine-all
```

That non-hashing command validates paths and inventory and reports total, retained, excluded, and duplicate-group counts. Without the explicit policy it fails on a collision. Materialization intentionally requires a one-pass full original-file hash verification and is therefore explicit:

```sh
/home/naoromi/miniforge3/envs/entropy/bin/python build_manifest_contract.py build-deepaction \
  --duplicate-policy quarantine-all \
  --output-dir /mnt/d/projects/proj-entropy/datasets/deepaction_v1_manifest_seed42_quarantine_v1 \
  --verify-source-hashes
```

The destination must not already exist. Publication is atomic and contains `source_manifest.csv`, metadata, and the small `excluded_duplicate_sources.csv` report. Every excluded occurrence records canonical and legacy IDs, original SHA-256, source path, label, prompt/group, legacy and normalized partition, reason, and duplicate group identifier/count. Metadata records the audit SHA-256, verification scope/count, retained/excluded/duplicate-group accounting, report row count, and report SHA-256. Hashing covers retained **and excluded** originals before anything is published; an audit mismatch or actual duplicate grouping that differs from the audit aborts. Retained partition assignments are copied unchanged, and prompts may legitimately have both retained and excluded rows as long as their retained rows keep their audited partition. The adapter does not force DeepAction's count distribution onto generic source manifests.

## Extraction index: `extraction_index` v1.0

A future extraction build writes `extraction_index.csv` and `extraction_index.metadata.json`. Its exact CSV header is:

```text
video_id,status,encoded_relative_path,encoded_sha256,artifact_relative_path,producer_schema_version,row_count,frame_count,failure_code,failure_message
```

The index is tied to one exact input source-manifest CSV SHA-256, one encoded-root reference, one artifact-root reference, one processing-profile object, and one extractor-identity object. It covers exactly the source-manifest IDs and has one row per ID. `producer_schema_version` is exactly `2.0`.

`success` requires safe encoded/artifact relative paths, an encoded-file SHA-256, and positive `row_count`/`frame_count`; it cannot have failure fields. `failed` requires `failure_code` and `failure_message` and must leave every output path/checksum/count empty. This keeps original sources distinct from encoded products and prevents failures from inventing outputs.

Current index validation checks contract/schema/version, metadata linkage, paths, IDs, enum/status values, dimensions, and declared checksum shapes. It intentionally does **not** inspect encoded bytes or artifact content/check their declared checksum: artifact-content verification is future work and is not claimed here.

`pack_wpp_dataset.py` consumes these versioned contracts rather than directly depending on an existing loader. It calls `verify_native_build()` before opening artifacts, requires every index row to be successful, and is documented separately in `bounded-wpp-dataset.md`; it remains a data layer, not a model/training integration.

## Native WPP extraction workflow (v1)

`build_native_wpp_extraction.py` is the partition-agnostic bridge from a **complete** approved source manifest to an `extraction_index`. It is an extraction producer, not a model packer. It invokes the separately maintained `entropy-cabac-wpp` producer unchanged, retaining its existing default profile (GOP 16, QP 28, no B-frames); it does not reuse historical `*_wpp_512_15fps*` assets and never pads, crops, or partitions the producer frame axis.

```sh
cd WPP_Classification_Model
/home/naoromi/miniforge3/envs/entropy/bin/python build_native_wpp_extraction.py \
  --source-manifest /mnt/d/projects/proj-entropy/datasets/deepaction_v1_manifest_seed42_quarantine_v1 \
  --source-root /mnt/d/projects/proj-entropy/datasets/deepaction_v1 \
  --output-dir /mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1
```

The command above is documentation only; it is a dataset-wide operation and must be deliberately approved before launch. `--output-dir` must not exist and should be outside the code repository. The current source contract requires complete source-ID coverage, so a bounded smoke must instead use its own temporary one-record synthetic source manifest; it must never publish a one-row index against the authoritative full manifest.

For each source, the runner rejects links/escapes, hashes the original immediately before a verified staging copy, and encodes only that staged copy. It records the source/staged/encoded SHA-256 chain. Source and encoded timing are derived from FFprobe **decoded video frames** and their `best_effort_timestamp`, not packet counts: this permits multi-packet/slice original frames while comparing display order correctly through B-frame reordering. The producer itself retains its stricter encoded-HEVC packet/slice checks. The runner compares exact `Fraction(best_effort_timestamp * time_base)`, requires equal decoded/display width/height and sample aspect ratio, and records coded geometry separately without requiring HEVC CTU-padded coded dimensions to equal source coded dimensions. Frame count and every start-normalized VFR timestamp must agree within one output time-base tick; this is pointwise (not average-FPS) validation and prevents cumulative drift. Missing or duplicate decoded-frame PTS fail.

The output contains `encoded/`, `artifacts/`, `logs/`, `build_ledger.json`, atomically published `extraction-index/`, and an owned private `.native-wpp-state/` journal. The ledger records profile/script hashes, executable commands and log hashes, geometry/timing checks, source/encoded hashes, and a hash inventory for each artifact. Schema-2.0 artifacts are checked before success is recorded: all required NPZ arrays have `CTU rows × T` shape and finite numeric values, every `rows.csv` cell exactly equals NPZ `row_bytes`, every `frames.csv` metadata field exactly equals its NPZ vector, PTS/counts/totals agree, and no 3,000-frame transformation occurs. The index metadata binds the ledger SHA-256 and its `parent_of_extraction_index_directory` path base; the ledger intentionally does not hash the index, avoiding a circular digest.

`load_extraction_index()` remains a lightweight contract loader: it validates declared paths, IDs, schema, and the index/source CSV linkage, but it does not read encoded or artifact bytes. Consumers needing native-content integrity must additionally run:

```sh
/home/naoromi/miniforge3/envs/entropy/bin/python verify_native_wpp_extraction.py \
  --index /mnt/d/projects/proj-entropy/derived/native_wpp/deepaction-native-v1/extraction-index \
  --source-manifest /mnt/d/projects/proj-entropy/datasets/deepaction_v1_manifest_seed42_quarantine_v1
```

The verifier checks the bound ledger digest and safely rooted encoded, artifact-tree, and command-log hashes; it rejects symlinks and unsafe declared paths. Generic extraction-index schema fields remain unchanged; native hash provenance is carried by the bound ledger metadata.

Run-level invalidity (manifest, output collision, tools, scripts) fails before publication. Per-video source, encoder, timing, extractor, or artifact failures produce exactly one `failed` row and processing continues; once every input ID has a result, a complete index is published and the command exits nonzero if any row failed.

## Safe extraction resume (state v3)

A fresh command remains fresh-only: its output directory must not exist. To recover an interrupted stateful build, use the same exact manifest, source root, producer scripts, and output directory:

```sh
python build_native_wpp_extraction.py ... --output-dir /outside/build --resume
# Retry terminal per-video failures only when explicitly requested:
python build_native_wpp_extraction.py ... --output-dir /outside/build --resume --retry-failed
```

The journal fingerprints the source-manifest CSV and metadata, ordered records/IDs, resolved native profile, producer/workflow/CLI/helper hashes, and runtime executable/dependency versions. A resume rejects any mismatch before cleanup, retry, or encoding. State v3 requires an immutable receipt reference on **every** completed attempt (including historical failures retained after `--retry-failed`) and validates the receipt tree as an exact inventory: missing, altered, replayed, symlinked, wrong-source/wrong-attempt, or unreferenced receipts fail closed. A receipt durably written just before a journal update is the sole recoverable crash boundary and is promoted only after its complete payload is validated; a journal-only terminal claim is never accepted. It rehashes **all** current originals before reuse, then verifies every committed success's encoded bytes, artifact tree/schema, and immutable attempt logs. Missing or altered successes fail closed and are never silently re-encoded. The journal is an integrity protocol rather than a cryptographic signature against a party able to rewrite all owned state and receipts. Internal `pending`, `running`, and `interrupted` states never appear in an extraction index; an index is published only for exact complete input coverage.

A build-wide cooperative lock lives beside the output directory and records PID/host/time/nonce. It is not stolen automatically: after a hard kill, inspect the owner and manually remove that exact lock only after confirming there is no writer. On Linux, encoder input, extractor encoded input, and extractor private artifact output are descriptor-pinned `/proc/self/fd/N/...` paths passed with inherited FDs. Fresh private bytes are atomically renamed into held `encoded/sha256` and `artifacts/sha256` parent FDs only after their logical directory inode bindings still match; a parent swap aborts the whole build rather than writing through a replacement pathname. This protects pathname traversal/publication, not a same-UID actor that rewrites contents through already-authorized FDs. Cleanup is limited to manifest-SHA-derived paths under this build's owned `encoded/`, `artifacts/`, `logs/`, and `.work/` roots; links are rejected. The journal blocks native verification while active/finalizing, so a previous public index cannot be consumed during a retry. Finalization writes a durable generation ledger and candidate index before replacing the public index; a later resume completes that publication without re-encoding committed successes.

A finalized compatible all-success state is a verify-only no-op on `--resume`. Stateful journal formats v1 and v2 are rejected, never upgraded or mutated. The separate legacy public completed-build shape with **no journal** is read-only compatible when it fully verifies; it is not a state-v1 journal. A legacy finalized build containing failures cannot be retried in place; use a new destination. No resume can use a pilot/subset manifest as if it were the retained 2,583-record manifest.
