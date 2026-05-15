use std::{collections::BTreeMap, path::Path, sync::Arc, time::SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use surrealkv::{Durability, LSMIterator, Options, Tree, TreeBuilder};
use thiserror::Error;

pub const DEFAULT_MAX_CONTEXT_BYTES: usize = 1_048_576;
pub const DEFAULT_MAX_CHUNKS: usize = 128;
pub const DEFAULT_RECENT_TAIL_CHUNKS: usize = 12;

#[derive(Debug, Error)]
pub enum ContextStoreError {
    #[error("surrealkv error: {0}")]
    SurrealKv(#[from] surrealkv::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid context input: {0}")]
    InvalidInput(String),
}

#[derive(Clone)]
pub struct SurrealKvContextStore {
    tree: Arc<Tree>,
}

impl SurrealKvContextStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContextStoreError> {
        let opts = Options::new()
            .with_path(path.as_ref().to_path_buf())
            .with_versioning(true, 0)
            .with_versioned_index(true)
            .with_block_cache_capacity(64 * 1024 * 1024)
            .with_max_memtable_size(128 * 1024 * 1024);
        let tree = TreeBuilder::with_options(opts).build()?;
        Ok(Self {
            tree: Arc::new(tree),
        })
    }

    pub async fn append(
        &self,
        tenant_id: &str,
        context_id: &str,
        records: Vec<ContextAppendRecord>,
    ) -> Result<ContextHead, ContextStoreError> {
        validate_component("tenant_id", tenant_id)?;
        validate_component("context_id", context_id)?;

        let records = records
            .into_iter()
            .filter(|record| !record.text.trim().is_empty())
            .collect::<Vec<_>>();
        if records.is_empty() {
            return Ok(self
                .head(tenant_id, context_id, None)?
                .unwrap_or_else(|| ContextHead::empty(tenant_id, context_id)));
        }

        let mut txn = self.tree.begin()?;
        txn.set_durability(Durability::Immediate);
        let head_key = head_key(tenant_id, context_id);
        let mut head = txn
            .get(head_key.clone())?
            .map(|bytes| serde_json::from_slice::<ContextHead>(&bytes))
            .transpose()?
            .unwrap_or_else(|| ContextHead::empty(tenant_id, context_id));

        for record in records {
            let revision = head.latest_revision + 1;
            let chunk = StoredContextChunk {
                tenant_id: tenant_id.to_string(),
                context_id: context_id.to_string(),
                revision,
                role: record.role,
                text_sha256: sha256_hex(record.text.as_bytes()),
                byte_len: record.text.len(),
                text: record.text,
                created_at_ms: now_ms(),
            };
            txn.set_at(
                chunk_key(tenant_id, context_id, revision),
                serde_json::to_vec(&chunk)?,
                revision,
            )?;
            head.latest_revision = revision;
            head.chunk_count += 1;
            head.updated_at_ms = chunk.created_at_ms;
        }

        txn.set_at(
            head_key,
            serde_json::to_vec(&head)?,
            head.latest_revision.max(1),
        )?;
        txn.commit().await?;
        Ok(head)
    }

    pub fn head(
        &self,
        tenant_id: &str,
        context_id: &str,
        rewind_revision: Option<u64>,
    ) -> Result<Option<ContextHead>, ContextStoreError> {
        validate_component("tenant_id", tenant_id)?;
        validate_component("context_id", context_id)?;

        let txn = self.tree.begin()?;
        let bytes = match rewind_revision {
            Some(revision) => txn.get_at(head_key(tenant_id, context_id), revision)?,
            None => txn.get(head_key(tenant_id, context_id))?,
        };
        bytes
            .map(|bytes| serde_json::from_slice::<ContextHead>(&bytes))
            .transpose()
            .map_err(ContextStoreError::from)
    }

    pub async fn assemble(
        &self,
        tenant_id: &str,
        context_id: &str,
        options: ContextAssemblyOptions,
    ) -> Result<ContextAssembly, ContextStoreError> {
        validate_component("tenant_id", tenant_id)?;
        validate_component("context_id", context_id)?;
        let normalized_options = options.normalized();

        let Some(head) = self.head(tenant_id, context_id, normalized_options.rewind_revision)?
        else {
            return Ok(ContextAssembly::empty_with_namespace(
                context_id,
                normalized_options.cache_namespace,
            ));
        };
        if head.latest_revision == 0 {
            return Ok(ContextAssembly::empty_with_namespace(
                context_id,
                normalized_options.cache_namespace,
            ));
        }

        let target_revision = normalized_options
            .rewind_revision
            .unwrap_or(head.latest_revision)
            .min(head.latest_revision);
        let policy_hash = assembly_policy_hash(&normalized_options);

        if normalized_options.cache_enabled
            && let Some(pack) =
                self.read_pack_at(tenant_id, context_id, &policy_hash, target_revision)?
        {
            if pack.revision == target_revision {
                return Ok(ContextAssembly {
                    context_id: context_id.to_string(),
                    cache_namespace: normalized_options.cache_namespace,
                    revision: target_revision,
                    text: pack.text,
                    included_chunks: pack.included_chunks,
                    included_bytes: pack.byte_len,
                    cache_hit: true,
                    base_cache_revision: Some(pack.revision),
                    tail_chunks: 0,
                });
            }

            if pack.revision < target_revision {
                let tail_chunks =
                    self.load_chunks(tenant_id, context_id, pack.revision + 1, target_revision)?;
                let mut text = pack.text;
                let tail_text = render_chunks(
                    choose_chunks(&tail_chunks, &normalized_options),
                    normalized_options
                        .max_context_bytes
                        .saturating_sub(text.len()),
                );
                if !tail_text.is_empty() {
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(&tail_text);
                }
                let text = trim_to_char_boundary(text, normalized_options.max_context_bytes);
                let assembly = ContextAssembly {
                    context_id: context_id.to_string(),
                    cache_namespace: normalized_options.cache_namespace.clone(),
                    revision: target_revision,
                    included_chunks: pack.included_chunks + tail_chunks.len(),
                    included_bytes: text.len(),
                    text,
                    cache_hit: true,
                    base_cache_revision: Some(pack.revision),
                    tail_chunks: tail_chunks.len(),
                };
                self.write_pack(tenant_id, context_id, &policy_hash, &assembly)
                    .await?;
                return Ok(assembly);
            }
        }

        let chunks = self.load_chunks(tenant_id, context_id, 1, target_revision)?;
        let selected = choose_chunks(&chunks, &normalized_options);
        let text = render_chunks(selected, normalized_options.max_context_bytes);
        let assembly = ContextAssembly {
            context_id: context_id.to_string(),
            cache_namespace: normalized_options.cache_namespace.clone(),
            revision: target_revision,
            included_chunks: chunks.len(),
            included_bytes: text.len(),
            text,
            cache_hit: false,
            base_cache_revision: None,
            tail_chunks: 0,
        };
        if normalized_options.cache_enabled {
            self.write_pack(tenant_id, context_id, &policy_hash, &assembly)
                .await?;
        }
        Ok(assembly)
    }

    fn load_chunks(
        &self,
        tenant_id: &str,
        context_id: &str,
        start_revision: u64,
        end_revision: u64,
    ) -> Result<Vec<StoredContextChunk>, ContextStoreError> {
        if start_revision > end_revision {
            return Ok(Vec::new());
        }

        let txn = self.tree.begin()?;
        let prefix = chunk_prefix(tenant_id, context_id);
        let mut end = prefix.clone();
        end.push(0xff);
        let mut iter = txn.range(prefix, end)?;
        let mut chunks = Vec::new();
        iter.seek_first()?;
        while iter.valid() {
            let value = iter.value()?;
            let chunk: StoredContextChunk = serde_json::from_slice(&value)?;
            if (start_revision..=end_revision).contains(&chunk.revision) {
                chunks.push(chunk);
            }
            iter.next()?;
        }
        chunks.sort_by_key(|chunk| chunk.revision);
        Ok(chunks)
    }

    fn read_pack_at(
        &self,
        tenant_id: &str,
        context_id: &str,
        policy_hash: &str,
        revision: u64,
    ) -> Result<Option<ContextPack>, ContextStoreError> {
        let txn = self.tree.begin()?;
        txn.get_at(pack_key(tenant_id, context_id, policy_hash), revision)?
            .map(|bytes| serde_json::from_slice::<ContextPack>(&bytes))
            .transpose()
            .map_err(ContextStoreError::from)
    }

    async fn write_pack(
        &self,
        tenant_id: &str,
        context_id: &str,
        policy_hash: &str,
        assembly: &ContextAssembly,
    ) -> Result<(), ContextStoreError> {
        if assembly.revision == 0 {
            return Ok(());
        }
        let mut txn = self.tree.begin()?;
        let pack = ContextPack {
            context_id: context_id.to_string(),
            cache_namespace: assembly.cache_namespace.clone(),
            revision: assembly.revision,
            policy_hash: policy_hash.to_string(),
            text: assembly.text.clone(),
            byte_len: assembly.text.len(),
            included_chunks: assembly.included_chunks,
            created_at_ms: now_ms(),
        };
        txn.set_at(
            pack_key(tenant_id, context_id, policy_hash),
            serde_json::to_vec(&pack)?,
            assembly.revision,
        )?;
        txn.commit().await?;
        Ok(())
    }

    pub async fn put_json(
        &self,
        namespace: &str,
        tenant_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), ContextStoreError> {
        validate_component("namespace", namespace)?;
        validate_component("tenant_id", tenant_id)?;
        validate_component("key", key)?;

        let mut txn = self.tree.begin()?;
        txn.set_durability(Durability::Immediate);
        txn.set(
            json_key(namespace, tenant_id, key),
            serde_json::to_vec(&value)?,
        )?;
        txn.commit().await?;
        Ok(())
    }

    pub fn get_json(
        &self,
        namespace: &str,
        tenant_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>, ContextStoreError> {
        validate_component("namespace", namespace)?;
        validate_component("tenant_id", tenant_id)?;
        validate_component("key", key)?;

        let txn = self.tree.begin()?;
        txn.get(json_key(namespace, tenant_id, key))?
            .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes))
            .transpose()
            .map_err(ContextStoreError::from)
    }

    pub async fn delete_json(
        &self,
        namespace: &str,
        tenant_id: &str,
        key: &str,
    ) -> Result<bool, ContextStoreError> {
        validate_component("namespace", namespace)?;
        validate_component("tenant_id", tenant_id)?;
        validate_component("key", key)?;

        let existing = self.get_json(namespace, tenant_id, key)?.is_some();
        if !existing {
            return Ok(false);
        }

        let mut txn = self.tree.begin()?;
        txn.set_durability(Durability::Immediate);
        txn.delete(json_key(namespace, tenant_id, key))?;
        txn.commit().await?;
        Ok(true)
    }

    pub fn list_json(
        &self,
        namespace: &str,
        tenant_id: &str,
    ) -> Result<Vec<serde_json::Value>, ContextStoreError> {
        validate_component("namespace", namespace)?;
        validate_component("tenant_id", tenant_id)?;

        let txn = self.tree.begin()?;
        let prefix = json_prefix(namespace, tenant_id);
        let mut end = prefix.clone();
        end.push(0xff);
        let mut iter = txn.range(prefix, end)?;
        let mut values = Vec::new();
        iter.seek_first()?;
        while iter.valid() {
            let value = iter.value()?;
            values.push(serde_json::from_slice::<serde_json::Value>(&value)?);
            iter.next()?;
        }
        Ok(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextAppendRecord {
    pub role: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextHead {
    pub tenant_id: String,
    pub context_id: String,
    pub latest_revision: u64,
    pub chunk_count: usize,
    pub updated_at_ms: u128,
}

impl ContextHead {
    fn empty(tenant_id: &str, context_id: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            context_id: context_id.to_string(),
            latest_revision: 0,
            chunk_count: 0,
            updated_at_ms: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextAssemblyOptions {
    pub query: Option<String>,
    pub cache_namespace: String,
    pub rewind_revision: Option<u64>,
    pub max_context_bytes: usize,
    pub max_chunks: usize,
    pub recent_tail_chunks: usize,
    pub cache_enabled: bool,
}

impl Default for ContextAssemblyOptions {
    fn default() -> Self {
        Self {
            query: None,
            cache_namespace: "default".to_string(),
            rewind_revision: None,
            max_context_bytes: DEFAULT_MAX_CONTEXT_BYTES,
            max_chunks: DEFAULT_MAX_CHUNKS,
            recent_tail_chunks: DEFAULT_RECENT_TAIL_CHUNKS,
            cache_enabled: true,
        }
    }
}

impl ContextAssemblyOptions {
    fn normalized(mut self) -> Self {
        self.max_context_bytes = self.max_context_bytes.clamp(1, DEFAULT_MAX_CONTEXT_BYTES);
        self.max_chunks = self.max_chunks.clamp(1, DEFAULT_MAX_CHUNKS);
        self.recent_tail_chunks = self.recent_tail_chunks.min(self.max_chunks);
        self.cache_namespace = normalize_cache_namespace(self.cache_namespace);
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextAssembly {
    pub context_id: String,
    pub cache_namespace: String,
    pub revision: u64,
    pub text: String,
    pub included_chunks: usize,
    pub included_bytes: usize,
    pub cache_hit: bool,
    pub base_cache_revision: Option<u64>,
    pub tail_chunks: usize,
}

impl ContextAssembly {
    fn empty_with_namespace(context_id: &str, cache_namespace: String) -> Self {
        Self {
            context_id: context_id.to_string(),
            cache_namespace,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct StoredContextChunk {
    tenant_id: String,
    context_id: String,
    revision: u64,
    role: String,
    text_sha256: String,
    byte_len: usize,
    text: String,
    created_at_ms: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ContextPack {
    context_id: String,
    #[serde(default)]
    cache_namespace: String,
    revision: u64,
    policy_hash: String,
    text: String,
    byte_len: usize,
    included_chunks: usize,
    created_at_ms: u128,
}

fn choose_chunks<'a>(
    chunks: &'a [StoredContextChunk],
    options: &ContextAssemblyOptions,
) -> Vec<&'a StoredContextChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let terms = options
        .query
        .as_deref()
        .map(query_terms)
        .unwrap_or_default();
    let recent_start = chunks
        .len()
        .saturating_sub(options.recent_tail_chunks)
        .min(chunks.len());
    let recent_revisions = chunks[recent_start..]
        .iter()
        .map(|chunk| chunk.revision)
        .collect::<std::collections::BTreeSet<_>>();

    let mut candidates = chunks
        .iter()
        .filter_map(|chunk| {
            let score = score_chunk(chunk, &terms);
            let is_recent = recent_revisions.contains(&chunk.revision);
            if terms.is_empty() || score > 0 || is_recent {
                let priority = score.saturating_mul(10_000)
                    + if is_recent { 5_000 } else { 0 }
                    + chunk.revision.min(4_999);
                Some((priority, chunk))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.revision.cmp(&left.1.revision))
    });

    let mut selected = BTreeMap::new();
    let mut bytes = 0usize;
    for (_, chunk) in candidates {
        if selected.len() >= options.max_chunks {
            break;
        }
        if !selected.is_empty() && bytes + chunk.byte_len > options.max_context_bytes {
            continue;
        }
        bytes += chunk.byte_len;
        selected.insert(chunk.revision, chunk);
    }
    selected.into_values().collect()
}

fn render_chunks(chunks: Vec<&StoredContextChunk>, max_bytes: usize) -> String {
    let mut rendered = String::new();
    for chunk in chunks {
        let part = format!(
            "[context revision={} role={}]\n{}",
            chunk.revision,
            chunk.role,
            chunk.text.trim()
        );
        if rendered.is_empty() {
            if part.len() <= max_bytes {
                rendered.push_str(&part);
            }
        } else if rendered.len() + part.len() + 2 <= max_bytes {
            rendered.push_str("\n\n");
            rendered.push_str(&part);
        }
    }
    rendered
}

fn score_chunk(chunk: &StoredContextChunk, terms: &[String]) -> u64 {
    if terms.is_empty() {
        return 0;
    }
    let lower = chunk.text.to_lowercase();
    terms
        .iter()
        .map(|term| lower.matches(term).count() as u64)
        .sum()
}

fn query_terms(query: &str) -> Vec<String> {
    let mut terms = query
        .to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|term| term.chars().count() >= 2)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn assembly_policy_hash(options: &ContextAssemblyOptions) -> String {
    let mut hasher = Sha256::new();
    hasher.update(options.cache_namespace.as_bytes());
    hasher.update([0]);
    hasher.update(options.max_context_bytes.to_be_bytes());
    hasher.update(options.max_chunks.to_be_bytes());
    hasher.update(options.recent_tail_chunks.to_be_bytes());
    if let Some(query) = &options.query {
        hasher.update(query.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn normalize_cache_namespace(value: String) -> String {
    let normalized = value.trim();
    if normalized.is_empty() {
        "default".to_string()
    } else {
        normalized.chars().take(256).collect()
    }
}

fn validate_component(name: &str, value: &str) -> Result<(), ContextStoreError> {
    if value.trim().is_empty() {
        return Err(ContextStoreError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    Ok(())
}

fn head_key(tenant_id: &str, context_id: &str) -> Vec<u8> {
    format!(
        "ctx/{}/{}/head",
        encode_component(tenant_id),
        encode_component(context_id)
    )
    .into_bytes()
}

fn chunk_prefix(tenant_id: &str, context_id: &str) -> Vec<u8> {
    format!(
        "ctx/{}/{}/chunk/",
        encode_component(tenant_id),
        encode_component(context_id)
    )
    .into_bytes()
}

fn chunk_key(tenant_id: &str, context_id: &str, revision: u64) -> Vec<u8> {
    let mut key = chunk_prefix(tenant_id, context_id);
    key.extend(format!("{revision:020}").as_bytes());
    key
}

fn pack_key(tenant_id: &str, context_id: &str, policy_hash: &str) -> Vec<u8> {
    format!(
        "ctx/{}/{}/pack/{policy_hash}",
        encode_component(tenant_id),
        encode_component(context_id)
    )
    .into_bytes()
}

fn json_prefix(namespace: &str, tenant_id: &str) -> Vec<u8> {
    format!(
        "json/{}/{}/",
        encode_component(namespace),
        encode_component(tenant_id)
    )
    .into_bytes()
}

fn json_key(namespace: &str, tenant_id: &str, key: &str) -> Vec<u8> {
    let mut prefix = json_prefix(namespace, tenant_id);
    prefix.extend(encode_component(key).as_bytes());
    prefix
}

fn encode_component(value: &str) -> String {
    hex::encode(value.as_bytes())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn trim_to_char_boundary(mut text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text.replace_range(..start, "");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_and_rewind_reads_previous_context_version() {
        let temp = tempfile::tempdir().unwrap();
        let store = SurrealKvContextStore::open(temp.path()).unwrap();

        let first = store
            .append(
                "tenant",
                "ctx",
                vec![ContextAppendRecord {
                    role: "user".to_string(),
                    text: "needle: red key".to_string(),
                }],
            )
            .await
            .unwrap();
        store
            .append(
                "tenant",
                "ctx",
                vec![ContextAppendRecord {
                    role: "assistant".to_string(),
                    text: "newer blue key".to_string(),
                }],
            )
            .await
            .unwrap();

        let assembly = store
            .assemble(
                "tenant",
                "ctx",
                ContextAssemblyOptions {
                    query: Some("needle".to_string()),
                    cache_namespace: "default".to_string(),
                    rewind_revision: Some(first.latest_revision),
                    max_context_bytes: 4096,
                    max_chunks: 8,
                    recent_tail_chunks: 2,
                    cache_enabled: true,
                },
            )
            .await
            .unwrap();

        assert!(assembly.text.contains("needle: red key"));
        assert!(!assembly.text.contains("newer blue key"));
    }

    #[tokio::test]
    async fn retrieval_finds_needle_in_large_haystack() {
        let temp = tempfile::tempdir().unwrap();
        let store = SurrealKvContextStore::open(temp.path()).unwrap();
        let mut records = Vec::new();
        for index in 0..512 {
            let text = if index == 377 {
                "needle marker: launch code violet".to_string()
            } else {
                format!("filler long-context memory line {index}")
            };
            records.push(ContextAppendRecord {
                role: "user".to_string(),
                text,
            });
        }
        store.append("tenant", "haystack", records).await.unwrap();

        let assembly = store
            .assemble(
                "tenant",
                "haystack",
                ContextAssemblyOptions {
                    query: Some("violet launch".to_string()),
                    cache_namespace: "default".to_string(),
                    max_context_bytes: 2048,
                    max_chunks: 6,
                    recent_tail_chunks: 2,
                    cache_enabled: true,
                    rewind_revision: None,
                },
            )
            .await
            .unwrap();

        assert!(assembly.text.contains("needle marker: launch code violet"));
        assert!(assembly.included_bytes <= 2048);
    }

    #[tokio::test]
    async fn context_pack_cache_reuses_common_prefix_and_appends_tail() {
        let temp = tempfile::tempdir().unwrap();
        let store = SurrealKvContextStore::open(temp.path()).unwrap();
        store
            .append(
                "tenant",
                "cache",
                vec![
                    ContextAppendRecord {
                        role: "system".to_string(),
                        text: "common project memory".to_string(),
                    },
                    ContextAppendRecord {
                        role: "user".to_string(),
                        text: "first scene".to_string(),
                    },
                ],
            )
            .await
            .unwrap();
        let first = store
            .assemble("tenant", "cache", ContextAssemblyOptions::default())
            .await
            .unwrap();
        assert!(!first.cache_hit);

        let second = store
            .assemble("tenant", "cache", ContextAssemblyOptions::default())
            .await
            .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.base_cache_revision, Some(first.revision));

        store
            .append(
                "tenant",
                "cache",
                vec![ContextAppendRecord {
                    role: "assistant".to_string(),
                    text: "small generated tail".to_string(),
                }],
            )
            .await
            .unwrap();
        let third = store
            .assemble("tenant", "cache", ContextAssemblyOptions::default())
            .await
            .unwrap();

        assert!(third.cache_hit);
        assert_eq!(third.base_cache_revision, Some(first.revision));
        assert!(third.text.contains("common project memory"));
        assert!(third.text.contains("small generated tail"));
        assert_eq!(third.tail_chunks, 1);
    }
}
