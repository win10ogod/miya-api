use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[derive(Clone)]
pub(crate) struct DurableStore {
    backend: Arc<Backend>,
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

enum Backend {
    Memory {
        objects: Mutex<BTreeMap<String, serde_json::Value>>,
        blobs: Mutex<BTreeMap<String, Vec<u8>>>,
    },
    Filesystem {
        root: PathBuf,
    },
}

impl DurableStore {
    pub(crate) fn memory() -> Self {
        Self {
            backend: Arc::new(Backend::Memory {
                objects: Mutex::new(BTreeMap::new()),
                blobs: Mutex::new(BTreeMap::new()),
            }),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub(crate) fn filesystem(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        std::fs::create_dir_all(root.join("objects")).map_err(|error| {
            format!(
                "failed to create durable object directory {}: {error}",
                root.display()
            )
        })?;
        std::fs::create_dir_all(root.join("blobs")).map_err(|error| {
            format!(
                "failed to create durable blob directory {}: {error}",
                root.display()
            )
        })?;
        Ok(Self {
            backend: Arc::new(Backend::Filesystem { root }),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    pub(crate) async fn put_json<T: Serialize>(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
        value: &T,
    ) -> Result<(), String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        let value = serde_json::to_value(value).map_err(|error| error.to_string())?;
        let key = memory_key(namespace, tenant_id, id);
        match self.backend.as_ref() {
            Backend::Memory { objects, .. } => {
                objects
                    .lock()
                    .map_err(|_| "durable object memory lock poisoned".to_string())?
                    .insert(key, value);
                Ok(())
            }
            Backend::Filesystem { root } => {
                let _guard = self.write_lock.lock().await;
                let path = object_path(root, namespace, tenant_id, id);
                write_atomic(
                    &path,
                    serde_json::to_vec(&value).map_err(|error| error.to_string())?,
                )
                .await
            }
        }
    }

    pub(crate) async fn get_json<T: DeserializeOwned>(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
    ) -> Result<Option<T>, String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        let value = match self.backend.as_ref() {
            Backend::Memory { objects, .. } => objects
                .lock()
                .map_err(|_| "durable object memory lock poisoned".to_string())?
                .get(&memory_key(namespace, tenant_id, id))
                .cloned(),
            Backend::Filesystem { root } => {
                let path = object_path(root, namespace, tenant_id, id);
                match tokio::fs::read(path).await {
                    Ok(bytes) => Some(
                        serde_json::from_slice::<serde_json::Value>(&bytes)
                            .map_err(|error| error.to_string())?,
                    ),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.to_string()),
                }
            }
        };
        value
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn list_json<T: DeserializeOwned>(
        &self,
        namespace: &str,
        tenant_id: &str,
    ) -> Result<Vec<T>, String> {
        validate_component("namespace", namespace)?;
        let values = match self.backend.as_ref() {
            Backend::Memory { objects, .. } => {
                let prefix = format!("{namespace}:{}:", tenant_hash(tenant_id));
                objects
                    .lock()
                    .map_err(|_| "durable object memory lock poisoned".to_string())?
                    .iter()
                    .filter(|(key, _)| key.starts_with(&prefix))
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>()
            }
            Backend::Filesystem { root } => {
                let directory = object_directory(root, namespace, tenant_id);
                let mut entries = match tokio::fs::read_dir(directory).await {
                    Ok(entries) => entries,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        return Ok(Vec::new());
                    }
                    Err(error) => return Err(error.to_string()),
                };
                let mut values = Vec::new();
                while let Some(entry) = entries
                    .next_entry()
                    .await
                    .map_err(|error| error.to_string())?
                {
                    if entry
                        .path()
                        .extension()
                        .and_then(|extension| extension.to_str())
                        != Some("json")
                    {
                        continue;
                    }
                    let bytes = tokio::fs::read(entry.path())
                        .await
                        .map_err(|error| error.to_string())?;
                    values.push(serde_json::from_slice(&bytes).map_err(|error| error.to_string())?);
                }
                values
            }
        };
        values
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<T>, _>>()
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn list_all_json<T: DeserializeOwned>(
        &self,
        namespace: &str,
    ) -> Result<Vec<T>, String> {
        validate_component("namespace", namespace)?;
        let values = match self.backend.as_ref() {
            Backend::Memory { objects, .. } => {
                let prefix = format!("{namespace}:");
                objects
                    .lock()
                    .map_err(|_| "durable object memory lock poisoned".to_string())?
                    .iter()
                    .filter(|(key, _)| key.starts_with(&prefix))
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>()
            }
            Backend::Filesystem { root } => {
                let namespace_directory = root.join("objects").join(namespace);
                let mut tenant_entries = match tokio::fs::read_dir(namespace_directory).await {
                    Ok(entries) => entries,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        return Ok(Vec::new());
                    }
                    Err(error) => return Err(error.to_string()),
                };
                let mut values = Vec::new();
                while let Some(tenant_entry) = tenant_entries
                    .next_entry()
                    .await
                    .map_err(|error| error.to_string())?
                {
                    if !tenant_entry
                        .file_type()
                        .await
                        .map_err(|error| error.to_string())?
                        .is_dir()
                    {
                        continue;
                    }
                    let mut entries = tokio::fs::read_dir(tenant_entry.path())
                        .await
                        .map_err(|error| error.to_string())?;
                    while let Some(entry) = entries
                        .next_entry()
                        .await
                        .map_err(|error| error.to_string())?
                    {
                        if entry
                            .path()
                            .extension()
                            .and_then(|extension| extension.to_str())
                            != Some("json")
                        {
                            continue;
                        }
                        let bytes = tokio::fs::read(entry.path())
                            .await
                            .map_err(|error| error.to_string())?;
                        values.push(
                            serde_json::from_slice(&bytes).map_err(|error| error.to_string())?,
                        );
                    }
                }
                values
            }
        };
        values
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<T>, _>>()
            .map_err(|error| error.to_string())
    }

    pub(crate) async fn delete_json(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
    ) -> Result<bool, String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        match self.backend.as_ref() {
            Backend::Memory { objects, .. } => Ok(objects
                .lock()
                .map_err(|_| "durable object memory lock poisoned".to_string())?
                .remove(&memory_key(namespace, tenant_id, id))
                .is_some()),
            Backend::Filesystem { root } => {
                let _guard = self.write_lock.lock().await;
                match tokio::fs::remove_file(object_path(root, namespace, tenant_id, id)).await {
                    Ok(()) => Ok(true),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
                    Err(error) => Err(error.to_string()),
                }
            }
        }
    }

    pub(crate) async fn put_blob(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
        bytes: Vec<u8>,
    ) -> Result<(), String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        match self.backend.as_ref() {
            Backend::Memory { blobs, .. } => {
                blobs
                    .lock()
                    .map_err(|_| "durable blob memory lock poisoned".to_string())?
                    .insert(memory_key(namespace, tenant_id, id), bytes);
                Ok(())
            }
            Backend::Filesystem { root } => {
                let _guard = self.write_lock.lock().await;
                write_atomic(&blob_path(root, namespace, tenant_id, id), bytes).await
            }
        }
    }

    pub(crate) async fn get_blob(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
    ) -> Result<Option<Vec<u8>>, String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        match self.backend.as_ref() {
            Backend::Memory { blobs, .. } => Ok(blobs
                .lock()
                .map_err(|_| "durable blob memory lock poisoned".to_string())?
                .get(&memory_key(namespace, tenant_id, id))
                .cloned()),
            Backend::Filesystem { root } => {
                match tokio::fs::read(blob_path(root, namespace, tenant_id, id)).await {
                    Ok(bytes) => Ok(Some(bytes)),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
                    Err(error) => Err(error.to_string()),
                }
            }
        }
    }

    pub(crate) async fn delete_blob(
        &self,
        namespace: &str,
        tenant_id: &str,
        id: &str,
    ) -> Result<bool, String> {
        validate_component("namespace", namespace)?;
        validate_component("id", id)?;
        match self.backend.as_ref() {
            Backend::Memory { blobs, .. } => Ok(blobs
                .lock()
                .map_err(|_| "durable blob memory lock poisoned".to_string())?
                .remove(&memory_key(namespace, tenant_id, id))
                .is_some()),
            Backend::Filesystem { root } => {
                let _guard = self.write_lock.lock().await;
                match tokio::fs::remove_file(blob_path(root, namespace, tenant_id, id)).await {
                    Ok(()) => Ok(true),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
                    Err(error) => Err(error.to_string()),
                }
            }
        }
    }
}

async fn write_atomic(path: &Path, bytes: Vec<u8>) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "durable path has no parent".to_string())?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| error.to_string())?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("object"),
        Uuid::new_v4()
    ));
    let mut file = tokio::fs::File::create(&temporary)
        .await
        .map_err(|error| error.to_string())?;
    file.write_all(&bytes)
        .await
        .map_err(|error| error.to_string())?;
    file.sync_all().await.map_err(|error| error.to_string())?;
    drop(file);
    if let Err(error) = tokio::fs::rename(&temporary, path).await {
        let _ = tokio::fs::remove_file(path).await;
        tokio::fs::rename(&temporary, path)
            .await
            .map_err(|_| error.to_string())?;
    }
    Ok(())
}

fn object_directory(root: &Path, namespace: &str, tenant_id: &str) -> PathBuf {
    root.join("objects")
        .join(namespace)
        .join(tenant_hash(tenant_id))
}

fn object_path(root: &Path, namespace: &str, tenant_id: &str, id: &str) -> PathBuf {
    object_directory(root, namespace, tenant_id).join(format!("{id}.json"))
}

fn blob_path(root: &Path, namespace: &str, tenant_id: &str, id: &str) -> PathBuf {
    root.join("blobs")
        .join(namespace)
        .join(tenant_hash(tenant_id))
        .join(id)
}

fn memory_key(namespace: &str, tenant_id: &str, id: &str) -> String {
    format!("{namespace}:{}:{id}", tenant_hash(tenant_id))
}

fn tenant_hash(tenant_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tenant_id.as_bytes());
    hex::encode(hasher.finalize())
}

fn validate_component(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 160
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(format!("invalid durable {name}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn filesystem_store_survives_reopen_for_objects_and_blobs() {
        let directory = tempfile::tempdir().unwrap();
        let store = DurableStore::filesystem(directory.path()).unwrap();
        store
            .put_json(
                "jobs",
                "tenant-a",
                "job-1",
                &serde_json::json!({"status": "queued"}),
            )
            .await
            .unwrap();
        store
            .put_blob("payloads", "tenant-a", "job-1", b"payload".to_vec())
            .await
            .unwrap();
        drop(store);

        let reopened = DurableStore::filesystem(directory.path()).unwrap();
        let object = reopened
            .get_json::<serde_json::Value>("jobs", "tenant-a", "job-1")
            .await
            .unwrap()
            .unwrap();
        let blob = reopened
            .get_blob("payloads", "tenant-a", "job-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(object["status"], "queued");
        assert_eq!(blob, b"payload");
    }
}
