// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::PathBuf;

use hf_hub::api::sync::Api;

use crate::error::EngineError;
use crate::model_ref::ModelRef;

/// Return a local path to the model's GGUF, downloading the whole file from
/// Hugging Face if needed. Downloads are cached by `hf-hub` under its default
/// cache directory; a local reference is returned unchanged.
pub fn resolve_model(model_ref: &ModelRef) -> Result<PathBuf, EngineError> {
    match model_ref {
        ModelRef::LocalPath(p) => Ok(p.clone()),
        ModelRef::HuggingFace { repo, file } => {
            let api = Api::new().map_err(|e| EngineError::Download(e.to_string()))?;
            let path = api
                .model(repo.clone())
                .get(file)
                .map_err(|e| EngineError::Download(e.to_string()))?;
            Ok(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_ref::ModelRef;

    #[test]
    fn local_ref_resolves_to_its_own_path() {
        let r = ModelRef::LocalPath(PathBuf::from("/tmp/x.gguf"));
        assert_eq!(resolve_model(&r).unwrap(), PathBuf::from("/tmp/x.gguf"));
    }
}
