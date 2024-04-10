use candle_transformers::models::bert::{BertModel, Config, DTYPE};

use anyhow::{Error as E, Result};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,

    /// The Hugging Face model
    #[arg(long)]
    pub model_id: Option<String>,

    #[arg(long)]
    pub revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Use the pytorch weights
    #[arg(long)]
    pub use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    pub n: usize,

    /// L2 normalization
    #[arg(long, default_value = "true")]
    pub normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu
    #[arg(long, default_value = "false")]
    pub approximate_gelu: bool,
}


impl Args {
    pub fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };

        let model = BertModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}
