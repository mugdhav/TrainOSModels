# SigLIP Fine-Tuning — MediaSearchMCP

Fine-tunes `google/siglip-so400m-patch14-384` on domain-specific image-caption pairs for the [MediaSearchMCP](https://github.com/mugdhav/local_media_search_mcp_server) project using LoRA adapters.

## Overview

- **Base model:** `google/siglip-so400m-patch14-384` (400M params, 384px)
- **Method:** LoRA (rank=16) on vision encoder attention layers (`q_proj`, `v_proj`) — ~10M trainable params; text encoder frozen
- **Loss:** SigLIP sigmoid contrastive loss
- **Output model:** `mugdhav/siglip-so400m-media-search-finetuned`

## Prerequisites

- Hugging Face account: `mugdhav` with write-access token saved as `HF_TOKEN` in Colab secrets
- Source dataset: `mugdhav/media-search-demo-files` (images + videos)
- Google Colab Pro (A10G GPU required for training notebook)

## Workflow

Run the notebooks in order:

### Step 1 — `01_prepare_captions.ipynb` (T4 GPU)
Downloads `mugdhav/media-search-demo-files`, generates text captions for each image/video frame using `Salesforce/blip-image-captioning-large`, and pushes to `mugdhav/media-search-demo-captioned`.

### Step 2 — `02_train_siglip.ipynb` (A10G GPU)
Loads the captioned dataset, applies LoRA to the SigLIP vision encoder, and trains for 3 epochs with:
- Batch size: 8, gradient accumulation: 4 (effective batch: 32)
- Learning rate: 2e-4 with cosine annealing + 50-step warmup
- Best checkpoint per epoch pushed to `mugdhav/siglip-so400m-media-search-finetuned`

### Step 3 — `03_verify_model.ipynb` (T4 GPU)
Compares cosine similarity of fine-tuned vs baseline model on 10 image-caption pairs.
**Pass condition:** fine-tuned avg similarity > baseline avg similarity.

## Known Blocker

The current dataset has only ~26 captioned items. The minimum for effective fine-tuning without severe overfitting is **500–2,000 images**. Before running the notebooks:

1. Upload more images/videos to the `media/` folder in the HF Space (`mugdhav/MediaSearchMCP`)
2. Re-run Step 1 to regenerate captions for the expanded dataset
3. Then proceed to Steps 2 and 3

## Post-Verification: Deploy Fine-Tuned Model

Once notebook 3 passes, update the production app:

1. In `mcp-servers/media-search-mcp/ai_indexer.py` line 39, change:
   ```python
   model_name: str = "google/siglip-so400m-patch14-384"
   ```
   to:
   ```python
   model_name: str = "mugdhav/siglip-so400m-media-search-finetuned"
   ```
2. Delete the `index/` directory to force re-indexing
3. Restart `app.py`
4. Push to HF Space: `git push huggingface hf_deploy:main`
