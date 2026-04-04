# SigLIP Fine-Tuning

Fine-tunes `google/siglip-so400m-patch14-384` on domain-specific image-caption pairs using LoRA adapters, for use in semantic media search applications.

## Overview

- **Base model:** `google/siglip-so400m-patch14-384` (400M params, 384px)
- **Method:** LoRA (rank=16) on vision encoder attention layers (`q_proj`, `v_proj`) — ~10M trainable params; text encoder frozen
- **Loss:** SigLIP sigmoid contrastive loss
- **Output:** your fine-tuned model pushed to your Hugging Face Hub

## Prerequisites

- Hugging Face account with a write-access token saved as `HF_TOKEN` in Colab secrets
- A source image/video dataset on Hugging Face Hub (see **Dataset Setup** below)
- Google Colab Pro (A10G GPU required for the training notebook)

## Dataset Setup

Create your own image dataset on Hugging Face Hub. As a reference, see [`mugdhav/media-search-demo-files`](https://huggingface.co/datasets/mugdhav/media-search-demo-files) for the expected structure — a dataset with an `image` column (and optionally a `video` column for video frames).

To create your own:
1. Upload your images/videos to a new HF dataset repo using `huggingface_hub` or the Hub web UI
2. Note the dataset ID (e.g. `your-username/your-media-dataset`)
3. Set that ID as `SOURCE_DATASET` in Notebook 1 before running

> **Note:** If your source dataset is gated, you must first visit the dataset page on Hugging Face and accept the access request with your account. The notebooks authenticate using your `HF_TOKEN` — the token alone is not sufficient without prior gate acceptance.

## Workflow

Run the notebooks in order:

### Step 1 — `01_prepare_captions.ipynb` (T4 GPU)
Downloads your source dataset, generates text captions for each image/video frame using `Salesforce/blip-image-captioning-large`, and pushes the captioned dataset to `{your-username}/media-search-demo-captioned`.

### Step 2 — `02_train_siglip.ipynb` (A10G GPU)
Loads the captioned dataset, applies LoRA to the SigLIP vision encoder, and trains for 3 epochs with:
- Batch size: 8, gradient accumulation: 4 (effective batch: 32)
- Learning rate: 2e-4 with cosine annealing + 50-step warmup
- Best checkpoint per epoch pushed to your specified output repo on HF Hub

For best results, use **500–2,000 domain-specific images** with quality captions. Training on fewer images risks overfitting.

### Step 3 — `03_verify_model.ipynb` (T4 GPU)
Compares cosine similarity of your fine-tuned model vs the baseline on 10 image-caption pairs.
**Pass condition:** fine-tuned avg similarity > baseline avg similarity.

## Post-Verification: Deploy Fine-Tuned Model

Once notebook 3 passes, update your production app to use the fine-tuned model:
1. Replace the base model name (`google/siglip-so400m-patch14-384`) with your fine-tuned model repo ID in your indexer
2. Delete any cached index files to force re-indexing with the new embeddings
3. Restart the application
