# SigLIP Fine-Tuning

## What and Why

SigLIP is a vision-language model that learns to match images to text descriptions. Out of the box it works well on general images, but when you need it to understand a specific domain — your own photos, product images, or media library — the generic model may miss nuanced semantic connections that matter for your use case.

Fine-tuning adapts the model to your data by teaching it the vocabulary and visual patterns specific to your domain. Rather than retraining the entire model (expensive and slow), this project uses **[Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/lora)** — a technique that inserts a small set of trainable weight adjustments (~10M parameters) into the vision encoder's attention layers while keeping everything else frozen. The result is a model that understands your images significantly better, trained in a fraction of the time and cost.

The three notebooks in this repo implement a complete fine-tuning pipeline:
1. **Generate captions** for your image dataset using [Bootstrapping Language-Image Pre-training (BLIP)](https://huggingface.co/Salesforce/blip-image-captioning-large).
2. **Fine-tune SigLIP** on those image-caption pairs using LoRA.
3. **Verify** that the fine-tuned model outperforms the baseline on your data.

---

## Prerequisites

- A [Hugging Face (HF)](https://huggingface.co) account with a [write-access token](https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens).
- A source image/video dataset uploaded to HF Hub (see **Dataset Setup** below).
- Google Colab Pro. An A10G Graphics Processing Unit (GPU) is required for the training notebook.

## Dataset Setup

Create your own image dataset on HF Hub. As a reference, see [`mugdhav/media-search-demo-files`](https://huggingface.co/datasets/mugdhav/media-search-demo-files) for the expected structure — a dataset with an `image` column (and optionally a `video` column for video frames).

To create your own:
1. Upload your images/videos to a new HF dataset repo using the [`huggingface_hub` Python library](https://huggingface.co/docs/huggingface_hub) or the [Hub web UI](https://huggingface.co/new-dataset).
2. Note the dataset repo ID — the `username/dataset-name` portion of the dataset URL (for example, `your-username/your-media-dataset` from `https://huggingface.co/datasets/your-username/your-media-dataset`).
3. Add that ID as a Colab Secret named `SOURCE_DATASET` before running Notebook 1 (see Step 3 below).

For best results, use **500–2,000 domain-specific images** with quality captions.

> **Note:** If your source dataset is gated, you must first visit the dataset page on HF and accept the access request with your account. The notebooks authenticate using your `HF_TOKEN` — the token alone is not sufficient without prior gate acceptance.

---

## Running the Notebooks

### Step 1 — Sign in to Google Colab

Go to [Google Colab](https://colab.research.google.com) and sign in with your Google account.

### Step 2 — Upload the notebooks

In Colab, go to **File → Upload notebook** and upload the notebooks from this repo one at a time as needed:
- `01_prepare_captions.ipynb`
- `02_train_siglip.ipynb`
- `03_verify_model.ipynb`

### Step 3 — Run Notebook 1: Generate Captions (T4 GPU)

1. Open `01_prepare_captions.ipynb`.
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).
3. Add the following secrets via the key icon (🔑) in the left sidebar — enable notebook access for each:
   - `HF_TOKEN` — your HF [write-access token](https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens).
   - `HF_USERNAME` — your HF username.
   - `SOURCE_DATASET` — repo ID of your source dataset (for example, `your-username/your-media-dataset`).
   - `OUTPUT_DATASET` — repo ID where the captioned dataset will be saved (for example, `your-username/output_ds`).
   - If your dataset is gated, accept the access request on the HF Hub page first.
4. In Cell 3, edit `IMAGE_COL` and `VIDEO_COL` directly in the code to match your dataset's column names.
5. Run all cells top to bottom with **Shift+Enter**.

This generates text captions for every image/video in your dataset using BLIP and pushes the result to your HF Hub.

### Step 4 — Run Notebook 2: Fine-Tune SigLIP (A10G GPU)

1. Open `02_train_siglip.ipynb`.
2. **Upgrade** the runtime to **A10G** (Runtime → Change runtime type → A10G small).
3. Add the following secrets via the key icon (🔑) in the left sidebar — enable notebook access for each:
   - `HF_TOKEN` — your HF [write-access token](https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens).
   - `HF_USERNAME` — your HF username.
   - `SOURCE_DATASET` — repo ID of the captioned dataset output from Notebook 1.
   - `OUTPUT_REPO` — repo ID where the fine-tuned model will be saved (for example, `your-username/output_model`).
4. In Cell 3, edit `BATCH_SIZE`, `GRAD_ACCUM`, `NUM_EPOCHS`, and `LORA_RANK` directly in the code if needed. If you run out of memory, set `BATCH_SIZE = 4` and `GRAD_ACCUM = 8`.
5. Run all cells top to bottom with **Shift+Enter**.

Training runs for 3 epochs with LoRA adapters on the vision encoder. The best checkpoint per epoch is pushed to your HF Hub.

### Step 5 — Run Notebook 3: Verify the Model (T4 GPU)

1. Open `03_verify_model.ipynb`.
2. Set the runtime to **T4 GPU**.
3. Add the following secrets via the key icon (🔑) in the left sidebar — enable notebook access for each:
   - `HF_TOKEN` — your HF [write-access token](https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens).
   - `HF_USERNAME` — your HF username.
   - `FINETUNED_REPO` — repo ID of the fine-tuned model output from Notebook 2.
   - `CAPTIONED_DATASET` — repo ID of the captioned dataset output from Notebook 1.
   - If your dataset is gated, accept the access request on the HF Hub page first.
4. Run all cells top to bottom with **Shift+Enter**.

The notebook compares [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of your fine-tuned model vs the baseline on 10 image-caption pairs.
**Pass condition:** fine-tuned avg similarity > baseline avg similarity.

---

## Post-Verification: Deploy the Fine-Tuned Model

Once Notebook 3 passes, update your production app to use the fine-tuned model:
1. Replace the base model name (`google/siglip-so400m-patch14-384`) with your fine-tuned model repo ID in your indexer.
2. Delete any cached index files to force re-indexing with the new embeddings.
3. Restart the application.

---

## Technical Overview

- **Base model:** [`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384) (400M params, 384px).
- **Method:** LoRA (rank=16) on vision encoder attention layers (`q_proj`, `v_proj`) — ~10M trainable params; text encoder frozen.
- **Loss:** SigLIP sigmoid contrastive loss.
- **Training:** 3 epochs, batch size 8 × gradient accumulation 4 (effective batch 32), lr 2e-4 with cosine annealing.
