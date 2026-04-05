# SigLIP Fine-Tuning — Kaggle Version

This directory contains Kaggle-compatible versions of the three fine-tuning notebooks. They are identical to the Hugging Face (HF) Colab versions except for the secrets API and GPU recommendations.

---

## What's Different from the Colab Version

| | Google Colab | Kaggle |
|---|---|---|
| Secrets API | `google.colab.userdata` | `kaggle_secrets.UserSecretsClient` |
| Secrets UI | Key icon (🔑) in left sidebar | Add-ons → Secrets |
| Notebook 1 & 3 GPU | T4 | P100 |
| Notebook 2 GPU | A10G (24GB) | P100 (16GB) — reduced batch size |
| Free tier | Limited hours | 30 hours/week GPU |

---

## Prerequisites

- A [Kaggle](https://www.kaggle.com) account.
- A HF account with a [write-access token](https://huggingface.co/docs/hub/en/security-tokens#what-are-user-access-tokens).
- A source image/video dataset uploaded to HF Hub (see the root `README.md` for dataset setup).

---

## Adding Secrets

In each notebook, open **Add-ons → Secrets** and add the following key-value pairs — enable notebook access for each:

| Secret name | Value |
|---|---|
| `HF_TOKEN` | Your HF write-access token. |
| `HF_USERNAME` | Your HF username. |
| `SOURCE_DATASET` | Repo ID of your source dataset (for example, `your-username/your-media-dataset`). |
| `OUTPUT_DATASET` | *(Notebook 1 only)* Repo ID for the captioned dataset output (for example, `your-username/output_ds`). |
| `OUTPUT_REPO` | *(Notebook 2 only)* Repo ID for the fine-tuned model output (for example, `your-username/output_model`). |
| `FINETUNED_REPO` | *(Notebook 3 only)* Repo ID of the fine-tuned model from Notebook 2. |
| `CAPTIONED_DATASET` | *(Notebook 3 only)* Repo ID of the captioned dataset from Notebook 1. |

---

## Running the Notebooks

### Step 1 — Notebook 1: Generate Captions (P100 GPU)

1. Open `01_prepare_captions.ipynb` in Kaggle.
2. Set accelerator to **GPU P100** (Settings → Accelerator → GPU P100).
3. Add secrets listed above for Notebook 1.
4. In Cell 3, edit `IMAGE_COL` and `VIDEO_COL` to match your dataset's column names.
5. Run all cells top to bottom with **Shift+Enter**.

### Step 2 — Notebook 2: Fine-Tune SigLIP (P100 GPU)

1. Open `02_train_siglip.ipynb` in Kaggle.
2. Set accelerator to **GPU P100**.
3. Add secrets listed above for Notebook 2.
4. In Cell 3, the default values are already set for P100 (`BATCH_SIZE = 2`, `GRAD_ACCUM = 16`). Edit `NUM_EPOCHS` and `LORA_RANK` if needed.
5. Run all cells top to bottom with **Shift+Enter**.

> **Note:** Kaggle does not offer an A10G GPU. The P100 has 16GB VRAM. The default `BATCH_SIZE = 2` and `GRAD_ACCUM = 16` maintain an effective batch size of 32 while staying within the memory limit.

### Step 3 — Notebook 3: Verify the Model (P100 GPU)

1. Open `03_verify_model.ipynb` in Kaggle.
2. Set accelerator to **GPU P100**.
3. Add secrets listed above for Notebook 3.
4. Run all cells top to bottom with **Shift+Enter**.

**Pass condition:** Fine-tuned avg similarity > baseline avg similarity.

---

## Post-Verification

Once Notebook 3 passes, the fine-tuned model is on HF Hub. Update your local app to use it:
1. Replace the base model name (`google/siglip-so400m-patch14-384`) with your fine-tuned model repo ID in your indexer.
2. Delete any cached index files to force re-indexing with the new embeddings.
3. Restart the application.
