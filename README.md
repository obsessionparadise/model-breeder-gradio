# 🧬 Model Breeder — Gradio Edition

A Gradio app version of the checkpoint-merging + image-generation workflow from
the source notebook. Streaming, NaN-safe, OOM-guarded, cancellable, works with
SD1.5, SDXL, Z-Image Turbo, Krea 2, and (experimentally) Anima, runs
identically **locally** or **on Google Colab** (with a public share link).

### What's included
| Tab | What it does |
|---|---|
| 📁 Files | Upload checkpoints / LoRAs / VAEs, browse `input/` and `output/` |
| 🧬 Checkpoint × Checkpoint | 8 merge methods (Weighted Sum, Add Difference, Sigmoid Blend, SLERP, Sum Twice, Triple Sum, DARE, XDARE), per-block weights, optional VAE baking, prediction-type guard |
| 🎨 Checkpoint + LoRA | Bake up to 4 LoRAs (LoRA / LoKr / LoHA) in one pass, per-block targeting, optional final blend back toward the base |
| 🧪 VAE Baker | Bake a standalone VAE into a checkpoint |
| 🔍 Metadata Reader | Read the recipe metadata this tool embeds in files it produces |
| 🖼️ Image Generator | SD1.5/SDXL single-file checkpoints, **or** Z-Image Turbo / Krea 2 / Anima via a HF repo id or local diffusers-folder path, with separate Text Encoder and VAE override fields; images stream into the gallery and save to disk one at a time as they finish, not all at once at the end |

Every merge/bake/generation is **preflight-checked** (RAM and disk space,
before any destructive work starts), **cancellable** mid-run via a Cancel
button, **single-flight** (only one heavy job runs at a time, process-wide,
so two jobs can't double up and exhaust memory), and **crash-safe** (any
failure — including a cancel — cleans up its own partial output file rather
than leaving a corrupt one behind). See **`CRITIQUE.md`** for the full
systematic audit this was built against: every identified weakness, why it
mattered, the fix, and how it was tested.

> **Note on scope:** this app ports the general-purpose merging/baking/generation
> *engine* faithfully. It intentionally does not include the source notebook's
> hard-coded character/anatomy recipe library — the block-weight presets here
> (Weighted Sum, UNet-only, Style-preserve, Conservative, Custom) are generic,
> content-neutral numeric presets. You can always type any 20-value block
> weight vector directly into the text box for full manual control.

---

## Generating with Z-Image Turbo / Krea 2 / Anima

These three ship as a **separate transformer + text-encoder + VAE**, not one
fused checkpoint file — the same modular layout ComfyUI uses
(`diffusion_models/` + `text_encoders/` + `vae/`). In the Image Generator
tab, pick one from the **Architecture** dropdown and the UI switches from
the single-file checkpoint picker to a **HF repo id or local diffusers-
folder path** field, pre-filled with the correct default repo:

| Architecture | Default repo | Recommended steps / CFG | Text encoder | VAE |
|---|---|---|---|---|
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` | 9 / 0.0 | Qwen3-4B | Flux-derived, 16ch |
| Krea 2 Turbo | `krea/Krea-2-Turbo` (gated — accept the license on the repo page first) | 8 / 0.0 | Qwen3-VL-4B-Instruct | Qwen-Image VAE, 16ch |
| Krea 2 Raw | `krea/Krea-2-Raw` (gated) | 52 / 3.5 | Qwen3-VL-4B-Instruct | Qwen-Image VAE, 16ch |
| Anima Base v1 (experimental) | _left blank — see below_ | ~40 / ~5.0 (community-reported) | Qwen3-0.6B-**Base** (not instruct) | Qwen-Image VAE, 16ch |

Switching architecture automatically resets the Steps/CFG sliders to the
values above. Krea 2 Raw's own model card says it's "not recommended for
inference use directly" — it exists as a base for fine-tuning/LoRA
training (LoRAs trained on Raw are meant to be used on Turbo instead).

Leave the repo field as the pre-filled default and the whole pipeline
downloads and caches automatically on first load (standard `diffusers`
behavior — needs internet access, and for Krea 2 you must accept the
license on the repo's Hugging Face page while logged in before the
download will work). Point it at a local folder under
`input/diffusers_repos/` instead if you've already downloaded a snapshot.

The **Text Encoder override** and **VAE override** fields let you swap an
individual component without touching the rest — e.g. try a different
fine-tuned VAE against the same transformer. The VAE override accepts a
repo id, a local folder, a filename from `input/vae/`, *or* a single
`.safetensors` file (VAEs are self-contained weights). The Text Encoder
override only accepts a repo id or a local folder under
`input/text_encoders/` — these are full instruction-tuned LLMs (Qwen3
variants), and loading one needs its tokenizer alongside the weights,
which a bare `.safetensors` file doesn't carry.

**Anima's repo field is deliberately left blank, not pre-filled.** Its
official repo (`circlestone-labs/Anima`) ships in ComfyUI's split-file
format (separate `diffusion_models/` / `text_encoders/` / `vae/` safetensors,
no `model_index.json`) — `diffusers`' `from_pretrained()` can't load that
layout directly, and there's no confirmed official diffusers pipeline for
Anima as of this writing. If you find or build a diffusers-format
conversion, you can type its repo id in as a best-effort attempt against
`Cosmos2TextToImagePipeline` (the architecture Anima is derived from). One
specific gotcha if you get this far: Anima's text encoder is the **base**
(non-instruction-tuned) Qwen3-0.6B checkpoint, not the standard chat/
instruct variant — using the wrong one loads without error but produces
poor results. If your installed `diffusers` doesn't have a needed pipeline
class yet for any of these three, you'll get a clear message telling you
to `pip install -U diffusers transformers` rather than a cryptic crash.

Generation streams live: a low-res approximate preview updates during
denoising itself (not just between images), a real progress bar tracks
step-by-step progress across the whole batch, and each finished image
saves to `output/images/` and appears in the gallery the moment it's done
— cancelling mid-batch keeps everything already generated.

## Reliability

- **Preflight checks** — every merge/bake/generate/checkpoint-load estimates
  the RAM and disk space it needs (including, for the newer architectures,
  the actual Hugging Face cache directory a download lands in — often a
  different disk than the app's own workspace) and refuses to start (with
  a clear message) rather than OOMing or filling the disk halfway through.
- **Cancellable** — every long-running job, including checkpoint loading,
  has a Cancel button. Cancelling a merge/bake cleans up its partial
  output file; cancelling a multi-image generation keeps every image
  already finished. (One honestly-stated limit: a Hugging Face download
  already in flight can't be interrupted mid-transfer — the library this
  app builds on doesn't expose a hook for that — so a cancel there takes
  effect once that transfer finishes rather than instantly.)
- **Single-flight, even if you close the tab mid-job** — only one heavy
  job runs at a time across the whole app; starting a second one while
  another is running gets an immediate "already running" message instead
  of silently competing for memory. This holds even if the job that's
  running gets abandoned (browser tab closed, connection dropped) — the
  lock stays correctly held for as long as the work is actually still
  happening in the background, not just for as long as someone's watching.
- **Streaming saves** — checkpoints are written tensor-by-tensor with
  periodic `fsync` to physical disk, and generated images save to disk one
  at a time as each finishes rather than only after a whole batch completes.
- **GPU-OOM-safe** — a CUDA out-of-memory error during generation is caught,
  the GPU cache is cleared, and you get an actionable message instead of a
  crashed session.

See **`CRITIQUE.md`** for the full audit and how each of these was tested.

## Run locally

```bash
git clone <this folder, or just copy app.py + engine.py + requirements.txt>
cd model_breeder_gradio
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python app.py
```

Then open the printed `http://127.0.0.1:7860` URL. Useful flags:

```bash
python app.py --share                 # also get a public gradio.live link
python app.py --server-name 0.0.0.0   # expose on your LAN
python app.py --port 8080             # custom port
```

If you have an NVIDIA GPU, install the CUDA build of torch **before**
`pip install -r requirements.txt` (see the note at the top of that file) —
otherwise you'll get the CPU build, which works but is much slower for
image generation.

## Run on Google Colab

In a Colab cell:

```python
!pip install -q gradio diffusers transformers accelerate safetensors omegaconf torchsde
# upload app.py and engine.py to the Colab file browser, or:
# !git clone <your repo> && %cd <your repo>
!python app.py --share
```

Colab is auto-detected — `share=True` turns on automatically even if you
forget the flag, so a public `https://xxxx.gradio.live` link appears in the
cell output. Click it to use the full UI from your phone or another tab.

**Tip:** to keep your merged models across Colab sessions, mount Google
Drive and point the workspace at it before launching:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['MODEL_BREEDER_DIR'] = '/content/drive/MyDrive/model_breeder_data'
!python app.py --share
```

## Workspace layout

By default the app creates `./model_breeder_data/` next to `app.py`,
organized into dedicated subfolders instead of one flat `input/` and one
flat `output/`:

```
input/
    checkpoints/       full SD1.5 / SDXL checkpoints (.safetensors, .ckpt)
    loras/              LoRA / LoKr / LoHA files
    vae/                 standalone VAE files
    text_encoders/      local folders for text-encoder overrides
    diffusers_repos/    local diffusers-format snapshots (Z-Image/Krea2/Anima)
output/
    checkpoints/         merge / LoRA-bake / VAE-bake / blend results
    images/               generated images
```

Each folder has its own short `README.md` explaining exactly what belongs
there. The **Files** tab's upload widget has a destination selector — leave
it on **Auto-detect** and it sorts each upload into the right subfolder by
inspecting its contents (the same way the app already tells checkpoints,
LoRAs, and VAEs apart internally), or pick a specific folder to override
that. Override the base location with the `MODEL_BREEDER_DIR` environment
variable (see the Drive example above) — the whole subfolder tree moves
with it.

**These folders are set up automatically — you never have to create them
yourself.** Three ways this happens, all equivalent and all safe to repeat:

1. The delivered project already ships with the full `model_breeder_data/`
   tree on disk (each leaf folder has a short `README.md` explaining what
   goes there — git/zip can't represent a truly empty folder, so that's
   what holds the place).
2. `app.py` calls `setup_workspace()` the moment it's imported, before the
   UI even builds — so even a bare checkout of just the `.py` files (e.g.
   pasted into a fresh Colab cell) gets the full folder tree on first launch.
3. You can also run it explicitly as its own install step, which is useful
   if you want the folders to exist *before* first launch (e.g. to drop
   model files in ahead of time):
   ```bash
   python setup_workspace.py                       # uses the default location
   python setup_workspace.py --dir /custom/path     # or a specific one
   ```
   This respects `MODEL_BREEDER_DIR` the same way `app.py` does, so both
   always agree on where the workspace lives. It never overwrites or
   deletes anything already there — safe to run as many times as you like.

To chain merges (e.g. merge A+B, then bake a LoRA into the result, then
blend that into C), copy or move the desired file from `output/checkpoints/`
into `input/checkpoints/` between steps — this keeps the two folders
unambiguous about "raw ingredients" vs. "results," matching how the
original notebook's Colab cells were organized into separate stages. You
don't strictly have to, though — the Image Generator tab lists checkpoints
from both `input/` and `output/` directly, so a fresh merge is immediately
usable without moving anything.

## Design system

The UI follows Material Design 3: a hand-tuned tonal color system (a
"Synthesis Violet" primary for merge/breed actions, "Catalyst Amber" for
generation/bake actions, both built as full 50–950 tonal ramps), Material's
own Roboto / Roboto Mono type pairing (mono reserved for technical values —
seeds, filenames, block-weight vectors, logs), a rounded shape scale (pill
buttons, 20px card corners, 14px input corners), and soft layered elevation
shadows instead of hard borders. Every tab is broken into labeled Material
surface cards instead of one long scrolling form, with a Material-style
top app bar and underline-indicator tabs. See `theme.py` for the full token
system if you want to reskin it — it's a self-contained `gr.Theme` plus one
CSS string, decoupled from the app logic in `app.py`.

## Files in this project

- `app.py` — Gradio UI and all wiring (`python app.py` to run)
- `engine.py` — the merge/bake/VAE/metadata/generation engine (no UI code, reusable from a script or notebook cell too)
- `guards.py` — RAM/disk preflight checks, cancellation, single-flight job lock, GPU-OOM handling, input sanitization
- `theme.py` — the Material 3 design system (color tokens, typography, shape, elevation) as a `gr.Theme` + CSS
- `setup_workspace.py` — creates the `input/`/`output/` folders (also importable; `app.py` calls it automatically)
- `CRITIQUE.md` — the systematic engineering audit: every identified weakness, the fix, and how it was tested
- `requirements.txt` — pip dependencies
- `LICENSE` — MIT
- `model_breeder_data/input/`, `model_breeder_data/output/` — the workspace folders, shipped pre-created

## License

MIT — see [`LICENSE`](./LICENSE). This project's own code is freely usable;
it doesn't grant any rights to the model weights you merge, bake, or
generate with — those remain under whatever license each model's own
publisher attaches to it (check the model card before redistributing
anything you produce with this tool, especially the gated Krea 2 repos).
