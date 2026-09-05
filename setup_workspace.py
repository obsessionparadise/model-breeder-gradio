"""
Model Breeder — Workspace setup
==================================
Creates (and documents) the folder structure this app reads and writes.
This is intentionally a separate, tiny, dependency-free module so folder
setup can happen in three equivalent ways and always agree with each other:

  1. Automatically, every time `app.py` starts (it imports and calls
     `setup_workspace()` before building the UI) — so the app always has
     somewhere to write even on a completely fresh checkout.
  2. Explicitly, as an install step: `python setup_workspace.py` — useful
     if you want the folders to exist *before* first launch (e.g. to drop
     model files in ahead of time, or as part of a scripted install).
  3. Implicitly shipped: the delivered project already includes the whole
     folder tree on disk (with a short README placeholder in each leaf
     folder, since git/zip can't represent an empty directory) — so opening
     the project folder shows the expected layout immediately, before
     you've run anything at all.

All three are idempotent and safe to run repeatedly — this never deletes or
overwrites anything that's already there.

Layout:
    input/
        checkpoints/       full SD1.5 / SDXL checkpoints (.safetensors, .ckpt)
        loras/              LoRA / LoKr / LoHA files
        vae/                 standalone VAE files
        text_encoders/      local HF-format folders for text-encoder overrides
        diffusers_repos/    local diffusers-format snapshots (Z-Image/Krea2/Anima)
    output/
        checkpoints/        merge / LoRA-bake / VAE-bake / blend results
        images/              generated images
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TOP_LEVEL_README = """\
# Model Breeder workspace

- **input/** — put your source files here (see the README in each subfolder
  for what goes where).
- **output/** — every merge, bake, and generated image lands here
  automatically.

This file exists only so the folder isn't empty before your first upload —
it's safe to ignore or delete.
"""

SUBFOLDER_READMES = {
    ('input', 'checkpoints'): """\
# input/checkpoints/

Full SD1.5 or SDXL model files — `.safetensors` or `.ckpt`. This is what the
Checkpoint x Checkpoint Merger, Checkpoint + LoRA Merger, and VAE Baker tabs
list in their model dropdowns, and what the Image Generator tab lists when
you pick the "Auto-detect (SD1.5 / SDXL)", "Stable Diffusion 1.5", or
"Stable Diffusion XL" architecture.

Not for Z-Image Turbo / Krea 2 / Anima — those don't ship as a single file;
see `input/diffusers_repos/` instead.
""",
    ('input', 'loras'): """\
# input/loras/

LoRA files — standard LoRA, LoKr, or LoHA, as `.safetensors`. Listed in the
Checkpoint + LoRA Merger tab's LoRA 1-4 dropdowns.
""",
    ('input', 'vae'): """\
# input/vae/

Standalone VAE files (`.safetensors`) with no attached UNet/text-encoder —
just the encoder/decoder. Listed in the VAE Baker tab and as the optional
"Bake VAE" choice in the Checkpoint x Checkpoint Merger tab.

For the newer architectures' VAE override field in the Image Generator tab,
a file placed here can also be selected by filename directly.
""",
    ('input', 'text_encoders'): """\
# input/text_encoders/

Local text-encoder folders for the Image Generator tab's "Text Encoder
override" field (Z-Image Turbo / Krea 2 / Anima only). Each subfolder here
should be a *complete* local HF-format snapshot -- the model weights *and*
its tokenizer/config files together, e.g.:

    input/text_encoders/qwen3-4b/
        config.json
        tokenizer.json
        tokenizer_config.json
        model.safetensors (or sharded .safetensors files + an index)
        ...

A bare `.safetensors` weights file on its own can't be used here -- these
are full instruction-tuned LLMs (Qwen3 variants), and loading one needs its
tokenizer alongside the weights, which a single weights file doesn't carry.
Typing a Hugging Face repo id directly into the override field (so it
downloads automatically) is usually simpler than assembling this by hand.
""",
    ('input', 'diffusers_repos'): """\
# input/diffusers_repos/

Local diffusers-format snapshots for Z-Image Turbo, Krea 2, or Anima -- the
newer architectures that ship as a whole repo (transformer + text_encoder +
tokenizer + vae + scheduler subfolders + a model_index.json), not one file.
Each subfolder here should be one complete snapshot, e.g.:

    input/diffusers_repos/z-image-turbo/
        model_index.json
        transformer/...
        text_encoder/...
        tokenizer/...
        vae/...
        scheduler/...

Point the Image Generator tab's repo/folder field at this subfolder's full
path. Most people won't need this -- typing the Hugging Face repo id
directly into that same field downloads and caches everything
automatically. This is for offline use, a custom local build, or a repo
you've already downloaded via `huggingface-cli download --local-dir`.
""",
    ('output', 'checkpoints'): """\
# output/checkpoints/

Results of merges, LoRA bakes, VAE bakes, and final blends land here
automatically. To chain operations (e.g. merge A+B, then bake a LoRA into
the result), copy or move the file you want to reuse from here back into
`input/checkpoints/` -- this keeps the two folders unambiguous about "raw
ingredients" vs. "results." The Image Generator tab also lists checkpoints
from here directly (prefixed `[output]`) so you can generate from a fresh
merge without moving it anywhere first.
""",
    ('output', 'images'): """\
# output/images/

Every image the Image Generator tab produces is saved here automatically,
one file per image, the moment it finishes rendering -- not held back until
a whole batch completes.
""",
}


@dataclass(frozen=True)
class Workspace:
    data_dir: Path
    input_dir: Path
    checkpoints_dir: Path
    loras_dir: Path
    vae_dir: Path
    text_encoders_dir: Path
    diffusers_repos_dir: Path
    output_dir: Path
    output_checkpoints_dir: Path
    output_images_dir: Path

    def __iter__(self):
        # Backward-compatible unpacking: data_dir, input_dir, output_dir = setup_workspace()
        # kept working for anything still expecting the old 3-tuple shape.
        return iter((self.data_dir, self.input_dir, self.output_dir))


def setup_workspace(data_dir=None, verbose: bool = False) -> Workspace:
    """Create the full input/output folder tree under data_dir if it
    doesn't already exist, and drop a short README in each leaf folder
    (and one at the top level) if missing. Returns a Workspace with every
    folder as a resolved Path attribute.

    data_dir defaults to the MODEL_BREEDER_DIR environment variable, falling
    back to ./model_breeder_data next to the current working directory --
    the same resolution app.py itself uses, so this can be called standalone
    before app.py ever runs and both will agree on the location.
    """
    if data_dir is None:
        data_dir = os.environ.get('MODEL_BREEDER_DIR', Path.cwd() / 'model_breeder_data')
    data_dir = Path(data_dir)
    input_dir = data_dir / 'input'
    output_dir = data_dir / 'output'

    dirs = {
        'input_dir': input_dir,
        'checkpoints_dir': input_dir / 'checkpoints',
        'loras_dir': input_dir / 'loras',
        'vae_dir': input_dir / 'vae',
        'text_encoders_dir': input_dir / 'text_encoders',
        'diffusers_repos_dir': input_dir / 'diffusers_repos',
        'output_dir': output_dir,
        'output_checkpoints_dir': output_dir / 'checkpoints',
        'output_images_dir': output_dir / 'images',
    }

    created = []
    for d in dirs.values():
        existed = d.exists()
        d.mkdir(parents=True, exist_ok=True)
        if not existed:
            created.append(d)

    _write_if_missing(input_dir / 'README.md', TOP_LEVEL_README)
    _write_if_missing(output_dir / 'README.md', TOP_LEVEL_README)
    for (top, sub), content in SUBFOLDER_READMES.items():
        target_dir = input_dir / sub if top == 'input' else output_dir / sub
        _write_if_missing(target_dir / 'README.md', content)

    if verbose:
        print(f'Workspace: {data_dir}')
        for name, d in dirs.items():
            tag = '(created)' if d in created else '(already existed)'
            print(f'  {d}  {tag}')

    return Workspace(data_dir=data_dir, **dirs)


def _write_if_missing(path: Path, content: str) -> None:
    try:
        if not path.exists():
            path.write_text(content, encoding='utf-8')
    except OSError:
        # Never let a missing README block the app from starting -- the
        # folders themselves are what actually matters.
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Create the input/output workspace folder tree Model Breeder uses.')
    parser.add_argument('--dir', type=str, default=None,
                         help='Workspace directory (default: $MODEL_BREEDER_DIR or ./model_breeder_data)')
    args = parser.parse_args()
    ws = setup_workspace(args.dir, verbose=True)
    print()
    print('Done. Drop checkpoints into input/checkpoints/, LoRAs into input/loras/, '
          'VAEs into input/vae/, then run: python app.py')
