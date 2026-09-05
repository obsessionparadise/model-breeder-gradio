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
