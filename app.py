"""
Model Breeder — Gradio Edition
================================
A Gradio front-end for the streaming Stable Diffusion checkpoint-merging
engine in engine.py: Checkpoint x Checkpoint merging (8 methods + per-block
weights + VAE baking), Checkpoint + LoRA baking, standalone VAE baking,
merge-metadata inspection, and SD1.5/SDXL/Z-Image/Krea2/Anima image
generation.

Run locally:
    pip install -r requirements.txt
    python app.py

Run on Google Colab:
    !pip install -q -r requirements.txt
    !python app.py --share
(or simply run this file in a notebook cell — it auto-detects Colab and
turns on share=True by itself.)

Data lives in ./model_breeder_data/{input,output} next to this file by
default, organized into subfolders (input/checkpoints, input/loras,
input/vae, input/text_encoders, input/diffusers_repos, output/checkpoints,
output/images — see setup_workspace.py). Override the base location with
the MODEL_BREEDER_DIR environment variable, e.g. point it at a mounted
Google Drive folder for persistence across sessions.
"""
from __future__ import annotations

import argparse
import os
import traceback
import warnings
from pathlib import Path

import gradio as gr

# Gradio 6 moved theme/css to launch()-time but still honors them on
# Blocks() for backward compatibility (verified: CSS renders correctly
# either way) — this just silences the resulting deprecation notice.
warnings.filterwarnings('ignore', message='.*moved from the Blocks constructor.*')

import engine
import guards
from theme import MATERIAL_THEME, MATERIAL_CSS
from setup_workspace import setup_workspace

# ─────────────────────────────────────────────────────────────────────────
# Workspace — the full folder tree is created (with a short README in each
# leaf folder) the moment this module is imported, so the app always has
# somewhere to read and write even on a completely fresh checkout.
# See setup_workspace.py for the layout and setup_workspace() itself.
# ─────────────────────────────────────────────────────────────────────────
WS = setup_workspace()
DATA_DIR = WS.data_dir
INPUT_DIR = WS.input_dir
CHECKPOINTS_DIR = WS.checkpoints_dir
LORAS_DIR = WS.loras_dir
VAE_DIR = WS.vae_dir
TEXT_ENCODERS_DIR = WS.text_encoders_dir
DIFFUSERS_REPOS_DIR = WS.diffusers_repos_dir
OUTPUT_DIR = WS.output_dir
OUTPUT_CKPT_DIR = WS.output_checkpoints_dir
OUTPUT_IMAGES_DIR = WS.output_images_dir


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────
# Shared small helpers
# ─────────────────────────────────────────────────────────────────────────
def _fmt_size(nbytes: int) -> str:
    mb = nbytes / 1e6
    return f'{mb/1000:.2f} GB' if mb >= 1000 else f'{mb:.1f} MB'


def list_files_markdown() -> str:
    def _table(folder):
        files = sorted(Path(folder).glob('*'))
        files = [f for f in files if f.is_file() and f.name != 'README.md']
        if not files:
            return '*(empty)*'
        rows = '\n'.join(f'| {f.name} | {_fmt_size(f.stat().st_size)} |' for f in files)
        return f'| File | Size |\n|---|---|\n{rows}'

    def _dirlist(folder):
        dirs = engine.list_local_dirs(folder)
        return ', '.join(f'`{d}`' for d in dirs) if dirs else '*(none)*'

    return (
        f'### input/checkpoints/\n\n{_table(CHECKPOINTS_DIR)}\n\n'
        f'### input/loras/\n\n{_table(LORAS_DIR)}\n\n'
        f'### input/vae/\n\n{_table(VAE_DIR)}\n\n'
        f'### input/text_encoders/  _(local folders — see README inside)_\n\n{_dirlist(TEXT_ENCODERS_DIR)}\n\n'
        f'### input/diffusers_repos/  _(local folders — see README inside)_\n\n{_dirlist(DIFFUSERS_REPOS_DIR)}\n\n'
        f'### output/checkpoints/\n\n{_table(OUTPUT_CKPT_DIR)}\n\n'
        f'### output/images/\n\n{_table(OUTPUT_IMAGES_DIR)}'
    )


UPLOAD_DESTINATIONS = {
    'Auto-detect (recommended)': None,
    'Checkpoint (SD1.5 / SDXL)': CHECKPOINTS_DIR,
    'LoRA': LORAS_DIR,
    'VAE': VAE_DIR,
}


def _classify_upload(src: Path) -> Path:
    """Content-sniff a single uploaded file to pick its destination
    subfolder. Only used when the user leaves the destination selector on
    Auto-detect; a LoRA/VAE are structurally distinguishable by their
    tensor keys, so this is reliable — anything that isn't clearly one of
    those two defaults to checkpoints/."""
    try:
        if engine.is_lora_file(src):
            return LORAS_DIR
        if engine.is_vae_file(src):
            return VAE_DIR
    except Exception:
        pass
    return CHECKPOINTS_DIR


def upload_to_input(files, destination_label):
    errors = []
    forced_dest = UPLOAD_DESTINATIONS.get(destination_label)
    if files:
        for f in files:
            src = Path(f.name if hasattr(f, 'name') else f)
            dest_dir = forced_dest if forced_dest is not None else _classify_upload(src)
            try:
                guards.atomic_copy_into(src, dest_dir)  # CRITIQUE.md C3: atomic, never a partial file
            except Exception as e:
                # One bad upload (e.g. disk full mid-copy) shouldn't silently
                # lose the rest of the batch — but the user needs to know.
                errors.append(f'⚠ Failed to upload {src.name}: {e}')
    md = list_files_markdown()
    if errors:
        md = '\n'.join(errors) + '\n\n' + md
    return md


def refresh_all_dropdowns():
    ckpts_in = engine.list_checkpoints(CHECKPOINTS_DIR) or []
    ckpts_out = engine.list_checkpoints(OUTPUT_CKPT_DIR) or []
    loras = engine.list_loras(LORAS_DIR) or []
    vaes = engine.list_vaes(VAE_DIR) or []
    # Image Generator can load either a fresh output/ checkpoint (a merge/
    # bake result) or an input/ checkpoint directly (no need to route a
    # plain checkpoint through a no-op merge just to generate from it).
    gen_ckpts = sorted([f'[input] {n}' for n in ckpts_in] + [f'[output] {n}' for n in ckpts_out])
    meta_ckpts = sorted([f'[input] {n}' for n in ckpts_in] + [f'[output] {n}' for n in ckpts_out])
    return ckpts_in, loras, vaes, gen_ckpts, meta_ckpts


def _resolve_meta_choice(choice: str) -> Path:
    if choice.startswith('[input] '):
        return CHECKPOINTS_DIR / choice[len('[input] '):]
    if choice.startswith('[output] '):
        return OUTPUT_CKPT_DIR / choice[len('[output] '):]
    return CHECKPOINTS_DIR / choice


def _resolve_gen_ckpt_choice(choice: str) -> Path:
    if choice.startswith('[input] '):
        return CHECKPOINTS_DIR / choice[len('[input] '):]
    if choice.startswith('[output] '):
        return OUTPUT_CKPT_DIR / choice[len('[output] '):]
    return OUTPUT_CKPT_DIR / choice


def _format_error(e: Exception) -> str:
    """CRITIQUE.md B3: expected, typed errors (resource guards, cancellation,
    plain validation) show just their message; anything else shows the full
    traceback so real bugs stay fully diagnosable.

    Uses the exception object's own __traceback__ rather than
    traceback.format_exc() deliberately: several callers retrieve `e` from
    a background-thread job result (engine.run_with_live_log /
    run_generation_stream) well after that thread's own except block has
    exited — at that point there is no "currently handled exception" in
    the calling frame, and traceback.format_exc() silently returns the
    useless string 'NoneType: None' instead of raising or warning. Reading
    the traceback off the exception object itself works correctly whether
    it was caught synchronously in this frame or handed back from a worker
    thread — verified both cases explicitly (see CRITIQUE.md round 2)."""
    if isinstance(e, (guards.InsufficientResourceError, guards.GpuOutOfMemoryError, ValueError)):
        return f'⚠ {e}'
    if isinstance(e, guards.OperationCancelled):
        return f'⏹ Cancelled — {e}'
    return ''.join(traceback.format_exception(type(e), e, e.__traceback__))


# ─────────────────────────────────────────────────────────────────────────
# Tab: Checkpoint x Checkpoint Merger
# ─────────────────────────────────────────────────────────────────────────
def merge_ui_recipe_change(recipe_name):
    return engine.CKPT_RECIPES.get(recipe_name, engine.CKPT_RECIPES['Custom (edit manually below)'])


def do_merge(model_a, model_b, model_c, method_label, alpha, beta,
             bw_text, pred_a, pred_b, vae_choice, out_name, fp16,
             use_alpha_ratio, alpha_ratio_text, use_beta_ratio, beta_ratio_text, dare_seed):
    log_box = ''
    # Yield shape: (log, progress, ckpt_a_dd_update, cancel_token_state, button_update)
    try:
        if not model_a or not model_b:
            yield 'Select both Model A and Model B first.', 0, gr.update(), None, gr.update()
            return
        if model_a == model_b:
            yield 'Model A and Model B must be different files.', 0, gr.update(), None, gr.update()
            return
        mode = engine.METHOD_LABEL_TO_MODE[method_label]
        pa, pb = CHECKPOINTS_DIR / model_a, CHECKPOINTS_DIR / model_b
        pc = CHECKPOINTS_DIR / model_c if model_c and model_c != '(none)' else None
        if mode == 1 and pc is None:
            yield 'Add Difference requires Model C.', 0, gr.update(), None, gr.update()
            return
        if mode in (4, 5) and pc is None:
            name = 'Sum Twice' if mode == 4 else 'Triple Sum'
            yield f'{name} requires Model C.', 0, gr.update(), None, gr.update()
            return

        bw_vec, err = engine.parse_bw(bw_text)
        if err and mode not in (4, 5, 6, 7):
            yield f'Block weights error: {err}', 0, gr.update(), None, gr.update()
            return
        if bw_vec is None:
            bw_vec = [1.0] * 20

        alpha_ratio_vec = None
        beta_ratio_vec = None
        if mode in (4, 5, 6, 7):
            if use_alpha_ratio:
                alpha_ratio_vec, err = engine.parse_ratio(alpha_ratio_text)
                if err:
                    yield f'Alpha block-ratio error: {err}', 0, gr.update(), None, gr.update()
                    return
            else:
                alpha_ratio_vec = [alpha] * 20
            if mode in (4, 5) and use_beta_ratio:
                beta_ratio_vec, err = engine.parse_ratio(beta_ratio_text)
                if err:
                    yield f'Beta block-ratio error: {err}', 0, gr.update(), None, gr.update()
                    return
            else:
                beta_ratio_vec = [beta] * 20
            if mode in (6, 7) and beta >= 0.98:
                yield 'Droprate (Beta) must be below 1.0.', 0, gr.update(), None, gr.update()
                return

        vae_path = VAE_DIR / vae_choice if vae_choice and vae_choice != '(none)' else None
        nm = guards.safe_filename(out_name, 'merged_checkpoint.safetensors')
        if not nm.endswith('.safetensors'):
            nm += '.safetensors'
        out_path = OUTPUT_CKPT_DIR / nm

        cancel_token = guards.CancelToken()

        def _target(progress_cb, log_cb):
            return engine.merge_checkpoints(
                pa, pb, pc, float(alpha), mode, out_path, bool(fp16), vae_path, bw_vec,
                progress_cb, log_cb, base_pred=pred_a, merge_pred=pred_b,
                beta=float(beta), alpha_ratio_vec=alpha_ratio_vec, beta_ratio_vec=beta_ratio_vec,
                dare_seed=int(dare_seed), cancel_token=cancel_token)

        try:
            yield log_box, 0, gr.update(), cancel_token, gr.update(interactive=False)
            state = None
            for state in engine.run_with_live_log(_target, job_name='Checkpoint merge'):
                log_box = '\n'.join(state['logs'])
                yield log_box, state['progress'], gr.update(), cancel_token, gr.update(interactive=False)
            if state and state.get('error'):
                yield (log_box + '\n' + _format_error(state['error'])), 0, gr.update(), None, gr.update(interactive=True)
            else:
                new_ckpts_out = engine.list_checkpoints(OUTPUT_CKPT_DIR)
                yield log_box, 100, gr.update(choices=new_ckpts_out, value=nm), None, gr.update(interactive=True)
        except Exception as e:
            yield log_box + '\n' + _format_error(e), 0, gr.update(), None, gr.update(interactive=True)
    except Exception as e:
        yield log_box + '\n' + _format_error(e), 0, gr.update(), None, gr.update(interactive=True)


# ─────────────────────────────────────────────────────────────────────────
# Tab: Checkpoint + LoRA Merger
# ─────────────────────────────────────────────────────────────────────────
def lora_ui_recipe_change(recipe_name):
    return engine.LORA_RECIPES.get(recipe_name, engine.LORA_RECIPES['Custom (edit manually below)'])


def do_lora_bake(base_ckpt, lora1, w1, lora2, w2, lora3, w3, lora4, w4,
                  unet_w, te_w, bw_text, do_final_blend, blend_bw_text, out_name, fp16):
    log_box = ''
    try:
        if not base_ckpt:
            yield 'Select a base checkpoint first.', 0, gr.update(), None, gr.update()
            return
        base_p = CHECKPOINTS_DIR / base_ckpt
        entries = [(lora1, w1), (lora2, w2), (lora3, w3), (lora4, w4)]
        loras_wts = [(str(LORAS_DIR / name), float(w)) for name, w in entries if name and name != '(none)']
        if not loras_wts:
            yield 'Select at least one LoRA.', 0, gr.update(), None, gr.update()
            return
        bw_vec, err = engine.parse_bw(bw_text)
        if err:
            yield f'Block weights error: {err}', 0, gr.update(), None, gr.update()
            return
        blend_bw_vec = None
        if do_final_blend:
            blend_bw_vec, err = engine.parse_bw(blend_bw_text)
            if err:
                yield f'Final blend block weights error: {err}', 0, gr.update(), None, gr.update()
                return

        nm = guards.safe_filename(out_name, 'lora_baked_checkpoint.safetensors')
        if not nm.endswith('.safetensors'):
            nm += '.safetensors'
        final_p = OUTPUT_CKPT_DIR / nm
        import time
        ts = time.time_ns()
        stage_out = OUTPUT_CKPT_DIR / f'_stage_{ts}.safetensors'

        meta = {
            'creator': 'Model Breeder (Gradio Edition)',
            'tool': 'Checkpoint + LoRA Merger',
            'base_checkpoint': base_p.name,
            'unet_weight': f'{unet_w:.4f}', 'te_weight': f'{te_w:.4f}',
            'loras': '; '.join(f'{Path(p).name} (weight={w:.4f})' for p, w in loras_wts),
            'block_weights_json': __import__('json').dumps([round(w, 4) for w in bw_vec]),
            'final_blend_enabled': str(bool(do_final_blend)),
            'output_precision': 'F16' if fp16 else 'F32',
        }

        cancel_token = guards.CancelToken()

        def _target(progress_cb, log_cb):
            engine.bake_lora_stage(base_p, loras_wts, bw_vec, stage_out, bool(fp16), progress_cb, log_cb,
                                    unet_w=float(unet_w), te_w=float(te_w),
                                    meta_extra=(None if do_final_blend else meta), cancel_token=cancel_token)
            if do_final_blend:
                log_cb('Running final blend against original base...')
                engine.blend_checkpoints(base_p, stage_out, blend_bw_vec, final_p, bool(fp16),
                                          progress_cb, log_cb, meta_extra=meta, cancel_token=cancel_token)
                if stage_out.exists():
                    stage_out.unlink()
            else:
                os.replace(str(stage_out), str(final_p))  # atomic rename, same filesystem

        try:
            yield log_box, 0, gr.update(), cancel_token, gr.update(interactive=False)
            state = None
            for state in engine.run_with_live_log(_target, job_name='LoRA bake'):
                log_box = '\n'.join(state['logs'])
                yield log_box, state['progress'], gr.update(), cancel_token, gr.update(interactive=False)
            if state and state.get('error'):
                for f in (stage_out,):
                    if f.exists():
                        try:
                            f.unlink()
                        except Exception:
                            pass
                yield (log_box + '\n' + _format_error(state['error'])), 0, gr.update(), None, gr.update(interactive=True)
            else:
                yield log_box, 100, gr.update(choices=engine.list_checkpoints(OUTPUT_CKPT_DIR), value=nm), None, gr.update(interactive=True)
        except Exception as e:
            yield log_box + '\n' + _format_error(e), 0, gr.update(), None, gr.update(interactive=True)
    except Exception as e:
        yield log_box + '\n' + _format_error(e), 0, gr.update(), None, gr.update(interactive=True)


# ─────────────────────────────────────────────────────────────────────────
# Tab: VAE Baker
# ─────────────────────────────────────────────────────────────────────────
def do_vae_bake(ckpt_choice, vae_choice, replace_existing, out_name, fp16):
    log_box = ''
    try:
        if not ckpt_choice or not vae_choice:
            yield 'Select both a checkpoint and a VAE.', 0, None, gr.update()
            return
        cp, vp = CHECKPOINTS_DIR / ckpt_choice, VAE_DIR / vae_choice
        nm = guards.safe_filename(out_name, f'{cp.stem}_vae-baked.safetensors')
        if not nm.endswith('.safetensors'):
            nm += '.safetensors'
        out_p = OUTPUT_CKPT_DIR / nm

        cancel_token = guards.CancelToken()

        def _target(progress_cb, log_cb):
            engine.bake_vae(cp, vp, out_p, bool(fp16), bool(replace_existing), progress_cb, log_cb,
                             cancel_token=cancel_token)

        try:
            yield log_box, 0, cancel_token, gr.update(interactive=False)
            state = None
            for state in engine.run_with_live_log(_target, job_name='VAE bake'):
                log_box = '\n'.join(state['logs'])
                yield log_box, state['progress'], cancel_token, gr.update(interactive=False)
            if state and state.get('error'):
                yield (log_box + '\n' + _format_error(state['error'])), 0, None, gr.update(interactive=True)
            else:
                yield log_box, 100, None, gr.update(interactive=True)
        except Exception as e:
            yield log_box + '\n' + _format_error(e), 0, None, gr.update(interactive=True)
    except Exception as e:
        yield log_box + '\n' + _format_error(e), 0, None, gr.update(interactive=True)


# ─────────────────────────────────────────────────────────────────────────
# Tab: Metadata Reader
# ─────────────────────────────────────────────────────────────────────────
def do_read_metadata(choice):
    if not choice:
        return 'Select a checkpoint first.'
    path = _resolve_meta_choice(choice)
    if not path.exists():
        return f'File not found: {path}'
    meta = engine.read_metadata(path)
    header = f'**File:** `{path.name}`  ({_fmt_size(path.stat().st_size)})\n\n'
    return header + engine.format_metadata_markdown(meta)


# ─────────────────────────────────────────────────────────────────────────
# Tab: Image Generator
# ─────────────────────────────────────────────────────────────────────────
# CRITIQUE.md C1: the generation session is created per-browser-session via
# gr.State(engine.ImageGenSession) in build_app(), NOT as a module-level
# global — a shared global would let concurrent users (this app is designed
# to run with --share, i.e. multi-user) silently swap out each other's
# loaded model mid-session.

ARCH_CHOICES = [(preset['label'], key) for key, preset in engine.ARCH_PRESETS.items()]


def _arch_is_legacy(arch_key: str) -> bool:
    return engine.ARCH_PRESETS.get(arch_key, {}).get('kind') == 'legacy_single_file'


def on_arch_change(arch_key):
    """Toggle which fields are visible: the input/-file dropdown for legacy
    SD1.5/SDXL checkpoints, vs. the repo-id/folder text field + text-encoder
    /VAE override fields for the newer component-based architectures
    (CRITIQUE.md E1/E2)."""
    preset = engine.ARCH_PRESETS.get(arch_key, {})
    legacy = preset.get('kind') == 'legacy_single_file'
    notes = preset.get('notes', '')
    default_repo = preset.get('default_repo', '')
    default_te = preset.get('default_text_encoder', '')
    return (
        gr.update(visible=legacy),                                    # legacy checkpoint dropdown
        gr.update(visible=not legacy, value=default_repo),            # repo/folder text field
        gr.update(visible=not legacy, value=default_te),              # text encoder override
        gr.update(visible=not legacy),                                # vae override
        gr.update(value=notes, visible=bool(notes)),                  # arch notes hint
        gr.update(value=preset.get('default_steps', 28)),             # steps
        gr.update(value=preset.get('default_guidance_scale', 7.0)),   # cfg
    )


def do_load_checkpoint(session, ckpt_name, repo_or_path, arch_key, text_encoder_override,
                        vae_override, scheduler_name):
    """Streaming generator: loading a multi-GB checkpoint or downloading a
    diffusers repo can take a while, so this streams log lines live instead
    of appearing all at once only once loading finishes. Cancellation is
    honestly scoped: it can stop the *next* loading stage (text encoder,
    VAE, or main pipeline) from starting, but can't interrupt a single
    Hugging Face download already in flight — huggingface_hub's high-level
    from_pretrained() doesn't expose a hook for that. What DOES fully
    protect against a stuck/abandoned download is that the JOB_LOCK is now
    held for the worker thread's actual lifetime (see
    engine.run_with_live_log's docstring) — so even an uninterruptible,
    still-running download correctly blocks a second heavy job from
    starting concurrently, which is the scenario that actually matters."""
    session = session or engine.ImageGenSession()
    log_box = ''

    legacy = _arch_is_legacy(arch_key)
    if legacy and not ckpt_name:
        yield session, 'Select a checkpoint first.', None, gr.update(interactive=False)
        return
    if not legacy and not (repo_or_path or '').strip():
        yield session, 'Enter a HF repo id or local diffusers-folder path.', None, gr.update(interactive=False)
        return

    # Legacy checkpoints can come from either input/checkpoints/ (generate
    # straight from an uploaded file, no merge required) or output/checkpoints/
    # (a merge/bake result) — the dropdown carries a [input]/[output] prefix
    # exactly like the Metadata Reader tab's dropdown.
    checkpoint_path = str(_resolve_gen_ckpt_choice(ckpt_name)) if legacy else (repo_or_path or '').strip()
    te_override = (text_encoder_override or '').strip() or None
    vae_ov = (vae_override or '').strip() or None

    # Resolve a bare filename/foldername against the dedicated subfolders
    # rather than requiring the user to type a full path.
    if vae_ov and not Path(vae_ov).exists():
        candidate = VAE_DIR / vae_ov
        if candidate.exists():
            vae_ov = str(candidate)
    if te_override and not Path(te_override).exists():
        candidate = TEXT_ENCODERS_DIR / te_override
        if candidate.exists():
            te_override = str(candidate)

    cancel_token = guards.CancelToken()

    def _target(progress_cb, log_cb):
        return session.load(
            checkpoint_path, scheduler_name, log_cb, progress_cb,
            arch_key=arch_key, text_encoder_override=te_override, vae_override=vae_ov,
            cancel_token=cancel_token)

    try:
        state = None
        for state in engine.run_with_live_log(_target, job_name='Load checkpoint'):
            log_box = '\n'.join(state['logs'])
            yield session, log_box, cancel_token, gr.update(interactive=False)
        if state and state.get('error'):
            yield session, log_box + '\n' + _format_error(state['error']), None, gr.update(interactive=False)
        else:
            yield session, log_box, None, gr.update(interactive=True)
    except Exception as e:
        yield session, log_box + '\n' + _format_error(e), None, gr.update(interactive=False)


def do_generate(session, prompt, neg_prompt, steps, cfg, width, height, seed, n_images, clip_skip,
                 scheduler_name):
    """Streaming generator (CRITIQUE.md D2/D3): yields a progressively-
    growing gallery, a live progress value, and a live approximate preview
    as each image renders — not just at image boundaries — and saves each
    PNG to disk immediately. Runs the actual generation in a background
    thread via engine.run_generation_stream, which is what makes real-time
    progress/preview possible at all: the diffusers step callback fires
    synchronously deep inside a single blocking pipe() call, so a plain
    (non-threaded) generator has no way to surface those until an entire
    image finishes."""
    if session is None or session.pipe is None:
        yield [], 'Load a checkpoint first.', None, None, 0, gr.update(interactive=True)
        return

    gallery_images = []
    log_box = ''
    cancel_token = guards.CancelToken()

    def _target(progress_cb, log_cb, preview_cb, image_cb):
        gen = session.generate_stream(
            prompt, neg_prompt, steps, cfg, width, height, seed, n_images, clip_skip,
            scheduler_name, progress_cb, log_cb, cancel_token=cancel_token, preview_cb=preview_cb)
        for kind, img, idx, this_seed in gen:
            out_p = OUTPUT_IMAGES_DIR / f'gen_{this_seed}_{idx}.png'
            try:
                img.save(str(out_p))
                log_cb(f'  saved -> {out_p.name}')
            except Exception as e:
                log_cb(f'  ⚠ could not save image {idx}: {e}')
            image_cb(img, idx, this_seed)

    try:
        yield gallery_images, '', None, cancel_token, 0, gr.update(interactive=False)
        state = None
        for state in engine.run_generation_stream(_target, job_name='Image generation'):
            gallery_images = [img for img, _idx, _seed in state['images']]
            log_box = '\n'.join(state['logs'])
            preview = state['preview'][0] if state['preview'] else None
            progress_pct = int(state['progress'] * 100)
            yield gallery_images, log_box, preview, cancel_token, progress_pct, gr.update(interactive=False)
        if state and state.get('error'):
            err = state['error']
            if isinstance(err, guards.OperationCancelled):
                log_box += f'\n⏹ Cancelled — kept {len(gallery_images)} image(s) already generated.'
            else:
                log_box += '\n' + _format_error(err)
            yield gallery_images, log_box, None, None, 0, gr.update(interactive=True)
        else:
            yield gallery_images, log_box, None, None, 100, gr.update(interactive=True)
    except Exception as e:
        yield gallery_images, log_box + '\n' + _format_error(e), None, None, 0, gr.update(interactive=True)


RES_PRESETS = {
    '512x512 (SD1.5 square)': (512, 512), '512x768 (SD1.5 portrait)': (512, 768),
    '768x512 (SD1.5 landscape)': (768, 512), '768x768 (SD1.5 square)': (768, 768),
    '1024x1024 (SDXL / DiT square)': (1024, 1024), '832x1216 (SDXL / DiT portrait)': (832, 1216),
    '1216x832 (SDXL / DiT landscape)': (1216, 832), '1344x768 (SDXL / DiT wide)': (1344, 768),
    '768x1344 (SDXL / DiT tall)': (768, 1344),
}


# ─────────────────────────────────────────────────────────────────────────
# Build the UI
# ─────────────────────────────────────────────────────────────────────────
def _card_title(text: str):
    gr.Markdown(text, elem_classes=['m3-card-title'])


def _hint(text: str):
    gr.Markdown(text, elem_classes=['m3-hint'])


def build_app() -> gr.Blocks:
    with gr.Blocks(title='Model Breeder — Gradio Edition', theme=MATERIAL_THEME, css=MATERIAL_CSS,
                    fill_width=False) as demo:

        # ── App bar ────────────────────────────────────────────────────
        gr.HTML(
            '<div class="m3-appbar">'
            '  <div class="m3-appbar-icon">🧬</div>'
            '  <div style="flex:1; min-width:0;">'
            '    <p class="m3-appbar-title">Model Breeder</p>'
            '    <p class="m3-appbar-subtitle">Checkpoint merging, LoRA baking, VAE baking, and image '
            '      generation for SD1.5, SDXL, Z-Image, Krea 2 &amp; Anima — streaming, NaN-safe, '
            '      OOM-guarded.</p>'
            '  </div>'
            f'  <span class="m3-chip">📁 {DATA_DIR}</span>'
            '</div>'
        )
        with gr.Group(elem_classes=['m3-card']):
            _card_title('System')
            sys_status_md = gr.Markdown(guards.system_status_markdown(DATA_DIR))
            sys_refresh_btn = gr.Button('Refresh status', elem_classes=['m3-btn-tonal'], size='sm')
            sys_refresh_btn.click(lambda: guards.system_status_markdown(DATA_DIR), outputs=sys_status_md)

        # ── Files tab ──────────────────────────────────────────────────
        with gr.Tab('📁  Files'):
            with gr.Group(elem_classes=['m3-card']):
                _card_title('Workspace')
                _hint('Files are organized by type — checkpoints, LoRAs, and VAEs each have their own '
                      'folder now for easier browsing. Leave the destination on Auto-detect and the app '
                      'sorts each upload by inspecting its contents; pick a specific folder to override that.')
                with gr.Row():
                    upload_dest_dd = gr.Dropdown(label='Destination folder', choices=list(UPLOAD_DESTINATIONS),
                                                 value='Auto-detect (recommended)', scale=1)
                    upload_widget = gr.File(label='Upload files', file_count='multiple', scale=2)
                files_md = gr.Markdown(list_files_markdown())
                with gr.Row():
                    refresh_files_btn = gr.Button('Refresh file list', elem_classes=['m3-btn-tonal'])
            upload_widget.upload(upload_to_input, inputs=[upload_widget, upload_dest_dd], outputs=files_md)
            refresh_files_btn.click(list_files_markdown, outputs=files_md)

        # ── Checkpoint x Checkpoint Merger ────────────────────────────
        with gr.Tab('🧬  Checkpoint × Checkpoint'):
            _hint('Merge two (or three) checkpoints. All 8 methods from the original notebook, per-block '
                  'weight control, optional VAE baking, NaN-safe streaming, prediction-type guard.')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Models')
                with gr.Row():
                    ckpt_a_dd = gr.Dropdown(label='Model A', choices=[])
                    ckpt_b_dd = gr.Dropdown(label='Model B', choices=[])
                    ckpt_c_dd = gr.Dropdown(label='Model C (optional)', choices=['(none)'], value='(none)')
                    merge_refresh_btn = gr.Button('⟳', scale=0, elem_classes=['m3-btn-icon'])
                with gr.Row():
                    pred_a_dd = gr.Dropdown(label='A prediction type', choices=['epsilon', 'v-pred'], value='epsilon')
                    pred_b_dd = gr.Dropdown(label='B prediction type', choices=['epsilon', 'v-pred'], value='epsilon')
                gr.Markdown('_Mismatch aborts the merge for all methods except DARE / XDARE, which are '
                            'designed to bridge them._')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Merge method & strength')
                method_dd = gr.Dropdown(label='Merge method', choices=[lbl for lbl, _ in engine.METHOD_OPTS],
                                         value=engine.METHOD_OPTS[0][0])
                with gr.Row():
                    alpha_sl = gr.Slider(0, 1, value=0.5, step=0.01, label='Alpha')
                    beta_sl = gr.Slider(0, 1, value=0.2, step=0.01,
                                        label='Beta (Sum Twice/Triple Sum ratio, or DARE droprate)')
                    dare_seed_num = gr.Number(value=0, precision=0, label='DARE/XDARE seed')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Per-block weights · Weighted Sum / Add Diff / Sigmoid / SLERP')
                bw_recipe_dd = gr.Dropdown(label='Recipe', choices=list(engine.CKPT_RECIPES),
                                            value='All Blocks Equal (standard full blend)')
                bw_text = gr.Textbox(label='Block weights (20 comma-separated values: BASE, IN00-08, MID00, OUT00-08)',
                                     value=engine.CKPT_RECIPES['All Blocks Equal (standard full blend)'])
                bw_recipe_dd.change(merge_ui_recipe_change, inputs=bw_recipe_dd, outputs=bw_text)

            with gr.Accordion('Block-ratio controls · Sum Twice / Triple Sum / DARE / XDARE', open=False,
                               elem_classes=['m3-card']):
                with gr.Row():
                    use_alpha_ratio_cb = gr.Checkbox(label='Alpha: use per-block ratio instead of slider', value=False)
                    use_beta_ratio_cb = gr.Checkbox(label='Beta: use per-block ratio (Sum Twice/Triple Sum only)', value=False)
                alpha_ratio_tb = gr.Textbox(label='Alpha block ratio (scalar or 20 comma-separated values)', value='1')
                beta_ratio_tb = gr.Textbox(label='Beta block ratio (scalar or 20 comma-separated values)', value='1')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Output')
                vae_dd = gr.Dropdown(label='Bake VAE (optional)', choices=['(none)'], value='(none)')
                with gr.Row():
                    out_name_tb = gr.Textbox(label='Output filename', value='merged_checkpoint.safetensors')
                    fp16_cb = gr.Checkbox(label='Save FP16 (recommended)', value=True)
                with gr.Row():
                    merge_btn = gr.Button('Merge', variant='primary', elem_classes=['m3-btn-filled'])
                    merge_cancel_btn = gr.Button('Cancel', elem_classes=['m3-btn-tonal'], scale=0)
                merge_progress = gr.Slider(0, 100, value=0, label='Progress', interactive=False,
                                           elem_classes=['m3-progress'])
                merge_log = gr.Textbox(label='Log', lines=14, max_lines=30, elem_classes=['m3-console'])
                merge_cancel_state = gr.State(None)

            def _merge_refresh():
                ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
                return (gr.update(choices=ckpts), gr.update(choices=ckpts),
                        gr.update(choices=['(none)'] + ckpts), gr.update(choices=['(none)'] + vaes))

            merge_refresh_btn.click(_merge_refresh, outputs=[ckpt_a_dd, ckpt_b_dd, ckpt_c_dd, vae_dd])
            merge_btn.click(
                do_merge,
                inputs=[ckpt_a_dd, ckpt_b_dd, ckpt_c_dd, method_dd, alpha_sl, beta_sl, bw_text,
                        pred_a_dd, pred_b_dd, vae_dd, out_name_tb, fp16_cb,
                        use_alpha_ratio_cb, alpha_ratio_tb, use_beta_ratio_cb, beta_ratio_tb, dare_seed_num],
                outputs=[merge_log, merge_progress, ckpt_a_dd, merge_cancel_state, merge_btn],
            )
            merge_cancel_btn.click(lambda tok: tok.cancel() if tok else None, inputs=merge_cancel_state, outputs=None)

        # ── Checkpoint + LoRA Merger ───────────────────────────────────
        with gr.Tab('🎨  Checkpoint + LoRA'):
            _hint('Bake up to 4 LoRAs (standard LoRA / LoKr / LoHA) into a base checkpoint in one pass, with '
                  'per-block weight targeting and an optional final blend back toward the original base. To '
                  'chain further stages, feed this tab\'s output back in as a new base checkpoint.')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Base checkpoint')
                with gr.Row():
                    base_ckpt_dd = gr.Dropdown(label='Base checkpoint (.safetensors)', choices=[], scale=4)
                    lora_refresh_btn = gr.Button('⟳', scale=0, elem_classes=['m3-btn-icon'])

            with gr.Group(elem_classes=['m3-card']):
                _card_title('LoRA stack')
                with gr.Row():
                    lora1_dd = gr.Dropdown(label='LoRA 1', choices=['(none)'], value='(none)')
                    w1_sl = gr.Slider(-2, 2, value=1.0, step=0.05, label='Weight')
                with gr.Row():
                    lora2_dd = gr.Dropdown(label='LoRA 2', choices=['(none)'], value='(none)')
                    w2_sl = gr.Slider(-2, 2, value=1.0, step=0.05, label='Weight')
                with gr.Row():
                    lora3_dd = gr.Dropdown(label='LoRA 3', choices=['(none)'], value='(none)')
                    w3_sl = gr.Slider(-2, 2, value=1.0, step=0.05, label='Weight')
                with gr.Row():
                    lora4_dd = gr.Dropdown(label='LoRA 4', choices=['(none)'], value='(none)')
                    w4_sl = gr.Slider(-2, 2, value=1.0, step=0.05, label='Weight')
                with gr.Row():
                    unet_w_sl = gr.Slider(0, 2, value=1.0, step=0.05, label='Global UNet weight multiplier')
                    te_w_sl = gr.Slider(0, 2, value=1.0, step=0.05, label='Global text-encoder weight multiplier')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Per-block weights')
                lora_bw_recipe_dd = gr.Dropdown(label='Recipe', choices=list(engine.LORA_RECIPES),
                                                 value='All Blocks Equal (standard bake)')
                lora_bw_text = gr.Textbox(label='Block weights (20 comma-separated values)',
                                          value=engine.LORA_RECIPES['All Blocks Equal (standard bake)'])
                lora_bw_recipe_dd.change(lora_ui_recipe_change, inputs=lora_bw_recipe_dd, outputs=lora_bw_text)

            with gr.Accordion('Final blend with original base (optional)', open=False, elem_classes=['m3-card']):
                final_blend_cb = gr.Checkbox(label='Enable final blend', value=False)
                blend_bw_text = gr.Textbox(
                    label='Blend block weights (A=original base, B=baked output; result = A*(1-w)+B*w)',
                    value='0,0,0,0.2,0.2,0.2,0.2,0,0,0,0.2,0.2,0.2,0.2,0.1,0.8,0.8,0.8,0.8,0.8')

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Output')
                with gr.Row():
                    lora_out_name_tb = gr.Textbox(label='Output filename', value='lora_baked.safetensors')
                    lora_fp16_cb = gr.Checkbox(label='Save FP16 (recommended)', value=True)
                with gr.Row():
                    lora_bake_btn = gr.Button('Bake', variant='primary', elem_classes=['m3-btn-filled'])
                    lora_cancel_btn = gr.Button('Cancel', elem_classes=['m3-btn-tonal'], scale=0)
                lora_progress = gr.Slider(0, 100, value=0, label='Progress', interactive=False,
                                          elem_classes=['m3-progress'])
                lora_log = gr.Textbox(label='Log', lines=14, max_lines=30, elem_classes=['m3-console'])
                lora_cancel_state = gr.State(None)

            def _lora_refresh():
                ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
                lo_opts = ['(none)'] + loras
                return (gr.update(choices=ckpts), gr.update(choices=lo_opts), gr.update(choices=lo_opts),
                        gr.update(choices=lo_opts), gr.update(choices=lo_opts))

            lora_refresh_btn.click(_lora_refresh, outputs=[base_ckpt_dd, lora1_dd, lora2_dd, lora3_dd, lora4_dd])
            lora_bake_btn.click(
                do_lora_bake,
                inputs=[base_ckpt_dd, lora1_dd, w1_sl, lora2_dd, w2_sl, lora3_dd, w3_sl, lora4_dd, w4_sl,
                        unet_w_sl, te_w_sl, lora_bw_text, final_blend_cb, blend_bw_text, lora_out_name_tb, lora_fp16_cb],
                outputs=[lora_log, lora_progress, base_ckpt_dd, lora_cancel_state, lora_bake_btn],
            )
            lora_cancel_btn.click(lambda tok: tok.cancel() if tok else None, inputs=lora_cancel_state, outputs=None)

        # ── VAE Baker ───────────────────────────────────────────────────
        with gr.Tab('🧪  VAE Baker'):
            _hint('Bakes a standalone VAE into a checkpoint that ships without one.')
            with gr.Group(elem_classes=['m3-card']):
                _card_title('Source files')
                with gr.Row():
                    vae_ckpt_dd = gr.Dropdown(label='Checkpoint (missing a VAE)', choices=[])
                    vae_vae_dd = gr.Dropdown(label='VAE to bake in', choices=[])
                    vae_refresh_btn = gr.Button('⟳', scale=0, elem_classes=['m3-btn-icon'])
                vae_replace_cb = gr.Checkbox(label='Replace existing VAE if the checkpoint already has one', value=False)

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Output')
                with gr.Row():
                    vae_out_name_tb = gr.Textbox(label='Output filename', value='')
                    vae_fp16_cb = gr.Checkbox(label='Save FP16 (recommended)', value=True)
                with gr.Row():
                    vae_bake_btn = gr.Button('Bake VAE', variant='primary', elem_classes=['m3-btn-filled'])
                    vae_cancel_btn = gr.Button('Cancel', elem_classes=['m3-btn-tonal'], scale=0)
                vae_progress = gr.Slider(0, 100, value=0, label='Progress', interactive=False,
                                         elem_classes=['m3-progress'])
                vae_log = gr.Textbox(label='Log', lines=10, max_lines=25, elem_classes=['m3-console'])
                vae_cancel_state = gr.State(None)

            def _vae_refresh():
                ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
                return gr.update(choices=ckpts), gr.update(choices=vaes)

            vae_refresh_btn.click(_vae_refresh, outputs=[vae_ckpt_dd, vae_vae_dd])
            vae_bake_btn.click(do_vae_bake,
                                inputs=[vae_ckpt_dd, vae_vae_dd, vae_replace_cb, vae_out_name_tb, vae_fp16_cb],
                                outputs=[vae_log, vae_progress, vae_cancel_state, vae_bake_btn])
            vae_cancel_btn.click(lambda tok: tok.cancel() if tok else None, inputs=vae_cancel_state, outputs=None)

        # ── Metadata Reader ─────────────────────────────────────────────
        with gr.Tab('🔍  Metadata Reader'):
            _hint('Read the recipe metadata this tool embeds in every checkpoint it produces — read-only, '
                  'never modifies the file.')
            with gr.Group(elem_classes=['m3-card']):
                _card_title('Checkpoint')
                with gr.Row():
                    meta_dd = gr.Dropdown(label='Checkpoint (input/ and output/)', choices=[], scale=4)
                    meta_refresh_btn = gr.Button('⟳', scale=0, elem_classes=['m3-btn-icon'])
                meta_read_btn = gr.Button('Read metadata', variant='primary', elem_classes=['m3-btn-filled'])

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Recipe')
                meta_md = gr.Markdown()

            def _meta_refresh():
                ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
                return gr.update(choices=meta_ckpts)

            meta_refresh_btn.click(_meta_refresh, outputs=meta_dd)
            meta_read_btn.click(do_read_metadata, inputs=meta_dd, outputs=meta_md)

        # ── Image Generator ─────────────────────────────────────────────
        with gr.Tab('🖼️  Image Generator'):
            _hint('SD1.5 &amp; SDXL load from a single checkpoint file. Z-Image Turbo, Krea 2, and Anima '
                  'ship as separate transformer + text-encoder + VAE components (like ComfyUI\'s modular '
                  'loading) — pick an architecture below to switch input modes. Uses GPU + FP16 if '
                  'available, otherwise CPU + BF16.')

            gen_session_state = gr.State(None)  # CRITIQUE.md C1: per-browser-session, not a shared global

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Architecture')
                arch_dd = gr.Dropdown(label='Model architecture', choices=ARCH_CHOICES, value='auto')
                arch_notes_md = gr.Markdown(visible=False, elem_classes=['m3-hint'])

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Checkpoint')
                with gr.Row():
                    gen_ckpt_dd = gr.Dropdown(label='Checkpoint (input/checkpoints/ or output/checkpoints/)',
                                              choices=[], scale=3, visible=True)
                    gen_repo_tb = gr.Textbox(label='HF repo id or local diffusers-folder path', scale=3,
                                             visible=False, placeholder='e.g. Tongyi-MAI/Z-Image-Turbo')
                    gen_sched_dd = gr.Dropdown(label='Scheduler', choices=list(engine.SCHED_MAP),
                                               value='DPM++ 2M Karras', scale=2)
                    gen_refresh_btn = gr.Button('⟳', scale=0, elem_classes=['m3-btn-icon'])
                with gr.Row():
                    gen_te_override_tb = gr.Textbox(
                        label='Text encoder override (HF repo id or local folder — optional)', visible=False,
                        placeholder='leave blank to use the architecture\'s default text encoder')
                    gen_vae_override_tb = gr.Textbox(
                        label='VAE override (repo id, local folder, or a filename from input/vae/ — optional)',
                        visible=False, placeholder='leave blank to use the architecture\'s default VAE')
                with gr.Row():
                    gen_load_btn = gr.Button('Load checkpoint', elem_classes=['m3-btn-tonal'])
                    gen_load_cancel_btn = gr.Button('Cancel', elem_classes=['m3-btn-tonal'], scale=0)
                gen_load_log = gr.Textbox(label='Load status', lines=3, elem_classes=['m3-console'])
                gen_load_cancel_state = gr.State(None)

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Generation settings')
                with gr.Row():
                    steps_sl = gr.Slider(1, 150, value=28, step=1, label='Steps')
                    cfg_sl = gr.Slider(0, 30, value=7.0, step=0.5, label='CFG scale')
                    clip_skip_sl = gr.Slider(1, 12, value=1, step=1, label='Clip skip (legacy SD1.5/SDXL only)')
                res_dd = gr.Dropdown(label='Resolution preset', choices=list(RES_PRESETS),
                                     value='512x512 (SD1.5 square)')
                with gr.Row():
                    width_num = gr.Number(value=512, precision=0, label='Width')
                    height_num = gr.Number(value=512, precision=0, label='Height')

                def _on_res(choice):
                    w, h = RES_PRESETS[choice]
                    return w, h
                res_dd.change(_on_res, inputs=res_dd, outputs=[width_num, height_num])

                with gr.Row():
                    seed_num = gr.Number(value=-1, precision=0, label='Seed (-1 = random)')
                    n_images_sl = gr.Slider(1, 8, value=1, step=1, label='Images')
                gr.Markdown('_Every value here is clamped to a safe range before generation starts, so a '
                            'stray huge number can\'t OOM the process._', elem_classes=['m3-hint'])

            arch_dd.change(
                on_arch_change, inputs=arch_dd,
                outputs=[gen_ckpt_dd, gen_repo_tb, gen_te_override_tb, gen_vae_override_tb, arch_notes_md,
                         steps_sl, cfg_sl],
            )

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Prompt')
                pos_prompt = gr.Textbox(label='Positive prompt', lines=3,
                                        value='masterpiece, best quality, highly detailed')
                neg_prompt = gr.Textbox(label='Negative prompt (not all architectures use this)', lines=3,
                                        value='lowres, bad anatomy, worst quality, low quality, jpeg artifacts, watermark')
                with gr.Row():
                    gen_btn = gr.Button('Generate', variant='primary', interactive=False, elem_classes=['m3-btn-filled'])
                    gen_cancel_btn = gr.Button('Cancel', elem_classes=['m3-btn-tonal'], scale=0)

            with gr.Group(elem_classes=['m3-card']):
                _card_title('Results')
                gr.Markdown('_Images appear here as soon as each one finishes — a batch of several images '
                            'saves progressively, not all at once at the end. The small preview updates '
                            'during denoising itself; it\'s a cheap approximation, not the final image.',
                            elem_classes=['m3-hint'])
                gen_progress = gr.Slider(0, 100, value=0, label='Progress', interactive=False,
                                         elem_classes=['m3-progress'])
                with gr.Row():
                    gen_preview_img = gr.Image(label='Live preview (approximate)', height=220,
                                               interactive=False, scale=1)
                    gen_gallery = gr.Gallery(label='Results (streamed as each image finishes)', columns=2,
                                             height='auto', elem_classes=['m3-gallery'], scale=2)
                gen_log = gr.Textbox(label='Log', lines=6, max_lines=20, elem_classes=['m3-console'])
                gen_cancel_state = gr.State(None)

            def _gen_refresh():
                ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
                return gr.update(choices=gen_ckpts)

            gen_refresh_btn.click(_gen_refresh, outputs=gen_ckpt_dd)
            gen_load_btn.click(
                do_load_checkpoint,
                inputs=[gen_session_state, gen_ckpt_dd, gen_repo_tb, arch_dd, gen_te_override_tb,
                        gen_vae_override_tb, gen_sched_dd],
                outputs=[gen_session_state, gen_load_log, gen_load_cancel_state, gen_btn],
            )
            gen_load_cancel_btn.click(lambda tok: tok.cancel() if tok else None,
                                      inputs=gen_load_cancel_state, outputs=None)
            gen_btn.click(
                do_generate,
                inputs=[gen_session_state, pos_prompt, neg_prompt, steps_sl, cfg_sl, width_num, height_num,
                        seed_num, n_images_sl, clip_skip_sl, gen_sched_dd],
                outputs=[gen_gallery, gen_log, gen_preview_img, gen_cancel_state, gen_progress, gen_btn],
            )
            gen_cancel_btn.click(lambda tok: tok.cancel() if tok else None, inputs=gen_cancel_state, outputs=None)

        # Populate all dropdowns once on load
        def _initial_load():
            ckpts, loras, vaes, gen_ckpts, meta_ckpts = refresh_all_dropdowns()
            lo_opts = ['(none)'] + loras
            return (
                gr.update(choices=ckpts), gr.update(choices=ckpts), gr.update(choices=['(none)'] + ckpts),
                gr.update(choices=['(none)'] + vaes),
                gr.update(choices=ckpts), gr.update(choices=lo_opts), gr.update(choices=lo_opts),
                gr.update(choices=lo_opts), gr.update(choices=lo_opts),
                gr.update(choices=ckpts), gr.update(choices=vaes),
                gr.update(choices=meta_ckpts),
                gr.update(choices=gen_ckpts),
            )

        demo.load(
            _initial_load,
            outputs=[ckpt_a_dd, ckpt_b_dd, ckpt_c_dd, vae_dd,
                     base_ckpt_dd, lora1_dd, lora2_dd, lora3_dd, lora4_dd,
                     vae_ckpt_dd, vae_vae_dd, meta_dd, gen_ckpt_dd],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Model Breeder — Gradio Edition')
    parser.add_argument('--share', action='store_true', help='Create a public gradio.live share link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--server-name', type=str, default=None,
                         help='Bind address, e.g. 0.0.0.0 to expose on your LAN')
    args = parser.parse_args()

    in_colab = _in_colab()
    demo = build_app()  # theme + custom CSS are applied inside build_app()
    demo.queue(default_concurrency_limit=2).launch(
        share=args.share or in_colab,
        server_port=args.port,
        server_name=args.server_name or ('0.0.0.0' if not in_colab else None),
        debug=in_colab,
    )


if __name__ == '__main__':
    main()
