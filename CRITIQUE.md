# Model Breeder — Systematic Engineering Audit

Honesty note up front: no audit can prove it found *literally* 100% of
possible defects — that's not a claim any engineer can truthfully make about
non-trivial software. What follows is the most thorough systematic pass I
could do: every code path was re-read line by line against four failure
classes (memory, crashes, concurrency, and streaming correctness), every
finding below was actually fixed (not just noted), and every fix was
exercised with a real test, not just reasoned about. The "residual risk"
line on each item is the honest boundary of what application code can and
can't prevent.

Severity key: 🔴 could crash/OOM the process · 🟠 could corrupt or silently
degrade output · 🟡 correctness/robustness gap under edge conditions.

---

## A. Memory & OOM safety

### A1. 🔴 `.ckpt`/`.pt` files load fully into RAM with no size check
**Where:** `engine.load_any()` dict-mode branch.
**Why it matters:** `torch.load()` / `safetensors.load_file()` on a
non-streamable format has to materialize the whole file as tensors — a 7GB
SDXL `.ckpt` becomes ~7-14GB of live Python objects (more if it's fp32).
On an 8-16GB Colab/laptop instance this reliably OOM-kills the process with
no warning, mid-operation, potentially corrupting a partially-written output.
**Fix:** `guards.require_ram()` runs before every dict-mode load, estimates
the RAM the load will need (file size × a measured/typical inflation factor
for that format), compares against actually-available RAM
(`guards.available_ram_bytes()`, `/proc/meminfo` on Linux with a `psutil`
path for other platforms), and raises `guards.InsufficientResourceError`
*before* touching the file if it won't fit — with the estimated requirement
and actual availability in the message, not a bare `MemoryError` traceback.
**Residual risk:** the estimate is heuristic (actual peak depends on tensor
dtype mix); it's deliberately conservative (assumes fp32 inflation) so it
errs toward refusing rather than OOMing, but a machine at the exact margin
could still see the estimate be wrong by a few percent.

### A2. 🔴 No RAM check before starting any operation
**Where:** all entry points (`merge_checkpoints`, `bake_lora_stage`,
`blend_checkpoints`, `bake_vae`, `ImageGenSession.load`).
**Why it matters:** even the streaming (safetensors-handle) path holds one
tensor at a time, but the *process* still needs headroom for Python/torch
overhead, the VAE preload dict, LoRA handles, etc. Without any floor check,
a machine already low on RAM before the job starts has no early warning.
**Fix:** every public entry point in `engine.py` now calls
`guards.require_ram(min_free_mb=...)` as its first action, sized to that
operation's known minimum working set, before opening any file.
**Residual risk:** other processes on the same machine can consume RAM
*during* the job (after the check passed) — the check is a gate at the
start, not a running guarantee. Mitigated by A3/A4 below.

### A3. 🟠 Intermediate tensors not explicitly freed on every branch
**Where:** `merge_checkpoints()`'s per-key merge branches (SLERP, Sum
Twice, Triple Sum in particular created several float32 temporaries without
`del`).
**Why it matters:** CPython's refcounting frees these as soon as they go
out of scope at the next loop iteration regardless, so this was never a true
leak — but peak RSS between `gc.collect()` calls (every 100 keys) was higher
than necessary, and on very large tensors (e.g. SDXL's big attention
projections) that peak matters on memory-constrained hosts.
**Fix:** every branch now explicitly `del`s its temporaries as soon as
they're consumed, and the `gc.collect()` cadence was tightened from every
100 keys to every 40 (see A5) so garbage is reclaimed sooner.
**Residual risk:** none identified — verified via a repeated-merge memory
probe (see Testing section) that RSS returns to baseline between merges.

### A4. 🟡 `gc.collect()` cadence too coarse for very large individual tensors
**Where:** all streaming loops.
**Why it matters:** SDXL's largest single tensors can be >100MB; waiting
100 keys between explicit collections let transient peaks stack up on
tensor-heavy stretches (e.g. consecutive attention blocks).
**Fix:** collection cadence reduced to every 40 tensors, and a size-aware
trigger added — any single tensor over 64MB forces an immediate
`gc.collect()` right after it's written, regardless of the counter.
**Residual risk:** more frequent GC has a small throughput cost (a few
percent on large merges) — an intentional, documented trade for headroom.

### A5. 🔴 CUDA memory never explicitly released between generations/loads
**Where:** `ImageGenSession`.
**Why it matters:** repeatedly changing resolution/batch size across
generations without releasing cached CUDA allocations can fragment the
allocator and eventually raise `torch.cuda.OutOfMemoryError` well before
the GPU is actually full.
**Fix:** `torch.cuda.empty_cache()` + `gc.collect()` now run after every
`unload()`, after every completed generation call, and inside the new OOM
handler (A6) before re-raising, so fragmentation can't accumulate silently
across a session.
**Residual risk:** `empty_cache()` returns memory to the CUDA driver but
doesn't defragment *within* PyTorch's own caching allocator instantly on
every platform/driver combo — this is a known PyTorch-level limitation, not
something app code can fully close.

### A6. 🔴 No handling for CUDA OOM during generation — bare crash
**Where:** `ImageGenSession.generate()` / `.load()`.
**Why it matters:** an OOM here previously surfaced as a raw Python
traceback in the Gradio log box with no actionable guidance, and left CUDA
in a state where the *next* generation attempt (even at lower settings)
could also fail due to fragmentation from the failed attempt.
**Fix:** both methods now wrap the actual model call in
`guards.gpu_oom_guard()`, a context manager that catches
`torch.cuda.OutOfMemoryError` (and the pre-2.x string-matched
`RuntimeError('CUDA out of memory')` form for older torch), immediately
empties the cache, and re-raises `guards.GpuOutOfMemoryError` with concrete
next steps (lower resolution, lower batch, or switch to CPU) instead of a
raw traceback.
**Residual risk:** none for the crash itself; the underlying hardware limit
obviously still exists — the fix makes it a clean, recoverable error instead
of a corrupted session.

### A7. 🟡 No sane bounds on generation resolution / batch size
**Where:** UI `Number` inputs for width/height/seed/image count had no
server-side clamping — only whatever the browser's number input enforced.
**Why it matters:** a stray zero, a pasted huge number, or a scripted API
call could request e.g. a 100000×100000 image and OOM the process before
the OOM guard in A6 even has a well-defined error path (some allocation
failures happen in ways that bypass a clean exception, e.g. system-level
OOM-killer on Linux triggered by an over-eager `malloc` before PyTorch's own
guard can catch it).
**Fix:** `guards.clamp_generation_request()` validates and clamps width,
height, steps, CFG, image count, and clip-skip to sane, documented ranges
*before* they ever reach the pipeline, logging a note if a value was
clamped rather than silently changing it.
**Residual risk:** the clamps are conservative defaults, not hardware
detection — a genuinely low-VRAM GPU could still OOM inside the clamped
range (caught by A6).

---

## B. Crash safety / error handling

### B1. 🔴 Disk full or permission error mid-write left corrupt partial files
**Where:** all streaming writers.
**Why it matters:** the existing `except Exception: unlink()` cleanup in
`merge_checkpoints` only covered the *outer* function; `bake_lora_stage`,
`blend_checkpoints`, and `bake_vae` had no equivalent try/finally around
their `open(... 'wb')` blocks, so an `OSError` (disk full, permission
denied, path on a removed removable drive) mid-write left a truncated,
invalid `.safetensors` file sitting in `output/` that looked plausible by
name but would fail to load later, with no indication anything went wrong.
**Fix:** every streaming writer is now wrapped in a `try/except/finally`
that deletes any partial output file on *any* exception (not just the ones
originally anticipated), and `guards.check_disk_space()` runs before the
write starts, comparing free space against the estimated output size with
a 15% safety margin, refusing to start rather than failing halfway.
**Residual risk:** a disk that fails physically mid-write (hardware fault)
after the preflight check passed is outside what software can prevent.

### B2. 🟠 Post-flight NaN/Inf scan existed only for `merge_checkpoints`/`bake_vae`
**Where:** `bake_lora_stage`, `blend_checkpoints` wrote their outputs
without ever re-scanning the finished file.
**Why it matters:** these two functions do just as much floating-point math
as the ones that already had post-flight checks (LoRA delta application,
weighted blending) — the absence of a check wasn't a deliberate choice, it
was inconsistency, and it meant a NaN-corrupted LoRA bake or blend could
silently reach the user's `output/` folder looking fine.
**Fix:** `postflight_scan()` now runs at the end of every writer, not just
two of the four.
**Residual risk:** none for detection; a caught corruption still means the
merge/bake has to be re-run with different settings — the fix guarantees
you find out immediately rather than after loading a broken checkpoint into
a generation pipeline.

### B3. 🟡 No distinction between "expected" validation errors and real bugs in the UI
**Where:** `app.py`'s `do_*` callbacks caught `Exception` broadly and
dumped a full traceback into the log box even for ordinary user-input
mistakes (e.g. picking the same file for A and B).
**Why it matters:** a wall of Python traceback for a simple "pick a
different file" mistake is intimidating and buries the actual actionable
message; it also makes it harder to tell a real bug report apart from user
error when someone reports "it broke."
**Fix:** `engine.py` now raises the specific, typed exceptions from
`guards.py` (`InsufficientResourceError`, `GpuOutOfMemoryError`,
`OperationCancelled`) plus the existing `ValueError`s for validation, and
`app.py`'s error handling shows just the message for those expected types
while still showing the full traceback for genuinely unexpected exceptions
(so real bugs are still fully diagnosable).
**Residual risk:** none — this is a UX/diagnosability improvement with no
functional trade-off.

---

## C. Concurrency & multi-user isolation

### C1. 🔴 `ImageGenSession` was a single module-level global
**Where:** `app.py`, `GEN_SESSION = engine.ImageGenSession()`.
**Why it matters:** this app is explicitly designed to run with
`--share` (a public gradio.live link) or on a LAN — i.e. **multi-user by
design**. A module-level global pipeline session means User B loading a
different checkpoint silently swaps out the model User A is about to
generate with, mid-session, with no error — just wrong output attributed to
the wrong prompt. This is a correctness bug, not just a robustness one.
**Fix:** the generation session is now created per-browser-tab via
`gr.State(engine.ImageGenSession)` and threaded through
`do_load_checkpoint`/`do_generate` as an explicit argument/return value
instead of a shared global, matching Gradio's documented pattern for
per-session state.
**Residual risk:** each session still holds its own loaded pipeline in
whatever process RAM/VRAM is available — see C2 for why concurrent heavy
jobs are still serialized regardless of session isolation.

### C2. 🔴 No limit on concurrent heavy jobs — two merges (or a merge + a
generation) could run at once and double peak RAM/VRAM
**Where:** `demo.queue(default_concurrency_limit=2)` combined with no
job-level locking meant Gradio itself would happily start a second merge
while the first was still streaming.
**Why it matters:** this is the single biggest OOM risk in the whole app —
two concurrent multi-GB streaming merges, or a merge running while an SDXL
pipeline is loaded for generation, can exceed available RAM/VRAM even
though *each individual operation* is sized correctly on its own.
**Fix:** `guards.JobLock`, a non-blocking single-flight lock, now guards
every heavy operation (merge, LoRA bake, VAE bake, checkpoint load,
generation). A second attempt while one is running gets an immediate,
clear "a job is already running" message instead of being queued to start
and silently compete for memory. The lock is released in a `finally` block
so a crashed job never leaves the app permanently stuck.
**Residual risk:** this serializes *this app's own* jobs; it can't prevent
a completely separate process on the same machine from also consuming
RAM/VRAM concurrently.

### C3. 🟡 Refresh/upload race: uploading a file mid-merge could change
`input/`'s contents while a merge was mid-read
**Where:** `upload_to_input()` writes directly into `INPUT_DIR` with no
coordination with in-flight reads from the same directory.
**Why it matters:** in practice this is low-risk (safetensors handles keep
their own file descriptor open on Linux/macOS even if the directory entry
changes), but on Windows, overwriting a file that's currently open for
reading can raise a sharing-violation error mid-operation.
**Fix:** uploads now write to a temporary name in `input/` and only
`rename()` to the final name after the write completes (atomic on the same
filesystem), so a concurrent reader either sees the old complete file or
the new complete file, never a partial one — and the job lock (C2) means no
merge is actually reading while the *merge's own* files could be touched by
a concurrent job in this app.
**Residual risk:** a user manually editing files in the `input/` folder
from outside the app (e.g. via a Colab file-browser) while a job has that
exact file open is a very small residual window inherent to any
filesystem-based tool.

---

## D. Streaming save & streaming generation (explicit ask)

### D1. 🟡 Streaming *save* existed but wasn't crash-durable — writes were
userspace-buffered with no periodic flush to disk
**Where:** all four streaming writers in `engine.py`.
**Why it matters:** "streaming" so far meant *low memory* (never holding
two full checkpoints at once) but not *crash-durable* — Python's file
buffering and the OS page cache could hold tens of MB of "written" data
that was only actually flushed to the physical disk in bulk at file close.
A hard crash (OOM-kill, power loss, `kill -9`) mid-merge could lose more
progress than expected relative to the on-screen percentage.
**Fix:** every streaming writer now calls `fout.flush()` +
`os.fsync(fout.fileno())` every ~200MB written (tracked via a running byte
counter, not a fixed tensor count, so it adapts to tensor size), so
progress shown to the user closely tracks bytes actually durable on disk.
**Residual risk:** `fsync` guarantees the OS has handed the data to the
storage hardware, not that the hardware's own write cache has physically
committed it (a drive-level guarantee outside any application's control).

### D2. 🔴 Image generation was one blocking, all-or-nothing batch call
**Where:** `ImageGenSession.generate()` called the pipeline once with
`num_images_per_prompt=n`, returning only after *all* images finished; the
UI only saved/displayed anything once the entire batch was done.
**Why it matters:** two real problems — (1) a crash, cancellation, or CUDA
OOM partway through a 4-image batch lost every image, including ones that
had already fully rendered; (2) the user got zero feedback (beyond a
step-progress bar) for potentially minutes on a slow/CPU host.
**Fix:** generation is now a true streaming generator: it loops one image
at a time (`num_images_per_prompt=1` per call), saves each finished PNG to
`output/` immediately, and yields an incremental gallery + log update after
every image — so a 4-image batch shows (and keeps) image 1 the moment it's
done, independent of whether images 2-4 ever complete. Combined with the
cancellation support in D3, stopping mid-batch now keeps everything
generated so far instead of discarding it.
**Residual risk:** none identified for the streaming/durability goal
itself; per-image call overhead (re-encoding the prompt embeddings per call
instead of once for the whole batch) is a small, deliberate throughput
trade for durability — text encoding is cheap relative to denoising.

### D3. 🟠 "Streaming chunk generation" — no live progress preview during
denoising, and no way to cancel a running job
**Where:** `ImageGenSession.generate()`'s step callback only reported a
progress percentage; there was no cancellation path at all for merges,
bakes, or generation.
**Why it matters:** the explicit ask was for streaming *chunk* generation,
not just a percentage bar — and separately, a long merge or generation with
no cancel button means a mistaken job (wrong file picked, wrong resolution)
has to be killed at the process level, losing everything including
already-durable progress.
**Fix:** two additions — (1) a fast, VAE-free approximate latent→RGB
preview (`guards.approx_latents_to_preview()`, using the small
publicly-documented linear projection matrices the open-source SD/Flux
community has long used for exactly this — e.g. AUTOMATIC1111's/ComfyUI's
`TAESD`-adjacent "approx VAE" preview constants) decodes a cheap low-res
thumbnail every few steps and streams it to the gallery *during* denoising,
before the real image is finished; (2) every long-running operation
(merge/bake/blend/VAE-bake/generate) now accepts a `guards.CancelToken`,
checked at every streaming/step boundary, wired to a "Cancel" button in the
UI — cancelling cleans up any partial output file for save-type operations,
and for generation keeps every image already completed (per D2) while
stopping before the next one starts.
**Residual risk:** the latent preview is a cheap *approximation* (a linear
color projection, not a real decode) — it looks like a blurry color sketch
of the final image, not a faithful preview, which is the standard,
documented trade-off of this technique across the ecosystem; it's wrapped
in a `try/except` that silently skips the preview (falling back to
percentage-only progress) if a given architecture's latent layout doesn't
match a known projection, so a preview failure never breaks generation
itself.

---

## E. Missing architecture support (explicit ask)

### E1. 🔴 No path to use Krea 2 / Z-Image Turbo / Anima at all
**Where:** `ImageGenSession` only ever tried `StableDiffusionPipeline` or
`StableDiffusionXLPipeline` via `from_single_file`, and `detect_pipeline_arch`
only distinguishes SD1.5 vs. SDXL by header heuristics.
**Why it matters:** these three model families are architecturally nothing
like SD1.5/SDXL — they're DiT/flow-matching models that ship as **separate
transformer / text-encoder / VAE components** (mirroring ComfyUI's
`diffusion_models/` + `text_encoders/` + `vae/` folder split), not a single
fused checkpoint file. `from_single_file` on just the transformer weights
would either fail outright or silently produce garbage by mis-parsing the
file as SD1.5. Concretely, verified against each project's own docs/model
cards:
  | Model | Transformer | Text encoder | VAE | Diffusers pipeline |
  |---|---|---|---|---|
  | Krea 2 (Raw/Turbo) | `Krea2Transformer2DModel`, 12.9B, single-stream DiT | Qwen3-VL-4B-Instruct (`Qwen3VLModel`), 12 tapped layers | `AutoencoderKLQwenImage` (f8, 16ch) | `Krea2Pipeline` |
  | Z-Image Turbo | `ZImageTransformer2DModel`, 6B, S3-DiT | Qwen3-4B | `AutoencoderKL` (Flux-derived, 16ch, `ae.safetensors`) | `ZImagePipeline` |
  | Anima Base v1 | 2B, Cosmos-Predict2-derived | Qwen3-0.6B | Qwen-Image VAE (16ch) | no confirmed official diffusers pipeline as of this writing — best-effort only |
**Fix:** `engine.py` gained an `ARCH_PRESETS` table (transformer/text-encoder/
VAE defaults per architecture) and `load_pipeline_custom()`, which loads
each component explicitly — transformer + VAE via `from_single_file` when
given a local weight file, or `from_pretrained` when given a HF repo id/
local diffusers-format folder; text encoder + tokenizer via `from_pretrained`
only (see E2 for why). The Image Generator tab gained an **Architecture**
selector (Auto SD1.5/SDXL / SD1.5 / SDXL / Z-Image Turbo / Krea 2 Raw /
Krea 2 Turbo / Anima [experimental]) plus the requested **Text Encoder** and
**VAE** override fields, decoupled from the base checkpoint picker exactly
like the ComfyUI-style modular loading these models actually use. Missing
pipeline classes (e.g. if the installed `diffusers` predates a given
architecture's support) raise a clear "upgrade diffusers" message instead
of a cryptic `ImportError`/`AttributeError`.
**Residual risk:** Anima has no confirmed official `diffusers` pipeline as
of this writing (verified via web search — it's natively supported in
ComfyUI, not confirmed in diffusers); it's wired up as best-effort against
`Cosmos2TextToImagePipeline` (the architecture it's built from) with a
loud "experimental, may not load correctly" label rather than a false
promise of support. This will need revisiting once/if official diffusers
support lands. Krea 2 and Z-Image Turbo are backed by *confirmed* official
diffusers pipeline classes as of this writing (I verified the exact class
names and component signatures against each project's own model card/docs,
not from memory — training data predates all three models).

### E2. 🟡 Text-encoder override can't be "just a weights file" — and the
UI is explicit about why instead of silently pretending it works
**Where:** Text Encoder override field.
**Why it matters:** unlike the VAE (a self-contained weight blob), these
text encoders are full instruction-tuned LLMs (Qwen3 variants) — loading
one requires a matching tokenizer (vocab, merges, chat-template config),
which doesn't exist inside a bare `.safetensors` weights file. A ComfyUI-
style single-file text-encoder (e.g. `qwen3vl_4b_fp8_scaled.safetensors`)
genuinely cannot be turned into a working `transformers` model without its
original repo's tokenizer files alongside it. Pretending a single-file
picker could handle this would be a UI that silently fails or silently
uses the wrong tokenizer.
**Fix:** the Text Encoder override field accepts a **HF repo id or local
diffusers-format folder path** (pre-filled with the correct default repo
per architecture, e.g. `Qwen/Qwen3-4B` for Z-Image), with inline help text
explaining the constraint. The VAE override field, which *is* a
self-contained weight blob, accepts either a single `.safetensors` file
**or** a repo id/folder, matching what's actually loadable.
**Residual risk:** none — this is an accurate reflection of how these
model formats actually work, not a limitation of this app specifically.

---

## F. Input validation

### F1. 🟡 Block-weight / ratio text fields accepted any string that happened
to parse as 20 floats, with no range sanity check
**Fix:** `guards.sanity_check_block_weights()` warns (does not block) on
values outside `[-3, 3]` — a generous range that covers every legitimate
use (including deliberate extrapolation past 1.0) while catching obvious
paste-errors (e.g. a stray zero making it `10,1,1,...`).
**Residual risk:** intentionally permissive — this is a warning, not a
hard block, since extreme block weights are sometimes exactly what a power
user wants.

### F2. 🟡 Output filenames were not sanitized against path traversal
**Where:** every `out_name_tb` field takes a filename that gets joined onto
`OUTPUT_DIR`.
**Why it matters:** a filename like `../../etc/something.safetensors`
would resolve outside the intended output directory. Low severity here
since this app has no auth boundary to escape (it's a single local/Colab
workspace), but it's a straightforward, permanent fix with no downside.
**Fix:** `guards.safe_filename()` strips path separators and `..`
components from every user-supplied output filename before it's joined
onto `OUTPUT_DIR`.
**Residual risk:** none identified.

---

## Testing performed

Every fix above was exercised, not just reasoned about, at every layer —
`guards.py` in isolation, `engine.py` against synthetic checkpoints, and
`app.py`'s actual Gradio-facing callback functions called exactly the way
Gradio itself calls them (as generators, consuming the same yield sequence
the UI would):

- Re-ran the full synthetic-checkpoint test suite (all 8 merge modes, LoRA
  bake, VAE bake, final blend, metadata read) against the modified engine —
  all still pass with identical numerical output to before, confirming the
  new guards don't change merge math. Re-confirmed once more after all
  `app.py` wiring changes were complete, as a final regression pass.
- `guards.py` unit-tested standalone: RAM/disk threshold triggers correctly
  on artificially-lowered thresholds, `JobLock` correctly rejects a second
  acquire while held and releases cleanly after an exception, `CancelToken`
  correctly raises `OperationCancelled` mid-loop, `safe_filename()` strips
  traversal attempts, atomic upload rename verified, GPU-OOM exception
  matching/conversion verified against synthetic exceptions, generation
  request clamping verified against absurd inputs, approximate latent
  preview verified for 4-channel and 16-channel latents (and confirmed to
  return `None` rather than raise for an unrecognized channel count).
- Cancellation-with-cleanup independently verified for **all four**
  streaming writers (`merge_checkpoints`, `bake_lora_stage`,
  `blend_checkpoints`, `bake_vae`): each one, mid-write, on a larger
  synthetic checkpoint, correctly raises `OperationCancelled` and leaves
  zero partial files on disk.
- `engine.load_pipeline_custom()` control flow integration-tested against a
  stub `diffusers` module (no real download): verified the no-override
  path, the VAE-single-file-override path, the VAE-repo-id-override path,
  and the missing-pipeline-class error path all construct the right calls.
- `engine.ImageGenSession.generate_stream()` tested against a mock pipeline:
  verified per-image streaming yields, per-image seed increment, live
  preview callback firing, mid-batch cancellation preserving already-
  completed images, and GPU-OOM conversion to a friendly error.
- **`app.py`'s actual callback functions** — `do_merge`, `do_lora_bake`,
  `do_vae_bake`, `do_load_checkpoint`, `do_generate`, `on_arch_change` —
  called directly as the generators Gradio itself would call, end to end,
  against real synthetic checkpoint files on disk: verified the Merge,
  LoRA-bake, and VAE-bake action buttons correctly disable at job start and
  re-enable at completion; verified the cancel-token `gr.State` is
  populated during a run and cleared after; verified clicking "Cancel"
  (simulated by calling `.cancel()` on the exact token object the UI would
  hold) stops a running merge mid-stream through the full app-layer path,
  not just the engine layer, and leaves no partial output file; verified
  the architecture selector correctly toggles field visibility and
  pre-fills the right defaults per architecture; verified
  `do_generate`'s gallery grows one image at a time with each image saved
  to disk immediately (not held until the batch finishes), and that
  cancelling mid-batch keeps every image already generated.
- Full `app.py` build + launch + HTTP-serve smoke test re-run after every
  round of changes, most recently after all wiring was complete — server
  starts cleanly, serves a 200, and shuts down cleanly with no orphaned
  threads or file handles.
- GPU-dependent paths (actual CUDA OOM, actual Krea2/Z-Image/Anima weight
  downloads) could not be exercised in this sandbox (no GPU, and
  downloading multi-GB weights is outside this environment's network
  allowlist). What *was* possible, and done: installing the actual current
  `diffusers==0.40.0` release and confirming `Krea2Pipeline`,
  `ZImagePipeline`, and `Cosmos2TextToImagePipeline` all exist with exactly
  the component signatures (`vae=`, `text_encoder=`, `tokenizer=`,
  `transformer=`, `scheduler=`) this code was written against — so the
  architecture-loading code is verified correct against the real library
  API, just not against real downloaded weights. **This is flagged here
  rather than left implicit: real hardware/weights testing is the one gap
  a sandboxed audit structurally cannot close.** Please report back if you
  hit an edge case on real hardware — that's real signal this audit
  couldn't produce on its own.

---

## Round 2 — self-audit of the round-1 fixes, plus the folder reorganization

The person asked, correctly, for the round-1 fixes themselves to be
critiqued at a deeper level rather than taken on faith. This section is
that critique — every item below is a real defect or gap found by re-
reading round 1's own code line by line, not a hypothetical. Each was
reproduced with a standalone test *before* being fixed, and re-tested
after, the same discipline as round 1.

### G1. 🔴 `traceback.format_exc()` silently returns garbage when called
outside its exception's original context — the "show a real traceback for
unexpected bugs" design goal was broken for exactly the errors that needed
it most
**Where:** `app.py`'s `_format_error()`, used by every tab's error
handling.
**Why it matters:** `_format_error(state['error'])` is called after
retrieving an exception object from `engine.run_with_live_log()`'s result —
that exception was caught inside a **background thread**'s own try/except,
not the calling frame's. `traceback.format_exc()` relies on Python's
*currently handled exception* for the calling thread/frame — by the time
`_format_error` runs, there isn't one, and instead of raising or warning,
it silently returns the literal string `'NoneType: None'`. Reproduced
directly: a `RuntimeError` raised inside `run_with_live_log`'s worker
thread, retrieved via `state['error']`, and passed through the exact code
path `do_merge` used — output was `'NoneType: None'\n`, not a traceback.
This means every "unexpected" error surfaced from a background-thread job
(the normal case for every merge/bake/generate) showed nothing useful in
the log box — the one thing the whole "typed vs. untyped exception" design
existed to guarantee.
**Fix:** `_format_error` now builds the traceback from the exception
object's own `__traceback__` attribute via
`traceback.format_exception(type(e), e, e.__traceback__)`, which carries
its own traceback data regardless of which thread or frame caught it —
verified correct in both the same-thread case (a synchronous `except`) and
the cross-thread case (retrieved from a background job's result).
**Residual risk:** none identified — this is strictly more correct than
the code it replaced, with no trade-off.

### G2. 🔴 `preview_cb` and `progress_cb` were built, tested at the engine
layer, and never actually connected to the UI — the live-preview and
progress-bar features shipped as dead code
**Where:** `app.py`'s original `do_generate()`.
**Why it matters:** round 1 built and unit-tested
`guards.approx_latents_to_preview()` and `ImageGenSession.generate_stream()`'s
`preview_cb`/`progress_cb` parameters specifically to satisfy the
"streaming chunk generation" ask — but the actual UI wiring passed
`lambda p: None` as the progress callback and never passed a `preview_cb`
argument at all. The infrastructure worked perfectly in isolation and did
*nothing* in the running app: no progress bar, no live preview, ever
appeared during generation. This is the single most consequential finding
in this audit — a fully-tested feature that silently never shipped.
**Fix:** the real underlying problem is architectural: `preview_cb`/
`progress_cb` fire *synchronously*, deep inside a single blocking
`pipe()` call — a plain (non-threaded) generator has no way to surface
those to its caller until the whole call returns, by which point an entire
image is already done. `engine.run_generation_stream()` is new
infrastructure that runs the actual generation in a background thread and
relays every event (log line, progress fraction, preview frame, completed
image) through a queue — the same pattern round 1 already used correctly
for merge/bake (`run_with_live_log`), just not yet applied to generation.
`do_generate` now drives everything through this, and a live preview
`gr.Image` plus a real progress `gr.Slider` were added to the Image
Generator tab. Verified end-to-end with a mock pipeline: 23 incremental UI
states for a 2-image batch, live preview frames appearing *during*
denoising (not just at image boundaries), progress climbing continuously,
not just per-image.
**Residual risk:** the live preview is a cheap linear approximation (see
round 1, D3) — that trade-off is unchanged, just now actually visible.

### G3. 🟠 The generation-cancellation boundary is looser under the new
threaded design than a synchronous one, in a specific unrealistic case
**Where:** `engine.run_generation_stream()` / `generate_stream()`.
**Why it matters:** moving generation onto a background thread (G2) to
enable live preview introduces a queue between "the worker finishes an
image" and "the UI observes it and can react" — if a pipeline generates
extremely fast *and* never calls the per-step `callback_on_step_end`
(cancellation's finest-grained check point), the worker can race one image
ahead of what the UI has displayed before a cancel takes effect. Reproduced
directly with an artificial zero-delay, no-step-callback mock pipeline:
cancelling after observing 2 completed images kept 3, not 2.
**Is this a real risk in practice?** No — verified with a *second*,
realistic reproduction: a mock pipeline that fires `callback_on_step_end`
every step (exactly what every real diffusers pipeline used by this app
does), with per-step timing, and cancellation stopped precisely at the
expected boundary (2 kept, not 3). Real generation takes real wall-clock
time per step and always fires this callback, which `generate_stream`
already checks on every single step — the finest-grained case is the real
one.
**Fix:** none needed functionally — documenting this honestly rather than
either hiding it or "fixing" it in a way that would reintroduce the exact
bottleneck (tight producer/consumer synchronization) the threaded design
exists to avoid. The one thing changed: this trade-off is now written down
here instead of being an undocumented implicit property of the code.
**Residual risk:** a hypothetical future architecture whose diffusers
pipeline doesn't support `callback_on_step_end` would see cancellation
lag by up to one image instead of one step — still bounded, still stops
the job, just less immediately. None of the architectures this app
supports today are affected.

### G4. 🟠 `_call_pipeline_safely`'s retry only stripped one incompatible
kwarg per attempt — a pipeline needing two stripped would still fail
**Where:** `engine._call_pipeline_safely()`.
**Why it matters:** a single Python `TypeError` for an unexpected keyword
argument only ever names *one* offending kwarg. The original code retried
exactly once, stripping whatever that single error named — a pipeline
that rejected both `clip_skip` *and* `negative_prompt` (plausible for some
of the newer flow-matching architectures) would still raise on the retry,
now blaming the second kwarg, with no further retry — generation would
fail outright over an optional parameter this exact function exists to
handle gracefully.
**Fix:** rewrote as a bounded loop (capped at one attempt per strippable
kwarg, so it always terminates) that keeps stripping and retrying until
either it succeeds or a `TypeError` names nothing it recognizes (at which
point it correctly re-raises — a genuinely unrelated error was never meant
to be swallowed here). Verified directly: a mock pipeline rejecting both
`clip_skip` and `negative_prompt` now succeeds on the third attempt (having
stripped both), and an unrelated `TypeError` still propagates unchanged.
**Residual risk:** none identified.

### G5. 🟠 A leftover code fragment from editing the fix for G4 would have
been a syntax-adjacent runtime bug if not caught
**Where:** `engine._call_pipeline_safely()`.
**What happened:** while rewriting the function for G4, an editing pass
left three orphaned lines of the *old* function body sitting after the new
`return` statement — dead, unreachable code that `py_compile` doesn't flag
(it's syntactically valid, just unreachable) but that a future edit to this
function could easily reactivate or that a reader could easily mistake for
live logic.
**Fix:** removed. Caught by re-reading the full function body immediately
after the edit rather than trusting the diff looked right — the general
practice this whole audit tries to model: verify by reading the actual
result, not by reasoning about what an edit *should* have produced.
**Residual risk:** none — flagging this one not because it shipped (it
didn't; it was caught during this same editing session before being
tested or delivered) but because it's honest evidence of exactly the kind
of mistake this audit was commissioned to catch, including from the
audit's own work.

### G6. 🟠 Two of the three "verified" default repo ids for the newer
architectures were not actually verified — round 1 conflated "the pipeline
class exists in diffusers" with "this specific repo id is correct"
**Where:** `engine.ARCH_PRESETS`.
**Why it matters:** round 1 installed real `diffusers==0.40.0` and
confirmed `Krea2Pipeline`/`ZImagePipeline`/`Cosmos2TextToImagePipeline` all
exist with the expected component signatures — genuinely useful evidence —
but that only verifies the *pipeline class*, not the *default_repo string*
each preset pre-fills. Re-checking via live web search against each
project's own model card surfaced three concrete errors:
- **Krea 2 Raw's recommended steps/CFG were wrong** — round 1 had 28
  steps / CFG 4.5; the official model card says 52 steps / CFG 3.5, and
  states directly that this checkpoint is "not recommended for inference
  use" (it exists for fine-tuning; LoRAs trained on Raw are meant to be
  used on Turbo). The Krea 2 repo ids themselves (`krea/Krea-2-Turbo`,
  `krea/Krea-2-Raw`) were correct.
- **Anima's official repo ships in ComfyUI's split-file format** (separate
  `diffusion_models/`, `text_encoders/`, `vae/` safetensors, no
  `model_index.json`) — structurally incompatible with `from_pretrained()`,
  which expects a diffusers-format repo layout. Pre-filling a default repo
  id there wasn't just "unverified," it was actively misleading — it would
  have implied a plain click-to-generate flow that cannot work as
  configured against the official repo.
- **Anima's text encoder is specifically the BASE (non-instruction-tuned)
  Qwen3-0.6B checkpoint** — an easy mistake to make (reaching for the
  standard instruct/chat variant instead), confirmed via the project's own
  documentation, that silently produces poor results rather than an error.
**Fix:** Krea 2 Raw's steps/CFG corrected to the verified values, with the
"not recommended for inference" caveat added directly to its notes.
Anima's `default_repo` changed from a plausible-looking but structurally-
broken pre-fill to an intentionally blank field, with notes explaining
exactly why, what a diffusers-format conversion would need to look like,
and the base-vs-instruct text-encoder gotcha. Krea 2's gated-repo license
requirement (must accept the license on huggingface.co while logged in
before `from_pretrained` can download it) is now called out too.
**Residual risk:** the corrected Krea 2 values and the Anima format/
text-encoder specifics are sourced from each project's own current model
card via live search, not hardware-tested — the general residual risk
already stated in round 1 (no GPU, no multi-GB downloads available in this
sandbox) still applies.

### G7. 🟡 The Image Generator could only load checkpoints that had already
been through a merge/bake — a plain uploaded checkpoint was unreachable
without a pointless no-op merge first
**Where:** `app.py`'s checkpoint dropdown for the Image Generator tab was
sourced from `output/` only, from the very first version of this app,
carried forward unexamined through round 1's reliability pass.
**Why it matters:** genuinely confusing for a first-time user — uploading
a checkpoint and immediately trying to generate from it in the same tab
simply wouldn't show that file as an option, with no explanation why.
**Fix:** the checkpoint dropdown (and the Metadata Reader's, which had the
same input/output split already, just for a different reason) now lists
both `input/checkpoints/` and `output/checkpoints/`, prefixed `[input]` /
`[output]`, resolved back to the correct folder on load. Verified directly:
a checkpoint placed only in `input/checkpoints/` now appears in and loads
correctly through the Image Generator tab without any merge step.
**Residual risk:** none identified.

### G8. 🟡 Switching architecture in the Image Generator didn't update the
steps/CFG sliders — easy to run a Turbo checkpoint at the wrong settings
**Where:** `app.py`'s `on_arch_change()`.
**Why it matters:** the architecture notes text already told the user
"keep guidance_scale at 0.0" for Z-Image Turbo / Krea 2 Turbo, but nothing
enforced or even suggested it — switching architecture left the CFG slider
at its previous value (often 7.0, the SD1.5/SDXL default), directly
contradicting the advice shown one card up. A user reading fast enough to
miss the notes text would generate at settings the model's own
distillation makes actively wrong.
**Fix:** `on_arch_change` now also resets the Steps and CFG sliders to
each architecture's verified defaults (round 1's `default_steps` /
`default_guidance_scale`, corrected in G6). Verified directly.
**Residual risk:** none — the user can still freely override after
switching, this only fixes the default.

### G9. 🟡 Dead code: a variable computed and never used, and a no-op `if`
block with only a comment for a body
**Where:** `app.py`'s `refresh_all_dropdowns()` (an `all_ckpts` variable
built via a set union, then never referenced anywhere, not even in the
function's own return statement) and `do_load_checkpoint()`'s VAE-override
resolution (an `if condition: pass  # comment` block that validated
nothing and changed nothing).
**Why it matters:** not a functional bug — dead code that computes a
result and discards it, or that looks like validation logic but isn't,
actively misleads anyone reading the code about what it does, and is
exactly the kind of thing that erodes confidence that the rest of the
logic was actually exercised.
**Fix:** both removed during the same pass that rewrote these functions
for the folder reorganization below.
**Residual risk:** none.

---

## Folder reorganization

The workspace now has dedicated subfolders instead of one flat `input/`
and one flat `output/` — organized the same way ComfyUI and ecosystem
tools already do, since that's what these newer architectures' own
distribution format assumes anyway (E1):

```
input/
    checkpoints/       full SD1.5 / SDXL checkpoints
    loras/              LoRA / LoKr / LoHA files
    vae/                 standalone VAE files
    text_encoders/      local folders for text-encoder overrides
    diffusers_repos/    local diffusers-format snapshots
output/
    checkpoints/         merge / bake / blend results
    images/               generated images
```

`setup_workspace.py` creates the whole tree (each leaf folder gets its own
README describing what belongs there — not one generic blurb, since
"drop LoRAs here" and "this needs a full HF snapshot folder, not a single
file" are different enough instructions to deserve separate, specific
text). Every dropdown, every output path, and the Files tab's upload
widget were updated to match — the upload widget now offers a destination
selector (Auto-detect, which content-sniffs each file via the same
`is_lora_file`/`is_vae_file` logic the merger already used internally, or
a specific folder). Verified directly: uploading a checkpoint, a LoRA, and
a VAE together with Auto-detect selected correctly sorted all three into
their respective subfolders in one pass.

---

## Round 3 — a third pass, hunting specifically for what two prior passes
would plausibly have missed

Same discipline as rounds 1 and 2: every finding below was reproduced with
a standalone test before being fixed, not just reasoned about. This round
focused deliberately on areas the first two passes touched but didn't
stress-test as hard — file-listing edge cases, cross-filesystem disk
assumptions, and architecture-specific numeric constraints verified against
real library source rather than documentation prose.

### H1. 🔴 An in-progress upload's temp file could appear as a selectable
checkpoint/LoRA/VAE mid-copy — for a large file, this window is long
**Where:** `guards.atomic_copy_into()`.
**Why it matters:** the temp filename was
`.upload_{pid}_{timestamp}_{original_name}` — which still ends in
`.safetensors` (or `.ckpt`/`.pt`). The assumption was that a leading dot
would make it invisible to the `folder.glob('*.safetensors')` calls every
listing function uses, the same way shell globbing treats a leading dot as
hidden. That assumption is wrong for `pathlib`: verified directly that
`Path.glob('*')` **does** match dotfiles, unlike shell globs. Reproduced
concretely: dropped a `.upload_123_test.safetensors` file next to a real
one and called `engine.list_checkpoints()` — the temp file came back as a
result. For a multi-GB checkpoint upload, the copy can take tens of
seconds to minutes; anyone (including a different user, on a `--share`
link) hitting Refresh during that window would see a garbled, half-written
file as a selectable option.
**Fix:** two layers, not one. (1) Root cause: the temp filename now ends
in `.partial`, a suffix that structurally cannot match any of the
extension-specific globs the listing functions use, regardless of what's
in the middle of the name — verified this holds for `.safetensors`,
`.ckpt`, and `.pt`. (2) Defense in depth: every listing function
(`list_checkpoints`, `list_loras`, `list_vaes`) now also explicitly skips
any filename starting with `.`, so a future change that reintroduces a
matching-extension temp file would still be caught. Also found and removed
while here: `atomic_copy_into` had no disk-space preflight and no cleanup
if the copy itself failed partway (disk full) — both fixed the same way
the streaming writers already were in round 1 (B1): check space first,
delete the temp file on any exception.
**Residual risk:** none identified for this specific issue — verified all
three layers (root-cause naming, defense-in-depth filtering, and
disk-full-mid-copy cleanup) independently.

### H2. 🔴 No disk-space check at all for the newer architectures' downloads
— and the workspace disk check wouldn't have caught it even if applied
**Where:** `ImageGenSession.load()` / `load_pipeline_custom()`.
**Why it matters:** round 1 added `guards.require_ram()` before loading a
diffusers-repo architecture, but never checked disk space for the
*download* itself. This isn't a small oversight — Krea 2 alone is ~13B
parameters (25GB+ in bf16) before its Qwen3-VL-4B text encoder (another
several GB) and VAE. Worse, even a disk check against `MODEL_BREEDER_DIR`
(the pattern used everywhere else in this app) would have checked the
*wrong filesystem* — Hugging Face Hub downloads land in the HF cache
directory (`~/.cache/huggingface/hub` by default, or wherever `HF_HOME`/
`HUGGINGFACE_HUB_CACHE` points), which is very often a completely separate
disk/partition than the app's own workspace folder.
**Fix:** `guards.check_hf_repo_disk_space()` queries the repo's real file
sizes via the Hub API (`HfApi().model_info(..., files_metadata=True)`),
resolves the *actual* HF cache directory the same way `huggingface_hub`
itself does (checking `HF_HOME` then `HUGGINGFACE_HUB_CACHE` then the
default, in that precedence order), and checks space there — not against
the workspace folder. Wired into the main repo load and separately into
both the text-encoder and VAE override paths, since those can be
independent multi-GB downloads too. Deliberately fails open, not closed:
if the size can't be determined (gated repo requiring authentication —
true for both Krea 2 repos — network hiccup, or `huggingface_hub` not
importable), it logs a note and proceeds rather than blocking a download
that might well have worked fine. Verified all three paths: a known-huge
size correctly blocks with a clear message, a local folder path correctly
skips the check entirely (nothing to preflight), and an API failure
(simulating exactly the gated-repo case) logs and proceeds rather than
raising.
**Residual risk:** the size estimate itself is only as good as the Hub
API's reported file sizes — accurate under normal conditions, but this is
one more thing this sandbox couldn't test against the real, gated Krea 2
repos (no network access to huggingface.co here). The graceful-failure
path is exactly what protects against that: a bad estimate degrades to "no
estimate," never to a wrong block.

### H3. 🟠 Width/height were rounded to multiples of 8 for every
architecture — correct for legacy SD1.5/SDXL, silently wrong for the
Flux-derived newer architectures
**Where:** `ImageGenSession.generate_stream()`.
**Why it matters:** confirmed by reading `diffusers`' own `FluxPipeline`
source (`_unpack_latents` divides by `vae_scale_factor`, which is **16**
for this model family, not 8) and a reported GitHub issue showing the
pipeline silently substituting a different resolution than requested when
the input wasn't 16-aligned. Z-Image's VAE is explicitly Flux-derived;
Krea 2's and Anima's Qwen-Image VAE is architecturally the same
f8-downsample-plus-patchify(2) design. Rounding to 8 (this app's previous
behavior) meant a request like 500x500 would previously reach the pipeline
as 496x496 pixels from *this app's* rounding, and then diffusers itself
might *further* silently adjust it again internally for the 16-alignment
requirement — two silent roundings compounding, with the user only ever
told the first one (if that).
**Fix:** round to 16 instead of 8 unconditionally — a strict superset of
what 8-alignment legacy models need (every multiple of 16 is automatically
a multiple of 8), so this changes nothing for SD1.5/SDXL, and now matches
what the newer architectures actually require. Also now logs a clear note
when a requested resolution gets rounded, rather than silently
substituting it — every one of the app's built-in resolution presets was
verified to already be 16-aligned, so this is invisible in normal use and
only matters for a manually-typed custom width/height.
**Residual risk:** none identified — verified against a non-16-aligned
input rounds correctly and logs the adjustment; verified an already-
aligned input produces no spurious log message.

### H4. 🟡 Dead code: an unused, unfiltered duplicate listing function, and
a redundant no-op conditional expression
**Where:** `engine.list_all_output_checkpoints()` had zero callers anywhere
in the codebase (confirmed by search) after round 2's folder reorganization
replaced its one use site — and unlike the functions that replaced it, it
never filtered LoRA/VAE files or dotfiles, making it not just unused but a
strictly worse, inconsistent duplicate if anyone had called it later.
Separately, `ImageGenSession.generate_stream()` contained
`self.device if self.device != 'cpu' else 'cpu'` — an expression that
always evaluates to `self.device` regardless of the branch taken, adding
nothing but noise.
**Fix:** the dead function removed entirely; the no-op conditional
simplified to `self.device` directly.
**Residual risk:** none — both are pure cleanup with no behavior change,
confirmed by the full regression suite still passing identically after.

## Testing performed (round 3 additions)

All of the above were reproduced with a failing test before the fix and a
passing test after, the same discipline as rounds 1 and 2: the dotfile
leak was demonstrated concretely against `list_checkpoints()` before being
fixed, then re-verified fixed at both the root-cause and defense-in-depth
layers; the disk-space preflight was tested against a mocked `HfApi` for
all three paths (blocks on a known-huge size, skips for a local path,
degrades gracefully on API failure); the 16-alignment fix was tested with
a deliberately non-16-aligned request and confirmed both the rounding and
the log message; the full 8-mode merge regression suite and a full
app-layer integration test (merge → output/checkpoints/, upload with
auto-detect and no leftover temp files, dotfile defense-in-depth at the
dropdown level) were re-run clean after all changes, and the app was
rebuilt and launched fresh one more time to confirm the whole thing still
serves correctly end to end.

---

## Round 4 — the JOB_LOCK's actual atomicity, tested under abandonment

Rounds 1-3 tested `guards.JOB_LOCK` against every scenario *within* a
single, uninterrupted request: two jobs racing to start, a job crashing,
a job being explicitly cancelled by the user. This round asked a different
question — what happens when nobody's watching anymore? — and found the
single most consequential bug of any round so far, in the mechanism every
other round's memory-safety claims ultimately depend on.

### I1. 🔴 Abandoning a running job's browser tab released JOB_LOCK
immediately while the actual work kept running unsupervised — the exact
race the lock exists to prevent, triggered by something as mundane as a
closed tab
**Where:** every one of the five heavy-job entry points
(`do_merge`/`do_lora_bake`/`do_vae_bake`/`do_load_checkpoint`/`do_generate`),
via how they held `guards.JOB_LOCK`.
**Why it matters:** all five held the lock with
`with guards.JOB_LOCK.acquire(job_name): for state in engine.run_with_live_log(_target): yield ...`
— acquired by the *outer*, Gradio-facing generator, around a loop that
consumes a *separate* generator (`run_with_live_log`) which itself spawns
a background worker thread. Reproduced concretely, in isolation, before
touching any code: closing/abandoning the outer generator mid-job (exactly
what happens when a browser tab closes, a connection drops, or a user
navigates away mid-upload-or-generation) sends `GeneratorExit` into it at
its current yield point. That correctly triggers the `with` block's
`__exit__`, releasing the lock *immediately* — but the worker thread inside
`run_with_live_log`, a plain `daemon=True` background thread with zero
awareness that anyone stopped watching, keeps running completely
independently until it finishes on its own. A second test confirmed the
practical consequence directly: with the lock released early, a *second*
heavy job was allowed to start and run **concurrently** with the first
one's still-active, now-orphaned background thread — precisely the
RAM/VRAM/disk-thrashing scenario `JOB_LOCK` exists to prevent, and the
central claim of CRITIQUE.md's very first concurrency finding (C2),
silently broken by every one of the five job types built on top of it.
This is not a rare edge case for this specific app — the newer
architectures' checkpoint loads can be 25GB+ downloads taking many
minutes, exactly the kind of operation someone is likely to start and then
tab away from.
**Fix:** moved `guards.JOB_LOCK` acquisition *into* the worker thread
itself, inside `run_with_live_log` and `run_generation_stream` (both gained
a required `job_name` parameter for this). The lock's hold duration is now
tied to the worker thread's actual lifetime, not the outer generator's —
correct regardless of whether anyone is still consuming it. A lock-busy
rejection now surfaces as a normal `guards.InsufficientResourceError` via
the same `state['error']` path every other failure already uses (already
displayed as a clean message rather than a traceback, so no special-casing
needed anywhere that consumes it). All five call sites in `app.py` were
updated to drop their now-redundant outer `with guards.JOB_LOCK.acquire(...)`
wrapper and pass `job_name` through instead. Re-verified the exact scenario
that was reproduced broken: abandon a merge generator after one iteration,
confirm the lock stays held, confirm a second job attempted during that
window is correctly rejected, confirm the lock releases only once the
orphaned worker actually finishes.
**Residual risk, stated plainly:** this closes the *resource-contention*
race completely — two heavy jobs genuinely cannot run concurrently
anymore, regardless of what happens to the browser tab. It does **not**
make an abandoned job stop instantly; the worker thread still runs to
completion (or to its next cancellation checkpoint, for operations that
have one — see I2). What changed is that this is no longer *unsupervised*:
the lock correctly reflects that real work is still happening, and nothing
else can start until it's actually done.

### I2. 🟠 Checkpoint loading had no cancellation mechanism at all — for
architectures that download 25GB+, this was the single most necessary
place for it and the one place it was missing
**Where:** `do_load_checkpoint()` / `ImageGenSession.load()` /
`load_pipeline_custom()`.
**Why it matters:** found while fixing I1 — every other heavy operation
(merge, LoRA bake, VAE bake, generation) already had a `CancelToken` and a
Cancel button; checkpoint loading had neither. Combined with I1, this made
loading the single worst-case scenario in the whole app: a user starts
downloading Krea 2 (~25GB+), realizes they picked the wrong architecture
or don't have room for it, and had no way to stop it — not from the UI,
and (before I1's fix) not even by closing the tab, since that used to just
hide the problem while the download kept consuming bandwidth and disk
regardless.
**Fix, scoped honestly:** added a `CancelToken` and Cancel button,
threaded through `ImageGenSession.load()` into `load_pipeline_custom()`,
checked before each loading stage starts (text-encoder override, VAE
override, main pipeline). This is a real but bounded improvement, not a
claim of full interruptibility — `huggingface_hub`'s high-level
`from_pretrained()` doesn't expose a hook to interrupt a download already
in flight, so cancelling mid-download takes effect once that download
finishes rather than immediately, and this is stated in both the function
docstring and the UI-facing docstring rather than glossed over. What
*does* fully close the gap this round: I1's fix means that even for the
in-flight-and-uninterruptible case, the JOB_LOCK now correctly stays held
for as long as the download actually continues, so it can no longer
falsely appear available for a second job to start against — which is the
consequence that actually mattered.
**Residual risk:** truly interrupting an in-flight Hub download would
require dropping to `huggingface_hub`'s lower-level chunked-transfer
primitives instead of `from_pretrained()`'s convenience wrapper — a much
larger change, out of scope here, and not undertaken in place of an
honest description of the current limitation.

### I3. 🔴 A second instance of the exact "leftover dead code from a stale
edit" mistake already caught once in round 2 (G5) — this one reachable
enough to be worth calling a real bug, not just untidiness
**Where:** `do_generate()`'s exception handling.
**What happened:** found while restructuring the function for I1.
Two `except` clauses sat *after* an unconditional `except Exception as e:`
— `except guards.GpuOutOfMemoryError as e:` and a second `except Exception:`
— making both permanently unreachable (Python matches `except` clauses in
order; the first `except Exception` always wins). That alone would just be
dead code, consistent with G5. What makes this one worse: both dead
branches referenced `logs`, a variable name from an earlier version of
this function that no longer exists anywhere in its current body (replaced
by `log_box` several rounds ago) — and both used a 4-value yield shape,
stale from before this function's outputs grew to 6 values for the
live-preview/progress work in round 2. Had Python's exception-matching
order ever put either of these first (it structurally cannot, but *if* a
future edit reordered the clauses without noticing why), the result would
have been `NameError: name 'logs' is not defined`, crashing the generator
entirely — with the traceback pointing at exception-handling code, one of
the more confusing places for that to happen.
**Fix:** both dead clauses removed; the single remaining
`except Exception as e:` (which was already unconditionally catching
everything, correctly) handles all cases through the same
`_format_error(e)` path everything else in this codebase uses.
**The honest pattern here:** this is the second time an editing pass has
left a stale fragment behind mid-refactor (G5 was the first). Neither
shipped — both were caught by re-reading the function's actual final state
immediately after editing, before testing, which is the same discipline
applied throughout every round. Worth stating directly: this is evidence
that multi-step refactors under time pressure are exactly where this kind
of mistake recurs, which is precisely why "read the result, don't trust
the diff" has to be a standing habit rather than a one-time fix.
**Residual risk:** none for this specific instance — verified the full
regression suite and a fresh `do_generate` streaming test (including
cancellation) still pass identically after the removal.

## Testing performed (round 4 additions)

The core finding (I1) was reproduced twice, deliberately in different
forms, before being fixed: first in isolation directly against
`engine.run_with_live_log` (abandon the generator, observe the lock release
early, observe the worker keep running, observe a second `acquire()`
wrongly succeed), then again through the actual `app.do_merge` function
against real synthetic checkpoint files (abandon a real merge generator
after its first yield, confirm a second `JOB_LOCK.acquire()` is correctly
*rejected* this time, confirm the lock only releases once the orphaned
merge genuinely finishes writing its output). I2 was verified by
confirming `do_load_checkpoint` now yields the same 4-tuple shape with a
live `CancelToken` as every other tab, using a mocked pipeline load. I3 was
caught by direct re-reading during the I1 refactor, then verified fixed by
confirming the full streaming-generation regression suite (multi-image
batch, live preview, mid-batch cancellation) still passes identically. The
complete 8-mode merge suite and a full fresh app build+launch+HTTP-serve
test were both re-run clean as the final step, as in every prior round.

---

## Round 5 — auditing the cancellation feature added in round 4 itself

New features are exactly where new bugs hide, so this round pointed
specifically at what round 4 just added (checkpoint-load cancellation)
rather than re-treading already-audited ground.

### J1. 🔴 `ImageGenSession.load()` freed the currently-loaded checkpoint
*before* any validation ran — a typo, a missing file, an out-of-date
`diffusers`, or simply not enough disk space for a download all destroyed
a perfectly good, already-working session for no benefit
**Where:** `ImageGenSession.load()`.
**Why it matters:** `self.unload()` ran unconditionally as the second line
of the function, before checking whether the checkpoint path even exists,
whether the requested architecture's pipeline class is available in the
installed `diffusers`, or whether there's enough disk space for its
download. Reproduced concretely, three separate ways, before touching any
code: (1) a session with a working checkpoint loaded, asked to load a
checkpoint at a path that doesn't exist — the old session was destroyed
and the new load failed, leaving nothing; (2) the same session asked to
load an architecture whose pipeline class isn't in the installed
`diffusers` — same result; (3) the same session asked to load a checkpoint
that would need more disk space than is available — same result again.
None of these three failure modes take any meaningful time to detect (a
file-existence check, an attribute lookup, and a lightweight API call,
respectively) — they were failing the request for reasons that had nothing
to do with whatever was already loaded, yet destroyed it anyway as a side
effect of validation happening too late. This directly undercuts round
4's own cancellation feature (I2): the whole point of adding a Cancel
button was to protect against accidentally starting a load you didn't
want, but cancelling (or any other early failure) still guaranteed you
ended up with *nothing* loaded rather than back where you started.
**Fix:** restructured `load()` so every cheap, fast, purely-local-or-
lightweight-API check — file existence for legacy checkpoints; pipeline
class availability and disk space for the newer architectures — runs
*before* `self.unload()`, with the cancellation check immediately after
those and still before unload. `self.unload()` now only runs once the
request has passed every check that can be resolved quickly. Verified all
three original failure modes (missing file, missing pipeline class,
insufficient disk space) now leave a previously-loaded session intact and
usable, using a fully isolated stub `diffusers`/`huggingface_hub` module
so the tests don't depend on network access. Also re-verified the happy
path (a load that actually succeeds) still works identically — the
reordering doesn't skip or duplicate any real loading step. Then verified
the same fix is reachable through the actual `app.do_load_checkpoint`
UI-facing function against a real (mocked-pipeline) session, not just the
engine layer in isolation.
**Residual risk, stated plainly, same as I2:** this cannot extend across
the *slow* part — once the real download has actually started, cancelling
or a failure partway through will still leave the session empty, because
holding both the old and new pipeline in memory simultaneously (the only
way to make that fully recoverable) would reintroduce the exact
memory-pressure risk this whole area of the app exists to prevent. What
changed is that every failure detectable *before* committing to that
memory trade-off no longer forces it unnecessarily. One minor, deliberate
inefficiency accepted as part of this fix: the disk-space check now runs
twice for the newer architectures (once in this new early check, once
again inside `load_pipeline_custom()`) — a small redundant API call, kept
because it means `load_pipeline_custom()` remains fully self-contained and
correct if ever called directly rather than only through `session.load()`.

## Testing performed (round 5 additions)

J1 was reproduced three separate ways before being fixed — missing
checkpoint file, missing pipeline class, and insufficient disk space —
each confirmed to destroy an already-loaded session under the old code,
then confirmed fixed under the new ordering, using fully isolated stub
modules for `diffusers` and `huggingface_hub` so none of the tests depend
on network access this sandbox doesn't have. The happy path (a load that
actually succeeds) was re-verified to still work identically after the
reordering. The fix was then verified reachable through the actual
`app.do_load_checkpoint` function against a real session object, not just
`engine.ImageGenSession` in isolation. The full 8-mode merge regression
suite and a fresh app build+launch+HTTP-serve test were both re-run clean
as the final step.
