"""
Model Breeder — Gradio Edition — Engine
========================================
Pure-python engine for streaming Stable Diffusion checkpoint merging, LoRA
baking, VAE baking, metadata inspection, and image generation. Ported from
the original "Model Breeder" Jupyter/Colab notebook, with all ipywidgets /
IPython.display UI code removed and replaced with plain callback hooks
(progress_cb(percent), log_cb(message)) so it can be driven from Gradio,
a CLI, or anything else.

Design choices carried over from the source notebook (and why):
  - Everything is streamed tensor-by-tensor straight to disk. Nothing ever
    holds two full checkpoints in RAM at once — this is what makes 6-7GB
    SDXL merges possible on a free Colab / laptop.
  - Every merged tensor is NaN/Inf-sanitised and clamped to the FP16-safe
    range before being written, and the whole output file is re-scanned
    after writing (post-flight) — if anything corrupt slipped through, the
    file is deleted rather than left on disk looking fine.
  - Legacy .ckpt files are pickles. torch.load(weights_only=True) is always
    tried first; only on failure does this fall back to the unsafe pickle
    path, with a loud, visible warning surfaced through log_cb — never
    silently.
"""
from __future__ import annotations

import gc
import json
import math
import os
import queue
import struct
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file as st_load_file

import guards

# ─────────────────────────────────────────────────────────────────────────
# Streaming-write durability helper (CRITIQUE.md D1): wraps a file handle
# and fsyncs to physical disk every ~200MB written, so the on-screen
# progress percentage tracks bytes actually durable, not just buffered.
# ─────────────────────────────────────────────────────────────────────────
class _DurableWriter:
    __slots__ = ('f', '_since_sync', '_sync_every')

    def __init__(self, f, sync_every_bytes: int = 200 * 1024 * 1024):
        self.f = f
        self._since_sync = 0
        self._sync_every = sync_every_bytes

    def write(self, data: bytes):
        self.f.write(data)
        self._since_sync += len(data)
        if self._since_sync >= self._sync_every:
            self.f.flush()
            os.fsync(self.f.fileno())
            self._since_sync = 0

    def final_sync(self):
        try:
            self.f.flush()
            os.fsync(self.f.fileno())
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
DTYPE_BYTES = {'F16': 2, 'F32': 4, 'BF16': 2, 'I32': 4, 'I64': 8,
               'U8': 1, 'I8': 1, 'F64': 8, 'BOOL': 1}
NONFLOAT_DTYPES = {'I64', 'I32', 'I16', 'I8', 'U8', 'U16', 'U32', 'U64', 'BOOL'}
TORCH2ST = {torch.int64: 'I64', torch.int32: 'I32', torch.int16: 'I16',
            torch.int8: 'I8', torch.uint8: 'U8', torch.bool: 'BOOL'}

BLOCK_NAMES = ['BASE', 'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08',
               'MID00', 'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08']

VAE_PFX = 'first_stage_model.'
VAE_MAP = [('encoder.', VAE_PFX + 'encoder.'), ('decoder.', VAE_PFX + 'decoder.'),
           ('quant_conv.', VAE_PFX + 'quant_conv.'), ('post_quant_conv.', VAE_PFX + 'post_quant_conv.')]

# Generic, content-neutral block-weight presets. 0.0 = keep 100% of Model A
# at that block; 1.0 = full alpha; values >1 extrapolate past B.
CKPT_RECIPES = {
    'All Blocks Equal (standard full blend)': ','.join(['1'] * 20),
    'UNet Only (keep A text encoder)': '0,' + ','.join(['1'] * 19),
    'Text Encoder Only': '1,' + ','.join(['0'] * 19),
    'Style Preserve (A keeps output blocks 4-8)': ('1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0'),
    'Conservative (half strength everywhere)': ','.join(['0.5'] * 20),
    'Custom (edit manually below)': ','.join(['1'] * 20),
}
LORA_RECIPES = {
    'All Blocks Equal (standard bake)': ','.join(['1'] * 20),
    'UNet Only (skip text encoder)': '0,' + ','.join(['1'] * 19),
    'Style Overlay (OUT04-OUT08 only)': ('0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1'),
    'Conservative (half strength everywhere)': ','.join(['0.5'] * 20),
    'Custom (edit manually below)': ','.join(['1'] * 20),
}

METHOD_OPTS = [
    ('Weighted Sum  (A*(1-a) + B*a)', 0),
    ('Add Difference  (A + (B-C)*a)', 1),
    ('Sigmoid Blend  (smooth S-curve)', 2),
    ('SLERP  (spherical interpolation)', 3),
    ('Sum Twice  (A,B via alpha -> C via beta)', 4),
    ('Triple Sum  (A:B:C = 1-(a+b) : a : b)', 5),
    ('DARE Merge  (drop-and-rescale, cross-base-model)', 6),
    ('XDARE  (DARE + CLIP-aware masking)', 7),
]
METHOD_LABEL_TO_MODE = {lbl: m for lbl, m in METHOD_OPTS}
MODE_NAMES = {0: 'Weighted Sum', 1: 'Add Difference', 2: 'Sigmoid Blend', 3: 'SLERP',
              4: 'Sum Twice', 5: 'Triple Sum', 6: 'DARE Merge', 7: 'XDARE'}


# ─────────────────────────────────────────────────────────────────────────
# Small shared helpers
# ─────────────────────────────────────────────────────────────────────────
def is_text_key(k: str) -> bool:
    return ('conditioner' in k or 'cond_stage_model' in k
            or 'text_encoder' in k or 'text_model' in k)


def blk_idx(k: str) -> int:
    """Map an SDXL/SD1.5 checkpoint key to block index 0-19. -1 = unknown."""
    if is_text_key(k):
        return 0
    for i in range(9):
        if f'input_blocks.{i}.' in k or f'input_blocks_{i}_' in k:
            return i + 1
    if 'middle_block' in k:
        return 10
    for i in range(9):
        if f'output_blocks.{i}.' in k or f'output_blocks_{i}_' in k:
            return 11 + i
    return -1


def block_cat(k: str):
    if k.startswith('first_stage_model.'):
        return 'vae', 0
    if is_text_key(k):
        return 'text', 0
    try:
        if 'model.diffusion_model.input_blocks.' in k:
            idx = int(k.split('model.diffusion_model.input_blocks.')[1].split('.')[0])
            return 'input', idx
        if 'model.diffusion_model.middle_block' in k:
            return 'mid', 0
        if 'model.diffusion_model.output_blocks.' in k:
            idx = int(k.split('model.diffusion_model.output_blocks.')[1].split('.')[0])
            return 'output', idx
    except (IndexError, ValueError):
        pass
    return 'other', 0


def parse_bw(s: str):
    try:
        vals = [float(x.strip()) for x in s.strip().split(',')]
    except Exception as e:
        return None, str(e)
    if len(vals) != 20:
        return None, f'Need exactly 20 comma-separated values, got {len(vals)}'
    return vals, None


def parse_ratio(s: str):
    """Accepts either a single scalar (uniform across all 20 blocks) or 20
    comma-separated values."""
    s = s.strip()
    try:
        if ',' not in s:
            v = float(s)
            return [v] * 20, None
        vals = [float(x.strip()) for x in s.split(',')]
        if len(vals) != 20:
            return None, f'Need 20 values, got {len(vals)}'
        return vals, None
    except Exception as e:
        return None, str(e)


def sf_header(path) -> dict:
    with open(str(path), 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw.decode('utf-8', 'ignore').rstrip('\x00'))


def peek_keys(path, n: Optional[int] = None):
    try:
        hdr = sf_header(path)
        keys = [k for k in hdr if k != '__metadata__']
        return keys[:n] if n else keys
    except Exception:
        return []


def is_lora_file(p: Path) -> bool:
    keys = peek_keys(p)
    return any('lora_down' in k or 'lora_up' in k or 'lokr_w1' in k or 'lokr_w2' in k
               or 'hada_w1_a' in k or 'hada_w2_a' in k for k in keys)


def is_vae_file(p: Path) -> bool:
    if 'vae' in p.name.lower():
        return True
    vp = ('encoder.', 'decoder.', 'quant_conv.', 'post_quant_conv.')
    keys = peek_keys(p, 40)
    return bool(keys) and all(any(k.startswith(x) for x in vp) for k in keys)


def list_checkpoints(folder) -> list:
    # Defense in depth against dotfiles (pathlib's glob('*'), unlike shell
    # globbing, DOES match hidden files — verified this concretely; a
    # partially-written upload temp file must never surface here even if
    # something upstream ever names one with a matching extension again).
    folder = Path(folder)
    return sorted(f.name for f in list(folder.glob('*.safetensors')) + list(folder.glob('*.ckpt'))
                  if not f.name.startswith('.') and not is_lora_file(f) and not is_vae_file(f))


def list_loras(folder) -> list:
    folder = Path(folder)
    return sorted(f.name for f in list(folder.glob('*.safetensors')) + list(folder.glob('*.pt'))
                  if not f.name.startswith('.') and is_lora_file(f))


def list_vaes(folder) -> list:
    folder = Path(folder)
    return sorted(f.name for f in list(folder.glob('*.safetensors')) + list(folder.glob('*.pt'))
                  if not f.name.startswith('.') and is_vae_file(f))


def list_local_dirs(folder) -> list:
    """Subdirectory names directly under `folder` — used for local diffusers-
    format snapshots (input/diffusers_repos/<name>/) and local text-encoder
    folders (input/text_encoders/<name>/), both of which need a full folder
    (weights + tokenizer/config), not a single file."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(d.name for d in folder.iterdir() if d.is_dir() and not d.name.startswith('.'))


def detect_arch(path) -> str:
    """Return 'sdxl', 'sd15', or 'unknown' from the first ~60 header keys."""
    try:
        keys = peek_keys(path, 60)
        if any('conditioner' in k for k in keys):
            return 'sdxl'
        if any('cond_stage_model' in k for k in keys):
            return 'sd15'
    except Exception:
        pass
    return 'unknown'


def load_any(path, security_log: Optional[Callable[[str], None]] = None):
    """Returns (handle_or_dict, mode) where mode is 'handle' (streamed
    safetensors) or 'dict' (fully loaded, for .ckpt/.pt)."""
    path = Path(path)
    try:
        h = safe_open(str(path), framework='pt', device='cpu')
        _ = len(list(h.keys()))
        return h, 'handle'
    except Exception:
        pass
    if path.suffix == '.ckpt':
        # SECURITY: legacy .ckpt files are Python pickles. weights_only=True
        # is always tried first; only on failure do we fall back to the
        # unsafe pickle path, and only with a loud, visible warning.
        try:
            sd = torch.load(str(path), map_location='cpu', weights_only=True)
        except Exception:
            if security_log:
                security_log(
                    f'⚠ SECURITY: {path.name} could not be loaded safely '
                    f'(weights_only=True failed) — falling back to legacy pickle load. '
                    f'Only do this for .ckpt files from a source you trust; a malicious '
                    f'.ckpt can execute arbitrary code on load. Prefer .safetensors.')
            sd = torch.load(str(path), map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k: v for k, v in sd.items() if hasattr(v, 'shape')}
        return sd, 'dict'
    return st_load_file(str(path), device='cpu'), 'dict'


def hkeys(h, m):
    return list(h.keys())


def hget(h, m, k):
    return h[k] if m == 'dict' else h.get_tensor(k)


def nan_safe(t, fp16: bool):
    if not torch.is_floating_point(t):
        return t
    t = torch.nan_to_num(t.float(), nan=0.0, posinf=65504.0, neginf=-65504.0).clamp(-65504.0, 65504.0)
    return t.half() if fp16 else t


def key_seed(key: str, base_seed: int) -> int:
    return (int(base_seed) + zlib.crc32(key.encode('utf-8'))) & 0xffffffff


def dare_mask(shape, keep_prob: float, seed: int):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    return torch.rand(shape, generator=g) < keep_prob


def clipxor_mask(delta, keep_prob: float):
    flat = delta.abs().flatten()
    k = max(1, int(round(keep_prob * flat.numel())))
    if k >= flat.numel():
        return torch.ones_like(delta, dtype=torch.bool)
    thresh = torch.topk(flat, k, largest=True).values.min()
    return delta.abs() >= thresh


def dare_delta(ta, tb, eff_alpha, droprate, seed, key, clip_max, use_magnitude_mask=False):
    delta = tb - ta
    keep_prob = max(1.0 - float(droprate), 1e-6)
    if droprate <= 1e-9:
        scaled = delta
    else:
        mask = (clipxor_mask(delta, keep_prob) if use_magnitude_mask
                else dare_mask(delta.shape, keep_prob, key_seed(key, seed)))
        scaled = delta * mask.float() / keep_prob
    t = ta + eff_alpha * scaled
    return t.clamp(-clip_max, clip_max)


def preflight_scan(handle, mode, label, log, path=None):
    bad = []
    int_dtypes = {'I32', 'I64', 'U8', 'I8', 'BOOL'}
    skip_set = set()
    if mode == 'handle' and path is not None:
        try:
            hdr = sf_header(path)
            skip_set = {k for k, v in hdr.items() if k != '__metadata__'
                        and v.get('dtype', 'F32') in int_dtypes}
        except Exception:
            pass
    for k in hkeys(handle, mode):
        if k in skip_set:
            continue
        t = hget(handle, mode, k)
        if torch.is_floating_point(t) and (torch.isnan(t).any() or torch.isinf(t).any()):
            bad.append(k)
    if bad:
        log(f'⚠ Pre-flight [{label}]: {len(bad)} NaN/Inf tensor(s) found — will sanitise')
    else:
        log(f'  Pre-flight [{label}]: clean')
    return bad


def postflight_scan(out_path: Path, log):
    bad = []
    n_total = 0
    with safe_open(str(out_path), framework='pt', device='cpu') as h:
        for k in h.keys():
            n_total += 1
            t = h.get_tensor(k)
            if torch.is_floating_point(t) and (torch.isnan(t).any() or torch.isinf(t).any()):
                bad.append(k)
    if bad:
        try:
            out_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f'Post-flight: {len(bad)} NaN/Inf tensor(s) found in output — file deleted.')
    log(f'  Post-flight: all {n_total} tensors clean')


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint x Checkpoint Merger  (8 methods, per-block weights, VAE bake)
# ─────────────────────────────────────────────────────────────────────────
def merge_checkpoints(pa, pb, pc, alpha, mode, out_path, fp16, vae_path, bw_vec,
                       progress_cb, log_cb, base_pred='epsilon', merge_pred='epsilon',
                       clip_max=65504.0, beta=0.0, alpha_ratio_vec=None, beta_ratio_vec=None,
                       dare_seed=0, cancel_token=None):
    pa, pb = Path(pa), Path(pb)
    pc = Path(pc) if pc else None
    out_path = Path(out_path)
    open_handles = []

    # CRITIQUE.md A2/B1: refuse to start rather than fail halfway.
    guards.require_ram(768, 'a checkpoint merge')
    guards.check_disk_space(out_path.parent, guards.estimate_output_bytes(pa, pb, pc, fp16=fp16),
                             'the merged checkpoint')
    guards.sanity_check_block_weights(bw_vec, log_cb)

    def _load(p):
        h, m = load_any(p, security_log=log_cb)
        if m == 'handle':
            open_handles.append(h)
        elif m == 'dict':
            guards.require_ram(guards.estimate_ckpt_ram_mb(p), f'loading {Path(p).name} fully into RAM')
        return h, m

    try:
        if mode in (6, 7):
            mismatch = base_pred.strip().lower() != merge_pred.strip().lower()
            log_cb(f'  Pred type: A={base_pred}  B={merge_pred}'
                   + ('  (mismatch allowed under DARE/XDARE)' if mismatch else '  OK'))
        else:
            if base_pred.strip().lower() != merge_pred.strip().lower():
                raise ValueError(
                    f'Prediction type mismatch: A={base_pred!r} B={merge_pred!r}. '
                    f'Epsilon and V-Pred models encode noise in opposite directions and '
                    f'cannot be safely merged with this method — use DARE/XDARE to bridge them.')
            log_cb(f'  Pred type: A={base_pred}  B={merge_pred}  OK')

        if alpha_ratio_vec is None:
            alpha_ratio_vec = [alpha] * 20
        if beta_ratio_vec is None:
            beta_ratio_vec = [beta] * 20

        log_cb(f'Opening A: {pa.name}')
        fa, ma = _load(pa)
        log_cb(f'Opening B: {pb.name}')
        fb, mb = _load(pb)
        fc, mc = (_load(pc) if pc else (None, None))
        if pc:
            log_cb(f'Opening C: {pc.name}')

        ka, kb = set(hkeys(fa, ma)), set(hkeys(fb, mb))
        kc = set(hkeys(fc, mc)) if fc else set()
        all_keys = sorted(ka | kb)
        out_dtype_str = 'F16' if fp16 else 'F32'
        log_cb(f'{len(all_keys)} keys · output {out_dtype_str}')
        progress_cb(5)

        arch_a, arch_b = detect_arch(pa), detect_arch(pb)
        if arch_a != 'unknown' and arch_b != 'unknown' and arch_a != arch_b:
            log_cb(f'⚠ Arch mismatch: A={arch_a}  B={arch_b} — keys unique to one model are copied verbatim')
        else:
            log_cb(f'  Arch: A={arch_a}  B={arch_b}  OK')

        preflight_scan(fa, ma, pa.name, log_cb, path=pa)
        preflight_scan(fb, mb, pb.name, log_cb, path=pb)
        if fc is not None:
            preflight_scan(fc, mc, pc.name, log_cb, path=pc)

        stats = {c: {'n': 0, 'skip': 0, 'guard': 0, 'asum': 0.0, 'bsum': 0.0}
                 for c in ('text', 'input', 'mid', 'output', 'vae', 'other')}

        vae_tensors = {}
        if vae_path:
            log_cb('Pre-loading VAE...')
            hv, mv = _load(vae_path)
            vkeys = hkeys(hv, mv)
            standalone = not any(k.startswith(VAE_PFX) for k in vkeys[:20])
            for vk in vkeys:
                t = hget(hv, mv, vk).float()
                if fp16:
                    t = t.half()
                sd_k = vk
                if standalone:
                    for src, dst in VAE_MAP:
                        if vk.startswith(src):
                            sd_k = dst + vk[len(src):]
                            break
                    else:
                        sd_k = VAE_PFX + vk
                vae_tensors[sd_k] = t
            gc.collect()
            log_cb(f'VAE ready — {len(vae_tensors)} tensors')
            progress_cb(10)

        def _shapes(path_, m_str, handle):
            if m_str == 'handle' and path_.suffix == '.safetensors':
                hdr = sf_header(path_)
                return {k: hdr[k]['shape'] for k in hdr if k != '__metadata__'}
            out = {}
            for k, v in handle.items():
                try:
                    out[k] = list(v.shape)
                except AttributeError:
                    pass
            return out

        sa = _shapes(pa, ma, fa)
        sb = _shapes(pb, mb, fb)

        def _nonfloat_dtypes(path_, m_str, handle):
            out = {}
            if m_str == 'handle' and path_.suffix == '.safetensors':
                try:
                    hdr = sf_header(path_)
                    for k, v in hdr.items():
                        if k == '__metadata__':
                            continue
                        dt = v.get('dtype', 'F32')
                        if dt in NONFLOAT_DTYPES:
                            out[k] = dt
                except Exception:
                    pass
            elif handle is not None:
                try:
                    for k, v in handle.items():
                        try:
                            if not torch.is_floating_point(v):
                                out[k] = TORCH2ST.get(v.dtype, 'I64')
                        except AttributeError:
                            pass
                except Exception:
                    pass
            return out

        nf_a = _nonfloat_dtypes(pa, ma, fa)
        nf_b = _nonfloat_dtypes(pb, mb, fb)

        vae_extra = [k for k in vae_tensors if k not in ka and k not in kb]
        write_keys = all_keys + vae_extra

        mode_name = MODE_NAMES.get(mode, 'Weighted Sum')
        meta = {
            'creator': 'Model Breeder (Gradio Edition)',
            'tool': 'Checkpoint x Checkpoint Merger',
            'merge_mode': mode_name,
            'merged_checkpoints': ' + '.join(p.name for p in (pa, pb, pc) if p),
            'checkpoint_a': pa.name, 'checkpoint_b': pb.name,
            'checkpoint_c': (pc.name if pc else 'none'),
            'block_weights_json': json.dumps([round(w, 4) for w in bw_vec]),
            'vae_baked': (Path(vae_path).name if vae_path else 'none'),
            'base_pred_type': base_pred, 'merge_pred_type': merge_pred,
            'output_precision': out_dtype_str,
            'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        }
        if mode in (4, 5):
            meta['beta'] = f'{beta:.4f}'
            meta['alpha_ratio_json'] = json.dumps([round(w, 4) for w in alpha_ratio_vec])
            meta['beta_ratio_json'] = json.dumps([round(w, 4) for w in beta_ratio_vec])
        elif mode in (6, 7):
            meta['dare_droprate'] = f'{beta:.4f}'
            meta['dare_seed'] = str(dare_seed)
            meta['alpha_ratio_json'] = json.dumps([round(w, 4) for w in alpha_ratio_vec])
            if mode == 7:
                meta['clipxor_note'] = 'best-effort reconstruction of CLIP-aware text-encoder masking'

        offset = 0
        hdr_dict = {'__metadata__': meta}
        for k in write_keys:
            if k in vae_tensors:
                shape = list(vae_tensors[k].shape)
            else:
                shape = sa.get(k, []) if k in ka else sb.get(k, [])
            if k in vae_tensors:
                key_dtype = out_dtype_str
            elif k in ka:
                key_dtype = nf_a.get(k, out_dtype_str)
            else:
                key_dtype = nf_b.get(k, out_dtype_str)
            nbytes = DTYPE_BYTES.get(key_dtype, 2) * max(math.prod(shape) if shape else 1, 1)
            hdr_dict[k] = {'dtype': key_dtype, 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
            offset += nbytes
        hdr_json = json.dumps(hdr_dict, separators=(',', ':')).encode('utf-8')
        pad = (8 - len(hdr_json) % 8) % 8
        hdr_bytes = hdr_json + b' ' * pad
        progress_cb(12)
        log_cb(f'Streaming merge -> {out_path.name}')
        n_wk = len(write_keys)

        with open(str(out_path), 'wb') as _raw_fout:
            fout = _DurableWriter(_raw_fout)
            fout.write(struct.pack('<Q', len(hdr_bytes)))
            fout.write(hdr_bytes)
            for i, k in enumerate(write_keys):
                if cancel_token is not None:
                    cancel_token.check()
                if k in vae_tensors:
                    t = vae_tensors[k]
                    if ma == 'dict' and k in fa:
                        del fa[k]
                    if mb == 'dict' and k in fb:
                        del fb[k]
                else:
                    ia, ib = k in ka, k in kb
                    cat, _cidx = block_cat(k)
                    cs = stats.setdefault(cat, {'n': 0, 'skip': 0, 'guard': 0, 'asum': 0.0, 'bsum': 0.0})
                    if (ia and k in nf_a) or (not ia and ib and k in nf_b):
                        cs['skip'] += 1
                        cs['n'] += 1
                        t = hget(fa if ia else fb, ma if ia else mb, k)
                    elif ia and ib:
                        ta_raw = hget(fa, ma, k)
                        tb_raw = hget(fb, mb, k)
                        if not (torch.is_floating_point(ta_raw) and torch.is_floating_point(tb_raw)
                                and ta_raw.shape == tb_raw.shape):
                            cs['skip'] += 1
                            cs['n'] += 1
                            t = ta_raw
                            if torch.is_floating_point(t):
                                t = t.half() if fp16 else t.float()
                        else:
                            bi = blk_idx(k)
                            if mode in (4, 5, 6, 7):
                                if 0 <= bi < 20:
                                    eff, eff_b = alpha_ratio_vec[bi], beta_ratio_vec[bi]
                                elif cat == 'vae':
                                    eff, eff_b = 0.0, 0.0
                                else:
                                    eff = alpha_ratio_vec[0] if alpha_ratio_vec else alpha
                                    eff_b = beta_ratio_vec[0] if beta_ratio_vec else beta
                            else:
                                if 0 <= bi < 20:
                                    bw = bw_vec[bi]
                                elif cat == 'vae':
                                    bw = 0.0
                                else:
                                    bw = 1.0
                                eff = alpha * bw
                            cs['n'] += 1
                            cs['asum'] += eff
                            if mode in (4, 5, 6, 7):
                                cs['bsum'] += eff_b
                            ta, tb = ta_raw.float(), tb_raw.float()
                            if torch.isnan(ta).any() or torch.isinf(ta).any():
                                ta = torch.nan_to_num(ta, nan=0., posinf=clip_max, neginf=-clip_max).clamp(-clip_max, clip_max)
                            if torch.isnan(tb).any() or torch.isinf(tb).any():
                                tb = torch.nan_to_num(tb, nan=0., posinf=clip_max, neginf=-clip_max).clamp(-clip_max, clip_max)
                            if mode == 0:
                                t = ta * (1.0 - eff) + tb * eff
                            elif mode == 1:
                                if fc and k in kc:
                                    tc = hget(fc, mc, k).float()
                                    t = ta + (tb - tc) * eff
                                    del tc
                                else:
                                    t = ta * (1.0 - eff) + tb * eff
                            elif mode == 2:
                                if eff == 0.0:
                                    t = ta
                                else:
                                    s = 1. / (1. + torch.exp(torch.tensor(-12. * (eff - .5)))).item()
                                    t = ta * (1.0 - s) + tb * s
                            elif mode == 3:
                                if eff == 0.0:
                                    t = ta
                                elif eff == 1.0:
                                    t = tb
                                else:
                                    v0, v1 = ta.reshape(-1), tb.reshape(-1)
                                    n0, n1 = torch.linalg.vector_norm(v0), torch.linalg.vector_norm(v1)
                                    if n0 < 1e-8 or n1 < 1e-8:
                                        t = ta * (1.0 - eff) + tb * eff
                                    else:
                                        dot = torch.clamp(torch.dot(v0, v1) / (n0 * n1), -1.0, 1.0)
                                        if torch.abs(dot) > 0.9995:
                                            t = ta * (1.0 - eff) + tb * eff
                                        else:
                                            omega = torch.acos(dot)
                                            so = torch.sin(omega)
                                            s0 = torch.sin((1.0 - eff) * omega) / so
                                            s1 = torch.sin(eff * omega) / so
                                            t = (s0 * v0 + s1 * v1).reshape(ta.shape)
                                            del so, omega, s0, s1, dot
                                    del v0, v1, n0, n1
                            elif mode == 4:
                                step1 = ta * (1.0 - eff) + tb * eff
                                if fc and k in kc:
                                    tc = hget(fc, mc, k).float()
                                    t = step1 * (1.0 - eff_b) + tc * eff_b
                                    del tc
                                else:
                                    t = step1
                                del step1
                            elif mode == 5:
                                if fc and k in kc:
                                    tc = hget(fc, mc, k).float()
                                    wA = 1.0 - (eff + eff_b)
                                    t = ta * wA + tb * eff + tc * eff_b
                                    del tc
                                else:
                                    t = ta * (1.0 - eff) + tb * eff
                            elif mode == 6:
                                t = dare_delta(ta, tb, eff, eff_b, dare_seed, k, clip_max, use_magnitude_mask=False)
                            else:
                                t = dare_delta(ta, tb, eff, eff_b, dare_seed, k, clip_max, use_magnitude_mask=(cat == 'text'))
                            if torch.isnan(t).any() or torch.isinf(t).any():
                                t = ta.clamp(-clip_max, clip_max)
                                cs['guard'] += 1
                            else:
                                t = t.clamp(-clip_max, clip_max)
                            del ta, tb
                            t = t.half() if fp16 else t
                    else:
                        cs['skip'] += 1
                        cs['n'] += 1
                        t = hget(fa if ia else fb, ma if ia else mb, k)
                        if torch.is_floating_point(t):
                            t = t.half() if fp16 else t.float()
                    if ma == 'dict' and k in fa:
                        del fa[k]
                    if mb == 'dict' and k in fb:
                        del fb[k]
                    if mc == 'dict' and fc and k in fc:
                        del fc[k]
                t = nan_safe(t, fp16)
                nbytes = t.numel() * t.element_size()
                fout.write(t.contiguous().numpy().tobytes())
                del t
                if i % 40 == 0 or nbytes > 64 * 1024 * 1024:
                    gc.collect()
                    progress_cb(int(12 + (i + 1) / n_wk * 76))
            fout.final_sync()
        del fa, fb, fc, vae_tensors
        gc.collect()
        progress_cb(98)
        log_cb('Post-flight scan...')
        postflight_scan(out_path, log_cb)
        progress_cb(100)
        log_cb(f'✓ Merge complete: {out_path.name}')
        active = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                  for k, v in stats.items()}
        log_cb('Per-category summary: ' + json.dumps(active))
        return stats
    except Exception:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        raise
    finally:
        for h in open_handles:
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint + LoRA baking  (standard LoRA / LoKr / LoHA; safetensors only)
# ─────────────────────────────────────────────────────────────────────────
D2C_XL = {
    'down_blocks_0_attentions_0': 'input_blocks_1_1', 'down_blocks_0_attentions_1': 'input_blocks_2_1',
    'down_blocks_1_attentions_0': 'input_blocks_4_1', 'down_blocks_1_attentions_1': 'input_blocks_5_1',
    'mid_block_attentions_0': 'middle_block_1', 'mid_block_0_attentions_0': 'middle_block_1',
    'up_blocks_0_attentions_0': 'output_blocks_0_1', 'up_blocks_0_attentions_1': 'output_blocks_1_1',
    'up_blocks_0_attentions_2': 'output_blocks_2_1', 'up_blocks_1_attentions_0': 'output_blocks_3_1',
    'up_blocks_1_attentions_1': 'output_blocks_4_1', 'up_blocks_1_attentions_2': 'output_blocks_5_1',
}
D2C_15 = {
    'down_blocks_0_attentions_0': 'input_blocks_1_1', 'down_blocks_0_attentions_1': 'input_blocks_2_1',
    'down_blocks_1_attentions_0': 'input_blocks_4_1', 'down_blocks_1_attentions_1': 'input_blocks_5_1',
    'down_blocks_2_attentions_0': 'input_blocks_7_1', 'down_blocks_2_attentions_1': 'input_blocks_8_1',
    'mid_block_attentions_0': 'middle_block_1', 'mid_block_0_attentions_0': 'middle_block_1',
    'up_blocks_0_attentions_0': 'output_blocks_3_1', 'up_blocks_0_attentions_1': 'output_blocks_4_1',
    'up_blocks_0_attentions_2': 'output_blocks_5_1', 'up_blocks_1_attentions_0': 'output_blocks_6_1',
    'up_blocks_1_attentions_1': 'output_blocks_7_1', 'up_blocks_1_attentions_2': 'output_blocks_8_1',
    'up_blocks_2_attentions_0': 'output_blocks_9_1', 'up_blocks_2_attentions_1': 'output_blocks_10_1',
    'up_blocks_2_attentions_2': 'output_blocks_11_1',
}


def is_te_key(k: str) -> bool:
    return any(x in k for x in ('conditioner', 'cond_stage_model', 'text_encoder'))


def bake_lora_stage(base_p, loras_wts, bw_vec, out_p, fp16, progress_cb, log_cb,
                     unet_w=1.0, te_w=1.0, meta_extra=None, cancel_token=None):
    """loras_wts: list of (lora_path, weight). Supports standard LoRA, LoKr, LoHA.
    Base checkpoint must be .safetensors (streaming requires the safetensors
    header format; for a .ckpt base, merge it with Checkpoint x Checkpoint
    Merger's Weighted Sum first, or convert it to .safetensors)."""
    base_p, out_p = Path(base_p), Path(out_p)
    if base_p.suffix != '.safetensors':
        raise ValueError('LoRA baking requires a .safetensors base checkpoint.')

    guards.require_ram(512, 'a LoRA bake')
    guards.check_disk_space(out_p.parent, guards.estimate_output_bytes(base_p, fp16=fp16), 'the baked checkpoint')
    guards.sanity_check_block_weights(bw_vec, log_cb)

    PFXS_XL = [('lora_unet_', 'model_diffusion_model_'), ('lora_te1_', 'conditioner_embedders_0_transformer_'),
               ('lora_te2_', 'conditioner_embedders_1_model_transformer_'), ('lora_te_', 'cond_stage_model_transformer_'),
               ('lora_text_encoder_', 'cond_stage_model_transformer_'), ('unet.', 'model_diffusion_model_'),
               ('text_encoder_2.', 'conditioner_embedders_1_model_transformer_'), ('text_encoder.', 'conditioner_embedders_0_transformer_')]
    PFXS_15 = [('lora_unet_', 'model_diffusion_model_'), ('lora_te_', 'cond_stage_model_transformer_'),
               ('lora_text_encoder_', 'cond_stage_model_transformer_'), ('unet.', 'model_diffusion_model_'),
               ('text_encoder.', 'cond_stage_model_transformer_')]

    log_cb(f'Base: {base_p.name}')
    progress_cb(4)

    def _b(k, s):
        return k[:-len(s)]

    handles, pair_maps = [], []
    for lp, lw in loras_wts:
        lp = Path(lp)
        log_cb(f'  LoRA {lp.name}  weight={lw:+.3f}')
        h = safe_open(str(lp), framework='pt', device='cpu')
        handles.append(h)
        pairs, dora = {}, set()
        for k in h.keys():
            if k.endswith('.lora_down.weight'):
                pairs.setdefault(_b(k, '.lora_down.weight'), {})['down'] = k
            elif k.endswith('.lora_up.weight'):
                pairs.setdefault(_b(k, '.lora_up.weight'), {})['up'] = k
            elif k.endswith('_lora_down_weight'):
                pairs.setdefault(_b(k, '_lora_down_weight'), {})['down'] = k
            elif k.endswith('_lora_up_weight'):
                pairs.setdefault(_b(k, '_lora_up_weight'), {})['up'] = k
            elif k.endswith('.alpha'):
                pairs.setdefault(_b(k, '.alpha'), {})['alpha'] = k
            elif k.endswith('_alpha') and not k.endswith('.alpha'):
                pairs.setdefault(_b(k, '_alpha'), {})['alpha'] = k
            elif k.endswith('.dora_scale'):
                dora.add(_b(k, '.dora_scale'))
            elif k.endswith('_dora_scale'):
                dora.add(_b(k, '_dora_scale'))
            elif k.endswith('.lokr_w1'):
                pairs.setdefault(_b(k, '.lokr_w1'), {})['lokr_w1'] = k
            elif k.endswith('.lokr_w1_a'):
                pairs.setdefault(_b(k, '.lokr_w1_a'), {})['lokr_w1_a'] = k
            elif k.endswith('.lokr_w1_b'):
                pairs.setdefault(_b(k, '.lokr_w1_b'), {})['lokr_w1_b'] = k
            elif k.endswith('.lokr_w2'):
                pairs.setdefault(_b(k, '.lokr_w2'), {})['lokr_w2'] = k
            elif k.endswith('.lokr_w2_a'):
                pairs.setdefault(_b(k, '.lokr_w2_a'), {})['lokr_w2_a'] = k
            elif k.endswith('.lokr_w2_b'):
                pairs.setdefault(_b(k, '.lokr_w2_b'), {})['lokr_w2_b'] = k
            elif k.endswith('.lokr_t2'):
                pairs.setdefault(_b(k, '.lokr_t2'), {})['lokr_t2'] = k
            elif k.endswith('.hada_w1_a'):
                pairs.setdefault(_b(k, '.hada_w1_a'), {})['hada_w1_a'] = k
            elif k.endswith('.hada_w1_b'):
                pairs.setdefault(_b(k, '.hada_w1_b'), {})['hada_w1_b'] = k
            elif k.endswith('.hada_w2_a'):
                pairs.setdefault(_b(k, '.hada_w2_a'), {})['hada_w2_a'] = k
            elif k.endswith('.hada_w2_b'):
                pairs.setdefault(_b(k, '.hada_w2_b'), {})['hada_w2_b'] = k
            elif k.endswith('.hada_t1'):
                pairs.setdefault(_b(k, '.hada_t1'), {})['hada_t1'] = k
            elif k.endswith('.hada_t2'):
                pairs.setdefault(_b(k, '.hada_t2'), {})['hada_t2'] = k
        pair_maps.append((pairs, dora, lw))
    progress_cb(10)

    try:
        with safe_open(str(base_p), framework='pt', device='cpu') as bf:
            ck_keys = list(bf.keys())
        is_xl = any('conditioner' in k for k in ck_keys[:50])
        PFXS = PFXS_XL if is_xl else PFXS_15
        D2C = D2C_XL if is_xl else D2C_15
        lu = {k[:-7].replace('.', '_'): k for k in ck_keys if k.endswith('.weight') or k.endswith('_weight')}

        def _res(b):
            bn = b.replace('.', '_').replace('__', '_').replace('_processor_', '_')
            bl = bn.lower()
            for lp_, sp_ in PFXS:
                lp_n = lp_.replace('.', '_')
                if bl.startswith(lp_n):
                    suf = bn[len(lp_n):]
                    m = lu.get(sp_ + suf)
                    if m:
                        return m
                    for d, c in D2C.items():
                        if suf.startswith(d + '_'):
                            m = lu.get(sp_ + c + suf[len(d):])
                            if m:
                                return m
                    for d, c in D2C.items():
                        if suf.startswith(c + '_'):
                            m = lu.get(sp_ + d + suf[len(c):])
                            if m:
                                return m
                    return None
            return None

        kd = {k: [] for k in ck_keys}
        skipped = 0
        for li, (pairs, dora, lw) in enumerate(pair_maps):
            for b, info in pairs.items():
                if b in dora and not any(x in info for x in ['down', 'up', 'lokr_w1', 'hada_w1_a']):
                    continue
                ck_k = _res(b)
                if ck_k is None:
                    skipped += 1
                    continue
                bi = blk_idx(ck_k)
                bw = bw_vec[bi] if 0 <= bi < 20 else 1.0
                if bw == 0.0:
                    continue
                te_m = te_w if is_te_key(ck_k) else unet_w
                kd[ck_k].append((handles[li], info, lw * bw * te_m))
        total_mapped = sum(len(v) for v in kd.values())
        log_cb(f'Mapped {total_mapped} LoRA op(s), skipped {skipped} unmatched key(s)')
        if total_mapped == 0 and any(len(p) for p, _d, _w in pair_maps):
            log_cb('⚠ ALL LoRA layers were blocked by the block-weight recipe (0 everywhere '
                   'they apply) — this bake will have zero effect. Check your block weights.')
        progress_cb(15)

        with open(str(base_p), 'rb') as f:
            hsz = struct.unpack('<Q', f.read(8))[0]
            hraw = f.read(hsz)
        bhdr = {k: v for k, v in json.loads(hraw.decode('utf-8', 'ignore').rstrip('\x00')).items() if k != '__metadata__'}

        odt = 'F16' if fp16 else 'F32'
        off = 0
        ohdr = {'__metadata__': (meta_extra or {})}
        nonfloat_keys = set()
        for k in ck_keys:
            shp = bhdr[k]['shape']
            src_dt = bhdr[k].get('dtype', 'F32')
            key_dtype = src_dt if src_dt in NONFLOAT_DTYPES else odt
            if src_dt in NONFLOAT_DTYPES:
                nonfloat_keys.add(k)
            nb = DTYPE_BYTES.get(key_dtype, 2) * max(math.prod(shp) if shp else 1, 1)
            ohdr[k] = {'dtype': key_dtype, 'shape': shp, 'data_offsets': [off, off + nb]}
            off += nb
        hj = json.dumps(ohdr, separators=(',', ':')).encode('utf-8')
        pad = (8 - len(hj) % 8) % 8
        hbytes = hj + b' ' * pad
        progress_cb(18)
        log_cb(f'Streaming bake -> {out_p.name}')

        n = len(ck_keys)
        try:
            with open(str(out_p), 'wb') as _raw_fout:
                fout = _DurableWriter(_raw_fout)
                fout.write(struct.pack('<Q', len(hbytes)))
                fout.write(hbytes)
                with safe_open(str(base_p), framework='pt', device='cpu') as bf:
                    for i, k in enumerate(ck_keys):
                        if cancel_token is not None:
                            cancel_token.check()
                        if k in nonfloat_keys:
                            traw = bf.get_tensor(k)
                            fout.write(traw.contiguous().numpy().tobytes())
                            del traw
                            if i % 40 == 0:
                                gc.collect()
                                progress_cb(int(18 + (i + 1) / n * 77))
                            continue
                        t = bf.get_tensor(k).float()
                        for lh, info, eff in kd[k]:
                            try:
                                if 'down' in info and 'up' in info:
                                    dn = lh.get_tensor(info['down']).float()
                                    up = lh.get_tensor(info['up']).float()
                                    dim = dn.shape[0]
                                    sc = lh.get_tensor(info['alpha']).item() / dim if 'alpha' in info else 1.0
                                    raw = (up.flatten(1)) @ (dn.flatten(1))
                                    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                                    if raw.numel() == t.numel():
                                        t = t + raw.reshape(t.shape) * eff * sc
                                    del dn, up, raw
                                elif 'lokr_w1' in info or 'lokr_w1_a' in info:
                                    w1 = ((lh.get_tensor(info['lokr_w1_a']).float() @ lh.get_tensor(info['lokr_w1_b']).float())
                                          if 'lokr_w1_a' in info else lh.get_tensor(info['lokr_w1']).float())
                                    if 'lokr_t2' in info:
                                        t2_ = lh.get_tensor(info['lokr_t2']).float()
                                        w2 = (torch.einsum('i j k l,i r,j s->r s k l', t2_,
                                                            lh.get_tensor(info['lokr_w2_a']).float(),
                                                            lh.get_tensor(info['lokr_w2_b']).float())
                                              if 'lokr_w2_a' in info else lh.get_tensor(info['lokr_w2']).float())
                                    elif 'lokr_w2_a' in info:
                                        w2 = lh.get_tensor(info['lokr_w2_a']).float() @ lh.get_tensor(info['lokr_w2_b']).float()
                                    else:
                                        w2 = lh.get_tensor(info['lokr_w2']).float()
                                    if w2.dim() == 4:
                                        w1 = w1.unsqueeze(2).unsqueeze(3)
                                    raw = torch.kron(w1, w2)
                                    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                                    r2 = raw.flatten(1) if raw.dim() > 2 else raw
                                    o2 = t.flatten(1) if t.dim() > 2 else t
                                    if r2.shape == o2.shape:
                                        sc = lh.get_tensor(info['alpha']).item() if 'alpha' in info else 1.0
                                        t = (o2 + r2 * eff * sc).reshape(t.shape)
                                    del w1, w2, raw, r2, o2
                                elif 'hada_w1_a' in info:
                                    w1a = lh.get_tensor(info['hada_w1_a']).float()
                                    w1b = lh.get_tensor(info['hada_w1_b']).float()
                                    w2a = lh.get_tensor(info['hada_w2_a']).float()
                                    w2b = lh.get_tensor(info['hada_w2_b']).float()
                                    if 'hada_t1' in info:
                                        t1_ = lh.get_tensor(info['hada_t1']).float()
                                        t2_ = lh.get_tensor(info['hada_t2']).float()
                                        m1 = torch.einsum('i j k l,i r,j s->r s k l', t1_, w1a, w1b)
                                        m2 = torch.einsum('i j k l,i r,j s->r s k l', t2_, w2a, w2b)
                                    else:
                                        m1, m2 = w1a @ w1b, w2a @ w2b
                                    raw = m1 * m2
                                    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                                    if raw.numel() == t.numel():
                                        sc = lh.get_tensor(info['alpha']).item() / w1a.shape[0] if 'alpha' in info else 1.0
                                        t = t + raw.reshape(t.shape) * eff * sc
                                    del w1a, w1b, w2a, w2b, m1, m2, raw
                            except Exception as e:
                                log_cb(f'  ⚠ skipped one LoRA op on {k}: {e}')
                        t = t.float().clamp(-65504.0, 65504.0)
                        t = torch.nan_to_num(t, nan=0.0, posinf=65504.0, neginf=-65504.0).clamp(-65504.0, 65504.0)
                        nbytes = t.numel() * (2 if fp16 else 4)
                        tout = t.half() if fp16 else t
                        fout.write(tout.contiguous().numpy().tobytes())
                        del t, tout
                        if i % 40 == 0 or nbytes > 64 * 1024 * 1024:
                            gc.collect()
                            progress_cb(int(18 + (i + 1) / n * 77))
                fout.final_sync()
            gc.collect()
            progress_cb(95)
            log_cb('Post-flight scan...')
            postflight_scan(out_p, log_cb)
            progress_cb(97)
            log_cb(f'✓ Bake done: {out_p.name}  ({out_p.stat().st_size / 1e6:.0f} MB)')
        except Exception:
            if out_p.exists():
                try:
                    out_p.unlink()
                except Exception:
                    pass
            raise
    finally:
        for h in handles:
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        gc.collect()


def blend_checkpoints(pa, pb, bw_vec, out_p, fp16, progress_cb, log_cb, meta_extra=None, cancel_token=None):
    """Final blend: result[block] = A*(1-w) + B*w, streamed. Safetensors only."""
    pa, pb, out_p = Path(pa), Path(pb), Path(out_p)
    if pa.suffix != '.safetensors' or pb.suffix != '.safetensors':
        raise ValueError('Final blend requires both files to be .safetensors.')
    guards.require_ram(512, 'a final blend')
    guards.check_disk_space(out_p.parent, guards.estimate_output_bytes(pa, pb, fp16=fp16), 'the blended checkpoint')
    guards.sanity_check_block_weights(bw_vec, log_cb)
    log_cb(f'Final blend: {pa.name}  x  {pb.name}')
    with safe_open(str(pa), framework='pt', device='cpu') as fa:
        keys_a = list(fa.keys())
    with open(str(pa), 'rb') as f:
        hsz = struct.unpack('<Q', f.read(8))[0]
        hraw = f.read(hsz)
    ahdr = {k: v for k, v in json.loads(hraw.decode('utf-8', 'ignore').rstrip('\x00')).items() if k != '__metadata__'}
    with open(str(pb), 'rb') as f:
        hsz = struct.unpack('<Q', f.read(8))[0]
        hraw = f.read(hsz)
    bhdr = {k: v for k, v in json.loads(hraw.decode('utf-8', 'ignore').rstrip('\x00')).items() if k != '__metadata__'}
    keys_b = set(bhdr.keys())
    ka_set = set(keys_a)
    all_keys = sorted(ka_set | keys_b)
    odt = 'F16' if fp16 else 'F32'
    off = 0
    ohdr = {'__metadata__': (meta_extra or {})}
    nonfloat_keys = set()
    for k in all_keys:
        if k in ka_set:
            shp, src_dt = ahdr[k]['shape'], ahdr[k].get('dtype', 'F32')
        else:
            shp, src_dt = bhdr[k]['shape'], bhdr[k].get('dtype', 'F32')
        key_dtype = src_dt if src_dt in NONFLOAT_DTYPES else odt
        if src_dt in NONFLOAT_DTYPES:
            nonfloat_keys.add(k)
        nb = DTYPE_BYTES.get(key_dtype, 2) * max(math.prod(shp) if shp else 1, 1)
        ohdr[k] = {'dtype': key_dtype, 'shape': shp, 'data_offsets': [off, off + nb]}
        off += nb
    hj = json.dumps(ohdr, separators=(',', ':')).encode('utf-8')
    pad = (8 - len(hj) % 8) % 8
    hbytes = hj + b' ' * pad
    progress_cb(8)
    log_cb(f'Streaming blend -> {out_p.name}')
    n = len(all_keys)
    try:
        with open(str(out_p), 'wb') as _raw_fout:
            fout = _DurableWriter(_raw_fout)
            fout.write(struct.pack('<Q', len(hbytes)))
            fout.write(hbytes)
            with safe_open(str(pb), framework='pt', device='cpu') as fb, \
                    safe_open(str(pa), framework='pt', device='cpu') as fa_ref:
                for i, k in enumerate(all_keys):
                    if cancel_token is not None:
                        cancel_token.check()
                    ia, ib = k in ka_set, k in keys_b
                    if k in nonfloat_keys:
                        traw = fa_ref.get_tensor(k) if ia else fb.get_tensor(k)
                        fout.write(traw.contiguous().numpy().tobytes())
                        del traw
                        if i % 40 == 0:
                            gc.collect()
                            progress_cb(int(8 + (i + 1) / n * 89))
                        continue
                    bi = blk_idx(k)
                    w = bw_vec[bi] if 0 <= bi < 20 else 0.0
                    if ia and ib and w != 0.0:
                        ta_raw, tb_raw = fa_ref.get_tensor(k), fb.get_tensor(k)
                        if not (torch.is_floating_point(ta_raw) and torch.is_floating_point(tb_raw)
                                and ta_raw.shape == tb_raw.shape):
                            t = ta_raw
                            if torch.is_floating_point(t):
                                t = t.half() if fp16 else t.float()
                            fout.write(t.contiguous().numpy().tobytes())
                            del t, ta_raw, tb_raw
                        else:
                            ta, tb = ta_raw.float(), tb_raw.float()
                            del ta_raw, tb_raw
                            ta = torch.nan_to_num(ta, nan=0.0, posinf=65504.0, neginf=-65504.0)
                            tb = torch.nan_to_num(tb, nan=0.0, posinf=65504.0, neginf=-65504.0)
                            t = ta * (1.0 - w) + tb * w
                            del ta, tb
                            t = torch.nan_to_num(t, nan=0.0, posinf=65504.0, neginf=-65504.0).clamp(-65504.0, 65504.0)
                            tout = t.half() if fp16 else t
                            fout.write(tout.contiguous().numpy().tobytes())
                            del t, tout
                    else:
                        t = fa_ref.get_tensor(k) if ia else fb.get_tensor(k)
                        if torch.is_floating_point(t):
                            t = t.half() if fp16 else t.float()
                        fout.write(t.contiguous().numpy().tobytes())
                        del t
                    if i % 40 == 0:
                        gc.collect()
                        progress_cb(int(8 + (i + 1) / n * 89))
            fout.final_sync()
        gc.collect()
        progress_cb(97)
        log_cb('Post-flight scan...')
        postflight_scan(out_p, log_cb)
        progress_cb(100)
        log_cb(f'✓ Blend done: {out_p.name}  ({out_p.stat().st_size / 1e6:.0f} MB)')
    except Exception:
        if out_p.exists():
            try:
                out_p.unlink()
            except Exception:
                pass
        raise


# ─────────────────────────────────────────────────────────────────────────
# VAE Baker
# ─────────────────────────────────────────────────────────────────────────
def bake_vae(ckpt_path, vae_path, out_path, fp16, replace_existing, progress_cb, log_cb, cancel_token=None):
    ckpt_path, vae_path, out_path = Path(ckpt_path), Path(vae_path), Path(out_path)
    if ckpt_path.suffix != '.safetensors':
        raise ValueError('VAE baking requires a .safetensors checkpoint.')
    guards.require_ram(384, 'a VAE bake')
    guards.check_disk_space(out_path.parent, guards.estimate_output_bytes(ckpt_path, fp16=fp16), 'the VAE-baked checkpoint')
    log_cb(f'Opening checkpoint: {ckpt_path.name}')
    with safe_open(str(ckpt_path), framework='pt', device='cpu') as hc:
        ckpt_keys = list(hc.keys())
    progress_cb(4)
    existing_vae = [k for k in ckpt_keys if k.startswith(VAE_PFX)]
    if existing_vae and not replace_existing:
        raise ValueError(f'Checkpoint already contains a VAE ({len(existing_vae)} tensors). '
                          f'Enable "Replace existing VAE" to overwrite it.')
    if existing_vae:
        log_cb(f'⚠ Replacing existing VAE ({len(existing_vae)} tensors)')

    log_cb('Loading VAE to bake in...')
    hv, mv = load_any(vae_path, security_log=log_cb)
    if mv == 'dict':
        guards.require_ram(guards.estimate_ckpt_ram_mb(vae_path), 'loading the VAE fully into RAM')
    vkeys = hkeys(hv, mv)
    standalone = not any(k.startswith(VAE_PFX) for k in vkeys[:20])
    vae_tensors = {}
    for vk in vkeys:
        t = hget(hv, mv, vk)
        if not hasattr(t, 'shape'):
            continue
        t = nan_safe(t, fp16)
        if standalone:
            sd_k = vk
            for src, dst in VAE_MAP:
                if vk.startswith(src):
                    sd_k = dst + vk[len(src):]
                    break
            else:
                sd_k = VAE_PFX + vk
        else:
            sd_k = vk
        vae_tensors[sd_k] = t
    if mv == 'handle':
        try:
            hv.__exit__(None, None, None)
        except Exception:
            pass
    if not vae_tensors:
        raise ValueError(f'No usable tensors found in VAE file: {vae_path.name}')
    log_cb(f'  VAE ready — {len(vae_tensors)} tensors')
    progress_cb(12)

    with open(str(ckpt_path), 'rb') as f:
        hsz = struct.unpack('<Q', f.read(8))[0]
        hraw = f.read(hsz)
    ckhdr = {k: v for k, v in json.loads(hraw.decode('utf-8', 'ignore').rstrip('\x00')).items() if k != '__metadata__'}

    base_keys = [k for k in ckpt_keys if k not in existing_vae]
    write_keys = base_keys + list(vae_tensors.keys())
    out_dtype_str = 'F16' if fp16 else 'F32'
    offset = 0
    hdr_dict = {'__metadata__': {}}
    for k in write_keys:
        shape = list(vae_tensors[k].shape) if k in vae_tensors else ckhdr.get(k, {}).get('shape', [])
        if k in vae_tensors:
            key_dtype = out_dtype_str
        else:
            src_dt = ckhdr.get(k, {}).get('dtype', 'F32')
            key_dtype = src_dt if src_dt in NONFLOAT_DTYPES else out_dtype_str
        nbytes = DTYPE_BYTES.get(key_dtype, 2) * max(math.prod(shape) if shape else 1, 1)
        hdr_dict[k] = {'dtype': key_dtype, 'shape': shape, 'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    hdr_json = json.dumps(hdr_dict, separators=(',', ':')).encode('utf-8')
    pad = (8 - len(hdr_json) % 8) % 8
    hdr_bytes = hdr_json + b' ' * pad
    progress_cb(15)
    log_cb(f'Streaming -> {out_path.name}  ({len(write_keys)} tensors)')
    n_wk = len(write_keys)
    try:
        with open(str(out_path), 'wb') as _raw_fout, safe_open(str(ckpt_path), framework='pt', device='cpu') as hc:
            fout = _DurableWriter(_raw_fout)
            fout.write(struct.pack('<Q', len(hdr_bytes)))
            fout.write(hdr_bytes)
            for i, k in enumerate(write_keys):
                if cancel_token is not None:
                    cancel_token.check()
                if k in vae_tensors:
                    t = vae_tensors[k]
                else:
                    t = nan_safe(hc.get_tensor(k), fp16)
                fout.write(t.contiguous().numpy().tobytes())
                del t
                if i % 40 == 0:
                    gc.collect()
                    progress_cb(int(15 + (i + 1) / n_wk * 78))
            fout.final_sync()
        del vae_tensors
        gc.collect()
        progress_cb(96)
        log_cb('Post-flight scan...')
        postflight_scan(out_path, log_cb)
        progress_cb(100)
        log_cb('✓ VAE baked successfully')
    except Exception:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        raise


# ─────────────────────────────────────────────────────────────────────────
# Metadata Reader (read-only)
# ─────────────────────────────────────────────────────────────────────────
def read_metadata(path) -> dict:
    path = Path(path)
    if path.suffix != '.safetensors':
        return {'error': 'Metadata inspection only works on .safetensors files.'}
    hdr = sf_header(path)
    return hdr.get('__metadata__', {}) or {}


def format_metadata_markdown(meta: dict) -> str:
    if not meta:
        return '_No metadata found in this file — it likely was not produced by this tool, ' \
               'or metadata was stripped by another tool._'
    if 'error' in meta:
        return f'_{meta["error"]}_'
    lines = []
    block_fields = [k for k in meta if k.endswith('_json') and 'block' in k.lower() or k.endswith('_ratio_json')]
    for k, v in meta.items():
        if k in block_fields:
            continue
        lines.append(f'- **{k}**: {v}')
    for k in block_fields:
        try:
            vals = json.loads(meta[k])
            rows = '\n'.join(f'| {n} | {v:.4f} |' for n, v in zip(BLOCK_NAMES, vals))
            lines.append(f'\n**{k}**\n\n| Block | Value |\n|---|---|\n{rows}')
        except Exception:
            lines.append(f'- **{k}**: (could not parse)')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Image Generation
# ─────────────────────────────────────────────────────────────────────────
SCHED_MAP = {
    'DPM++ 2M Karras': ('DPMSolverMultistepScheduler', True),
    'DPM++ SDE Karras': ('DPMSolverSDEScheduler', True),
    'Euler a': ('EulerAncestralDiscreteScheduler', False),
    'Euler': ('EulerDiscreteScheduler', False),
    'DDIM': ('DDIMScheduler', False),
    'UniPC': ('UniPCMultistepScheduler', False),
}

# Architecture presets (CRITIQUE.md E1). "legacy_single_file" covers the
# original SD1.5/SDXL single-checkpoint path unchanged. "diffusers_repo"
# covers the newer DiT/flow-matching families, which ship as a HF-repo-
# style bundle of separate transformer/text_encoder/tokenizer/vae/scheduler
# components rather than one fused checkpoint — verified via live web search
# against each project's own Hugging Face model card/docs and, where
# possible, by installing the current diffusers release and confirming the
# pipeline classes actually exist with the expected component signatures
# (not from training-data memory, since all three post-date this app's
# knowledge cutoff).
ARCH_PRESETS = {
    'auto': {'label': 'Auto-detect (SD1.5 / SDXL)', 'kind': 'legacy_single_file'},
    'sd15': {'label': 'Stable Diffusion 1.5', 'kind': 'legacy_single_file', 'force_arch': 'sd15'},
    'sdxl': {'label': 'Stable Diffusion XL', 'kind': 'legacy_single_file', 'force_arch': 'sdxl'},
    'zimage_turbo': {
        'label': 'Z-Image Turbo (Tongyi-MAI, 6B S3-DiT)',
        'kind': 'diffusers_repo', 'pipeline_class': 'ZImagePipeline',
        'default_repo': 'Tongyi-MAI/Z-Image-Turbo',
        'default_text_encoder': 'Qwen/Qwen3-4B',
        'default_guidance_scale': 0.0, 'default_steps': 9,
        'notes': ('Verified: ships as a ready-to-use diffusers repo (transformer + '
                  'Qwen3-4B text encoder + Flux-derived 16-channel VAE), loads with a '
                  'plain from_pretrained call. Turbo is distilled — keep guidance_scale '
                  'at 0.0 and steps around 9.'),
    },
    'krea2_turbo': {
        'label': 'Krea 2 Turbo (12.9B single-stream DiT, distilled)',
        'kind': 'diffusers_repo', 'pipeline_class': 'Krea2Pipeline',
        'default_repo': 'krea/Krea-2-Turbo',
        'default_text_encoder': 'Qwen/Qwen3-VL-4B-Instruct',
        'default_guidance_scale': 0.0, 'default_steps': 8,
        'notes': ('Verified against the official model card: 8 steps, guidance_scale '
                  '0.0. Gated on Hugging Face — you must accept the license on the repo '
                  'page (logged in) before from_pretrained can download it.'),
    },
    'krea2_raw': {
        'label': 'Krea 2 Raw (12.9B single-stream DiT, full-step)',
        'kind': 'diffusers_repo', 'pipeline_class': 'Krea2Pipeline',
        'default_repo': 'krea/Krea-2-Raw',
        'default_text_encoder': 'Qwen/Qwen3-VL-4B-Instruct',
        'default_guidance_scale': 3.5, 'default_steps': 52,
        'notes': ('Verified against the official model card: 52 steps, guidance_scale '
                  '3.5. Krea\'s own model card says this checkpoint is "not recommended '
                  'for inference use" directly — it exists as a base for fine-tuning/LoRA '
                  'training (LoRAs trained on Raw are meant to be used on Turbo instead). '
                  'Also gated — accept the license on the repo page first.'),
    },
    'anima': {
        'label': 'Anima Base v1 (2B, Cosmos-Predict2-derived) — experimental',
        'kind': 'diffusers_repo', 'pipeline_class': 'Cosmos2TextToImagePipeline',
        'default_repo': '',
        'default_text_encoder': None,
        'default_guidance_scale': 5.0, 'default_steps': 40,
        'notes': ('⚠ Left blank deliberately, not defaulted to a repo id — here\'s why. '
                  'Anima\'s official repo (circlestone-labs/Anima) ships in ComfyUI\'s '
                  'split-file format (separate diffusion_models/ + text_encoders/ + vae/ '
                  'safetensors, no model_index.json), which diffusers\' from_pretrained() '
                  'cannot load directly — it expects a diffusers-format repo layout. There '
                  'is no confirmed official diffusers pipeline for Anima as of this '
                  'writing. If you find or build a diffusers-format conversion (community '
                  'ones occasionally appear, e.g. search "Anima diffusers" on Hugging '
                  'Face), you can type its repo id here as a best-effort attempt via '
                  'Cosmos2TextToImagePipeline — the architecture Anima is derived from. '
                  'One more specific gotcha if you do get this far: Anima\'s text encoder '
                  'is the BASE (non-instruction-tuned) Qwen3-0.6B checkpoint, not the '
                  'standard chat/instruct one people usually reach for — using the wrong '
                  'variant will load without error but produce poor results. Steps/CFG '
                  'above are community-reported starting points (30-50 steps, CFG 4-6), '
                  'not from an official inference config. Recommended samplers per the '
                  'model card: er_sde, euler_a, or dpmpp_2m_sde_gpu.'),
        'experimental': True,
    },
}


def detect_pipeline_arch(path) -> str:
    path = Path(path)
    try:
        hdr = sf_header(path)
        keys = [k for k in hdr if k != '__metadata__']
        if any(k.startswith('conditioner.') or 'add_embedding' in k for k in keys):
            return 'sdxl'
        if path.stat().st_size > 5.5e9 and not any(k.startswith('cond_stage_model.') for k in keys):
            return 'sdxl'
    except Exception:
        pass
    return 'sd15'


def detect_device():
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = None
        return 'cuda', torch.float16, name
    return 'cpu', torch.bfloat16, None


def set_scheduler(pipe, name):
    import diffusers
    sname, karras = SCHED_MAP.get(name, ('DPMSolverMultistepScheduler', True))
    cls = getattr(diffusers, sname, diffusers.DPMSolverMultistepScheduler)
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config, use_karras_sigmas=karras)
    except ImportError:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    except TypeError:
        try:
            pipe.scheduler = cls.from_config(pipe.scheduler.config)
        except ImportError:
            pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    except Exception:
        pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


def load_pipeline_custom(arch_key, repo_or_path, text_encoder_override, vae_override,
                          dtype, log_cb, progress_cb, cancel_token=None):
    """Loads one of the newer DiT/flow-matching architectures (CRITIQUE.md E1).
    Primary path: `PipelineClass.from_pretrained(repo_or_path)` loads every
    component (transformer/text_encoder/tokenizer/vae/scheduler) from one HF
    repo id or local diffusers-format folder, exactly matching each project's
    own documented usage. Overrides, when given, replace individual
    components (CRITIQUE.md E2: the VAE override accepts a single weight
    file OR a repo/folder; the text-encoder override must be a repo/folder,
    since a bare weights file has no tokenizer).

    cancel_token, if given, is checked before each stage (text-encoder
    override, VAE override, main pipeline) starts — honestly scoped: this
    can stop the *next* stage from starting, but cannot interrupt a single
    from_pretrained() call already in flight, since huggingface_hub's
    high-level download API doesn't expose a cancellation hook. A cancel
    requested mid-download will take effect once that download finishes
    (successfully or not) rather than immediately — a real limitation of
    the underlying library, not something worth pretending around."""
    import importlib
    diffusers = importlib.import_module('diffusers')
    preset = ARCH_PRESETS[arch_key]
    pipe_cls_name = preset['pipeline_class']
    pipe_cls = getattr(diffusers, pipe_cls_name, None)
    if pipe_cls is None:
        raise RuntimeError(
            f'Your installed `diffusers` does not have {pipe_cls_name} yet. This '
            f'architecture needs a very recent diffusers release — try: '
            f'pip install -U diffusers transformers')
    progress_cb(10)
    guards.check_hf_repo_disk_space(repo_or_path, log_cb)
    kwargs = {'torch_dtype': dtype}

    if text_encoder_override:
        if cancel_token is not None:
            cancel_token.check()
        log_cb(f'  Loading text-encoder override: {text_encoder_override}')
        guards.check_hf_repo_disk_space(text_encoder_override, log_cb)
        from transformers import AutoTokenizer, AutoModel
        kwargs['tokenizer'] = AutoTokenizer.from_pretrained(text_encoder_override)
        kwargs['text_encoder'] = AutoModel.from_pretrained(text_encoder_override, torch_dtype=dtype)
        progress_cb(35)

    if vae_override:
        if cancel_token is not None:
            cancel_token.check()
        log_cb(f'  Loading VAE override: {vae_override}')
        vae_cls = getattr(diffusers, 'AutoencoderKLQwenImage', None) if arch_key.startswith(('krea2', 'anima')) \
            else getattr(diffusers, 'AutoencoderKL', None)
        if vae_cls is None:
            vae_cls = diffusers.AutoencoderKL
        vp = str(vae_override)
        if vp.endswith(('.safetensors', '.ckpt', '.pt')) and Path(vp).exists():
            kwargs['vae'] = vae_cls.from_single_file(vp, torch_dtype=dtype)
        else:
            guards.check_hf_repo_disk_space(vp, log_cb)
            kwargs['vae'] = vae_cls.from_pretrained(vp, torch_dtype=dtype)
        progress_cb(55)

    if cancel_token is not None:
        cancel_token.check()
    log_cb(f'  Loading {pipe_cls_name} from "{repo_or_path}" ...')
    progress_cb(60)
    pipe = pipe_cls.from_pretrained(repo_or_path, **kwargs)
    progress_cb(90)
    return pipe


def _call_pipeline_safely(pipe, kwargs, log_cb):
    """Some pipeline classes don't accept every kwarg (e.g. clip_skip is a
    legacy-UNet-pipeline-only concept; some flow-matching pipelines don't
    take negative_prompt). A single TypeError only ever names one bad
    kwarg at a time, so this retries in a bounded loop — strip whichever
    kwarg the error blames, try again, repeat — rather than a single retry
    that would still fail outright on a second incompatible kwarg."""
    stripped = dict(kwargs)
    removed_total = []
    STRIPPABLE = ('clip_skip', 'negative_prompt', 'callback_on_step_end',
                  'callback_on_step_end_tensor_inputs')
    for _attempt in range(len(STRIPPABLE) + 1):
        try:
            return pipe(**stripped)
        except TypeError as e:
            msg = str(e)
            removed = [k for k in STRIPPABLE if k in stripped and k in msg]
            if not removed:
                if removed_total:
                    log_cb(f'  (already retried without {", ".join(removed_total)} — '
                           f'this error is unrelated, not retrying further)')
                raise
            for k in removed:
                del stripped[k]
            removed_total.extend(removed)
    log_cb(f'  (this pipeline doesn\'t accept {", ".join(removed_total)} — retrying without them)')
    return pipe(**stripped)


class ImageGenSession:
    """Holds one loaded diffusers pipeline so repeated generations don't
    reload the checkpoint from disk each time. CRITIQUE.md C1: intended to
    be instantiated per-session (e.g. via gr.State), not as a shared
    module-level global — a shared global would let concurrent users
    silently swap out each other's loaded model."""

    def __init__(self):
        self.pipe = None
        self.arch = None
        self.kind = None
        self.device = 'cpu'
        self.dtype = torch.bfloat16
        self.loaded_name = None

    def unload(self):
        if self.pipe is not None:
            try:
                self.pipe.to('cpu')
            except Exception:
                pass
            self.pipe = None
        guards.release_gpu_memory()  # CRITIQUE.md A5

    def load(self, checkpoint_path, scheduler_name, log_cb, progress_cb=lambda p: None,
              arch_key='auto', text_encoder_override=None, vae_override=None, cancel_token=None):
        guards.require_ram(1024, 'loading an image-generation pipeline')
        dev, dtype, gpu_name = detect_device()
        preset = ARCH_PRESETS.get(arch_key, ARCH_PRESETS['auto'])

        # Every CHEAP, FAST check runs BEFORE unload() — a missing file, a
        # typo'd repo id, a diffusers version too old for this architecture,
        # or simply not enough disk space for the download should never
        # cost you a checkpoint you already had loaded and working.
        # Reproduced this concretely before fixing it: a typo'd filename
        # used to wipe out an already-loaded session even though the typo
        # was detectable instantly, with zero actual loading work done.
        # This can't be extended across the *slow* part (the download
        # itself) without holding two full pipelines in memory at once,
        # which would defeat the entire reason unload() exists — so a
        # cancellation or failure once the real download has started will
        # still leave the session empty. That residual gap is real and
        # documented here rather than silently accepted as unavoidable
        # everywhere.
        if preset['kind'] == 'legacy_single_file':
            path = Path(checkpoint_path)
            if not path.exists():
                raise ValueError(f'Checkpoint file not found: {path}')
        else:
            repo_or_path = checkpoint_path or preset['default_repo']
            if not repo_or_path:
                raise ValueError('No Hugging Face repo id or local folder path given for this architecture.')
            pipe_cls_name = preset['pipeline_class']
            import importlib
            diffusers_mod = importlib.import_module('diffusers')
            if getattr(diffusers_mod, pipe_cls_name, None) is None:
                raise RuntimeError(
                    f'Your installed `diffusers` does not have {pipe_cls_name} yet. This '
                    f'architecture needs a very recent diffusers release — try: '
                    f'pip install -U diffusers transformers')
            guards.check_hf_repo_disk_space(repo_or_path, log_cb)
        if cancel_token is not None:
            cancel_token.check()

        self.unload()

        with guards.gpu_oom_guard('loading the checkpoint'):
            if preset['kind'] == 'legacy_single_file':
                path = Path(checkpoint_path)
                arch = preset.get('force_arch') or detect_pipeline_arch(path)
                progress_cb(20)
                log_cb(f'Loading {path.name} as {arch.upper()} on {dev.upper()}...')
                if arch == 'sdxl':
                    from diffusers import StableDiffusionXLPipeline
                    pipe = StableDiffusionXLPipeline.from_single_file(
                        str(path), torch_dtype=dtype, use_safetensors=path.suffix == '.safetensors',
                        low_cpu_mem_usage=True)
                else:
                    from diffusers import StableDiffusionPipeline
                    pipe = StableDiffusionPipeline.from_single_file(
                        str(path), torch_dtype=dtype, load_safety_checker=False, low_cpu_mem_usage=True)
                progress_cb(70)
                set_scheduler(pipe, scheduler_name)
                loaded_label = path.name
            else:
                arch = arch_key
                repo_or_path = checkpoint_path or preset['default_repo']
                log_cb(f'Loading {preset["label"]} on {dev.upper()}...')
                if preset.get('experimental'):
                    log_cb(f'⚠ {preset.get("notes", "This architecture is experimental.")}')
                pipe = load_pipeline_custom(arch_key, repo_or_path, text_encoder_override,
                                            vae_override, dtype, log_cb, progress_cb,
                                            cancel_token=cancel_token)
                loaded_label = str(repo_or_path)

            if dev == 'cpu':
                torch.set_num_threads(os.cpu_count() or 4)
            pipe.to(dev)
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            if dev == 'cuda' and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

        self.pipe, self.arch, self.kind = pipe, arch, preset['kind']
        self.device, self.dtype, self.loaded_name = dev, dtype, loaded_label
        dev_label = f'GPU ({gpu_name})' if dev == 'cuda' and gpu_name else dev.upper()
        dtype_label = 'FP16' if dtype == torch.float16 else 'BF16'
        progress_cb(100)
        log_cb(f'✓ {loaded_label} -> {dev_label} · {dtype_label} · {arch.upper() if isinstance(arch, str) else arch}')
        return arch, dev_label

    def generate_stream(self, prompt, negative_prompt, steps, cfg, width, height, seed, n_images,
                         clip_skip, scheduler_name, progress_cb, log_cb,
                         cancel_token=None, preview_cb=None):
        """Streaming generator (CRITIQUE.md D2/D3): yields ('preview', PIL.Image, index)
        during denoising and ('image', PIL.Image, index, seed) as soon as each
        image finishes — the caller saves/displays each one immediately rather
        than waiting for the whole batch. Raises guards.OperationCancelled
        cleanly if cancelled between images (already-yielded images are kept
        by the caller; this function itself holds nothing back)."""
        if self.pipe is None:
            raise ValueError('Load a checkpoint first.')
        import random
        width, height, steps, cfg, n_images, clip_skip = guards.clamp_generation_request(
            width, height, steps, cfg, n_images, clip_skip, log_cb=log_cb)
        seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2 ** 31 - 1)
        seed = seed % 2147483647

        is_legacy = self.kind == 'legacy_single_file'
        if is_legacy:
            set_scheduler(self.pipe, scheduler_name)
        # Alignment: legacy SD1.5/SDXL only need multiples of 8, but the
        # newer Flux-family-derived architectures (Z-Image confirmed; Krea2/
        # Anima's Qwen-Image VAE is architecturally the same f8+patchify(2)
        # design) use vae_scale_factor=16 internally and silently *round*
        # any non-16-aligned request rather than erroring (verified against
        # diffusers' own FluxPipeline source and a reported issue showing
        # exactly this silent substitution). Rounding to 16 here up front
        # means what this app tells the user it's generating is what
        # actually gets generated, for every architecture — a strict
        # superset of the 8-alignment legacy models need, so this changes
        # nothing for them.
        gen_w = max(64, width // 16 * 16)
        gen_h = max(64, height // 16 * 16)
        cs = int(clip_skip)
        clip_skip_val = (cs - 1) if cs > 1 else None
        if (gen_w, gen_h) != (width, height):
            log_cb(f'  (requested {width}x{height} rounded to {gen_w}x{gen_h} — '
                   f'required to be a multiple of 16)')
        log_cb(f'Generating {n_images} image(s) at {gen_w}x{gen_h}, {steps} steps, seed={seed}...')

        for img_idx in range(n_images):
            if cancel_token is not None:
                cancel_token.check()
            this_seed = seed + img_idx
            gen = torch.Generator(self.device).manual_seed(this_seed)

            def _cb(pipe, step, timestep, cb_kwargs):
                if cancel_token is not None:
                    cancel_token.check()
                frac = (img_idx + (step + 1) / max(steps, 1)) / n_images
                progress_cb(min(0.97, frac))
                if preview_cb is not None and step % 3 == 0 and 'latents' in cb_kwargs:
                    prev = guards.approx_latents_to_preview(cb_kwargs['latents'])
                    if prev is not None:
                        preview_cb(prev, img_idx)
                return cb_kwargs

            call_kwargs = dict(
                prompt=prompt, width=gen_w, height=gen_h,
                num_inference_steps=steps, guidance_scale=float(cfg), num_images_per_prompt=1,
                generator=gen, callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=['latents'],
            )
            if negative_prompt:
                call_kwargs['negative_prompt'] = negative_prompt
            if is_legacy:
                call_kwargs['clip_skip'] = clip_skip_val

            with guards.gpu_oom_guard(f'generating image {img_idx + 1}/{n_images}'):
                out = _call_pipeline_safely(self.pipe, call_kwargs, log_cb)
            img = out.images[0]
            del out
            guards.release_gpu_memory()
            yield ('image', img, img_idx, this_seed)

        log_cb(f'✓ Generated {n_images} image(s) — seed {seed}')

    def generate(self, prompt, negative_prompt, steps, cfg, width, height, seed, n_images,
                 clip_skip, scheduler_name, progress_cb, log_cb, cancel_token=None):
        """Non-streaming convenience wrapper over generate_stream() for
        callers that just want the final list (kept for backward
        compatibility / simple scripting use)."""
        results, seeds = [], None
        for kind, img, idx, this_seed in self.generate_stream(
                prompt, negative_prompt, steps, cfg, width, height, seed, n_images,
                clip_skip, scheduler_name, progress_cb, log_cb, cancel_token=cancel_token):
            if kind == 'image':
                results.append(img)
                if seeds is None:
                    seeds = this_seed
        return results, seeds


# ─────────────────────────────────────────────────────────────────────────
# Live-log runner: executes a long task in a background thread and yields
# incremental (logs, progress_percent, done, error) snapshots so a UI can
# stream progress instead of blocking until completion.
# ─────────────────────────────────────────────────────────────────────────
def run_with_live_log(target: Callable, job_name: str, poll_interval: float = 0.25):
    """target(progress_cb, log_cb) -> result. Yields dicts:
    {'logs': [...], 'progress': 0-100, 'done': bool, 'result': Any, 'error': Exception|None}

    job_name is passed straight to guards.JOB_LOCK, acquired **inside the
    worker thread** rather than by the caller — deliberately, not an
    accident of refactoring. If the caller acquired the lock itself around
    the loop that consumes this generator, closing/abandoning that outer
    generator early (a disconnected browser tab, a dropped connection —
    anything that makes Python send GeneratorExit into it) releases the
    lock immediately via the context manager's __exit__, while this
    function's background thread keeps running completely independently,
    orphaned, until it finishes on its own — verified this concretely
    happens with a direct test before restructuring around it. Acquiring
    the lock here instead ties its hold duration to the worker thread's
    actual lifetime, which is correct regardless of whether anyone is
    still watching. A lock-busy rejection now arrives as a normal
    `guards.InsufficientResourceError` via the 'error' key of the final
    yielded dict, same as any other failure — no special-casing needed by
    callers, since guards.InsufficientResourceError is already displayed
    distinctly (its message only, no traceback) wherever this project
    formats errors.
    """
    q: "queue.Queue" = queue.Queue()
    result_holder = {}

    def progress_cb(p):
        q.put(('progress', p))

    def log_cb(msg):
        q.put(('log', str(msg)))

    def worker():
        try:
            with guards.JOB_LOCK.acquire(job_name):
                result_holder['result'] = target(progress_cb, log_cb)
        except Exception as e:
            result_holder['error'] = e
            q.put(('log', f'✗ {type(e).__name__}: {e}'))
        finally:
            q.put(('done', None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    logs = []
    progress_val = 0
    while True:
        try:
            kind, payload = q.get(timeout=poll_interval)
        except queue.Empty:
            yield {'logs': list(logs), 'progress': progress_val, 'done': False,
                   'result': None, 'error': None}
            continue
        if kind == 'progress':
            progress_val = payload
        elif kind == 'log':
            logs.append(payload)
        elif kind == 'done':
            yield {'logs': list(logs), 'progress': progress_val, 'done': True,
                   'result': result_holder.get('result'), 'error': result_holder.get('error')}
            return
        yield {'logs': list(logs), 'progress': progress_val, 'done': False,
               'result': None, 'error': None}


def run_generation_stream(target: Callable, job_name: str, poll_interval: float = 0.2):
    """Like run_with_live_log, but for image generation specifically: the
    step-level progress and the approximate live preview
    (guards.approx_latents_to_preview) both fire synchronously *inside* a
    single blocking pipe() call, deep inside diffusers' own step loop — a
    plain generator has no way to surface those to its caller until the
    call returns, which is only once a whole image is done. Running the
    actual generation in a background thread and relaying every event
    through a queue is what makes real-time progress and live preview
    during a single image's denoising actually reach the UI, rather than
    the progress_cb/preview_cb callbacks firing into the void.

    job_name works exactly as in run_with_live_log — guards.JOB_LOCK is
    acquired inside the worker thread, not by the caller, so its hold
    duration matches the worker's actual lifetime even if the outer
    (Gradio-facing) generator is abandoned early. See run_with_live_log's
    docstring for the concrete race this closes.

    target(progress_cb, log_cb, preview_cb, image_cb) -> None, run in a
    background thread. Yields incremental dicts:
    {'logs': [...], 'progress': 0.0-1.0, 'preview': (PIL.Image, idx) | None,
     'images': [(PIL.Image, idx, seed), ...], 'done': bool, 'error': Exception | None}
    'images' accumulates (like 'logs') so a caller building a gallery only
    needs to diff against what it already has; 'preview' is the single most
    recent preview frame (superseded frames are simply dropped, which is
    correct for a live preview — no reason to render stale ones)."""
    q: "queue.Queue" = queue.Queue()
    result_holder = {}

    def progress_cb(p):
        q.put(('progress', p))

    def log_cb(msg):
        q.put(('log', str(msg)))

    def preview_cb(img, idx):
        q.put(('preview', (img, idx)))

    def image_cb(img, idx, seed):
        q.put(('image', (img, idx, seed)))

    def worker():
        try:
            with guards.JOB_LOCK.acquire(job_name):
                target(progress_cb, log_cb, preview_cb, image_cb)
        except Exception as e:
            result_holder['error'] = e
            q.put(('log', f'✗ {type(e).__name__}: {e}'))
        finally:
            q.put(('done', None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    logs, images = [], []
    progress_val, preview = 0.0, None
    while True:
        try:
            kind, payload = q.get(timeout=poll_interval)
        except queue.Empty:
            yield {'logs': list(logs), 'progress': progress_val, 'preview': preview,
                   'images': list(images), 'done': False, 'error': None}
            continue
        if kind == 'progress':
            progress_val = payload
        elif kind == 'log':
            logs.append(payload)
        elif kind == 'preview':
            preview = payload
        elif kind == 'image':
            images.append(payload)
            preview = None  # the just-finished image supersedes its own in-progress preview
        elif kind == 'done':
            yield {'logs': list(logs), 'progress': progress_val, 'preview': None,
                   'images': list(images), 'done': True, 'error': result_holder.get('error')}
            return
        yield {'logs': list(logs), 'progress': progress_val, 'preview': preview,
               'images': list(images), 'done': False, 'error': None}
