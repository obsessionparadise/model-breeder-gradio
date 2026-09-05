# output/checkpoints/

Results of merges, LoRA bakes, VAE bakes, and final blends land here
automatically. To chain operations (e.g. merge A+B, then bake a LoRA into
the result), copy or move the file you want to reuse from here back into
`input/checkpoints/` -- this keeps the two folders unambiguous about "raw
ingredients" vs. "results." The Image Generator tab also lists checkpoints
from here directly (prefixed `[output]`) so you can generate from a fresh
merge without moving it anywhere first.
