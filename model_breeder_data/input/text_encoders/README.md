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
