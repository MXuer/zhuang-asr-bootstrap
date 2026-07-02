# Decisions

- Use `use_star=True` MMS alignment as the V1 starting point because it excludes non-text speech/noise better than `use_star=False`.
- Run MMS alignment in `/home/duhu/anaconda3/envs/qwen3-asr/bin/python` with `torch==2.6.0+cu124`; the base environment segfaulted on longer `forced_align` runs.
- Keep Chinese sentence punctuation `，。？！` in text outputs. Map ASCII punctuation to Chinese equivalents where needed: `, -> ，`, `. ; -> 。`, `? -> ？`, `! -> ！`.
- Remove quote characters from training text.
- VAD correction should adjust alignment boundaries but must keep any silence that still belongs inside the spoken span. Silence is allowed inside a segment.
- Every dataset version needs an audit manifest and a validation report before it can be used for training.
- V1 is real read-speech only. OpenVoice, noise, speed perturb, and partial-span augmentation are later versions, not part of V1 clean.

