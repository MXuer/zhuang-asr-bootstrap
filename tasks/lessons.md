# Lessons

- Do not conclude MMS alignment cannot handle long audio until the exact environment is checked. `qwen3-asr` succeeded where base Python segfaulted.
- Do not use filesystem alphabetical order for Bible book order. Use the fixed NT order: `MAT MRK LUK JHN ACT ROM 1CO 2CO GAL EPH PHP COL 1TH 2TH 1TI 2TI TIT PHM HEB JAS 1PE 2PE 1JN 2JN 3JN JUD REV`.
- Do not strip punctuation before confirming the downstream component needs it stripped. MMS aligner handles punctuation internally.
- For long background jobs, call the environment Python directly instead of wrapping with `conda run`, and write resumable outputs.

