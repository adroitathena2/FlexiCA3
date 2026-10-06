# tests/golden

`replay.jsonl` is intentionally empty (0 lines): it is the replay stub that
`REPLAY_PATH` defaults to (`core/config.py`). `JsonlReplayStore` treats a
missing or empty file as zero recorded entries (blank lines are skipped, a
miss returns `None` and the caller degrades visibly), so an empty stub keeps
the suite green until a live capture populates it via `record()`.

Do NOT hand-write entries here: only `record()` output from a genuine model
call is legitimate evidence. See `docs/samples/README.md` for the committed
canonical trace, which lives separately.
