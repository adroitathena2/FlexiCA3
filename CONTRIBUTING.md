# Contributing to Paytriq

## Branches and PRs — one member, one branch

- Each member works on their **own branch** (`<login>/what-you-are-doing`)
  and opens a **pull request** for review. No direct pushes to `main`.
- Keep PRs small and scoped; one rubric area per PR beats one mega-PR.

## Commits — your login, your authorship

- Commit **only as yourself** (your own login / `user.name`). Never commit
  on behalf of another member.
- Write messages that say what changed and why, not just "fix".

## History is append-only

- **No history rewriting**: no `push --force`, no rebase of shared branches,
  no amendments to commits other members have pulled. If a mistake lands,
  fix it with a new commit or a revert.
- `traces/` is gitignored and never committed; the curated copy in
  `docs/samples/` is the only trace artefact that belongs in git.

## Before opening a PR

```powershell
ruff check .
python -m pytest tests/unit/test_eval.py tests/unit/test_observability.py -q
```

Secrets (`.env`, API keys) are never committed — see `.env.example`.
