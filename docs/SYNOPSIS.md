# Paytriq — synopsis

Campus sponsorship dealmaking is still done by hand: a student organiser cold-mails
nearby businesses, quotes a flat fee copied from last year's fest, and negotiates over
email with no record of what was promised or why a number was chosen. The previous
prototype automated the shape of that workflow while keeping its failure modes — a
hardcoded list of invented businesses with sequential fake phone numbers, one flat
price for every event regardless of footfall, a reply router that tested for "yes"
before it tested for price objections, and a human approval gate that defaulted to
`approve=True`.

Paytriq replaces that pipeline with seven reasoning agents (A1 Discovery … A7 Arbiter)
coordinating through a typed append-only blackboard under a LangGraph `StateGraph`.
Every routing decision goes through a named decision backend (`clef → gemini → rules`)
and names its source; every outbound email and MoU release parks at a human gate
backed by a durable approval row; every figure in the final ROI report carries a
stated assumption. The execution trace — including `distinct_gap_values`, a timing
statistic that distinguishes a genuine capture from a hand-written one — is the
system's primary output, not an afterthought.

Built for a student fest organiser who needs sponsors without inventing them: where the proposal sketched a pipeline, this run replaces the invented business list with evidence-backed discovery, the flat fee with footfall-priced tiers, the yes-first reply router with a proper intent classifier, and the default-approve gate with a durable human approval row. To try it, run `uvicorn api.main:app --port 18780`, read the interactive contract at `/docs`, open the dependency-free console in `frontend/` to prefill the demo event and drive the seven stages (approving the SEND and MOU gates when they park); architecture, contracts, and the evaluation with its headline negative finding are in `docs/DESIGN.md`, `docs/INPUT.md`, `docs/OUTPUT.md`, and `docs/EVALUATION.md`.
