# Hyperclients — LinkedIn Buyer Lead Qualification Engine (v2)

One job: find LinkedIn posts from **genuine buyers** of a service you sell,
reject everything else, score what's left, and show only real leads in the
UI. Precision matters more than volume — a false positive costs more trust
than a missed lead. Every filter, prompt and gate in this repo exists to
protect precision; the engine never pads a result with weak matches.

```
backend/                    FastAPI app (Python 3.12+)
  main.py                   routes, background search worker, static UI mount
  models.py                 LeadType / TimeWindow / schemas (single source of truth)
  discovery/
    base.py                 DiscoveryClient interface + RawPost + shared parsing
    serp_client.py          Google SERP (Serper.dev-style) implementation of the interface
  query_builder.py          high-signal per-lead-type LinkedIn queries (§6)
  prefilter.py              deterministic cheap rejects before AI spend
  classifier.py             LLM structured classification (DeepSeek by default), fail-closed (§7)
  scoring.py                weighted score + hard acceptance gates (§8)
  engine.py                 iterative exact-count orchestrator (§9)
  db.py                     Supabase store + in-memory store (same protocol)
  geography.py              country normalization (alias/ISO/city table)
  testing/mock_providers.py offline demo corpus + deterministic classifier
  tests/                    pytest suite (offline — no keys needed)
frontend/index.html         single-file UI (no build step)
supabase/schema.sql         Postgres schema (run once in Supabase)
```

## The precision pipeline

```
user request ─▶ search row ─▶ query set (3-5 natural phrasings per lead type,
     each paired with negative seller terms: -"we offer" -"our services"
     -"book a call" -"dm us" -"we specialize" -"we help")
   ─▶ country normalized to a canonical code (geography.py alias table)
   ─▶ Google SERP search: "<query> site:linkedin.com/posts after:YYYY-MM-DD"
        (Serper.dev; after-date recomputed fresh per request, §3)
   ─▶ dedupe by canonical post URL (every lead carries one — the UI offers
        "Open post ↗" + Copy URL per row)
   ─▶ deterministic prefilter (cheap drops: job ads, sellers, job seekers,
        marketplaces, advice content — only when no buyer phrase is present)
   ─▶ LLM structured classification (DeepSeek deepseek-chat, direction-of-intent, §2 traps in the prompt)
   ─▶ hard gates: type is a buyer AND buying-not-selling AND service match ≥ 50
        AND intent ≥ recommendation AND weighted score ≥ 60 AND model is_qualified
   ─▶ slice to EXACTLY N best-scoring leads (never more, never padded)
   ─▶ persist to Supabase ─▶ UI table + past searches + window re-filter
```

## Discovery — Google SERP over LinkedIn posts (§0)

Discovery is **not** LinkedIn-native scraping. It queries Google via a SERP
API (Serper.dev by default) with:

```
<buyer query + negative terms> site:linkedin.com/posts after:YYYY-MM-DD
```

The `after:` date is recomputed from "now" on every request from the chosen
time window (§3) — it is never hardcoded or stored as a value.

**Designed-for trade-offs (they are provider behavior, not bugs):**

- **Coverage is partial and inconsistent.** Google does not fully or promptly
  crawl `linkedin.com/posts`, so a tight exact-phrase query + `site:` + a
  narrow window will often return zero results even when matching posts
  exist. The engine therefore runs 3–5 phrasings per search and treats a
  zero-hit search as a *normal shortage*, never as a failure — the UI
  explains it.
- **The 24h window structurally underperforms** — Google's crawl lag on
  LinkedIn is typically 1–3 days. The UI shows a note under the 24h option;
  treat 7d/14d/28d as the windows with usable volume.
- **SERP relevance is a starting filter, not a verdict.** Every result still
  passes the deterministic prefilter and the LLM classifier, which catch
  the agency self-promotion / job-seeker posts that happen to rank.
- **Swappable vendor:** everything sits behind `DiscoveryClient.search_posts
  (queries, since)`, so swapping the SERP vendor later never touches the rest
  of the codebase.

## Generalization — this is not a single-niche tool

`service` and `country` are free-text inputs; nothing in the codebase branches
on what the service *is*. Query templates interpolate `{service}` as an opaque
string, the prefilter keywords and classifier prompt teach *generic* sell-side
patterns and a domain-agnostic direction-of-intent test, and the test suite
runs the whole pipeline against unrelated services (plumber/Nairobi-KE,
UX designer/Toronto-CA, wedding photographer/Mumbai-IN) with zero code changes.
The only service mentions in the classifier prompt are the §7 few-shot pairs,
which exist purely to teach the reasoning pattern.

**Country normalization** (`backend/geography.py`): whatever the user types —
full name ("United States"), ISO code ("US"), abbreviation ("USA", "u.s.a."),
or a city implying a country ("Nairobi" → KE, "Mumbai" → IN, "Toronto" → CA) —
is mapped case/space/accent-insensitively to a canonical ISO alpha-2 code via
a broad alias table and stored on the search row. Unknown input degrades
gracefully to the raw free-text location signal (never a silent hard reject).

## The post URL is a first-class, guaranteed field

Every accepted lead has one: the SERP client drops results whose link is not a
`linkedin.com/posts/` URL; the engine refuses URL-less candidates
defensively; leads are deduped, stored and re-sliced on the canonical post URL
(`leads.post_url` is unique); the API returns it; and the UI shows
"Open post ↗" plus a **Copy URL** button per row.

## Setup

1. Python 3.12+: `python -m venv .venv` then activate, `pip install -r backend/requirements.txt`.
2. Create a Supabase project; run `supabase/schema.sql` in its SQL editor.
3. `cd backend && copy .env.example .env` and fill in:
   - `DEEPSEEK_API_KEY` (https://platform.deepseek.com — classification uses
     `deepseek-chat`; `LLM_PROVIDER=deepseek` is the default),
   - `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`,
   - `SERPER_API_KEY` from https://serper.dev (free tier; each SERP query costs
     1 of your monthly searches).
4. Run: `uvicorn main:app --reload --port 8000` (from `backend/`).
5. Open http://127.0.0.1:8000 — the single-file UI is served from `frontend/`.

**Offline demo (no keys):** set `MOCK_MODE=1` in `.env`. The app then uses an
in-memory store, a mock discovery corpus (with the §2 traps baked in) and a
deterministic classifier, so you can click through the whole flow locally.
Without `MOCK_MODE`, discovery and classification are fail-closed: a search
started while config is missing returns `503` naming exactly what to set.

## Config reference

| Env | Default | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `deepseek` | `deepseek` \| `openai` |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` | — / `deepseek-chat` | Classification LLM (DeepSeek) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek OpenAI-compatible endpoint |
| `OPENAI_API_KEY` / `OPENAI_MODEL` (legacy) | — / `gpt-4o` | Only when `LLM_PROVIDER=openai` |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | — | Postgres persistence |
| `SERPER_API_KEY` | — | Google SERP access (https://serper.dev) |
| `SERPER_BASE_URL` | `https://google.serper.dev` | Any Serper-compatible endpoint |
| `SERPER_SITE_RESTRICTION` | `linkedin.com/posts` | Appended to every query as `site:` |
| `SERPER_RESULTS_PER_QUERY` | 10 | `num` results per query (free tier caps at 10) |
| `SERPER_GL` / `SERPER_HL` | — / `en` | Optional Google country/language |
| `DISCOVERY_PROVIDER` | `auto` | `auto` \| `serp` \| `mock` |
| `MOCK_MODE` | `0` | 1 = offline demo (memory store + mock providers) |
| `MAX_SEARCHES_PER_DAY` | 20 | Usage budget behind `GET /api/usage` |
| `MIN_OVERALL_SCORE` / `MIN_SERVICE_MATCH` / `MIN_INTENT_STRENGTH` | 60 / 50 / `recommendation` | Hard gates |
| `ENGINE_MAX_ITERATIONS` / `ENGINE_DEADLINE_SECONDS` / `ENGINE_EARLY_STOP_EMPTY_ROUNDS` | 30 / 1800 / 3 | Exact-count loop caps |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/search` | `{service, country, lead_type, time_window, leads_needed}` → runs in background, returns `search_id` |
| `GET` | `/api/search/{id}/status` | progress poll (`completed` / `failed`) |
| `GET` | `/api/leads?search_id=&time_window=&status=` | fetch / re-filter saved leads by window (`24h|7d|14d|28d`) |
| `PATCH` | `/api/leads/{id}` | `{status, notes}` |
| `GET` | `/api/searches` | past searches |
| `GET` | `/api/usage` | remaining searches today |

The time window is converted to a concrete `after` date **computed fresh from
"now" on every request** (rounded down to the widest bucket guaranteeing at
least one full day) — it is never hardcoded or stored as a value. The window
is stored on the search row so results can be re-filtered or re-run later,
and the UI's secondary filter bar re-slices already-saved leads by
`posted_at` without re-running a search.

**Exact-count loop:** whatever number the user asks for (10/25/50/100), the
engine keeps generating fresh query phrasings and re-scanning until it has
collected exactly N qualified leads — it never overdelivers (it slices to
exactly N) and it keeps looping (not a fixed small number of rounds) up to
`ENGINE_MAX_ITERATIONS` (30) or the deadline. It only stops short when
discovery is genuinely dry (several rounds in a row adding nothing new) and
reports the shortage instead of padding with weak matches. The window options
offered in the UI are 7d/14d/28d.

## Tests

All offline, no keys required:

```bash
cd backend
python -m pytest tests -q
```

Covers models/time-window math, country normalization (aliases/ISO/cities/
accent-folding + graceful fallback), query quality per lead type with §6
negative pairing across unrelated services, prefilter rules, scoring gates,
LLM structured-output validation + fail-closed isolation (DeepSeek json_object
and OpenAI json_schema request shapes), the Serper
client (query construction with fresh `after:` date, organic-result mapping,
quiet-zero-hit behavior, loud provider errors) with HTTP fully mocked, the
full engine loop (exact-N delivery, no padding, early stop, provider-error
failures, zero-hit shortages, dedupe), whole-pipeline generalization over ≥3
unrelated services/countries, and the HTTP API end to end.

## How the classification prompt protects precision (§2 traps)

The classifier is built around the **direction-of-intent** question — who
needs vs who offers — with explicit examples of every never-a-lead category:
sellers/offerings ("we offer…", "DM me for…", white-label pitches), job
seekers ("open to work"), talent marketplaces / recruiting-sellers (recruiting
freelancers as inventory is **not** a buyer), agency self-promotion ("our
agency can help" is a seller even though the word "agency" appears — never
`our_agency`), and thought leadership with no procurement action. The
classification LLM (DeepSeek `deepseek-chat` by default) must
return structured JSON fields; every response is re-validated with
Pydantic, and any failure (unavailable model, timeout, schema violation) drops
the candidate — never guesses it into "qualified".
