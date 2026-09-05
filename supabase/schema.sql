-- ===========================================================================
-- Hyperclients — Lead Qualification Engine
-- Supabase / Postgres schema. Run this in the Supabase SQL editor (or psql).
-- ===========================================================================

create extension if not exists pgcrypto;

-- ---------------------------------------------------------------------------
-- searches: one row per user search request
-- ---------------------------------------------------------------------------
create table if not exists searches (
  id uuid primary key default gen_random_uuid(),
  service text not null,
  country text default '',
  lead_type text check (lead_type in ('need_freelancer','hiring_buyer','our_agency')),
  time_window text check (time_window in ('24h','7d','14d','28d')),
  leads_needed int not null,
  -- lifecycle: queued | running | completed | failed | no_results
  status text not null default 'queued',
  found_count int not null default 0,
  accepted_count int not null default 0,
  scanned_count int not null default 0,
  error text,
  created_at timestamptz not null default now(),
  finished_at timestamptz
);

create index if not exists idx_searches_created_at on searches (created_at desc);

-- ---------------------------------------------------------------------------
-- leads: qualified posts only (everything else is rejected upstream)
-- ---------------------------------------------------------------------------
create table if not exists leads (
  id uuid primary key default gen_random_uuid(),
  search_id uuid references searches(id) on delete cascade,
  lead_type text check (lead_type in ('need_freelancer','hiring_buyer','our_agency')),
  time_window text check (time_window in ('24h','7d','14d','28d')),
  post_url text not null unique,          -- canonical post identity (dedupe)
  author_name text,
  author_profile_url text,
  post_text text,
  post_date date,                          -- re-filter-by-window queries hit this
  overall_quality_score numeric,
  service_match_score numeric,
  intent_strength text,
  status text not null default 'new',      -- new | contacted | replied | not_a_fit
  notes text,
  created_at timestamptz not null default now()
);

-- The UI re-filters saved leads by time window on post_date directly.
create index if not exists idx_leads_post_date on leads (post_date desc);
create index if not exists idx_leads_search_id on leads (search_id);
create index if not exists idx_leads_status on leads (status);

-- ---------------------------------------------------------------------------
-- Optional: row-level security
-- This app talks to Supabase with the service-role key (server-side only), so
-- RLS is off by default. If you ever expose the API directly to browsers,
-- enable RLS and add policies; never put the service-role key in frontend code.
-- ---------------------------------------------------------------------------
-- alter table searches enable row level security;
-- alter table leads enable row level security;
