-- Explicit Data API access for fresh databases after Supabase's default-grant
-- change on October 30, 2026. Grant only on named application tables; preserve
-- existing privileges, RLS policies, data, and defaults for future objects.

grant select, insert, update, delete on table
  public.videos,
  public.credits,
  public.video_jobs,
  public.billing_credit_events,
  public.video_access_overrides,
  public.analytics_events,
  public.video_transcript_segments,
  public.api_keys,
  public.api_credits,
  public.api_billing_events,
  public.api_usage_events,
  public.mcp_oauth_clients,
  public.mcp_oauth_authorization_requests,
  public.mcp_oauth_authorization_codes,
  public.mcp_oauth_access_tokens,
  public.mcp_oauth_refresh_tokens,
  public.api_trial_grants,
  public.web_processing_charges
to service_role;

-- Preserve the owner operations already defined by the RLS policies.
grant select, insert, update, delete on table
  public.videos,
  public.credits,
  public.video_jobs,
  public.video_transcript_segments
to authenticated;

grant select on table
  public.billing_credit_events,
  public.video_access_overrides,
  public.api_credits,
  public.api_billing_events,
  public.api_usage_events,
  public.api_trial_grants,
  public.web_processing_charges
to authenticated;

grant select, delete on table public.api_keys to authenticated;
