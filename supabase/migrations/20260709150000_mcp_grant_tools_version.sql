-- One-time re-consent for the Claude MCP connector tool surface.
--
-- The durable MCP OAuth grant has no dedicated connections table: consent is
-- carried by the authorization code minted at approval and then by the
-- access/refresh token rows that share a connection_id. Refresh rotation
-- revokes and re-creates rows in both token tables with the same
-- connection_id, so the approved-tools version must live on those rows and be
-- propagated on every rotation.
--
-- approved_tools_version semantics:
--   1 = the historical four-tool approval screen
--       (upload_video, get_video_status, list_videos, search_video)
--   2 = the current six-tool approval screen
--       (adds get_transcript and get_frames)
--
-- Existing rows default to 1: any token or in-flight authorization code
-- created before this migration was consented against the four-tool list.
-- Token validation rejects versions older than the server-side constant
-- MCP_APPROVED_TOOLS_VERSION (src/api/mcp_oauth.py), which forces a one-time
-- OAuth reconnect without any mass token deletion.

alter table public.mcp_oauth_authorization_codes
  add column if not exists approved_tools_version integer not null default 1;

alter table public.mcp_oauth_access_tokens
  add column if not exists approved_tools_version integer not null default 1;

alter table public.mcp_oauth_refresh_tokens
  add column if not exists approved_tools_version integer not null default 1;
