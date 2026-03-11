"""First-party product analytics event tracking."""
from __future__ import annotations

from src.db.supabase import get_client
from src.utils.logging import get_logger

logger = get_logger(__name__)

ALLOWED_EVENTS = {
    "signup_complete",
    "video_submitted",
    "video_ready",
    "search_run",
    "search_success",
    "checkout_started",
    "checkout_success",
}


def track(
    event_name: str,
    *,
    user_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert a product analytics event. Fire-and-forget: never raises."""
    if event_name not in ALLOWED_EVENTS:
        logger.warning("analytics.track called with unknown event_name=%s", event_name)
        return

    if event_name == "signup_complete" and user_id is None:
        logger.warning("analytics.track rejecting signup_complete with user_id=None")
        return

    try:
        client = get_client()
        client.table("analytics_events").insert({
            "event_name": event_name,
            "user_id": user_id,
            "metadata": metadata or {},
        }).execute()
    except Exception as exc:
        # Duplicate-key from partial unique index on signup_complete is expected.
        exc_str = str(exc)
        if "analytics_events_signup_complete_user_unique" in exc_str:
            logger.debug("Duplicate signup_complete for user_id=%s (no-op)", user_id)
            return
        logger.warning("analytics.track failed event=%s: %s", event_name, exc)
