"""Native workspace authorization, evidence boundaries and MCP resource metadata."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from mcp import ClientSession

from src.api.workspace import LearningView, UI_URI
from tests.api.test_mcp import _authorized_headers, _issue_access_token, _run_mcp_session
from tests.api.conftest import _upload_video_record


def sample_view(kind="study-guide"):
    data = {
        "kind": kind, "video_id": "00000000-0000-4000-8000-000000000001", "title": "Dot products",
        "coverage": {"kind": "excerpt", "start_s": 0, "end_s": 30},
        "sources": [{"id": "definition", "kind": "transcript", "start_s": 2, "end_s": 8, "summary": "Coordinate definition"}],
    }
    if kind == "study-guide":
        data["sections"] = [{"title": "The rule", "origin": "lecture", "paragraphs": ["v·w is the sum of coordinate products."], "citations": ["definition"]}]
    if kind == "flashcards":
        data["cards"] = [{"front": "Compute (3,4)·(2,0)", "back": "6", "rationale": "3×2+4×0=6", "citations": ["definition"]}]
    if kind == "playground":
        data["playground"] = {"model": "dot-product-2d", "question": "What changes the sign?", "model_limit": "Real Euclidean 2D vectors.", "extension_note": "Generated teaching model.", "initial": {"v": [3, 4], "w": [2, 0]}, "citations": ["definition"]}
    if kind == "presentation":
        data["slides"] = [{"title": "The rule", "bullets": ["Multiply corresponding coordinates and add."], "notes": "Ask for a prediction.", "origin": "lecture", "citations": ["definition"]}]
    return data


@pytest.mark.parametrize("change", [
    lambda d: d.update(video_id="not-a-video-uuid"),
    lambda d: d["coverage"].update(end_s=-1),
    lambda d: d["sources"][0].update(start_s=float("nan")),
    lambda d: d["sections"][0].update(citations=["invented"]),
    lambda d: d["sections"][0].update(html="<script>execute()</script>"),
    lambda d: d["sections"][0].update(paragraphs=["https://example.test/?X-Amz-Signature=secret"]),
    lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])),
])
def test_invalid_or_capability_bearing_view_rejected(change):
    data = sample_view(); change(data)
    with pytest.raises(ValueError):
        LearningView.model_validate(data)


def test_generated_cards_need_reasoning_and_unknown_fields_fail():
    data = sample_view("flashcards"); data["cards"][0].pop("rationale")
    with pytest.raises(ValueError, match="reasoning"):
        LearningView.model_validate(data)


def test_source_markup_is_data_not_instructions():
    data = sample_view(); data["sections"][0]["paragraphs"] = ["<script>source quotation</script>"]
    assert LearningView.model_validate(data).sections[0].paragraphs == ["<script>source quotation</script>"]


def test_incomplete_finite_case_table_rejected():
    data = sample_view("playground")
    data["playground"] = {
        "model": "reviewed-cases", "question": "What assumption matters?", "model_limit": "Reviewed cases only.", "extension_note": "Generated comparison.",
        "controls": [{"id": "condition", "label": "Condition", "options": [{"id": "yes", "label": "True"}, {"id": "no", "label": "False"}]}],
        "cases": [{"when": {"condition": "yes"}, "title": "First case", "outcome": "Supported", "explanation": "From the definition", "assumptions": ["The condition holds"], "citations": ["definition"]}] * 2,
    }
    with pytest.raises(ValueError, match="every control combination"):
        LearningView.model_validate(data)


def test_workspace_returns_owned_library_and_actual_allowance_without_media_capabilities(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    from src.api.app import _video_record_to_response
    monkeypatch.setattr("src.api.app.v1_list_my_videos", lambda identity: [_video_record_to_response(_upload_video_record("00000000-0000-4000-8000-000000000001", status="ready"))])
    monkeypatch.setattr("src.api.app.v1_api_billing_summary", lambda identity: SimpleNamespace(model_dump=lambda **kw: {"api_units_balance": 17, "unit_cost_index_video": 9}))

    async def callback(session):
        return await session.call_tool("open_workspace", {})
    result = _run_mcp_session(_authorized_headers(token), callback)
    assert not result.isError
    assert result.structuredContent["allowance"]["unit_cost_index_video"] == 9
    assert "source_url" not in result.structuredContent["videos"][0]
    assert "config" in result.meta["vmf"]


def test_playback_checks_owner_and_hides_signed_url_from_model(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    observed = []

    def get(video_id, identity):
        observed.append((video_id, identity.user_id))
        from src.api.app import _video_record_to_response
        response = _video_record_to_response(_upload_video_record("00000000-0000-4000-8000-000000000001", status="ready"))
        return response.model_copy(update={"source_url": "https://private.test/lesson?X-Amz-Signature=secret"})
    monkeypatch.setattr("src.api.app.v1_get_video", get)

    async def callback(session):
        return await session.call_tool("get_workspace_video", {"video_id": "requested"})
    result = _run_mcp_session(_authorized_headers(token), callback)
    assert observed == [("requested", "user_123")]
    assert "source_url" not in result.structuredContent["video"]
    assert "X-Amz" not in result.content[0].text
    assert "X-Amz" in result.meta["vmf"]["source_url"]


def test_render_checks_ownership_and_rejects_not_ready(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    from fastapi import HTTPException

    def forbidden(video_id, identity):
        raise HTTPException(404, "Video not found")
    monkeypatch.setattr("src.api.app.v1_get_video", forbidden)

    async def callback(session):
        return await session.call_tool("render_learning_view", {"view": sample_view()})
    result = _run_mcp_session(_authorized_headers(token), callback)
    assert result.isError
    monkeypatch.setattr("src.api.app.v1_get_video", lambda **kwargs: SimpleNamespace(status="processing"))
    result = _run_mcp_session(_authorized_headers(token), callback)
    assert result.isError and "ready" in result.content[0].text


def test_render_same_video_companions_are_validated_before_retrieval(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    other = sample_view("flashcards"); other["video_id"] = "00000000-0000-4000-8000-000000000002"
    monkeypatch.setattr("src.api.app.v1_get_video", lambda **kw: pytest.fail("Should reject before reading"))

    async def callback(session):
        return await session.call_tool("render_learning_view", {"view": sample_view(), "companion_views": [other]})
    result = _run_mcp_session(_authorized_headers(token), callback)
    assert result.isError and "same video" in result.content[0].text


def test_prepared_view_and_companions_round_trip_without_leaking_playback(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    from src.api.app import _video_record_to_response

    def owned(video_id, identity):
        assert video_id == "00000000-0000-4000-8000-000000000001"
        assert identity.user_id == "user_123"
        response = _video_record_to_response(_upload_video_record(video_id, status="ready"))
        return response.model_copy(update={"source_url": "https://private.test/lesson?X-Amz-Signature=secret"})
    monkeypatch.setattr("src.api.app.v1_get_video", owned)

    async def callback(session):
        return await session.call_tool("render_learning_view", {
            "view": sample_view(), "companion_views": [sample_view("flashcards"), sample_view("playground"), sample_view("presentation")],
        })
    rendered = _run_mcp_session(_authorized_headers(token), callback)
    assert not rendered.isError
    assert rendered.structuredContent["view"]["kind"] == "study-guide"
    assert [v["kind"] for v in rendered.structuredContent["companion_views"]] == ["flashcards", "playground", "presentation"]
    assert rendered.structuredContent["view"]["sections"][0]["bullets"] == []
    assert "X-Amz" not in str(rendered.structuredContent)
    assert "X-Amz" not in rendered.content[0].text
    assert "X-Amz" in rendered.meta["vmf"]["source_url"]


def test_ui_resource_csp_and_entrypoint_metadata_survive_real_mcp_transport(monkeypatch, mcp_oauth_store, tmp_path):
    token = _issue_access_token(monkeypatch)
    fixture = tmp_path / "workspace.html"; fixture.write_text("<!doctype html><title>Workspace test</title>")
    monkeypatch.setattr("src.api.mcp.UI_PATH", fixture)
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://account.r2.cloudflarestorage.com/path")

    async def callback(session: ClientSession):
        tools = await session.list_tools()
        resources = await session.list_resources()
        resource = await session.read_resource(UI_URI)
        return tools, resources, resource
    tools, resources, resource = _run_mcp_session(_authorized_headers(token), callback)
    opener = next(t for t in tools.tools if t.name == "open_workspace")
    assert not opener.inputSchema.get("required")
    assert opener.meta["openai/ui"]["entrypoints"] == [{"type": "global"}, {"type": "thread"}]
    assert opener.meta["ui"]["resourceUri"] == UI_URI
    content = resource.contents[0]
    descriptor = next(r for r in resources.resources if str(r.uri) == UI_URI)
    assert descriptor.meta == content.meta
    assert content.mimeType == "text/html;profile=mcp-app"
    assert content.meta["ui"]["csp"] == {"connectDomains": ["https://account.r2.cloudflarestorage.com"], "resourceDomains": ["https://account.r2.cloudflarestorage.com"]}
    assert content.meta["openai/ui"]["availableDisplayModes"] == ["fullscreen"]


@pytest.mark.parametrize("endpoint, expected", [
    ("", []),
    ("https://scoped.r2.cloudflarestorage.com/path", ["https://scoped.r2.cloudflarestorage.com"]),
])
def test_resource_policy_is_exact_and_import_safe(monkeypatch, endpoint, expected):
    from src.api.mcp import _workspace_resource_metadata
    monkeypatch.setenv("R2_ENDPOINT_URL", endpoint)
    assert _workspace_resource_metadata()["ui"]["csp"] == {
        "connectDomains": expected, "resourceDomains": expected,
    }
