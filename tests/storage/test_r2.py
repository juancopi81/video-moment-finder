from __future__ import annotations

from pathlib import Path

import boto3

from src.storage.config import R2Config
from src.storage.r2 import R2Store, thumbnail_key


class FakePaginator:
    def __init__(self, pages_by_prefix: dict[str, list[dict]]) -> None:
        self._pages_by_prefix = pages_by_prefix

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        return self._pages_by_prefix.get(prefix, [])


class FakeClient:
    def __init__(self, pages_by_prefix: dict[str, list[dict]] | None = None) -> None:
        self.uploaded: list[tuple[str, str, dict]] = []
        self.deleted: list[dict] = []
        self._pages_by_prefix = pages_by_prefix or {}

    def upload_fileobj(self, fileobj, bucket: str, key: str, ExtraArgs: dict) -> None:  # noqa: N803
        fileobj.read()
        self.uploaded.append((bucket, key, ExtraArgs))

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_objects_v2"
        return FakePaginator(self._pages_by_prefix)

    def delete_objects(self, Bucket: str, Delete: dict) -> None:  # noqa: N803
        self.deleted.extend(Delete.get("Objects", []))


def _make_config(public_url: str | None = None) -> R2Config:
    return R2Config(
        endpoint_url="https://r2.example.com",
        access_key_id="access",
        secret_access_key="secret",
        bucket_name="video-thumbnails",
        public_url=public_url,
    )


def test_thumbnail_key() -> None:
    assert thumbnail_key("video_a", 7) == "thumb/video_a/thumb_00007.jpg"


def test_upload_thumbnail_builds_url(tmp_path, monkeypatch) -> None:
    fake_client = FakeClient()
    monkeypatch.setenv("R2_UPLOAD_WORKERS", "1")
    monkeypatch.setenv("TQDM_DISABLE", "1")

    def fake_boto_client(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(boto3, "client", fake_boto_client)

    config = _make_config(public_url="https://cdn.example.com")
    store = R2Store(config)

    thumb_path = tmp_path / "thumb.jpg"
    thumb_path.write_bytes(b"\xff\xd8\xff")

    result = store.upload_thumbnail("video_a", 3, thumb_path)

    assert result.key == "thumb/video_a/thumb_00003.jpg"
    assert result.url == "https://cdn.example.com/thumb/video_a/thumb_00003.jpg"
    assert fake_client.uploaded


def test_upload_thumbnails_multiple(tmp_path, monkeypatch) -> None:
    fake_client = FakeClient()
    monkeypatch.setenv("R2_UPLOAD_WORKERS", "1")
    monkeypatch.setenv("TQDM_DISABLE", "1")

    def fake_boto_client(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(boto3, "client", fake_boto_client)

    store = R2Store(_make_config())

    thumbs: list[tuple[int, Path]] = []
    for idx in range(2):
        path = tmp_path / f"thumb_{idx}.jpg"
        path.write_bytes(b"\xff\xd8\xff")
        thumbs.append((idx, path))

    results = store.upload_thumbnails("video_b", thumbs)
    assert len(results) == 2
    assert len(fake_client.uploaded) == 2


def test_delete_video_thumbnails(monkeypatch) -> None:
    pages_by_prefix = {
        "thumb/video_a/": [
            {"Contents": [{"Key": "thumb/video_a/thumb_00000.jpg"}]},
        ],
    }
    fake_client = FakeClient(pages_by_prefix)
    monkeypatch.setenv("R2_UPLOAD_WORKERS", "1")
    monkeypatch.setenv("TQDM_DISABLE", "1")

    def fake_boto_client(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(boto3, "client", fake_boto_client)

    store = R2Store(_make_config())
    deleted = store.delete_video_thumbnails("video_a")

    assert deleted == 1
    assert len(fake_client.deleted) == 1


def test_delete_video_thumbnails_no_objects(monkeypatch) -> None:
    fake_client = FakeClient(pages_by_prefix={})
    monkeypatch.setenv("R2_UPLOAD_WORKERS", "1")
    monkeypatch.setenv("TQDM_DISABLE", "1")

    def fake_boto_client(*args, **kwargs):
        return fake_client

    monkeypatch.setattr(boto3, "client", fake_boto_client)

    store = R2Store(_make_config())
    deleted = store.delete_video_thumbnails("video_a")

    assert deleted == 0
    assert fake_client.deleted == []
