"""Cloudflare R2 storage for video thumbnails."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.config import Config

from src.storage.config import R2Config


class R2StorageError(RuntimeError):
    """Raised when R2 operations fail."""


@dataclass(frozen=True)
class UploadResult:
    """Result of a thumbnail upload."""

    key: str
    url: str


def thumbnail_key(video_id: str, frame_index: int) -> str:
    """Generate R2 key for a thumbnail."""
    return f"{video_id}/thumb_{frame_index:05d}.jpg"


class R2Store:
    """Cloudflare R2 storage for video thumbnails."""

    def __init__(self, config: R2Config) -> None:
        self._config = config
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def _build_url(self, key: str) -> str:
        """Build public URL for a key."""
        if self._config.public_url:
            return f"{self._config.public_url.rstrip('/')}/{key}"
        return f"{self._config.endpoint_url}/{self._config.bucket_name}/{key}"

    def upload_thumbnail(
        self,
        video_id: str,
        frame_index: int,
        file_path: Path,
    ) -> UploadResult:
        """Upload a single thumbnail to R2."""
        if not file_path.exists():
            raise R2StorageError(f"Thumbnail file not found: {file_path}")

        key = thumbnail_key(video_id, frame_index)

        try:
            with file_path.open("rb") as f:
                self._client.upload_fileobj(
                    f,
                    self._config.bucket_name,
                    key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
        except Exception as exc:
            raise R2StorageError(f"Failed to upload thumbnail: {exc}") from exc

        return UploadResult(key=key, url=self._build_url(key))

    def upload_thumbnails(
        self,
        video_id: str,
        thumbnails: list[tuple[int, Path]],
    ) -> list[UploadResult]:
        """Upload multiple thumbnails to R2."""
        results = []
        for frame_index, file_path in thumbnails:
            result = self.upload_thumbnail(video_id, frame_index, file_path)
            results.append(result)
        return results

    def delete_video_thumbnails(self, video_id: str) -> int:
        """Delete all thumbnails for a video. Returns count of deleted objects."""
        prefix = f"{video_id}/"

        try:
            # List all objects with this prefix
            paginator = self._client.get_paginator("list_objects_v2")
            objects_to_delete = []

            for page in paginator.paginate(
                Bucket=self._config.bucket_name,
                Prefix=prefix,
            ):
                if "Contents" in page:
                    objects_to_delete.extend(
                        [{"Key": obj["Key"]} for obj in page["Contents"]]
                    )

            if not objects_to_delete:
                return 0

            # Delete in batches of 1000 (S3 limit)
            deleted_count = 0
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                self._client.delete_objects(
                    Bucket=self._config.bucket_name,
                    Delete={"Objects": batch},
                )
                deleted_count += len(batch)

            return deleted_count
        except Exception as exc:
            raise R2StorageError(f"Failed to delete thumbnails: {exc}") from exc
