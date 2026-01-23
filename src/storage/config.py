"""Configuration for storage backends."""
from __future__ import annotations

import os
from dataclasses import dataclass


class StorageConfigError(ValueError):
    """Raised when storage configuration is invalid or missing."""


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    url: str | None
    api_key: str | None
    collection_name: str
    in_memory: bool = False

    @classmethod
    def from_env(cls, collection_name: str = "video_frames") -> QdrantConfig:
        """Create config from environment variables."""
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")

        if not url:
            raise StorageConfigError("QDRANT_URL environment variable is required")

        return cls(url=url, api_key=api_key, collection_name=collection_name)

    @classmethod
    def in_memory(cls, collection_name: str = "video_frames") -> QdrantConfig:
        """Create in-memory config for testing."""
        return cls(url=None, api_key=None, collection_name=collection_name, in_memory=True)


@dataclass(frozen=True)
class R2Config:
    """Configuration for Cloudflare R2 storage."""

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    public_url: str | None = None

    @classmethod
    def from_env(cls) -> R2Config:
        """Create config from environment variables."""
        endpoint_url = os.environ.get("R2_ENDPOINT_URL")
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get("R2_BUCKET_NAME")
        public_url = os.environ.get("R2_PUBLIC_URL")

        missing = []
        if not endpoint_url:
            missing.append("R2_ENDPOINT_URL")
        if not access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not bucket_name:
            missing.append("R2_BUCKET_NAME")

        if missing:
            raise StorageConfigError(
                f"Missing required R2 environment variables: {', '.join(missing)}"
            )

        return cls(
            endpoint_url=endpoint_url,  # type: ignore[arg-type]
            access_key_id=access_key_id,  # type: ignore[arg-type]
            secret_access_key=secret_access_key,  # type: ignore[arg-type]
            bucket_name=bucket_name,  # type: ignore[arg-type]
            public_url=public_url,
        )
