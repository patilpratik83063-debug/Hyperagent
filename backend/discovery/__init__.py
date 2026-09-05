"""Discovery providers (swappable implementations behind DiscoveryClient)."""
from .base import (
    DiscoveryClient,
    DiscoveryConfigError,
    DiscoveryError,
    RawPost,
    SearchBatchResult,
    canonical_post_url,
    parse_posted_at,
)
from .serp_client import SerperDiscoveryClient

__all__ = [
    "DiscoveryClient",
    "DiscoveryConfigError",
    "DiscoveryError",
    "RawPost",
    "SearchBatchResult",
    "canonical_post_url",
    "parse_posted_at",
    "SerperDiscoveryClient",
]
