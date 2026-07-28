"""Interfaces and implementations for uploading images referenced by spans."""

from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.images.traceloop_image_uploader import TraceloopImageUploader

__all__ = ["ImageUploader", "TraceloopImageUploader"]
