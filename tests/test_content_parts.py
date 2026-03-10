"""Tests for multimodal content utilities."""

from __future__ import annotations

from llm_expose.core.content_parts import (
    build_local_attachment_descriptor,
    content_has_images,
    extract_image_urls,
    extract_invocation_attachments,
    normalize_mcp_content,
    strip_image_parts,
)


def test_normalize_mcp_content_converts_base64_image_to_image_url() -> None:
    content = [
        {"type": "text", "text": "Result"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        },
    ]

    normalized = normalize_mcp_content(content)

    assert normalized[0] == {"type": "text", "text": "Result"}
    assert normalized[1]["type"] == "image_url"
    assert normalized[1]["image_url"]["url"].startswith("data:image/png;base64,AAAA")


def test_strip_image_parts_removes_images_and_preserves_text_parts() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            ],
        }
    ]

    stripped, removed = strip_image_parts(messages)

    assert removed == 1
    assert stripped[0]["content"] == [{"type": "text", "text": "hello"}]
    assert content_has_images(stripped[0]["content"]) is False


def test_extract_image_urls_reads_openai_image_parts() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    assert extract_image_urls(content) == [
        "https://example.com/a.jpg",
        "data:image/png;base64,AAAA",
    ]


def test_extract_invocation_attachments_returns_data_url_descriptors() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
    ]

    attachments = extract_invocation_attachments(content)

    assert len(attachments) == 2
    assert attachments[0]["kind"] == "image"
    assert attachments[0]["source_type"] == "data_url"
    assert attachments[0]["media_type"] == "image/png"
    assert attachments[0]["size_bytes"] == 3
    assert attachments[1]["source_type"] == "url"
    assert attachments[1]["url"] == "https://example.com/a.jpg"


def test_build_local_attachment_descriptor_respects_path_redaction(tmp_path) -> None:
    image_file = tmp_path / "frame.jpg"
    image_file.write_bytes(b"abc")

    redacted = build_local_attachment_descriptor(image_file, kind="image", include_path=False)
    visible = build_local_attachment_descriptor(
        image_file,
        kind="image",
        include_path=True,
        attachment_ref="att_test",
    )

    assert redacted["source_type"] == "local_path"
    assert redacted["path"] is None
    assert redacted["attachment_ref"] is None
    assert visible["path"] == str(image_file.resolve())
    assert visible["attachment_ref"] == "att_test"
    assert visible["filename"] == "frame.jpg"
    assert visible["size_bytes"] == 3
