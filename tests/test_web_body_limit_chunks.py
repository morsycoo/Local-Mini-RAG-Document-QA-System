import asyncio
import json
from types import SimpleNamespace

from app.web.main import RequestBodyLimitMiddleware


def make_scope(limit=1024):
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(max_request_body_bytes=limit)
        )
    )
    return {
        "type": "http",
        "path": "/api/test",
        "method": "POST",
        "headers": [],
        "app": app,
    }


def test_chunked_body_without_content_length_is_rejected_over_limit():
    async def run_test():
        chunks = [
            {
                "type": "http.request",
                "body": b"x" * 600,
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"y" * 425,
                "more_body": False,
            },
        ]
        downstream_calls = []
        sent_messages = []

        async def receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.disconnect"}

        async def send(message):
            sent_messages.append(message)

        async def downstream_app(scope, receive, send):
            downstream_calls.append(True)

        middleware = RequestBodyLimitMiddleware(downstream_app)
        await middleware(make_scope(), receive, send)

        assert downstream_calls == []
        assert sent_messages[0]["type"] == "http.response.start"
        assert sent_messages[0]["status"] == 413

        headers = dict(sent_messages[0]["headers"])
        assert headers.get(b"x-request-id")
        assert headers.get(b"x-content-type-options") == b"nosniff"

        assert sent_messages[1]["type"] == "http.response.body"
        assert json.loads(sent_messages[1]["body"]) == {
            "error": "Request body too large."
        }

    asyncio.run(run_test())


def test_chunked_body_without_content_length_within_limit_is_allowed():
    async def run_test():
        chunks = [
            {
                "type": "http.request",
                "body": b"x" * 400,
                "more_body": True,
            },
            {
                "type": "http.request",
                "body": b"y" * 400,
                "more_body": False,
            },
        ]
        received_chunks = []
        sent_messages = []

        async def receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.disconnect"}

        async def send(message):
            sent_messages.append(message)

        async def downstream_app(scope, receive, send):
            first = await receive()
            second = await receive()

            received_chunks.extend([first, second])

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            await send({
                "type": "http.response.body",
                "body": b"ok",
            })

        middleware = RequestBodyLimitMiddleware(downstream_app)
        await middleware(make_scope(), receive, send)

        assert len(received_chunks) == 2
        assert received_chunks[0]["body"] == b"x" * 400
        assert received_chunks[0]["more_body"] is True
        assert received_chunks[1]["body"] == b"y" * 400
        assert received_chunks[1]["more_body"] is False

        assert sent_messages[0]["status"] == 200
        assert sent_messages[1]["body"] == b"ok"

    asyncio.run(run_test())
