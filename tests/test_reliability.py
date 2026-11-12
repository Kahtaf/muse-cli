import argparse
import io
import os
import stat
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from curl_cffi.requests.websockets import WebSocketTimeout
from muse_cli import cli, gateway, update


class SendBoundaryTest(unittest.TestCase):
    def test_failed_history_boundary_does_not_send_or_return_an_old_reply(self):
        for error in (gateway.GatewayError(503, "unavailable"), TimeoutError("history timeout")):
            with self.subTest(error=type(error).__name__):
                gw = MagicMock()
                gw.call_json.side_effect = [error, {"chat_events": [{
                    "seq": 9, "event_name": "message.assistant", "payload": {
                        "display_text": "old answer", "status": "completed"}}]}]
                args = argparse.Namespace(text="new question", thread="test", wait=1, download=None)
                with patch.object(cli, "load_config", return_value={"node_id": "test"}), \
                        patch.object(cli, "connect", return_value=gw), patch.object(cli, "out"), \
                        self.assertRaises(type(error)):
                    cli.cmd_send(args)
                gw._open.assert_not_called()
                gw.close.assert_called_once()


def response(body, end_body):
    sf = gateway.ServiceFrame(stream_id=7)
    sf.response.status = 200
    sf.response.body = body
    sf.response.end_body = end_body
    return sf


class ResponseCompletionTest(unittest.TestCase):
    def test_deadline_after_headers_rejects_incomplete_body(self):
        gw = gateway.Gateway.__new__(gateway.Gateway)
        gw._open = MagicMock(return_value=7)
        gw._read_frame = MagicMock(return_value=response(b"partial", False))
        with patch.object(gateway.time, "time", side_effect=[0, 0, 2]), \
                patch.object(gateway.time, "monotonic", side_effect=[0, 0, 2]), \
                self.assertRaises(TimeoutError):
            gw.request("identity", timeout=1)

    def test_completed_empty_body_is_valid(self):
        gw = gateway.Gateway.__new__(gateway.Gateway)
        gw._open = MagicMock(return_value=7)
        gw._read_frame = MagicMock(return_value=response(b"", True))
        self.assertEqual(gw.request("identity"), b"")

    def test_completed_chunked_body_is_valid(self):
        gw = gateway.Gateway.__new__(gateway.Gateway)
        gw._open = MagicMock(return_value=7)
        last = gateway.ServiceFrame(stream_id=7)
        last.body_chunk.data = b"second"
        last.body_chunk.end_body = True
        gw._read_frame = MagicMock(side_effect=[response(b"first", False), last])
        self.assertEqual(gw.request("identity"), b"firstsecond")

    def make_gateway(self):
        gw = gateway.Gateway.__new__(gateway.Gateway)
        gw._recv_lock = threading.Lock()
        gw._chunks = {}
        gw.ws = MagicMock()
        gw.noise = MagicMock()
        gw.noise.decrypt.side_effect = lambda data: data
        return gw

    def test_socket_wait_receives_the_remaining_deadline(self):
        gw = self.make_gateway()
        frame = gateway.NoiseTransportFrame(total_chunks=1, payload=gateway.ServiceResponse(
            payload=response(b"done", True).SerializeToString()).SerializeToString())
        gw.ws.recv.return_value = (frame.SerializeToString(), 0)
        with patch.object(gateway.time, "monotonic", return_value=10):
            self.assertEqual(gw._read_frame(deadline=11).response.body, b"done")
        self.assertEqual(gw.ws.recv.call_args.kwargs, {"timeout": 1})

    def test_socket_timeout_becomes_cli_timeout(self):
        gw = self.make_gateway()
        gw.ws.recv.side_effect = WebSocketTimeout("synthetic idle socket")
        with patch.object(gateway.time, "monotonic", return_value=10), \
                self.assertRaises(TimeoutError):
            gw._read_frame(deadline=11)

    def test_transport_chunks_do_not_restart_the_deadline(self):
        gw = self.make_gateway()
        frame = gateway.NoiseTransportFrame(chunk_id=1, chunk_index=0,
                                             total_chunks=2, payload=b"first")
        gw.ws.recv.return_value = (frame.SerializeToString(), 0)
        with patch.object(gateway.time, "monotonic", side_effect=[0, 2]), \
                self.assertRaises(TimeoutError):
            gw._read_frame(deadline=1)
        self.assertEqual(gw.ws.recv.call_count, 1)


class BrowserScopeTest(unittest.TestCase):
    def test_lookalike_urls_are_not_muse_tabs(self):
        for url in ("https://muse.ai.evil.invalid/", "https://notmuse.ai/",
                    "https://example.com/?next=muse.ai", "http://muse.ai/"):
            with self.subTest(url=url), patch.object(cli, "_browser_run", return_value={
                    "tabs": [{"id": "1", "url": url}]}), \
                    patch.object(cli.time, "sleep"):
                jar, error = cli._browser_cookies()
                self.assertIsNone(jar)
                self.assertEqual(error, "no muse.ai tab open in Chrome")

    def test_only_exact_host_cookies_are_returned(self):
        jar = [{"name": "hatch_sess", "value": "synthetic", "domain": d}
               for d in ("muse.ai", ".muse.ai", ".notmuse.ai", ".other.muse.ai")]
        with patch.object(cli, "_browser_run", side_effect=[
                {"tabs": [{"id": "1", "url": "https://muse.ai/"}]}, {}, {"cookies": jar}]):
            found, error = cli._browser_cookies()
        self.assertIsNone(error)
        self.assertEqual([c["domain"] for c in found], ["muse.ai", ".muse.ai"])

    def test_cookie_jar_loader_scopes_domains_and_keeps_httponly(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as fh:
            fh.write("# Netscape HTTP Cookie File\n")
            fh.write("#HttpOnly_.muse.ai\tTRUE\t/\tTRUE\t0\thatch_sess\tsynthetic\n")
            fh.write(".other.muse.ai\tTRUE\t/\tTRUE\t0\twrong_host\tother\n")
            fh.write(".notmuse.ai\tTRUE\t/\tTRUE\t0\tlookalike\tother\n")
            fh.flush()
            self.assertEqual(gateway.load_cookies(fh.name), "hatch_sess=synthetic")

    def test_cookie_jar_accepts_browser_session_expiry_and_skips_expired_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as fh:
            fh.write(".muse.ai\tTRUE\t/\tTRUE\t-1\thatch_sess\tsynthetic\n")
            fh.write(".muse.ai\tTRUE\t/\tTRUE\t1\texpired\told\n")
            fh.flush()
            self.assertEqual(gateway.load_cookies(fh.name), "hatch_sess=synthetic")


class CookiePersistenceTest(unittest.TestCase):
    def export(self, root, jar):
        path = os.path.join(root, "cookies.txt")
        with patch.object(cli, "CONFIG_DIR", root), patch.object(cli, "COOKIES_FILE", path), \
                patch.object(cli.browsers, "supported", return_value=False), \
                patch.object(cli.shutil, "which", return_value="agent-browser"), \
                patch.object(cli, "_browser_cookies", return_value=(jar, None)), \
                patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            cli.cmd_auth_export(argparse.Namespace(browser="agent-browser"))

    def test_backup_is_private_even_if_the_original_was_not(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "cookies.txt")
            with open(path, "w") as fh:
                fh.write("hatch_sess=synthetic-old")
            os.chmod(path, 0o644)
            self.export(root, [{"name": "hatch_sess", "value": "synthetic-new", "domain": ".muse.ai"}])
            for name in (path, path + ".bak"):
                self.assertEqual(stat.S_IMODE(os.stat(name).st_mode), 0o600)

    def test_foreign_session_does_not_replace_an_existing_login(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "cookies.txt")
            with open(path, "w") as fh:
                fh.write("hatch_sess=synthetic-old")
            with self.assertRaises(SystemExit):
                self.export(root, [{"name": "hatch_sess", "value": "wrong", "domain": ".notmuse.ai"}])
            with open(path) as fh:
                self.assertEqual(fh.read(), "hatch_sess=synthetic-old")


class ForkUpdateTest(unittest.TestCase):
    def test_uv_update_keeps_the_fork_source(self):
        with patch.object(update.sys, "prefix", "/example/uv/tools/muse-cli"):
            argv = update.upgrade_argv()
        self.assertEqual(argv[:5], ["uv", "tool", "install", "--force", "--upgrade"])
        self.assertEqual(argv[-1], "git+https://github.com/Kahtaf/muse-cli.git@fix/agent-reliability")

    def test_fork_does_not_offer_pypi_update_notices(self):
        with patch.object(update.sys.stdout, "isatty", return_value=True), \
                patch.object(update.sys.stderr, "isatty", return_value=True):
            self.assertFalse(update._should_check())


if __name__ == "__main__":
    unittest.main()
