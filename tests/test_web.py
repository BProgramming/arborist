"""fetch_url / crawl_docs against a small local docs site (no internet needed)."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import web
from conftest import FakeApprover, write
from tools import Decision

LONG = "\n".join(f"Paragraph {i}: " + "words " * 30 for i in range(200))

PAGES = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /docs/secret.html\n"),
    "/docs/index.html": ("text/html", """<!doctype html><html><head><title>Demo Docs</title>
        <script>var tracking = "SCRIPT_TEXT";</script><style>.x{color:red}</style></head>
        <body><nav>NAV_TEXT <a href="/docs/a.html">A</a></nav>
        <main><h1>Welcome</h1><p>Start with the <a href="a.html">guide</a>,
        the <a href="b.html">redirected page</a> or <a href="sub/c.html#part">deep page</a>.</p>
        <p><a href="/other/x.html">other section</a> <a href="https://external.example/">external</a>
        <a href="mailto:me@example.com">mail</a> <a href="logo.png">logo</a>
        <a href="secret.html">secret</a> <a href="long.html">long</a></p></main>
        <footer>FOOTER_TEXT</footer></body></html>"""),
    "/docs/a.html": ("text/html", """<html><head><title>Guide A</title></head><body>
        <div role="navigation">SIDEBAR_TEXT</div>
        <article><h2>Installing</h2><p>Run the   installer   twice.</p>
        <pre>pip install demo
    indented line</pre>
        <ul><li>first item</li><li>second <code>inline_code()</code></li></ul>
        <table><tr><th>flag</th><th>meaning</th></tr><tr><td>-v</td><td>verbose output</td></tr></table>
        <p>The async context manager closes the pool.</p>
        <a href="sub/c.html">deep</a></article></body></html>"""),
    "/docs/b2.html": ("text/html", "<title>B after redirect</title><main><p>" + "B content. " * 30 + "</p></main>"),
    "/docs/sub/c.html": ("text/html", "<title>Deep C</title><p>" + "Deep content about pools. " * 20
                         + '</p><a href="d.html">deeper</a>'),
    "/docs/sub/d.html": ("text/html", "<title>Deeper D</title><p>too deep</p>"),
    "/docs/secret.html": ("text/html", "<title>Secret</title><p>robots says no</p>"),
    "/docs/long.html": ("text/plain", LONG),
    "/docs/data.bin": ("application/octet-stream", "\x00\x01"),
    "/other/x.html": ("text/html", "<title>Other</title><p>out of scope</p>"),
}
REDIRECTS = {"/docs/b.html": "/docs/b2.html"}


@pytest.fixture
def site():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append(self.path)
            if self.path in REDIRECTS:
                self.send_response(302)
                self.send_header("Location", REDIRECTS[self.path])
                self.end_headers()
                return
            if self.path == "/docs/to-localhost":
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/docs/a.html")
                self.end_headers()
                return
            if self.path not in PAGES:
                self.send_error(404)
                return
            ctype, body = PAGES[self.path]
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    yield base, hits
    server.shutdown()


@pytest.fixture(autouse=True)
def no_crawl_delay(monkeypatch):
    monkeypatch.setattr(web, "CRAWL_DELAY", 0)


@pytest.fixture
def web_toolbox(make_toolbox):
    def make(approver=None, **limits):
        return make_toolbox(approver, web_allow_private=True, **limits)
    return make


# ------------------------------------------------------------ html to text ---

def test_html_to_text_keeps_content_and_drops_chrome():
    _, html = PAGES["/docs/a.html"]
    page = web.html_to_page(html, "http://h/docs/a.html")
    t = page.text
    assert page.title == "Guide A"
    assert "## Installing" in t and "Run the installer twice." in t
    assert "```\npip install demo\n    indented line\n```" in t  # <pre> kept verbatim
    assert "- first item" in t and "`inline_code()`" in t
    assert "| -v | verbose output" in t
    assert "SIDEBAR_TEXT" not in t
    assert page.links == ["http://h/docs/sub/c.html"]


def test_tables_and_wrapped_list_items_stay_on_one_line():
    html = """<main><table><tr><td><div><p>name</p></div></td><td><p>value one</p></td></tr>
        <tr><td></td><td></td></tr></table>
        <ul><li><p>wrapped item</p></li></ul>""" + "<p>filler text.</p>" * 20 + "</main>"
    text = web.html_to_page(html, "http://h/").text
    assert "| name | value one" in text
    assert "- wrapped item" in text
    assert not any(line.strip() in ("|", "| |", "-") for line in text.splitlines())


def test_main_column_is_preferred_and_scripts_dropped():
    _, html = PAGES["/docs/index.html"]
    page = web.html_to_page(html, "http://h/docs/index.html")
    for junk in ("SCRIPT_TEXT", "NAV_TEXT", "FOOTER_TEXT", "color:red"):
        assert junk not in page.text
    assert "# Welcome" in page.text
    # Links resolved, fragments dropped, non-http skipped, nav links excluded.
    assert "http://h/docs/sub/c.html" in page.links
    assert not any(l.startswith("mailto:") or "#" in l for l in page.links)


def test_normalize_and_scope():
    assert web.normalize("../b.html#x", "https://Docs.Example.com/a/c.html") == "https://docs.example.com/b.html"
    assert web.normalize("javascript:alert(1)") is None
    assert web.scope_prefix("https://h/docs/guide/intro.html") == "/docs/guide/"
    assert web.scope_prefix("https://h/docs/") == "/docs/"
    start = "https://h/docs/index.html"
    assert web.in_scope("https://h/docs/sub/x.html", start, "/docs/")
    assert not web.in_scope("https://h/other/x.html", start, "/docs/")
    assert not web.in_scope("https://h2/docs/x.html", start, "/docs/")
    assert not web.in_scope("https://h/docs/logo.png", start, "/docs/")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.1.2.3", "192.168.0.10",
                                  "169.254.169.254", "::1", "::ffff:127.0.0.1", "0.0.0.0"])
def test_private_addresses_are_refused(host):
    with pytest.raises(web.WebError, match="private or local"):
        web.check_address(host, 80, allow_private=False)


def test_public_address_is_allowed():
    web.check_address("8.8.8.8", 443, allow_private=False)  # literal: no DNS needed


def test_search_ranks_lines_by_matching_words():
    pages = [web.Page("u1", "t", "nothing here\nthe async pool closes\nasync only"),
             web.Page("u2", "t", "pool async context manager closes")]
    hits = web.search_pages(pages, "async context manager", limit=5)
    assert hits[0] == "u2: pool async context manager closes"
    assert all("nothing" not in h for h in hits)


# --------------------------------------------------------------- fetch_url ---

def test_every_new_page_asks_and_shows_the_url_and_reason(site, web_toolbox):
    base, hits = site
    approver = FakeApprover()
    tb = web_toolbox(approver)
    out = tb.call("fetch_url", {"url": f"{base}/docs/a.html", "reason": "install steps"}).content
    assert out.startswith("Page: Guide A")
    assert "not instructions" in out and "## Installing" in out
    assert "Links on this page" in out and "/docs/sub/c.html" in out
    tb.call("fetch_url", {"url": f"{base}/docs/index.html"})
    # Same host, second page: asked again, no blanket approval for the site.
    assert [u for _, u in approver.fetch_requests] == [f"{base}/docs/a.html", f"{base}/docs/index.html"]
    assert approver.fetch_details == [("install steps", ""), ("", "")]


def test_reading_more_of_a_page_already_fetched_does_not_ask(site, web_toolbox):
    base, hits = site
    approver = FakeApprover()
    tb = web_toolbox(approver, web_page_chars=200)
    tb.call("fetch_url", {"url": f"{base}/docs/a.html"})
    before = len(hits)
    out = tb.call("fetch_url", {"url": f"{base}/docs/a.html", "offset": 100}).content
    assert "Guide A" in out and len(approver.fetch_requests) == 1 and len(hits) == before


def test_declining_one_page_does_not_block_the_next(site, web_toolbox):
    base, _ = site
    approver = FakeApprover(fetch=[Decision(False), Decision(True)])
    tb = web_toolbox(approver)
    assert "declined" in tb.call("fetch_url", {"url": f"{base}/docs/a.html"}).content
    assert tb.call("fetch_url", {"url": f"{base}/docs/index.html"}).content.startswith("Page:")


def test_declined_host_is_never_contacted(site, web_toolbox):
    base, hits = site
    tb = web_toolbox(FakeApprover(fetch=[Decision(False, "not that site")]))
    out = tb.call("fetch_url", {"url": f"{base}/docs/a.html"}).content
    assert out.startswith("error:") and "declined" in out and "not that site" in out
    assert hits == []
    assert not tb.web_tainted


def test_preapproved_hosts_skip_the_prompt(site, web_toolbox):
    base, _ = site
    approver = FakeApprover()
    tb = web_toolbox(approver, web_allowed_hosts=["127.0.0.1"])
    assert tb.call("fetch_url", {"url": f"{base}/docs/a.html"}).content.startswith("Page:")
    assert approver.fetch_requests == []


def test_private_hosts_are_refused_by_default(site, make_toolbox):
    base, hits = site
    out = make_toolbox().call("fetch_url", {"url": f"{base}/docs/a.html"}).content
    assert "private or local address" in out and hits == []


def test_redirect_to_a_new_host_needs_approval(site, web_toolbox):
    base, _ = site
    approver = FakeApprover(fetch=[Decision(True), Decision(False)])  # 127.0.0.1 yes, localhost no
    out = web_toolbox(approver).call("fetch_url", {"url": f"{base}/docs/to-localhost"}).content
    assert [h for h, _ in approver.fetch_requests] == ["127.0.0.1", "localhost"]
    assert "declined to let you visit" in out and "localhost" in out


def test_same_host_redirect_is_followed(site, web_toolbox):
    base, _ = site
    out = web_toolbox().call("fetch_url", {"url": f"{base}/docs/b.html"}).content
    assert "Page: B after redirect" in out and f"URL: {base}/docs/b2.html" in out


def test_long_pages_come_in_parts(site, web_toolbox):
    base, hits = site
    tb = web_toolbox(web_page_chars=3000)
    first = tb.call("fetch_url", {"url": f"{base}/docs/long.html"}).content
    assert "Paragraph 0:" in first and "continue with offset=" in first
    offset = int(first.rsplit("offset=", 1)[1].split("]")[0])
    second = tb.call("fetch_url", {"url": f"{base}/docs/long.html", "offset": offset}).content
    assert second.split("\n\n", 1)[1].lstrip("\n").startswith("Paragraph")  # cut on a line break
    assert hits.count("/docs/long.html") == 1  # served from the cache
    assert "past the end" in tb.call("fetch_url", {"url": f"{base}/docs/long.html", "offset": 10**7}).content


@pytest.mark.parametrize("path, message", [
    ("/docs/data.bin", "not a text page"),
    ("/docs/missing.html", "HTTP 404"),
])
def test_fetch_errors_are_reported(site, web_toolbox, path, message):
    base, _ = site
    assert message in web_toolbox().call("fetch_url", {"url": base + path}).content


def test_non_http_urls_are_rejected(web_toolbox):
    tb = web_toolbox()
    for url in ("file:///etc/passwd", "ftp://example.com/x", "not a url"):
        assert "not an http(s) URL" in tb.call("fetch_url", {"url": url}).content


# -------------------------------------------------------------- crawl_docs ---

def test_crawl_stays_in_scope_obeys_robots_and_caches(site, web_toolbox):
    base, hits = site
    tb = web_toolbox()
    out = tb.call("crawl_docs", {"url": f"{base}/docs/index.html", "query": "async context manager"}).content
    assert "Crawled" in out and "not instructions" in out
    assert "Guide A" in out and "B after redirect" in out and "Deep C" in out and "long.html" in out
    assert "/other/" not in out and "external.example" not in out
    assert "robots.txt disallows" in out and "/docs/secret.html" not in hits
    assert "Deeper D" in out  # index -> sub/c.html -> sub/d.html: depth 2, the default cap
    assert "Best matches" in out and "async context manager closes the pool" in out
    before = len(hits)
    assert tb.call("fetch_url", {"url": f"{base}/docs/a.html"}).content.startswith("Page: Guide A")
    assert len(hits) == before  # cached
    assert any(line.startswith("fetched 1/") for line in tb.output)  # progress shown


def test_a_crawl_is_one_approval_showing_its_scope(site, web_toolbox):
    base, hits = site
    approver = FakeApprover()
    tb = web_toolbox(approver)
    out = tb.call("crawl_docs", {"url": f"{base}/docs/index.html", "reason": "find the pool API"}).content
    assert "Crawled" in out and len(hits) > 2
    assert approver.fetch_requests == [("127.0.0.1", f"{base}/docs/index.html")]
    reason, scope = approver.fetch_details[0]
    assert reason == "find the pool API" and f"under {base}/docs/" in scope
    # The approval ends with the call: a new page afterwards asks again.
    tb.call("fetch_url", {"url": f"{base}/other/page.html"})
    assert len(approver.fetch_requests) == 2


def test_declined_crawl_contacts_nothing(site, web_toolbox):
    base, hits = site
    out = web_toolbox(FakeApprover(fetch=[Decision(False)])).call(
        "crawl_docs", {"url": f"{base}/docs/index.html"}).content
    assert "declined" in out and hits == []


def test_crawl_caps_are_enforced(site, web_toolbox):
    base, _ = site
    tb = web_toolbox(web_max_pages=2, web_max_depth=1)
    out = tb.call("crawl_docs", {"url": f"{base}/docs/index.html", "max_pages": 50, "max_depth": 9}).content
    assert out.count(" chars)") == 2
    assert "stopped at 2 pages" in out


def test_crawl_depth_limit(site, web_toolbox):
    base, _ = site
    out = web_toolbox().call("crawl_docs", {"url": f"{base}/docs/index.html", "max_depth": 1}).content
    assert "Deep C" in out and "Deeper D" not in out


def test_crawl_of_a_broken_start_page_is_an_error(site, web_toolbox):
    base, _ = site
    assert "HTTP 404" in web_toolbox().call("crawl_docs", {"url": f"{base}/docs/nope.html"}).content


# ----------------------------------------------------- web content and edits ---

def test_web_content_makes_auto_edits_ask_again(site, ws_root, web_toolbox):
    base, _ = site
    write(ws_root, "a.py", "x = 1\n")
    approver = FakeApprover()
    tb = web_toolbox(approver, approve_edits="auto")
    tb.call("read_file", {"path": "a.py"})
    tb.call("edit_file", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert approver.change_requests == []  # auto mode: no prompt yet

    tb.call("fetch_url", {"url": f"{base}/docs/a.html"})
    assert tb.notices and "edits will ask" in tb.notices[0]
    tb.call("edit_file", {"path": "a.py", "old_string": "x = 2", "new_string": "x = 3"})
    assert len(approver.change_requests) == 1

    tb.reset()  # /clear
    tb.call("read_file", {"path": "a.py"})
    tb.call("edit_file", {"path": "a.py", "old_string": "x = 3", "new_string": "x = 4"})
    assert len(approver.change_requests) == 1


def test_always_answer_does_not_survive_web_content(site, ws_root, web_toolbox):
    base, _ = site
    write(ws_root, "a.py", "n = 0\n")
    approver = FakeApprover(change=[Decision(True, always=True)])
    tb = web_toolbox(approver)
    tb.call("fetch_url", {"url": f"{base}/docs/a.html"})
    tb.call("read_file", {"path": "a.py"})
    for i in range(2):
        tb.call("edit_file", {"path": "a.py", "old_string": f"n = {i}", "new_string": f"n = {i + 1}"})
    assert len(approver.change_requests) == 2


def test_web_tools_can_be_disabled(web_toolbox):
    tb = web_toolbox(web_enabled=False)
    names = {t["function"]["name"] for t in tb.schemas}
    assert "fetch_url" not in names and "crawl_docs" not in names and "read_file" in names
    assert "unknown tool" in tb.call("fetch_url", {"url": "https://example.com"}).content
