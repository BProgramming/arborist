"""Fetch web pages and crawl documentation sites for the model.

Standard library only. Pages are reduced to plain text (headings, lists and
code blocks kept; scripts, styles and site navigation dropped) and cached for
the session, so the model can page through them without re-fetching.

Safety, since this hands outside text to a model that can edit your files:
  * http/https only; hosts that resolve to private, loopback or link-local
    addresses are refused (so a page can't point it at your router, your
    LAN, or the llama-server itself). The check happens again on redirects.
  * Every request is checked with the Toolbox before it's sent, so each visit
    can need its own approval (see Toolbox._allow).
  * Crawls stay on one host under one path prefix, obey robots.txt, pause
    between requests and stop at a page cap.
"""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

USER_AGENT = "arborist/0.1 (local coding assistant; reads documentation)"
MAX_BYTES = 3 * 1024 * 1024
CRAWL_DELAY = 0.3  # seconds between crawl requests; be polite

TEXT_TYPES = {"text/html", "application/xhtml+xml", "text/plain", "text/markdown",
              "text/x-markdown", "text/x-rst", "application/json", "text/xml", "application/xml"}
# Links to these are never worth crawling.
SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".pdf", ".zip", ".gz",
    ".tgz", ".bz2", ".xz", ".7z", ".tar", ".whl", ".exe", ".dmg", ".msi", ".deb", ".rpm",
    ".mp3", ".mp4", ".webm", ".mov", ".woff", ".woff2", ".ttf", ".eot", ".css", ".js", ".map",
)


class WebError(Exception):
    """A fetch problem the model can be told about."""


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- urls ---

def normalize(url: str, base: str | None = None) -> str | None:
    """Absolute http(s) URL without its #fragment, or None if it isn't one."""
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url.strip())
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def scope_prefix(url: str) -> str:
    """The directory a crawl is limited to: /docs/guide/intro.html -> /docs/guide/."""
    path = urlsplit(url).path or "/"
    return path if path.endswith("/") else path.rsplit("/", 1)[0] + "/"


def in_scope(url: str, start: str, prefix: str) -> bool:
    a, b = urlsplit(url), urlsplit(start)
    return (a.netloc == b.netloc and (a.path or "/").startswith(prefix)
            and not a.path.lower().endswith(SKIP_EXTENSIONS))


def check_address(host: str, port: int | None, allow_private: bool) -> None:
    """Refuse hosts that resolve to anything but public addresses.

    (The connection resolves the name again, so a DNS server that changes its
    answer in between could slip past this. That's an accepted limitation.)
    """
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise WebError(f"can't resolve {host}: {e.strerror or e}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise WebError(f"{host} resolves to a private or local address ({ip}); "
                           "only public websites can be fetched")


# ---------------------------------------------------------- html to text ---

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer",
            "aside", "form", "button", "iframe", "select", "head", "object", "canvas"}
    BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "dl",
             "dt", "dd", "blockquote", "hr", "figure", "figcaption", "details", "summary"}
    VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed",
            "source", "track", "wbr"}
    MAIN = {"main", "article"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.title = ""
        self.links: list[str] = []
        self._seen_links: set[str] = set()
        self._stack: list[tuple[str, bool, bool]] = []  # (tag, skips, is_main)
        self._skip = 0
        self._in_title = False
        self._pre = 0
        self._main_depth = 0
        self._cell = 0  # inside <td>/<th>: block breaks become spaces, keeping rows on one line
        self.all: list[str] = []
        self.main: list[str] = []

    def _emit(self, s: str) -> None:
        if self._skip:
            return
        self.all.append(s)
        if self._main_depth:
            self.main.append(s)

    def _newline(self, n: int = 1) -> None:
        self._emit(" " if self._cell and not self._pre else "\n" * n)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title" and not self.title:
            self._in_title = True
        if tag == "base" and a.get("href"):
            self.base = urljoin(self.base, a["href"])
        if tag == "a" and a.get("href") and not self._skip:
            link = normalize(a["href"], self.base)
            if link and link not in self._seen_links:
                self._seen_links.add(link)
                self.links.append(link)
        if tag in self.VOID:
            if tag in ("br", "hr"):
                self._newline()
            return
        # Pages sometimes mark the main column with a role instead of <main>.
        skips = tag in self.SKIP or a.get("role") in ("navigation", "banner", "contentinfo") \
            or a.get("aria-hidden") == "true"
        is_main = tag in self.MAIN or a.get("role") == "main"
        self._stack.append((tag, skips, is_main))
        if skips:
            self._skip += 1
        if is_main:
            self._main_depth += 1
        if re.fullmatch(r"h[1-6]", tag):
            self._newline(2)
            self._emit("#" * int(tag[1]) + " ")
        elif tag == "li":
            self._newline()
            self._emit("- ")
        elif tag == "pre":
            self._pre += 1
            self._newline(2)
            self._emit("```\n")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag in ("td", "th"):
            self._emit(" | ")
            self._cell += 1
        elif tag in self.BLOCK:
            self._newline()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        # Close back to the matching open tag; HTML often leaves tags unclosed.
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                for t, skips, is_main in reversed(self._stack[i:]):
                    self._close(t, skips, is_main)
                del self._stack[i:]
                break

    def _close(self, tag, skips, is_main):
        if re.fullmatch(r"h[1-6]", tag):
            self._newline()
        elif tag == "pre":
            self._emit("\n```")
            self._newline(2)
            self._pre = max(0, self._pre - 1)
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag in ("td", "th"):
            self._cell = max(0, self._cell - 1)
        elif tag in self.BLOCK or tag == "li":
            self._newline()
        if skips:
            self._skip -= 1
        if is_main:
            self._main_depth -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._pre:
            self._emit(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))


def _tidy(chunks: list[str]) -> str:
    out, in_code = [], False
    for line in "".join(chunks).split("\n"):
        if line.strip() == "```":
            in_code = not in_code
            out.append("```")
        elif in_code:
            out.append(line.rstrip())
        else:
            line = re.sub(r" {2,}", " ", line).strip()
            if "|" in line and not line.strip("| "):
                continue  # a table row with nothing in it
            out.append(line)
    text = "\n".join(out)
    # "- " then a paragraph on the next line (<li><p>…) -> one bullet line.
    text = re.sub(r"^-\n+(?=\S)", "- ", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_page(html: str, url: str) -> Page:
    p = _TextExtractor(url)
    p.feed(html)
    p.close()
    main, everything = _tidy(p.main), _tidy(p.all)
    # Prefer the main column when the page marks one and it has real content.
    text = main if len(main) >= 200 else everything
    return Page(url=url, title=re.sub(r"\s+", " ", p.title).strip() or url, text=text, links=p.links)


# --------------------------------------------------------------- fetching ---

class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, check: Callable[[str], None]):
        super().__init__()
        self.check = check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = normalize(newurl, req.full_url)
        if url is None:
            raise WebError(f"redirected to an unsupported URL: {newurl}")
        self.check(url)
        return super().redirect_request(req, fp, code, msg, headers, url)


class Fetcher:
    """Fetches and caches pages. `allow` is called with every URL before it's
    requested (redirect targets and robots.txt included) and raises WebError
    to stop it. Cached pages aren't requested again, so they aren't checked."""

    def __init__(self, allow: Callable[[str], None], timeout: float = 15,
                 allow_private: bool = False, max_bytes: int = MAX_BYTES):
        self.allow = allow
        self.timeout = timeout
        self.allow_private = allow_private
        self.max_bytes = max_bytes
        self.cache: dict[str, Page] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _check(self, url: str) -> None:
        parts = urlsplit(url)
        self.allow(url)
        check_address(parts.hostname, parts.port, self.allow_private)

    def _open(self, url: str):
        self._check(url)
        opener = urllib.request.build_opener(_CheckedRedirects(self._check))
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        })
        try:
            return opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise WebError(f"HTTP {e.code} {e.reason} for {url}") from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, WebError):
                raise e.reason from None
            raise WebError(f"couldn't fetch {url}: {e.reason}") from None
        except (TimeoutError, socket.timeout):
            raise WebError(f"timed out after {self.timeout:g}s fetching {url}") from None
        except (ValueError, http.client.HTTPException) as e:
            raise WebError(f"couldn't fetch {url}: {e}") from None

    def fetch(self, url: str) -> Page:
        norm = normalize(url)
        if norm is None:
            raise WebError(f"not an http(s) URL: {url}")
        if norm in self.cache:
            return self.cache[norm]
        try:
            with self._open(norm) as resp:
                final = normalize(resp.geturl()) or norm
                ctype = resp.headers.get_content_type()
                charset = resp.headers.get_content_charset() or "utf-8"
                if ctype not in TEXT_TYPES and not ctype.startswith("text/"):
                    raise WebError(f"{final} is {ctype}, not a text page")
                data = resp.read(self.max_bytes + 1)
        except (TimeoutError, socket.timeout):
            raise WebError(f"timed out after {self.timeout:g}s reading {norm}") from None
        truncated = len(data) > self.max_bytes
        try:
            body = data[: self.max_bytes].decode(charset, errors="replace")
        except LookupError:
            body = data[: self.max_bytes].decode("utf-8", errors="replace")
        if ctype in ("text/html", "application/xhtml+xml"):
            page = html_to_page(body, final)
        else:
            page = Page(url=final, title=final, text=body.strip())
        if truncated:
            page.text += f"\n\n[page cut at {self.max_bytes // 1024 // 1024} MB]"
        self.cache[norm] = self.cache[final] = page
        return page

    def robots_allows(self, url: str) -> bool:
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        if key not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                with self._open(key + "/robots.txt") as resp:
                    rp.parse(resp.read(512 * 1024).decode("utf-8", errors="replace").splitlines())
                self._robots[key] = rp
            except WebError:
                self._robots[key] = None  # no robots.txt (or unreadable): allowed
        rp = self._robots[key]
        return rp is None or rp.can_fetch(USER_AGENT, url)

    def crawl(self, start: str, max_pages: int, max_depth: int,
              progress: Callable[[str], None] = lambda s: None) -> tuple[list[Page], list[str]]:
        """Breadth-first crawl of `start`'s directory on its host."""
        start = normalize(start)
        if start is None:
            raise WebError("not an http(s) URL")
        prefix = scope_prefix(start)
        pages, problems = [], []
        queue, seen = deque([(start, 0)]), {start}
        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            if not self.robots_allows(url):
                problems.append(f"robots.txt disallows {url}")
                continue
            cached = url in self.cache
            try:
                page = self.fetch(url)
            except WebError as e:
                if not pages and url == start:
                    raise  # the page they asked for failed: that's the answer
                problems.append(str(e))
                continue
            pages.append(page)
            progress(f"{len(pages)}/{max_pages}  {page.url}")
            if depth < max_depth:
                for link in page.links:
                    if link not in seen and in_scope(link, start, prefix):
                        seen.add(link)
                        queue.append((link, depth + 1))
            if queue and not cached:
                time.sleep(CRAWL_DELAY)
        if queue:
            problems.append(f"stopped at {max_pages} pages; {len(queue)} more links were in scope")
        return pages, problems


# ---------------------------------------------------------------- search ---

def search_pages(pages: list[Page], query: str, limit: int) -> list[str]:
    """Lines that best match the query's words, best first, as 'url: line'."""
    terms = {t for t in re.findall(r"[\w.]+", query.lower()) if len(t) >= 2}
    if not terms:
        return []
    need = 1 if len(terms) <= 2 else (len(terms) + 1) // 2
    scored = []
    for page_no, page in enumerate(pages):
        for line_no, line in enumerate(page.text.splitlines()):
            low = line.lower()
            score = sum(1 for t in terms if t in low)
            if score >= need and line.strip():
                shown = line.strip() if len(line) <= 240 else line.strip()[:240] + "…"
                scored.append((-score, page_no, line_no, f"{page.url}: {shown}"))
    scored.sort()
    return [s[3] for s in scored[:limit]]
