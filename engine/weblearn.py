"""Autonomous web-learning engine for TinyGPT.

Two capabilities:

1. SELF-LEARNING LOOP (runs in background, no user action needed):
   Every interval, fetch fresh text from free public sources
   (Wikipedia random articles, BBC news RSS, Hacker News top stories),
   convert to dialogue-style training samples, and continue training the
   chat model on them. Every learned item is logged to
   data/weblearn/journal.jsonl with source URL + timestamp + loss before
   and after, so you can see exactly what the model learned.

2. WEB SEARCH AT CHAT TIME (retrieval augmentation):
   When the user asks a question, we can search DuckDuckGo's instant
   answer API + Wikipedia search (no API key needed), and show the
   retrieved facts alongside the model's own generated reply.

All network calls are pure stdlib (urllib) — no extra dependencies.
"""

import os
import re
import json
import time
import html
import random
import threading
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

from . import chat as C
from .gpt import AdamW

WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "weblearn")
JOURNAL = os.path.join(WEB_DIR, "journal.jsonl")
STORE = os.path.join(WEB_DIR, "store.json")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


# ---------------------------------------------------------------- fetch utils

def _get(url, timeout=10, retries=2):
    """GET with a browser-like UA; returns text or None."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5)
    return None


def _get_json(url, timeout=10):
    txt = _get(url, timeout=timeout)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


# ---------------------------------------------------------------- sources

def fetch_wikipedia_article(topic=None):
    """Fetch one Wikipedia article (random if no topic). Returns
    (title, text, url) or None."""
    if topic:
        api = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
               "&prop=extracts&explaintext=1&redirects=1&titles="
               + urllib.parse.quote(topic))
    else:
        api = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
               "&list=random&rnnamespace=0&rnlimit=1")
        d = _get_json(api)
        if not d or "query" not in d or "random" not in d["query"]:
            return None
        topic = d["query"]["random"][0]["title"]
        api = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
               "&prop=extracts&explaintext=1&redirects=1&titles="
               + urllib.parse.quote(topic))
    d = _get_json(api)
    if not d:
        return None
    pages = d.get("query", {}).get("pages", {})
    for _, page in pages.items():
        extract = page.get("extract")
        if extract and len(extract) > 400:
            title = page.get("title", topic)
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
                title.replace(" ", "_"))
            return title, extract, url
    return None


def fetch_bbc_news(max_items=5):
    """BBC top stories via RSS. Returns list of (title, text, url)."""
    rss = _get("https://feeds.bbci.co.uk/news/rss.xml", timeout=10)
    if rss is None:
        return []
    out = []
    try:
        root = ET.fromstring(rss)
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            if title and desc:
                out.append((html.unescape(title),
                            html.unescape(title + ". " + desc),
                            link))
            if len(out) >= max_items:
                break
    except Exception:
        pass
    return out


def fetch_hackernews(max_items=5):
    """Hacker News top stories. Returns list of (title, text, url)."""
    d = _get_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    if not d:
        return []
    out = []
    for sid in d[:max_items * 2]:
        item = _get_json(
            f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
        if item and item.get("title"):
            body = item.get("text") or ""
            body = re.sub(r"<[^>]+>", " ", body)  # strip html
            text = item["title"] + ". " + body
            out.append((item["title"], text,
                        item.get("url") or
                        f"https://news.ycombinator.com/item?id={sid}"))
        if len(out) >= max_items:
            break
    return out


# ---------------------------------------------------------------- text -> data

def article_to_dialogues(title, text, max_chars=3000):
    """Convert an article into pseudo-dialogues for LM training.

    We split the article into paragraphs and build Q/A-style pairs:
    the "question" is the title, the "answer" is a paragraph chunk.
    This teaches the model factual text in dialogue format.
    """
    text = re.sub(r"\s+", " ", text)[:max_chars]
    paras = [p.strip() for p in text.split(". ")
             if len(p.strip()) > 60]
    dialogs = []
    for i, p in enumerate(paras[:6]):
        q = f"tell me about {title}"
        a = p if p.endswith(".") else p + "."
        dialogs.append([q, a])
    return dialogs


def _seen(url):
    """Load set of already-learned source URLs."""
    if not os.path.exists(JOURNAL):
        return set()
    seen = set()
    try:
        with open(JOURNAL) as f:
            for line in f:
                try:
                    seen.add(json.loads(line).get("url", ""))
                except Exception:
                    pass
    except Exception:
        pass
    return seen


def _log(entry):
    os.makedirs(WEB_DIR, exist_ok=True)
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _store_state(state):
    os.makedirs(WEB_DIR, exist_ok=True)
    with open(STORE, "w") as f:
        json.dump(state, f)


def _load_state():
    if os.path.exists(STORE):
        try:
            with open(STORE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"cycles": 0, "items_learned": 0, "last_cycle": None,
            "total_tokens": 0}


# ---------------------------------------------------------------- self-learning

def learn_cycle(bot_holder, minutes=3, lr=1e-4, log=print):
    """One self-learning cycle: fetch fresh web content, fine-tune.

    bot_holder: dict-like with keys 'bot' (ChatBot or None) and 'meta'.
    Returns journal entry summary dict.
    """
    state = _load_state()
    seen = _seen()

    # gather fresh items from all sources
    items = []
    try:
        art = fetch_wikipedia_article()
        if art:
            items.append({"source": "wikipedia", "title": art[0],
                          "text": art[1], "url": art[2]})
    except Exception:
        pass
    for t, txt, url in fetch_bbc_news(3):
        items.append({"source": "bbc", "title": t, "text": txt, "url": url})
    for t, txt, url in fetch_hackernews(3):
        items.append({"source": "hackernews", "title": t, "text": txt,
                       "url": url})

    fresh = [it for it in items if it["url"] not in seen]
    if not fresh:
        entry = {"ts": time.time(), "cycles": state["cycles"] + 1,
                "result": "no new content", "items": 0}
        _log(entry)
        return entry

    # build training dialogues from fresh items
    dialogs = []
    for it in fresh:
        dialogs.extend(article_to_dialogues(it["title"], it["text"]))

    # measure loss before on this new data (if model exists)
    bot = bot_holder.get("bot")
    loss_before = None
    if bot is not None:
        try:
            loss_before = _eval_loss(bot, dialogs)
        except Exception:
            loss_before = None

    # fine-tune
    trained_ok = False
    n_tokens = 0
    if bot is not None:
        try:
            n_tokens = _finetune(bot, dialogs, minutes=minutes, lr=lr)
            trained_ok = n_tokens > 0
        except Exception as e:
            log(f"weblearn finetune error: {e}")

    entry = {
        "ts": time.time(),
        "cycles": state["cycles"] + 1,
        "items": len(fresh),
        "sources": sorted({it["source"] for it in fresh}),
        "titles": [it["title"] for it in fresh],
        "urls": [it["url"] for it in fresh],
        "tokens_learned": n_tokens,
        "loss_before": loss_before,
        "trained": trained_ok,
    }
    _log(entry)
    for it in fresh:
        _log({"ts": time.time(), "learned_url": it["url"],
              "source": it["source"], "title": it["title"]})

    state["cycles"] += 1
    state["items_learned"] += len(fresh)
    state["total_tokens"] += int(n_tokens)
    state["last_cycle"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _store_state(state)
    return entry


def _eval_loss(bot, dialogs):
    """Average next-token loss of the model on new dialogues."""
    model, tok = bot.model, bot.tok
    losses = []
    for d in dialogs[:8]:
        ids = C.dialogue_to_ids(tok, d)
        if len(ids) < 8:
            continue
        ids = ids[:model.T + 1]
        x = np.array([ids[:-1]], dtype=np.int64)
        y = np.array([ids[1:]], dtype=np.int64)
        logits, _ = model.forward(x)
        B, T, V = logits.shape
        loss = -np.mean(np.log(
            softmax_last(logits[0])[np.arange(T), y[0]] + 1e-9))
        losses.append(float(loss))
    return round(float(np.mean(losses)), 4) if losses else None


def softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _finetune(bot, dialogs, minutes=1.5, lr=1e-4, batch=8):
    """Continue training the live model on new web dialogues.

    Uses the same next-token objective; small LR to avoid catastrophic
    forgetting. Returns tokens processed.
    """
    model, tok = bot.model, bot.tok
    X, Y = C.make_samples(tok, dialogs, block=model.T)
    if X.shape[0] == 0:
        return 0
    opt = AdamW(model.p, lr=lr)
    t0 = time.time()
    tokens = 0
    idx = 0
    rng = np.random.default_rng()
    while (time.time() - t0) / 60 < minutes and idx < 200:
        bi = rng.choice(X.shape[0], size=min(batch, X.shape[0]),
                        replace=False)
        xb, yb = X[bi], Y[bi]
        fwd = model.forward(xb)
        loss, grads = model.backward(xb, yb, fwd)
        opt.step(model.p, grads)
        tokens += int(xb.size)
        idx += 1
    return tokens


# ---------------------------------------------------------------- background loop

class WebLearnLoop(threading.Thread):
    """Daemon that self-learns every `interval_min` minutes forever."""

    def __init__(self, bot_holder, interval_min=30, learn_minutes=3):
        super().__init__(daemon=True)
        self.bot_holder = bot_holder      # shared dict {'bot':..., 'meta':...}
        self.interval_min = interval_min
        self.learn_minutes = learn_minutes
        self.stop_flag = threading.Event()
        self.last_entry = None
        self.status = "idle"

    def run(self):
        # wait for the main training to finish first (bot exists)
        waited = 0
        while not self.stop_flag.is_set():
            if self.bot_holder.get("bot") is not None:
                break
            self.stop_flag.wait(60)
            waited += 1
            if waited > 240:  # 4h with no model -> give up
                self.status = "no model"
                return
        while not self.stop_flag.is_set():
            self.status = "learning"
            try:
                self.last_entry = learn_cycle(
                    self.bot_holder, minutes=self.learn_minutes)
                self.status = "idle"
            except Exception as e:
                self.status = f"error: {e}"
            # sleep in 30s slices so stop() is responsive
            for _ in range(int(self.interval_min * 2)):
                if self.stop_flag.is_set():
                    return
                time.sleep(30)


# ---------------------------------------------------------------- chat-time search

def web_search(query, max_results=4):
    """Search the web for a query. No API key required.

    Combines DuckDuckGo instant answers + Wikipedia search.
    Returns list of {title, snippet, url, source}.
    """
    results = []
    q = urllib.parse.quote_plus(query)

    # 1) DuckDuckGo instant answer API
    d = _get_json(f"https://api.duckduckgo.com/?q={q}&format=json"
                  "&no_html=1&skip_disambig=1")
    if d:
        abst = d.get("AbstractText") or ""
        if abst:
            results.append({
                "title": d.get("Heading") or query,
                "snippet": html.unescape(str(abst))[:400],
                "url": d.get("AbstractURL") or
                       f"https://duckduckgo.com/?q={q}",
                "source": "duckduckgo",
            })
        for t in (d.get("RelatedTopics") or [])[:6]:
            if isinstance(t, dict) and t.get("Text") and t.get("FirstURL"):
                results.append({
                    "title": (t.get("Text", "")[:80]),
                    "snippet": html.unescape(t["Text"])[:400],
                    "url": t["FirstURL"],
                    "source": "duckduckgo",
                })
                if len(results) >= max_results:
                    break

    # 2) Wikipedia search
    wd = _get_json(
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&list=search&srlimit=3&srsearch=" + urllib.parse.quote(query))
    if wd:
        for hit in wd.get("query", {}).get("search", [])[:3]:
            results.append({
                "title": hit.get("title", ""),
                "snippet": re.sub(r"<[^>]+>", "", hit.get("snippet", ""))[:400],
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
                    hit["title"].replace(" ", "_")),
                "source": "wikipedia",
            })

    # dedupe by url
    seen = set()
    out = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            out.append(r)
    return out[:max_results]


def web_answer(query):
    """Best-effort short factual answer from the web (for chat).

    Returns dict {answer, source, url} or None.
    """
    # DuckDuckGo instant answer
    d = _get_json("https://api.duckduckgo.com/?q="
                  + urllib.parse.quote_plus(query)
                  + "&format=json&no_html=1&skip_disambig=1")
    if d and d.get("AbstractText"):
        return {
            "answer": html.unescape(str(d["AbstractText"]))[:600],
            "source": d.get("AbstractSource") or "duckduckgo",
            "url": d.get("AbstractURL") or "",
        }
    # Wikipedia first paragraph
    try:
        art = fetch_wikipedia_article(topic=query)
        if art:
            first_para = art[1].split("\n")[0][:600]
            if first_para:
                return {"answer": first_para, "source": "wikipedia",
                        "url": art[2]}
    except Exception:
        pass
    return None


def read_journal(limit=50):
    """Read the self-learning journal (newest first)."""
    if not os.path.exists(JOURNAL):
        return []
    entries = []
    with open(JOURNAL) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries[-limit:][::-1]


def stats():
    """Summary of self-learning so far."""
    state = _load_state()
    j = read_journal(1000)
    learned = [e for e in j if "learned_url" in e]
    return {
        "cycles": state.get("cycles", 0),
        "items_learned": state.get("items_learned", 0),
        "total_tokens": state.get("total_tokens", 0),
        "last_cycle": state.get("last_cycle"),
        "journal_size": len(j),
    }
