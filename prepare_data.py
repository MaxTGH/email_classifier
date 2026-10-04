"""Step 0: build data/emails.csv with two columns: text, label.

Two sources:
  python prepare_data.py --demo               # public dataset, works right away
  python prepare_data.py --mbox path/to.mbox  # your own Gmail (Google Takeout export)
  add --out path/to/emails.csv to save somewhere other than data/emails.csv

For your own Gmail, each inbox message is labelled bank / news / other by the
rules below: sender domains and keywords. Edit the lists to fit your inbox.
The model then learns to generalise those rules from the text alone.
"""
import argparse
import mailbox
import re
from email.header import decode_header, make_header
from email.utils import parseaddr
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "emails.csv"
MAX_CHARS = 2000  # the model only reads ~256 tokens anyway

# Labelling rules for --mbox (inbox mail only), checked in this order: bank, news, other.
# Domains also match subdomains (chase.com matches alerts.chase.com); full
# addresses (name@example.com) match only that sender.
BANK_DOMAINS = (
    "chase.com", "bankofamerica.com", "wellsfargo.com", "capitalone.com", "americanexpress.com",
    "aexp.com", "citi.com", "discover.com", "usbank.com", "ally.com", "sofi.com", "schwab.com",
    "fidelity.com", "paypal.com", "venmo.com", "creditkarma.com", "experian.com", "synchrony.com",
)
BANK_WORDS = (
    "statement is ready", "payment due", "payment received", "autopay", "available balance",
    "credit score", "credit summary", "direct deposit", "account alert", "credit card", "bank account",
)
NEWS_DOMAINS = (
    "nytimes.com", "wsj.com", "washingtonpost.com", "bloomberg.com", "reuters.com", "apnews.com",
    "cnn.com", "bbc.com", "bbc.co.uk", "theguardian.com", "axios.com", "politico.com", "npr.org",
    "theatlantic.com", "economist.com", "ft.com", "substack.com", "morningbrew.com",
)
NEWS_WORDS = ("breaking news", "top stories", "headlines", "morning briefing", "daily briefing", "newsletter")
SKIP_LABELS = ("Sent", "Spam", "Trash", "Draft", "Chat")


def demo_data():
    from datasets import load_dataset

    # 20 Newsgroups: old-school internet messages. We keep 4 topics and give
    # them inbox-style names.
    keep = {
        "comp.sys.mac.hardware": "tech",
        "rec.sport.baseball": "sports",
        "sci.med": "health",
        "misc.forsale": "shopping",
    }
    rows = []
    for split in ("train", "test"):
        for ex in load_dataset("SetFit/20_newsgroups", split=split):
            if ex["label_text"] in keep and len(ex["text"].strip()) > 20:
                rows.append({"text": ex["text"][:MAX_CHARS], "label": keep[ex["label_text"]]})
    return pd.DataFrame(rows)


def decode(value):
    try:
        return str(make_header(decode_header(value or "")))
    except Exception:
        return value or ""


def body_of(msg):
    """Prefer the plain-text part; fall back to HTML with tags stripped."""
    parts = msg.walk() if msg.is_multipart() else [msg]
    plain, html = None, None
    for part in parts:
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            continue
        if ctype == "text/plain" and plain is None:
            plain = text
        elif ctype == "text/html" and html is None:
            html = re.sub(r"<style.*?</style>|<[^>]+>", " ", text, flags=re.S)
    text = plain or html or ""
    return re.sub(r"\s+", " ", text).strip()


def from_domain(sender, domains):
    address = parseaddr(sender)[1].lower()
    domain = address.rpartition("@")[2]
    return any(address == d if "@" in d else domain == d or domain.endswith("." + d) for d in domains)


def label_of(gmail_labels, sender, subject, body):
    tags = {t.strip() for t in gmail_labels.split(",")}
    if "Inbox" not in tags or tags & set(SKIP_LABELS):
        return None  # only inbox mail; your own replies in inbox threads are tagged Sent too
    words = f"{subject} {body[:500]}".lower()
    if from_domain(sender, BANK_DOMAINS) or any(w in words for w in BANK_WORDS):
        return "bank"
    if from_domain(sender, NEWS_DOMAINS) or any(w in words for w in NEWS_WORDS):
        return "news"
    return "other"


def model_text(sender, subject, body):
    """The exact text the model sees. gmail_labeler.py uses this too, so
    live emails are formatted the same way as the training data."""
    return f"From: {sender}\nSubject: {subject}\n\n{body}"[:MAX_CHARS]


def mbox_data(path):
    rows = []
    for msg in mailbox.mbox(path):
        sender, subject, body = decode(msg["From"]), decode(msg["Subject"]), body_of(msg)
        label = label_of(decode(msg.get("X-Gmail-Labels", "")), sender, subject, body)
        if label is None:
            continue
        rows.append({"text": model_text(sender, subject, body), "label": label})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true")
    group.add_argument("--mbox")
    ap.add_argument("--out", type=Path, default=OUT, help="where to write the CSV")
    args = ap.parse_args()

    df = demo_data() if args.demo else mbox_data(args.mbox)
    df = df.drop_duplicates("text").sample(frac=1, random_state=0)  # shuffle
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} messages to {args.out}\n")
    print(df["label"].value_counts().to_string())
