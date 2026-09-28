"""Step 5: label your real Gmail inbox with the fine-tuned model.

Run it whenever you like; it only touches inbox mail in the Primary tab from
the last few days that it hasn't labelled yet, and adds labels like
Classifier/Bank.

  python 05_gmail_labeler.py --dry-run     # classify and print counts, change nothing
  python 05_gmail_labeler.py               # add the labels
  python 05_gmail_labeler.py --days 30     # look further back

One-time setup: put your Google Cloud OAuth "Desktop app" client in
credentials.json. The first run opens a browser to grant access and saves
token.json. Keep both files private. Output is counts only, never email text.
"""
import argparse
import base64
import email
from collections import Counter
from pathlib import Path

import torch
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prepare_data import body_of, decode, model_text

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]  # read mail + change labels
PARENT = "Classifier"
MAX_LEN = 256  # same as 03_finetune.py
RETRIES = 8  # on Gmail rate limits, wait and retry with growing delays instead of crashing

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=7, help="how far back to look")
ap.add_argument("--max", type=int, default=500, help="most emails to process in one run")
ap.add_argument("--threshold", type=float, default=0.7, help="skip emails the model is less sure about")
ap.add_argument("--dry-run", action="store_true", help="classify but don't add labels")
args = ap.parse_args()


def gmail_service():
    creds = Credentials.from_authorized_user_file("token.json", SCOPES) if Path("token.json").exists() else None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:  # apps left in "Testing" lose their login after 7 days
            creds = None
    if not creds or not creds.valid:
        creds = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES).run_local_server(port=0)
    Path("token.json").write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def label_ids(gmail, names, create):
    """Map each category to its Gmail label id, creating Classifier/<Name> if needed."""
    existing = {l["name"]: l["id"] for l in gmail.users().labels().list(userId="me").execute(num_retries=RETRIES)["labels"]}
    for name in [PARENT] + [f"{PARENT}/{n.title()}" for n in names]:
        if name not in existing and create:
            existing[name] = gmail.users().labels().create(userId="me", body={"name": name}).execute(num_retries=RETRIES)["id"]
    return {n: existing[f"{PARENT}/{n.title()}"] for n in names if f"{PARENT}/{n.title()}" in existing}


def inbox_ids(gmail):
    ids, page = [], None
    while len(ids) < args.max:
        resp = gmail.users().messages().list(
            userId="me", q=f"in:inbox category:primary newer_than:{args.days}d", pageToken=page, maxResults=100
        ).execute(num_retries=RETRIES)
        ids += [m["id"] for m in resp.get("messages", [])]
        page = resp.get("nextPageToken")
        if not page:
            break
    return ids[: args.max]


def fetch_texts(gmail, ids, done_ids):
    """Download each email and format it like the training data. Skips ones already labelled."""
    out = {}
    for i, mid in enumerate(ids, 1):
        print(f"  fetched {i}/{len(ids)}", end="\r", flush=True)
        raw = gmail.users().messages().get(userId="me", id=mid, format="raw").execute(num_retries=RETRIES)
        if done_ids & set(raw.get("labelIds", [])):
            continue
        msg = email.message_from_bytes(base64.urlsafe_b64decode(raw["raw"]))
        out[mid] = model_text(decode(msg["From"]), decode(msg["Subject"]), body_of(msg))
    print()
    return out


def classify(texts):
    tok = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model").eval()
    results = []
    for i in range(0, len(texts), 16):
        print(f"  classified {i}/{len(texts)}", end="\r", flush=True)
        enc = tok(texts[i : i + 16], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            probs = model(**enc).logits.softmax(-1)
        conf, idx = probs.max(-1)
        results += [(model.config.id2label[j], c) for j, c in zip(idx.tolist(), conf.tolist())]
    print(f"  classified {len(texts)}/{len(texts)}")
    return results


if __name__ == "__main__":
    categories = list(AutoModelForSequenceClassification.from_pretrained("model").config.id2label.values())
    gmail = gmail_service()
    cat_ids = label_ids(gmail, categories, create=not args.dry_run)
    done_ids = set(cat_ids.values())

    ids = inbox_ids(gmail)
    texts = fetch_texts(gmail, ids, done_ids)
    print(f"Primary inbox emails from the last {args.days} days: {len(ids)}, not yet labelled: {len(texts)}")
    if not texts:
        raise SystemExit

    by_label, unsure = {}, 0
    for mid, (label, conf) in zip(texts, classify(list(texts.values()))):
        if conf < args.threshold:
            unsure += 1
        else:
            by_label.setdefault(label, []).append(mid)

    if not args.dry_run:
        for label, mids in by_label.items():
            for i in range(0, len(mids), 1000):  # batchModify takes up to 1000 ids per call
                gmail.users().messages().batchModify(
                    userId="me", body={"ids": mids[i : i + 1000], "addLabelIds": [cat_ids[label]]}
                ).execute(num_retries=RETRIES)

    counts = Counter({label: len(mids) for label, mids in by_label.items()})
    verb = "Would label" if args.dry_run else "Labelled"
    print(f"{verb} {sum(counts.values())}: " + ", ".join(f"{n} {l}" for l, n in counts.most_common()))
    print(f"Left alone (confidence < {args.threshold:.0%}): {unsure}")
