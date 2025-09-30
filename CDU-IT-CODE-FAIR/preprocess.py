#!/usr/bin/env python3
"""
NT Tourism Reviews — Preprocessing Toolkit
==========================================
Cleans and prepares the unified raw CSV (`nt_reviews.csv` by default) produced by
your collectors (Google/Blogs/YouTube) for downstream modeling and visualization.

Major steps:
A) Schema check, drop empties
B) Exact & (optional) near-duplicate filtering (TF‑IDF + NearestNeighbors)
C) Text normalization (HTML/entities, URL/@/# extraction-removal, emoji/control removal,
   optional de-elongation, case)
D) Rating/timestamp normalization (to ISO-UTC)
E) Feature enrichment (source_type, char/word/sent counts, place_id, time features)
F) Safety flags (PII redaction counts, basic profanity flag)
G) Blog chunk expansion (long posts -> medium-sized chunks)
H) Optional balancing: per-place cap and minimum source mix quota
I) Group-safe train/val/test split by place_id (or object fallback)
J) Clean report + write CSV

Usage
-----
# simple run with defaults (requires langdetect & scikit-learn for some steps)
python preprocess.py --in nt_reviews.csv --out nt_reviews_clean.csv --english-only

"""

import re, html, json, math, warnings, argparse, sys
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# ---------- A) Load & schema sanity ----------
EXPECTED_COLS = ["source","object","text","rating","timestamp","url","author","extra_json"]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the raw CSV has the expected columns. Missing columns are added as None."""
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = None
    return df[EXPECTED_COLS].copy()

def drop_empty_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where text is empty/whitespace after basic coercion."""
    s = df["text"].astype(str).str.strip()
    return df[s.ne("")]

# ---------- Text cleanup & normalization ----------
EMOJI_RE = re.compile("["                       
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
"]+", flags=re.UNICODE)
CTRL_RE  = re.compile(r"[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]+")
TAG_RE   = re.compile(r"<[^>]+>")
URL_RE   = re.compile(r"(https?://[^\s]+)")
MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_\.]+)")
HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")
ZW_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")  # zero-width chars

def strip_html_entities_tags(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def strip_emoji_ctrl(text: str) -> str:
    t = EMOJI_RE.sub("", text or "")
    t = CTRL_RE.sub(" ", t)
    t = ZW_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_and_remove_patterns(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """Extract URLs, @mentions, #hashtags and remove them from text."""
    urls = URL_RE.findall(text or "")
    mentions = MENTION_RE.findall(text or "")
    hashtags = HASHTAG_RE.findall(text or "")
    # remove them from text
    t = URL_RE.sub(" ", text or "")
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(), urls, mentions, hashtags

def normalize_case(text: str, lower: bool) -> str:
    return text.lower() if lower else text

def de_elongate(word: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", word)

def de_elongate_text(text: str) -> str:
    return " ".join(de_elongate(w) for w in (text or "").split())

# ---------- Field normalization ----------
def coerce_rating(x):
    """Coerce to [1..5] float; else NaN."""
    try:
        f = float(x)
        return f if 1.0 <= f <= 5.0 else np.nan
    except Exception:
        return np.nan

def to_iso_utc(x) -> Optional[str]:
    """Parse timestamps of various shapes to ISO-8601 UTC string or None."""
    try:
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(dt): return None
        return dt.isoformat()
    except Exception:
        return None

def derive_time_feats(df: pd.DataFrame, tz_local: str = "Australia/Darwin") -> pd.DataFrame:
    """
    Create time-based features. Keeps normalized UTC timestamp and adds local seasonal features.
    NOTE: We convert tz-aware UTC -> tz-naive local using tz_convert(None) (not tz_localize(None)).
    """
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Drop tz by converting to naive UTC (consistent storage)
    df["timestamp_iso"] = ts_utc.dt.tz_convert(None)
    # Localize for seasonal features; fall back to UTC if tz database missing
    try:
        ts_local = ts_utc.dt.tz_convert(tz_local)
    except Exception:
        ts_local = ts_utc
    df["year"] = ts_local.dt.year
    df["month"] = ts_local.dt.month
    df["dow"] = ts_local.dt.dayofweek
    df["hour"] = ts_local.dt.hour
    # Top End dry season roughly May–Oct
    df["dry_season_flag"] = df["month"].between(5, 10).astype(int)
    return df

# ---------- Deduplication ----------
def dedup_exact(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["source","text"])

def dedup_near(df: pd.DataFrame, text_col="text", threshold=0.90, max_neighbors=5) -> pd.DataFrame:
    """
    Optional near-duplicate removal using sklearn NearestNeighbors on TF-IDF.
    Silently skips if sklearn is unavailable or dataset too small.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        warnings.warn("sklearn not available; skipping near-duplicate filtering.")
        return df

    texts = df[text_col].astype(str).tolist()
    if len(texts) < 50:
        return df

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(max_neighbors, len(texts)), metric="cosine").fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    # cosine distance -> similarity
    sims = 1 - dists
    to_drop = set()
    for i, row in enumerate(idxs):
        if i in to_drop: continue
        for j, sim in zip(row[1:], sims[i][1:]):  # skip self
            if sim >= threshold:
                to_drop.add(j)
    keep_mask = ~pd.Series(range(len(df))).isin(to_drop)
    return df.loc[keep_mask].copy()

# ---------- Source enrichment ----------
def extract_place_id(extra_json):
    try:
        if pd.isna(extra_json): return None
        obj = json.loads(extra_json) if isinstance(extra_json, str) else extra_json
        return obj.get("place_id")
    except Exception:
        return None

def add_source_features(df: pd.DataFrame) -> pd.DataFrame:
    df["source_type"] = df["source"].fillna("").map(
        lambda s: "google" if "google" in s else ("youtube" if "youtube" in s else ("blog" if "blog" in s else s))
    )
    df["char_len"] = df["text"].astype(str).str.len()
    df["word_len"] = df["text"].astype(str).str.split().map(len)
    df["sent_count"] = df["text"].astype(str).str.count(r"[.!?]+") + 1
    df["place_id"] = df["extra_json"].map(extract_place_id)
    return df

# ---------- Safety/quality ----------
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
AU_PHONE_RE = re.compile(r"(?:(?:\+?61|0)[2-478])\s?\d{2,4}\s?\d{3}\s?\d{3}")
BASIC_PROFANITY = {"damn","hell","shit","crap","bloody","bastard"}  # extend as needed

def flag_and_redact_pii(text: str) -> Tuple[str, int, int]:
    emails = len(EMAIL_RE.findall(text))
    phones = len(AU_PHONE_RE.findall(text))
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = AU_PHONE_RE.sub("[PHONE]", t)
    return t, emails, phones

def flag_profanity(text: str) -> int:
    toks = re.findall(r"[A-Za-z']+", (text or "").lower())
    return int(any(tok in BASIC_PROFANITY for tok in toks))

# ---------- Label construction ----------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    # infer sentiment label from Google stars when present
    def star_to_sentiment(x):
        if pd.isna(x): return None
        if x <= 2: return "neg"
        if x == 3: return "neu"
        return "pos"  # 4-5
    df["label_star"] = df["rating"]
    df["label_sentiment"] = df["rating"].map(star_to_sentiment)
    return df

# ---------- Balancing helpers ----------
def cap_per_group(df: pd.DataFrame, group_col: str, cap: int) -> pd.DataFrame:
    if cap is None or cap <= 0: return df
    return (df.sort_values("timestamp_iso")
              .groupby(group_col, group_keys=False)
              .head(cap))

def enforce_source_mix(df: pd.DataFrame, min_frac: Dict[str,float]) -> pd.DataFrame:
    """Downsample dominant sources to guarantee minimum fraction for others."""
    n = len(df)
    targets = {k: int(v*n) for k,v in min_frac.items()}
    parts = []
    used = 0
    for src, tgt in targets.items():
        part = df[df["source_type"]==src].sample(min(tgt, (df["source_type"]==src).sum()), random_state=42)
        parts.append(part); used += len(part)
    remaining = df.drop(pd.concat(parts).index) if parts else df
    need = max(0, n - used)
    if need>0 and len(remaining)>0:
        parts.append(remaining.sample(min(need, len(remaining)), random_state=42))
    return pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

# ---------- Group-safe splits ----------
def train_val_test_split_group(
    df: pd.DataFrame, group_col: str, ratios=(0.8, 0.1, 0.1), seed=42
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Split by groups (place_id/object) so the same place doesn't leak across splits."""
    rng = np.random.RandomState(seed)
    groups = pd.Series(df[group_col].fillna(df["object"]).astype(str).unique())
    groups = groups.sample(frac=1.0, random_state=seed).tolist()
    n = len(groups)
    n_train = int(ratios[0]*n); n_val = int(ratios[1]*n)
    g_train = set(groups[:n_train])
    g_val = set(groups[n_train:n_train+n_val])
    g_test = set(groups[n_train+n_val:])
    idx_train = df[df[group_col].fillna(df["object"]).astype(str).isin(g_train)].index
    idx_val   = df[df[group_col].fillna(df["object"]).astype(str).isin(g_val)].index
    idx_test  = df[df[group_col].fillna(df["object"]).astype(str).isin(g_test)].index
    return idx_train, idx_val, idx_test

# ---------- Blog segmentation (paragraph/word chunks) ----------
def chunk_text_by_words(text: str, min_w=60, max_w=120) -> List[str]:
    words = (text or "").split()
    chunks, i = [], 0
    while i < len(words):
        j = min(len(words), i + max_w)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j
    # drop tiny trailing chunks
    return [c for c in chunks if len(c.split()) >= min_w] or ([text] if text else [])

def expand_blog_chunks(df: pd.DataFrame, min_w=60, max_w=120) -> pd.DataFrame:
    blogs = df[df["source_type"]=="blog"].copy()
    others = df[df["source_type"]!="blog"].copy()
    if blogs.empty: return df
    rows = []
    for _, r in blogs.iterrows():
        parts = chunk_text_by_words(r["text"], min_w=min_w, max_w=max_w)
        for i, p in enumerate(parts):
            nr = r.copy()
            nr["parent_url"] = r["url"]
            nr["chunk_ix"] = i
            nr["text"] = p
            rows.append(nr)
    out = pd.concat([others, pd.DataFrame(rows)], ignore_index=True)
    return out

# ---------- Language filter (optional) ----------
def keep_english_only(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    try:
        from langdetect import detect
    except Exception:
        warnings.warn("langdetect not installed; skipping language filter.")
        return df
    mask = df[text_col].astype(str).map(lambda t: (detect(t) == "en") if t and len(t.strip())>=20 else True)
    return df[mask].copy()

# ---------- Master pipeline ----------
def preprocess_csv(
    in_csv: str,
    out_csv: str,
    *,
    min_chars: int = 15,
    lowercase: bool = False,
    remove_emojis: bool = True,
    de_elongate_words: bool = False,
    english_only: bool = False,
    apply_near_dedup: bool = True,
    near_dup_threshold: float = 0.90,
    blog_chunking: bool = True,
    blog_min_w: int = 60,
    blog_max_w: int = 120,
    per_place_cap: Optional[int] = None,   # e.g., 200
    enforce_mix: Optional[Dict[str,float]] = None,  # e.g., {"youtube":0.15, "blog":0.10}
    write_report: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    df = ensure_schema(df)
    df["text"] = df["text"].astype(str).str.replace("\u0000"," ", regex=False)

    # A/B) schema & exact dedup
    df = drop_empty_texts(df)
    df = dedup_exact(df)

    # C) text cleanup
    #  - html/entities & tags
    df["text"] = df["text"].map(strip_html_entities_tags)
    #  - PII redaction & flags
    pii_res = df["text"].map(flag_and_redact_pii)
    df["text"] = [t for (t,_,__) in pii_res]
    df["pii_emails"] = [e for (_,e,__) in pii_res]
    df["pii_phones"] = [p for (_,_,p) in pii_res]
    #  - extract urls/@/#
    extracted = df["text"].map(extract_and_remove_patterns)
    df["text"] = [x[0] for x in extracted]
    df["urls_in_text"] = [",".join(x[1]) if x[1] else None for x in extracted]
    df["mentions"] = [",".join(x[2]) if x[2] else None for x in extracted]
    df["hashtags"] = [",".join(x[3]) if x[3] else None for x in extracted]
    #  - emoji/control removal
    if remove_emojis:
        df["text"] = df["text"].map(strip_emoji_ctrl)
    else:
        df["text"] = df["text"].map(lambda t: CTRL_RE.sub(" ", t or ""))

    #  - de-elongate (soooo -> soo)
    if de_elongate_words:
        df["text"] = df["text"].map(de_elongate_text)
    #  - case
    df["text"] = df["text"].map(lambda s: normalize_case(s, lowercase))

    # Drop ultra-short noise
    if min_chars and min_chars>0:
        df = df[df["text"].str.len() >= min_chars]

    # D) normalize rating & timestamp
    df["rating"] = df["rating"].map(coerce_rating)
    df["timestamp"] = df["timestamp"].map(to_iso_utc)

    # E) enrich
    df = add_source_features(df)
    df = derive_time_feats(df)
    df["profanity_flag"] = df["text"].map(flag_profanity)

    # Optional sentiment label from stars
    df = add_labels(df)

    # L) language filter
    if english_only:
        df = keep_english_only(df)

    # J) blog chunking
    if blog_chunking:
        df = expand_blog_chunks(df, min_w=blog_min_w, max_w=blog_max_w)

    # B) near-duplicate filter (post-clean)
    if apply_near_dedup:
        df = dedup_near(df, text_col="text", threshold=near_dup_threshold, max_neighbors=5)

    # H) balance (optional)
    if per_place_cap:
        group_col = "place_id"
        # fallback to object if place_id missing
        if "place_id" not in df.columns or df["place_id"].isna().all():
            group_col = "object"
        df = cap_per_group(df, group_col, per_place_cap)

    if enforce_mix:
        df = enforce_source_mix(df, enforce_mix)

    # Final exact dedup safety
    df = dedup_exact(df).reset_index(drop=True)

    # Report
    if write_report:
        rpt = {
            "rows": len(df),
            "by_source": df["source_type"].value_counts().to_dict(),
            "rating_nonnull": int(df["rating"].notna().sum()),
            "time_min": str(pd.to_datetime(df["timestamp"], errors="coerce").min()),
            "time_max": str(pd.to_datetime(df["timestamp"], errors="coerce").max()),
            "short_texts_dropped_min_chars": int(min_chars),
            "blog_chunks_rows": int(df["source_type"].eq("blog").sum()),
            "near_dedup_threshold": near_dup_threshold if apply_near_dedup else None,
        }
        print("=== CLEAN REPORT ===")
        for k,v in rpt.items():
            print(f"{k}: {v}")

    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote cleaned dataset -> {out_csv}  (rows={len(df)})")
    return df

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess NT Tourism reviews CSV.")
    ap.add_argument("--in", dest="in_csv", default="nt_reviews.csv", help="Input CSV")
    ap.add_argument("--out", dest="out_csv", default="nt_reviews_clean.csv", help="Output CSV")
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--no-emojis", dest="remove_emojis", action="store_false")
    ap.add_argument("--de-elongate", action="store_true")
    ap.add_argument("--english-only", action="store_true")
    ap.add_argument("--no-english-only", dest="english_only", action="store_false")
    ap.add_argument("--near-dup-threshold", type=float, default=0.90)
    ap.add_argument("--no-near-dup", dest="apply_near_dedup", action="store_false")
    ap.add_argument("--blog-chunking", action="store_true", default=True)
    ap.add_argument("--no-blog-chunking", dest="blog_chunking", action="store_false")
    ap.add_argument("--blog-min-w", type=int, default=60)
    ap.add_argument("--blog-max-w", type=int, default=120)
    ap.add_argument("--per-place-cap", type=int, default=None)
    ap.add_argument("--min-blog-frac", type=float, default=None)
    ap.add_argument("--min-youtube-frac", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    enforce_mix = None
    if args.min_blog_frac is not None or args.min_youtube_frac is not None:
        enforce_mix = {}
        if args.min_blog_frac is not None:
            enforce_mix["blog"] = args.min_blog_frac
        if args.min_youtube_frac is not None:
            enforce_mix["youtube"] = args.min_youtube_frac

    preprocess_csv(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        min_chars=args.min_chars,
        lowercase=args.lowercase,
        remove_emojis=args.remove_emojis,
        de_elongate_words=args.de_elongate,
        english_only=args.english_only,
        apply_near_dedup=args.apply_near_dedup,
        near_dup_threshold=args.near_dup_threshold,
        blog_chunking=args.blog_chunking,
        blog_min_w=args.blog_min_w,
        blog_max_w=args.blog_max_w,
        per_place_cap=args.per_place_cap,
        enforce_mix=enforce_mix,
        write_report=True
    )

if __name__ == "__main__":
    # If executed directly, run with CLI args
    main()
