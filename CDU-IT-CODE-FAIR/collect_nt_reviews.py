#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect NT/Darwin tourism reviews & posts
Sources:
  - Google Maps (Places API) — enhanced discovery + dual-sort reviews
  - YouTube comments (search + channel uploads fallback)
  - Blogs (RSS discovery + curated fallback via trafilatura)
"""

import os
import time
import csv
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import urllib.robotparser as urobot

import requests
import pandas as pd
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil import parser as dateparser
import googlemaps
from dotenv import load_dotenv

# -----------------------------
# Global config
# -----------------------------
DARWIN_CENTER = (-12.4634, 130.8456)  # Darwin CBD
SEARCH_RADIUS_METERS = 60000          # used for nearby/text search
API_SLEEP = 0.6                       # polite pacing for APIs
CRAWL_SLEEP = 0.5                     # pacing for HTTP fetches

REQUEST_UA = "Mozilla/5.0 (research; CDU IT Code Fair; +educational)"
REQ_TIMEOUT = 20
# Speed toggle (default off; set in main from --fast)
FAST_MODE = False


# Expanded seed list to offset the 5-review cap per place
SEED_PLACE_NAMES = [
    # Darwin & surrounds
    "Mindil Beach", "Mindil Beach Sunset Market", "Darwin Waterfront", "Wave Lagoon Darwin",
    "Stokes Hill Wharf", "Cullen Bay Marina", "Aquascene", "Deckchair Cinema",
    "Museum and Art Gallery of the Northern Territory", "George Brown Darwin Botanic Gardens",
    "Charles Darwin National Park", "East Point Reserve", "Darwin Military Museum",
    "Defence of Darwin Experience", "Fannie Bay Gaol Museum", "RFDS Darwin Tourist Facility",
    "Casuarina Coastal Reserve", "Leanyer Recreation Park", "Nightcliff Jetty", "Parap Markets",
    "Berry Springs Nature Park", "Territory Wildlife Park", "Howard Springs Nature Park",
    "Crocosaurus Cove", "Mason Gallery", "Austin Lane", "Goyder Park", "1934 Qantas Hangar",
    "St Mary's Star of the Sea Cathedral", "WWII Oil Storage Tunnels", "Darwin Aviation Museum",
    "Crocodylus Park", "Sea Darwin", "Darwin Harbour Cruises", "Spectacular Jumping Crocodile Cruise",
    "Matt Wright Explore the Wild", "Lameroo Beach", "Lee Point", "Nightcliff Foreshore",
    "Dripstone Cliffs", "Rapid Creek Markets",

    # Litchfield NP
    "Litchfield National Park", "Wangi Falls", "Florence Falls", "Buley Rockhole", "Tolmer Falls",
    "Magnetic Termite Mounds",

    # Kakadu NP
    "Kakadu National Park", "Bowali Visitor Centre", "Ubirr", "Nourlangie", "Yellow Water", "Gunlom Falls",

    # Tiwi & day trips
    "Tiwi Islands", "Tiwi Design Art Centre", "Jilamara Arts and Crafts", "Adelaide River Jumping Crocodile Cruises",

    # Katherine / Nitmiluk (Top End extension)
    "Nitmiluk Gorge", "Nitmiluk Cruises", "Cutta Cutta Caves", "Mataranka Thermal Pool",
]

# -----------------------------
# Data container
# -----------------------------
@dataclass
class Record:
    source: str          # google_maps | youtube | blog
    object: str          # place_name | query/channel | blog_url
    text: str
    rating: Optional[float]
    timestamp: Optional[str]
    url: Optional[str]
    author: Optional[str]
    extra_json: Optional[str]

# -----------------------------
# Helpers
# -----------------------------
def safe_parse_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return dateparser.parse(s).isoformat()
    except Exception:
        return None

def robots_allows(url: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urobot.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except Exception:
        return False

# -----------------------------
# Google Maps (Places API)  — ENHANCED DISCOVERY + DUAL-SORT REVIEWS
# -----------------------------
def google_places_client(api_key: str) -> googlemaps.Client:
    return googlemaps.Client(key=api_key)

# Areas to scan (Darwin + popular NT regions)
SCAN_CENTERS = [
    (-12.4634, 130.8456),  # Darwin CBD
    (-12.9248, 131.3074),  # Litchfield NP
    (-12.6700, 132.8300),  # Kakadu (Jabiru/Ubirr area)
    (-14.3140, 132.4200),  # Katherine / Nitmiluk
]
SCAN_RADIUS_KM = 45   # coverage radius around centers
GRID_STEP_KM   = 15   # grid granularity (smaller => more coverage / more calls)

# Attraction-like place types
ATTRACTION_TYPES = ["tourist_attraction", "park", "museum", "zoo", "aquarium", "art_gallery"]

# Keywords to surface tours/markets/lookouts/waterfalls/etc.
DISCOVERY_KEYWORDS = [
    "tour", "cruise", "jumping crocodile", "harbour cruise",
    "market", "sunset market", "waterfall", "lookout", "swimming hole",
    "wildlife", "gallery", "art", "war museum", "aviation", "tunnel",
    "botanic gardens", "national park", "crocodile", "river cruise"
]

def _deg_per_km_lat() -> float:
    return 1.0 / 110.574

def _deg_per_km_lng(lat: float) -> float:
    return 1.0 / (111.320 * math.cos(math.radians(lat)))

def generate_grid(center_lat: float, center_lng: float, radius_km: float = 40, step_km: float = 15):
    """Yield lat/lng points in a simple grid covering ~radius_km."""
    lat_step = step_km * _deg_per_km_lat()
    lng_step = step_km * _deg_per_km_lng(center_lat)
    n_steps = max(1, int(radius_km // step_km))
    for dy in range(-n_steps, n_steps + 1):
        for dx in range(-n_steps, n_steps + 1):
            yield (center_lat + dy * lat_step, center_lng + dx * lng_step)

def nearby_search_collect(gclient: googlemaps.Client, loc, type_=None, keyword=None):
    """Call places_nearby with optional type/keyword; returns list of result dicts."""
    try:
        resp = gclient.places_nearby(location=loc, radius=SEARCH_RADIUS_METERS, type=type_, keyword=keyword)
        out = resp.get("results", [])
        while "next_page_token" in resp:
            time.sleep(2.0)  # required delay
            resp = gclient.places_nearby(page_token=resp["next_page_token"])
            out.extend(resp.get("results", []))
        return out
    except Exception as e:
        print(f"[Google] nearby_search error at {loc} type={type_} kw={keyword}: {e}")
        return []

def text_search_collect(gclient: googlemaps.Client, query: str, loc=None):
    """Text search with optional location bias; returns list of result dicts."""
    try:
        kwargs = dict(query=query)
        if loc:
            kwargs.update(location=loc, radius=SEARCH_RADIUS_METERS)
        resp = gclient.places(**kwargs)
        out = resp.get("results", [])
        while "next_page_token" in resp:
            time.sleep(2.0)
            resp = gclient.places(page_token=resp["next_page_token"])
            out.extend(resp.get("results", []))
        return out
    except Exception as e:
        print(f"[Google] text_search error '{query}': {e}")
        return []

def discover_place_candidates(gclient: googlemaps.Client) -> Dict[str, Dict[str, Any]]:
    """
    Discover many attraction-like places across grid points & queries.
    Returns: {place_id: {'name':..., 'user_ratings_total':..., 'rating':..., 'from':..., 'latlng':(lat,lng)}}
    """
    candidates: Dict[str, Dict[str, Any]] = {}

    # 1) Grid scan with attraction types
    for (clat, clng) in SCAN_CENTERS:
        for pt in generate_grid(clat, clng, radius_km=SCAN_RADIUS_KM, step_km=GRID_STEP_KM):
            for t in ATTRACTION_TYPES:
                results = nearby_search_collect(gclient, pt, type_=t, keyword=None)
                for r in results:
                    pid = r.get("place_id")
                    if not pid:
                        continue
                    meta = candidates.get(pid, {})
                    meta.update({
                        "name": r.get("name"),
                        "user_ratings_total": r.get("user_ratings_total", 0) or 0,
                        "rating": r.get("rating"),
                        "from": "nearby",
                        "latlng": tuple(r.get("geometry", {}).get("location", {}).values()) if r.get("geometry") else None
                    })
                    candidates[pid] = meta
            time.sleep(API_SLEEP)

    # 2) Grid scan with keywords (no type), to catch tours/markets/etc.
    for (clat, clng) in SCAN_CENTERS:
        for pt in generate_grid(clat, clng, radius_km=SCAN_RADIUS_KM, step_km=GRID_STEP_KM):
            for kw in DISCOVERY_KEYWORDS:
                results = nearby_search_collect(gclient, pt, type_=None, keyword=kw)
                for r in results:
                    pid = r.get("place_id")
                    if not pid:
                        continue
                    meta = candidates.get(pid, {})
                    meta.update({
                        "name": r.get("name"),
                        "user_ratings_total": r.get("user_ratings_total", 0) or 0,
                        "rating": r.get("rating"),
                        "from": "nearby_kw",
                        "latlng": tuple(r.get("geometry", {}).get("location", {}).values()) if r.get("geometry") else None
                    })
                    candidates[pid] = meta
            time.sleep(API_SLEEP)

    # 3) Text queries for seed attractions with regional suffixes
    for name in SEED_PLACE_NAMES:
        for suffix in ("Darwin NT", "Northern Territory", "Australia"):
            q = f"{name} {suffix}"
            results = text_search_collect(gclient, q, loc=DARWIN_CENTER)
            for r in results:
                pid = r.get("place_id")
                if not pid:
                    continue
                meta = candidates.get(pid, {})
                meta.update({
                    "name": r.get("name"),
                    "user_ratings_total": r.get("user_ratings_total", 0) or 0,
                    "rating": r.get("rating"),
                    "from": "text",
                    "latlng": tuple(r.get("geometry", {}).get("location", {}).values()) if r.get("geometry") else None
                })
                candidates[pid] = meta
        time.sleep(API_SLEEP)

    return candidates

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
def place_details_reviews_sorted(gclient: googlemaps.Client, place_id: str, sort: str):
    # sort: 'most_relevant' or 'newest'
    return gclient.place(place_id=place_id, fields=["name", "url", "reviews"], reviews_sort=sort)

def collect_google_reviews(gclient: googlemaps.Client, max_places: int) -> List[Record]:
    print("[Google] Discovering attraction candidates (grid + keywords + text)…")
    cand = discover_place_candidates(gclient)

    ranked = sorted(
        cand.items(),
        key=lambda kv: (kv[1].get("user_ratings_total", 0), kv[1].get("rating", 0.0)),
        reverse=True
    )
    place_ids_list = [pid for pid, _meta in ranked][:max_places]
    print(f"[Google] Will fetch details for {len(place_ids_list)} places.")

    records: List[Record] = []
    for pid in tqdm(place_ids_list, desc="Google Place Details"):
        try:
            seen = set()
            combined_reviews: List[Record] = []

            # Only 1 sort in FAST mode; 2 sorts otherwise
            sorts = ("most_relevant",) if FAST_MODE else ("most_relevant", "newest")

            for sort in sorts:
                details = place_details_reviews_sorted(gclient, pid, sort)
                result = details.get("result", {}) or {}
                pname = result.get("name")
                purl  = result.get("url")
                reviews = result.get("reviews", []) or []

                for rv in reviews:
                    key = (rv.get("author_name"), rv.get("time"), rv.get("text") or rv.get("original_text", {}).get("text"))
                    if key in seen:
                        continue
                    seen.add(key)

                    text = rv.get("text") or rv.get("original_text", {}).get("text") or ""
                    rating = rv.get("rating")
                    author = rv.get("author_name")
                    ts = rv.get("time")
                    iso_ts = None
                    if ts:
                        try:
                            iso_ts = pd.to_datetime(int(ts), unit="s").isoformat()
                        except Exception:
                            iso_ts = None

                    combined_reviews.append(Record(
                        source="google_maps",
                        object=pname or "",
                        text=(text or "").strip(),
                        rating=float(rating) if rating is not None else None,
                        timestamp=iso_ts,
                        url=purl,
                        author=author,
                        extra_json=json.dumps({"place_id": pid, "reviews_sort": sort}, ensure_ascii=False)
                    ))
                time.sleep(API_SLEEP)

            records.extend(combined_reviews)
            time.sleep(API_SLEEP)

        except Exception as e:
            print(f"[Google] details error for place_id={pid}: {e}")
            continue

    return records


# --------- YouTube Data API (comments as micro-reviews) ----------
from googleapiclient.discovery import build as yt_build_service
from googleapiclient.errors import HttpError

YOUTUBE_SEARCH_QUERIES = [
    "Mindil Beach Darwin review", "Mindil Beach sunset",
    "Darwin Waterfront review", "Crocosaurus Cove review",
    "Berry Springs Nature Park swimming", "Fannie Bay Gaol Darwin",
    "Litchfield National Park review", "Kakadu National Park review",
    "Nightcliff Jetty sunset", "Parap Markets food"
]

# Fallback channels 
YOUTUBE_CHANNEL_IDS = [
    "UC7WtIGE1kqtxl8qkGYnNLRA",  # Tourism NT
    "UCFYZoIRSuSKc0qgzPPnHx6A",  # Mindil Beach Sunset Market
    "UC9zBaxBlszAQZ7jMFh3qJyw",  # Mindil Beach Markets (alt)
]

def yt_build(api_key: str):
    return yt_build_service("youtube", "v3", developerKey=api_key)

def yt_search_video_ids(yt, query: str, max_results: int = 10):
    resp = yt.search().list(q=query, part="id", type="video", maxResults=max_results).execute()
    return [it["id"]["videoId"] for it in resp.get("items", []) if it.get("id", {}).get("videoId")]

def yt_uploads_playlist_id(yt, channel_id: str):
    try:
        resp = yt.channels().list(id=channel_id, part="contentDetails").execute()
        items = resp.get("items", [])
        if not items:
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError:
        return None

def yt_playlist_video_ids(yt, playlist_id: str, max_videos=30):
    vids = []
    if not playlist_id:
        return vids
    req = yt.playlistItems().list(part="contentDetails", playlistId=playlist_id, maxResults=50)
    while req and len(vids) < max_videos:
        resp = req.execute()
        for it in resp.get("items", []):
            vid = it.get("contentDetails", {}).get("videoId")
            if vid:
                vids.append(vid)
        req = yt.playlistItems().list_next(req, resp)
    return vids[:max_videos]

def yt_filter_commentable_videos(yt, video_ids):
    kept = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        if not chunk:
            continue
        resp = yt.videos().list(part="status,statistics", id=",".join(chunk)).execute()
        for it in resp.get("items", []):
            vid = it.get("id")
            status = it.get("status", {}) or {}
            stats  = it.get("statistics", {}) or {}
            made_for_kids = bool(status.get("madeForKids") or status.get("selfDeclaredMadeForKids"))
            try:
                comment_count = int(stats.get("commentCount", 0) or 0)
            except Exception:
                comment_count = 0
            if (not made_for_kids) and comment_count > 0:
                kept.append(vid)
    return kept

def yt_fetch_comments(yt, video_id: str, max_total: int = 100):
    out = []
    req = yt.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        order="relevance",
        textFormat="plainText",
    )
    while req and len(out) < max_total:
        try:
            resp = req.execute()
        except HttpError as e:
            msg = str(e)
            if "commentsDisabled" in msg or "forbidden" in msg:
                return []
            raise
        for item in resp.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "text": sn.get("textDisplay", "") or "",
                "author": sn.get("authorDisplayName"),
                "timestamp": sn.get("publishedAt"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })
        req = yt.commentThreads().list_next(req, resp)
    return out[:max_total]

def collect_youtube(api_key: str,
                    queries: list,
                    per_query_videos=8,
                    per_video_comments=50,
                    fallback_channels=YOUTUBE_CHANNEL_IDS,
                    fallback_channel_videos=12):
    if not api_key:
        print("[YouTube] Skipped (no YOUTUBE_API_KEY).")
        return []

    yt = yt_build(api_key)
    records: List[Record] = []

    any_search_succeeded = False
    for q in tqdm(queries, desc="YouTube queries"):
        try:
            vids = yt_search_video_ids(yt, q, max_results=per_query_videos)
        except HttpError as e:
            print(f"[YouTube] search blocked for '{q}': {e}")
            vids = []
        if not vids:
            continue
        any_search_succeeded = True
        vids = yt_filter_commentable_videos(yt, vids)
        for vid in vids:
            try:
                comments = yt_fetch_comments(yt, vid, max_total=per_video_comments)
            except HttpError as e:
                print(f"[YouTube] skip video {vid}: {e}")
                comments = []
            for c in comments:
                records.append(Record(
                    source="youtube",
                    object=q,
                    text=c["text"],
                    rating=None,
                    timestamp=c["timestamp"],
                    url=c["url"],
                    author=c["author"],
                    extra_json=None
                ))

    if not any_search_succeeded or not records:
        print("[YouTube] Falling back to channel uploads…")
        for ch in fallback_channels:
            upl = yt_uploads_playlist_id(yt, ch)
            vids = yt_playlist_video_ids(yt, upl, max_videos=fallback_channel_videos)
            vids = yt_filter_commentable_videos(yt, vids)
            for vid in vids:
                try:
                    comments = yt_fetch_comments(yt, vid, max_total=per_video_comments)
                except HttpError as e:
                    print(f"[YouTube] skip video {vid}: {e}")
                    comments = []
                for c in comments:
                    records.append(Record(
                        source="youtube",
                        object=f"channel:{ch}",
                        text=c["text"],
                        rating=None,
                        timestamp=c["timestamp"],
                        url=c["url"],
                        author=c["author"],
                        extra_json=None
                    ))
    return records

# -----------------------------
# Blogs (RSS + curated fallback via trafilatura)
# -----------------------------
import re
import feedparser
import trafilatura

BLOG_SITES = [
    "https://www.nomadasaurus.com",
    "https://campermate.com/en/blog",
    "https://www.australiantraveller.com",
    "https://kelseyinlondon.com",
    "https://perthtravelers.com",
    "https://northernterritory.com/articles",
    "https://norther.com.au/journal",
    "https://www.aussiemob.com",
]

BLOG_SEED_URLS = [
    "https://www.nomadasaurus.com/places-to-visit-in-the-northern-territory/",
    "https://www.nomadasaurus.com/best-northern-territory-road-trips/",
    "https://campermate.com/en/blog/post/getaway-guides/nt/9-epic-road-trip-routes-in-the-northern-territory",
    "https://www.australiantraveller.com/nt/best-nt-road-trips/",
    "https://kelseyinlondon.com/northern-territory-itinerary/",
    "https://perthtravelers.com/northern-territory-travel-guide/",
    "https://northernterritory.com/articles/10-takeaway-ideas-in-darwin",
    "https://northernterritory.com/articles/7-unique-foods-to-try-in-northern-australia-and-where-to-find-them",
    "https://norther.com.au/journal/",
    "https://www.aussiemob.com/best-road-trips-in-the-northern-territory/",
]

NT_KEYWORDS = [
    "northern territory", "darwin", "kakadu", "litchfield", "tiwi",
    "nitmiluk", "katherine", "top end", "mindil", "parap", "berry springs"
]

def guess_feed_candidates(site_url: str):
    base = site_url.rstrip("/")
    candidates = {
        f"{base}/feed/",
        f"{base}/rss/",
        f"{base}?feed=rss2",
        f"{base}/category/australia/northern-territory/feed/",
        f"{base}/tag/northern-territory/feed/",
        f"{base}/tag/darwin/feed/",
        f"{base}/tag/kakadu/feed/",
        f"{base}/tag/litchfield/feed/",
    }
    if re.search(r"/(articles|journal)$", base):
        candidates.add(f"{base}/feed/")
    return list(candidates)

def is_valid_feed(url: str) -> bool:
    try:
        r = requests.get(url, headers={"User-Agent": REQUEST_UA}, timeout=REQ_TIMEOUT)
        if r.status_code >= 400:
            return False
        ctype = r.headers.get("Content-Type", "").lower()
        if "xml" in ctype or "rss" in ctype or "atom" in ctype:
            return True
        txt = r.text[:1000].lower()
        return ("<rss" in txt) or ("<feed" in txt)
    except Exception:
        return False

def discover_feeds(site_url: str):
    feeds = []
    for cand in guess_feed_candidates(site_url):
        if is_valid_feed(cand):
            feeds.append(cand)
    # dedupe, preserve order
    seen = set(); out = []
    for f in feeds:
        if f not in seen:
            out.append(f); seen.add(f)
    return out

def entry_mentions_nt(title: str, summary: str) -> bool:
    blob = f"{title} {summary}".lower()
    return any(k in blob for k in NT_KEYWORDS)

def collect_blogs_smart(max_articles: int = 40):
    urls_to_fetch: List[str] = []

    # Try RSS feeds first
    for site in BLOG_SITES:
        feeds = discover_feeds(site)
        for feed_url in feeds:
            fp = feedparser.parse(feed_url)
            for e in fp.entries:
                link = e.get("link")
                title = e.get("title", "")
                summary = e.get("summary", "")
                if link and entry_mentions_nt(title, summary):
                    urls_to_fetch.append(link)

    # Fallback to curated URLs
    if not urls_to_fetch:
        urls_to_fetch = BLOG_SEED_URLS[:]

    # Deduplicate and cap
    urls_to_fetch = list(dict.fromkeys(urls_to_fetch))[:max_articles]

    records: List[Record] = []
    for u in tqdm(urls_to_fetch, desc="Blogs smart"):
        if not robots_allows(u):
            continue
        try:
            html = requests.get(u, headers={"User-Agent": REQUEST_UA}, timeout=REQ_TIMEOUT).text
            text = trafilatura.extract(html, url=u) or ""
            text = text.strip()
            if not text:
                continue
            records.append(Record(
                source="blog",
                object=u,
                text=text,
                rating=None,
                timestamp=None,
                url=u,
                author=None,
                extra_json=None
            ))
            time.sleep(CRAWL_SLEEP)
        except Exception as e:
            print(f"[Blogs] fail {u}: {e}")
            continue
    return records

# -----------------------------
# Export & main
# -----------------------------
def dedupe_and_export(records: List[Record], out_csv: str):
    rows = []
    for r in records:
        if not r.text:
            continue
        rows.append({
            "source": r.source,
            "object": r.object,
            "text": r.text.replace("\u0000", " ").strip(),
            "rating": r.rating,
            "timestamp": r.timestamp,
            "url": r.url,
            "author": r.author,
            "extra_json": r.extra_json
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.drop_duplicates(subset=["source", "text"], inplace=True)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"[OK] Wrote {len(df)} rows -> {out_csv}")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Collect NT/Darwin tourism reviews & posts (Google + Blogs + YouTube)")
    parser.add_argument("--output", default="nt_reviews.csv", help="Output CSV path")
    parser.add_argument("--max_places", type=int, default=30, help="Max Google places to fetch")
    parser.add_argument("--max_blogs", type=int, default=15, help="Max blog articles to fetch")
    # after existing parser.add_argument(...) lines
    parser.add_argument("--fast", action="store_true",
                        help="Faster discovery: Darwin-only, smaller grid, fewer keywords, single-sort reviews")
    parser.add_argument("--skip_youtube", action="store_true",
                        help="Skip YouTube collection (speed up)")
    parser.add_argument("--skip_blogs", action="store_true",
                        help="Skip blogs collection (speed up)")

    args = parser.parse_args()

    global FAST_MODE, API_SLEEP, SCAN_CENTERS, SCAN_RADIUS_KM, GRID_STEP_KM, DISCOVERY_KEYWORDS
    FAST_MODE = bool(args.fast)

    if FAST_MODE:
        # Faster Google discovery
        API_SLEEP = 0.3                        # a bit snappier
        SCAN_CENTERS = [(-12.4634, 130.8456)]  # Darwin CBD only
        SCAN_RADIUS_KM = 20                    # smaller search radius
        GRID_STEP_KM = 25                      # coarser grid (fewer points)
        DISCOVERY_KEYWORDS = ["market", "tour"]  # minimal keywords (skip waterfalls/lookouts etc.)

    skip_youtube = bool(args.skip_youtube)
    skip_blogs = bool(args.skip_blogs)
    google_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    yt_key     = os.getenv("YOUTUBE_API_KEY", "")

    records: List[Record] = []

    # Google Maps
    if google_key:
        print("[Google] Collecting Google Maps reviews…")
        gclient = googlemaps.Client(key=google_key)
        records.extend(collect_google_reviews(gclient, max_places=args.max_places))
    else:
        print("[Google] Skipped (no GOOGLE_PLACES_API_KEY).")

    # Blogs
    if not skip_blogs:
        print("[Blogs] Collecting via smart RSS + fallback URLs…")
        try:
            records.extend(collect_blogs_smart(max_articles=args.max_blogs))
        except Exception as e:
            print(f"[Blogs] Skipped due to error: {e}")
    else:
        print("[Blogs] Skipped (--skip_blogs).")

    # YouTube
    if not skip_youtube:
        print("[YouTube] Collecting comments as tourist micro-reviews…")
        try:
            # lighter defaults in fast mode
            yt_videos = 4 if FAST_MODE else 8
            yt_comments = 20 if FAST_MODE else 50
            records.extend(collect_youtube(
                yt_key,
                YOUTUBE_SEARCH_QUERIES,
                per_query_videos=yt_videos,
                per_video_comments=yt_comments
            ) or [])
        except Exception as e:
            print(f"[YouTube] Skipped due to error: {e}")
    else:
        print("[YouTube] Skipped (--skip_youtube).")

    # Export
    dedupe_and_export(records, args.output)

if __name__ == "__main__":
    main()
