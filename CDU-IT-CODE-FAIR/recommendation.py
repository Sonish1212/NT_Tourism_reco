# NT Tourism — Modeling + Themes + Recommendations (integrated actions)
import re, json, warnings
from collections import Counter
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVecTopics



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG = 42

# -------------------- Load Clean Data --------------------
DF = pd.read_csv("nt_reviews_clean.csv")

# Minimal schema guardrails
for c in ["source","source_type","object","text","rating","label_sentiment",
          "place_id","parent_url","timestamp_iso","timestamp"]:
    if c not in DF.columns:
        DF[c] = None

# Cleanup + dedupe
DF["text"] = DF["text"].astype(str)
DF = DF[DF["text"].str.strip().ne("")].copy()
DF = DF.drop_duplicates(subset=["text","source","object"], keep="first").reset_index(drop=True)

# Timestamp (UTC) & recency window
ts = pd.to_datetime(DF["timestamp_iso"].fillna(DF["timestamp"]), errors="coerce", utc=True)
DF["_ts"] = ts
today = pd.Timestamp.utcnow()
last12 = today - pd.Timedelta(days=365)

# -------------------- Labels from stars (fallback) --------------------
def star_to_sentiment(x):
    try:
        f = float(x)
    except Exception:
        return None
    if f <= 2: return "neg"
    if f == 3: return "neu"
    return "pos"

if "label_sentiment" not in DF.columns or DF["label_sentiment"].isna().all():
    DF["label_sentiment"] = DF["rating"].map(star_to_sentiment)
else:
    DF["label_sentiment"] = DF["label_sentiment"].where(
        DF["label_sentiment"].notna(),
        DF["rating"].map(star_to_sentiment)
    )

# -------------------- Group key (place + article) --------------------
def build_groups(df):
    g = df["place_id"].astype(str).where(df["place_id"].notna(), df["object"].astype(str))
    pu = df["parent_url"].astype(str).where(df["parent_url"].notna(), "")
    return (g.fillna("UNK") + "||" + pu.fillna(""))

groups_all = build_groups(DF)

# -------------------- Labeled subset & leakage-free split --------------------
LAB = DF[DF["label_sentiment"].notna()].copy()
y_all = LAB["label_sentiment"]
groups_lab = build_groups(LAB)

from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    splitter = sgkf.split(LAB["text"], y_all, groups=groups_lab)
except Exception:
    print("[WARN] StratifiedGroupKFold unavailable; using GroupKFold fallback.")
    gkf = GroupKFold(n_splits=5)
    splitter = gkf.split(LAB["text"], y_all, groups=groups_lab)

# Assign fold id
fold_id = np.empty(len(LAB), dtype=int)
for k, (_, test_idx) in enumerate(splitter):
    fold_id[test_idx] = k

# Pick VAL/TEST folds with most minorities (neg+neu)
fc = (pd.DataFrame({"fold": fold_id, "y": LAB["label_sentiment"].values})
        .groupby(["fold","y"]).size().unstack(fill_value=0))
for c in ["neg","neu","pos"]:
    if c not in fc.columns: fc[c] = 0
fc["minor_score"] = fc["neg"] + fc["neu"]

VAL_FOLD  = int(fc["minor_score"].idxmax())
TEST_FOLD = int(fc.drop(index=VAL_FOLD)["minor_score"].idxmax())

train_mask = ~np.isin(fold_id, [VAL_FOLD, TEST_FOLD])
val_mask   =  (fold_id == VAL_FOLD)
test_mask  =  (fold_id == TEST_FOLD)

train_idx = np.where(train_mask)[0]
val_idx   = np.where(val_mask)[0]
test_idx  = np.where(test_mask)[0]

LAB_train = LAB.iloc[train_idx].copy()
LAB_val   = LAB.iloc[val_idx].copy()
LAB_test  = LAB.iloc[test_idx].copy()

print("[SPLIT*] chosen VAL fold:", VAL_FOLD, "| TEST fold:", TEST_FOLD)
print("[SPLIT] sizes:", len(LAB_train), len(LAB_val), len(LAB_test))
print("[SPLIT] label counts:",
      LAB_train["label_sentiment"].value_counts().to_dict(),
      LAB_val["label_sentiment"].value_counts().to_dict(),
      LAB_test["label_sentiment"].value_counts().to_dict())

# -------------------- Oversample minorities on TRAIN --------------------
def oversample_to(df, ycol="label_sentiment", target_per_class=120, random_state=RNG):
    frames = []
    for cls, grp in df.groupby(ycol):
        if len(grp) < target_per_class:
            grp = resample(grp, replace=True, n_samples=target_per_class, random_state=random_state)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)

print("[BAL] train class counts (before):", LAB_train["label_sentiment"].value_counts().to_dict())
target = int(min(150, max(60, LAB_train["label_sentiment"].value_counts().median())))
LAB_train_bal = oversample_to(LAB_train, target_per_class=target, random_state=RNG)
print("[BAL] train class counts (after):",  LAB_train_bal["label_sentiment"].value_counts().to_dict())

# -------------------- Model: word+char TF-IDF + gentler class weights --------------------


feats = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b"
    )),
    ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,6),
        min_df=2
    )),
])

pipe = Pipeline([
    ("feats", feats),
    ("clf", LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        class_weight={"pos": 1.0, "neu": 5.0, "neg": 7.0},
        C=2.0,
        random_state=RNG
    ))
])

pipe.fit(LAB_train_bal["text"], LAB_train_bal["label_sentiment"])

# -------------------- Threshold tuning with neutral band --------------------
def predict_neu_band(P, classes, tau_pos=0.60, tau_neg=0.50, neu_max=0.58, margin=0.08):
    """
    - If max prob < neu_max  OR  (max - second) < margin  -> 'neu'
    - Else if argmax is 'pos' and prob >= tau_pos        -> 'pos'
    - Else if argmax is 'neg' and prob >= tau_neg        -> 'neg'
    - Else                                               -> 'neu'
    """
    pos_i = classes.index("pos"); neg_i = classes.index("neg")
    maxp = P.max(axis=1)
    arg  = P.argmax(axis=1)
    second = np.partition(P, -2, axis=1)[:, -2]

    out = []
    for i, row in enumerate(P):
        if (maxp[i] < neu_max) or ((row[arg[i]] - second[i]) < margin):
            out.append("neu")
        elif arg[i] == pos_i and row[pos_i] >= tau_pos:
            out.append("pos")
        elif arg[i] == neg_i and row[neg_i] >= tau_neg:
            out.append("neg")
        else:
            out.append("neu")
    return np.array(out)

classes = pipe.classes_.tolist()
P_val = pipe.predict_proba(LAB_val["text"])

best = (0.0, 0.60, 0.50, 0.58, 0.08)
for tau_pos in [0.55, 0.60, 0.65]:
    for tau_neg in [0.45, 0.50, 0.55]:
        for neu_max in [0.55, 0.58, 0.60, 0.62]:
            for margin in [0.06, 0.08, 0.10, 0.12]:
                yhat = predict_neu_band(P_val, classes, tau_pos, tau_neg, neu_max, margin)
                f1 = f1_score(LAB_val["label_sentiment"], yhat, average="macro", zero_division=0)
                if f1 > best[0]:
                    best = (f1, tau_pos, tau_neg, neu_max, margin)

print(f"[THRESH+] best val macro-F1={best[0]:.3f} "
      f"at tau_pos={best[1]:.2f}, tau_neg={best[2]:.2f}, neu_max={best[3]:.2f}, margin={best[4]:.2f}")

print("\n[VAL] classification report (thresholded+band)")
print(classification_report(
    LAB_val["label_sentiment"],
    predict_neu_band(P_val, classes, best[1], best[2], best[3], best[4]),
    digits=3, zero_division=0
))

P_test = pipe.predict_proba(LAB_test["text"])
print("\n[TEST] classification report (thresholded+band)")
print(classification_report(
    LAB_test["label_sentiment"],
    predict_neu_band(P_test, classes, best[1], best[2], best[3], best[4]),
    digits=3, zero_division=0
))

# === Use tuned thresholds for dataset labeling ===
P_all = pipe.predict_proba(DF["text"])
DF["model_label"] = predict_neu_band(P_all, classes, best[1], best[2], best[3], best[4])
DF["model_conf"]  = P_all.max(axis=1)

CONF_THRESH = 0.65
DF["final_sentiment"] = DF["label_sentiment"]
mask_unlabeled = DF["final_sentiment"].isna()
DF.loc[mask_unlabeled & (DF["model_conf"] >= CONF_THRESH), "final_sentiment"] = DF.loc[mask_unlabeled, "model_label"]

# -------------------- Themes via NMF topics --------------------

vec_topics = TfidfVecTopics(ngram_range=(1,2), min_df=5, max_df=0.9,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
X_topics = vec_topics.fit_transform(DF["text"])
n_topics = 12
nmf = NMF(n_components=n_topics, random_state=RNG, init="nndsvda", max_iter=400)
W = nmf.fit_transform(X_topics)
H = nmf.components_
terms = np.array(vec_topics.get_feature_names_out())

def topic_labels(H, terms, topk=6):
    labels = []
    for t in range(H.shape[0]):
        top_terms = terms[H[t].argsort()[::-1][:topk]]
        labels.append(", ".join(top_terms))
    return labels

topic_names = topic_labels(H, terms, topk=6)
DF["topic_id"] = W.argmax(axis=1)
DF["topic_label"] = [topic_names[i] for i in DF["topic_id"]]

# -------------------- Priority scoring per place --------------------
place_key = DF["place_id"].astype(str).where(DF["place_id"].notna(), DF["object"].astype(str)).fillna("UNKNOWN")
DF["_place"] = place_key
DF["is_recent"] = DF["_ts"] >= last12

# Aggregates
n_total   = DF.groupby("_place")["text"].size().rename("n_total")
pos_share = DF.groupby("_place")["final_sentiment"].apply(lambda s: (s=="pos").mean()).rename("pos_share")
neg_share = DF.groupby("_place")["final_sentiment"].apply(lambda s: (s=="neg").mean()).rename("neg_share")
recent_share = DF.groupby("_place")["is_recent"].mean().rename("recent_share")

# Topic distribution per place
topic_counts = DF.pivot_table(index="_place", columns="topic_label", values="text", aggfunc="count").fillna(0)

# Priority = (neg - pos) * log(1+n) * (0.5 + 0.5*recent)
prio = (neg_share.fillna(0) - pos_share.fillna(0)) * np.log1p(n_total) * (0.5 + 0.5*recent_share)
priority = pd.concat([n_total, pos_share, neg_share, recent_share, prio.rename("priority")], axis=1)\
             .sort_values("priority", ascending=False)

# -------------------- ACTIONS (integrated, NT-specific) --------------------
CATEGORY_PATTERNS = {
    "heat_shade": r"\b(shade|shade\s?sail|shaded|mister|misting|sun\s?exposure|hot|heat|humid|muggy|water (?:bubbler|refill)|cool(?:er)? hours)\b",
    "parking": r"\b(park(?:ing)?|car ?park|carpark|parking\s?lot|spaces?\b|spot\b)\b",
    "queues_crowding": r"\b(queue|line[- ]?up|long\s?line|wait(?:ing)?|crowd(?:ed|s)?|packed|busy)\b",
    "pricing_value": r"\b(price|priced|expensive|overpriced|cost(?:ly)?|value\s?for\s?money|ticket|entry\s?fee|fees?)\b",
    "signage_wayfinding": r"\b(sign(?:age)?|map(?:s)?|wayfinding|direction(?:s)?|info(?:rmation)?|QR|guide|label)\b",
    "service_staff": r"\b(staff|service|ranger|guide|helpful|friendly|rude|attitude|customer\s?service)\b",
    "amenities_clean": r"\b(clean|dirty|toilet|bathroom|restroom|loo|amenit(?:y|ies)|bin|trash|rubbish|litter|smell|soap|toilet\s?closed)\b",
    "accessibility": r"\b(access(?:ible|ibility)?|wheelchair|pram|stroller|ramp|steps?\b|steep|mobility)\b",
    "safety_info": r"\b(safety|warning|alert|danger|closure|closed|wet\s?season|track)\b",
    "wildlife_risks": r"\b(croc(?:odile)?|stinger|jelly(?:fish)?|box\s?jelly|marine\s?stinger|mosquito(?:es)?|midg(?:e|y)|sandfl(?:y|ies))\b",
    "food_vendors": r"\b(food|drink|caf[eé]|coffee|stall|vendor|menu|option(?:s)?|EFTPOS|card\s?only|cash\s?only|eftpos\s?down)\b",
    "events": r"\b(event|market|festival|night\s?market|show)\b",
    "lighting_transport": r"\b(lighting|lights|dark|night|bus|public\s?transport|shuttle)\b",
    "fishing_gear": r"\b(fish(?:ing)?|bait|rod|mackerel|trevally|queenfish|shark)\b",
}
CATEGORY_ACTIONS = {
    "heat_shade": "Heat: add shade sails/misters & water bubblers; promote cooler visit hours.",
    "parking": "Parking: overflow wayfinding; clear signage; live availability via QR/site.",
    "queues_crowding": "Queues: timed entry & surge staffing; marked lines; live crowd info via QR.",
    "pricing_value": "Value: NT resident/family bundles; off-peak discounts; list inclusions clearly.",
    "signage_wayfinding": "Wayfinding: onsite maps & multilingual signs; QR with live status/closures.",
    "service_staff": "Service: staff & cultural-safety training; surge roster in peak season.",
    "amenities_clean": "Amenities: more cleaning cycles/bins; stock toilets; quick-issue QR reporting.",
    "accessibility": "Accessibility: ramps & pram routes; mark accessible paths on maps/QR.",
    "safety_info": "Ops comms: publish closures/track conditions (wet season) + alternatives via QR.",
    "wildlife_risks": "Marine/croc safety: clearer stinger/croc signage; season-specific alerts; repellents info.",
    "food_vendors": "Food/EFTPOS: diversify cuisines; shaded dining; ensure reliable EFTPOS; manage cash-only stalls.",
    "events": "Programming: extend dry-season hours; shoulder-season events with local & First Nations partners.",
    "lighting_transport": "Access: better lighting on paths; publish bus/shuttle options; event shuttles.",
    "fishing_gear": "Anglers: info on species/seasons; bait/gear availability; tide/current safety tips.",
}
CAT_REGEX = {k: re.compile(v, flags=re.I) for k, v in CATEGORY_PATTERNS.items()}

def categorize_text_blob(text: str) -> Counter:
    counts = Counter()
    if not isinstance(text, str) or not text.strip():
        return counts
    for cat, rx in CAT_REGEX.items():
        hits = rx.findall(text)
        if hits:
            counts[cat] += len(hits)
    return counts

def build_signal_text_for_place(df_place: pd.DataFrame) -> str:
    neg = df_place[df_place["final_sentiment"] == "neg"]["text"].astype(str).tolist()
    neu = df_place[df_place["final_sentiment"] == "neu"]["text"].astype(str).tolist()
    topics = df_place["topic_label"].astype(str).tolist()
    return " \n ".join([" ".join(neg[:200]), " ".join(neu[:150]), " ".join(topics[:50])])

# topic summary per place (for display & heuristic)
top_topic_text = (DF.groupby("_place")["topic_label"]
                    .apply(lambda s: " | ".join(s.value_counts().head(5).index))
                    .rename("top_topics"))

# Heuristic actions from topics/place name (to avoid empties)
PLACE_HINTS = {
    "market": ["queues_crowding","food_vendors","amenities_clean","lighting_transport"],
    "beach":  ["safety_info","heat_shade","food_vendors"],
    "waterfront": ["safety_info","food_vendors","signage_wayfinding"],
    "jetty":  ["lighting_transport","safety_info","fishing_gear"],
    "park":   ["heat_shade","amenities_clean","signage_wayfinding"],
    "lagoon": ["safety_info","amenities_clean","food_vendors"],
    "museum": ["signage_wayfinding","amenities_clean"],
    "cathedral":["signage_wayfinding","accessibility"],
    "gallery":["signage_wayfinding","amenities_clean"],
    "national park":["safety_info","signage_wayfinding","heat_shade"],
}

def infer_actions_from_topics_and_place(topics: str, place_name: str):
    acts = []
    blob = f"{topics or ''} {place_name or ''}"
    for cat, rx in CAT_REGEX.items():
        if rx.search(blob):
            acts.append(CATEGORY_ACTIONS[cat])
    low = (place_name or "").lower()
    for key, cats in PLACE_HINTS.items():
        if key in low:
            for c in cats:
                act = CATEGORY_ACTIONS[c]
                if act not in acts: acts.append(act)
    return acts[:5]

# Build per-place signals and actions
place_sig = {pk: build_signal_text_for_place(sub) for pk, sub in DF.groupby("_place")}
place_cat_counts = {pk: categorize_text_blob(sig) for pk, sig in place_sig.items()}

def actions_from_counts(cat_counts: Counter, top_k=3):
    if not cat_counts: return []
    top = [c for c, _ in cat_counts.most_common(top_k)]
    return [CATEGORY_ACTIONS[c] for c in top if c in CATEGORY_ACTIONS]

# -------------------- Per-place recommendations --------------------
PL = priority.copy()
PL = PL.join(top_topic_text, how="left")

# main actions from real text; fallback to topics/place hints
actions_list = []
for pk in PL.index:
    main = actions_from_counts(place_cat_counts.get(pk, Counter()), top_k=3)
    if not main or len(main) < 2:
        add = infer_actions_from_topics_and_place(PL.at[pk,"top_topics"], pk)
        # merge & dedupe
        merged = []
        for a in (main + add):
            if a not in merged: merged.append(a)
        actions_list.append(merged[:5] if merged else ["Add on-site QR feedback to pinpoint issues; review staffing/amenities at peak times."])
    else:
        actions_list.append(main)
PL["suggested_actions"] = actions_list

# Human-readable place_name (prefer Google)
google_titles = (DF[DF["source_type"]=="google"]
                 .groupby("place_id")["object"]
                 .agg(lambda s: s.value_counts().index[0] if len(s)>0 else None))
PL["place_name"] = PL.index.to_series().map(google_titles).fillna(PL.index.to_series())

# -------------------- Filter to tourism places (integrated) --------------------
NON_TOURISM_RE = re.compile(r"\b(Woolworths|Coles|IGA|Aldi|Supermarket|Chemist|Pharmacy|BWS|Liquorland)\b", re.I)
is_non_tourism = PL["place_name"].astype(str).str.contains(NON_TOURISM_RE)
PL = PL.loc[~is_non_tourism].copy()

# Align topic_counts to filtered places
topic_counts = topic_counts.loc[topic_counts.index.isin(PL.index)].copy()

# -------------------- System (NT-wide) themes with robust actions --------------------
DF["is_recent"] = DF["_ts"] >= last12
theme_rows = []
for topic, sub in DF.groupby("topic_label"):
    n = len(sub)
    pos = (sub["final_sentiment"]=="pos").mean()
    neg = (sub["final_sentiment"]=="neg").mean()
    recent = sub["is_recent"].mean()
    priority_theme = (neg - pos) * np.log1p(n) * (0.5 + 0.5*recent)
    signal = build_signal_text_for_place(sub)
    cats = categorize_text_blob(signal)
    acts = actions_from_counts(cats, top_k=3)
    if not acts:
        acts = infer_actions_from_topics_and_place(topic, "")
    theme_rows.append({
        "topic_label": topic,
        "n": n,
        "pos_share": pos,
        "neg_share": neg,
        "recent_share": recent,
        "priority": priority_theme,
        "suggested_actions": acts[:5]
    })
theme = pd.DataFrame(theme_rows).sort_values("priority", ascending=False)

# -------------------- Save Outputs (same filenames) --------------------
PL.reset_index(inplace=True)
PL.rename(columns={"index":"place_key"}, inplace=True)

PL.to_csv("reco_places_priority.csv", index=False)
topic_counts.to_csv("place_topic_counts.csv")
theme.to_csv("reco_system_themes.csv", index=False)

print("\n[OK] Wrote:")
print(" - reco_places_priority.csv (tourism-only, topics + enriched actions)")
print(" - place_topic_counts.csv (filtered to tourism places)")
print(" - reco_system_themes.csv (themes with system-level actions)")
