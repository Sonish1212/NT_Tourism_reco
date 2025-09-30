# NT_Tourism_reco
Data-driven project for the CDU-IT Code Fair: collecting, preprocessing, and analyzing NT tourism reviews (Google Maps, YouTube, Blogs) to generate actionable recommendations and system-level insights for making the Northern Territory a stronger tourism hub.
This project was developed for the CDU-IT Code Fair to support the Northern Territory in becoming a stronger tourism hub.

We built an end-to-end pipeline that turns raw public reviews into actionable insights:

Data Collection

Google Maps reviews (via API)

Travel blogs (RSS + full-text extraction)

YouTube video metadata & comments

Preprocessing

Clean & normalize text (emoji removal, PII redaction, HTML strip)

Deduplicate (exact + near-duplicate filtering)

Label sentiment (from Google star ratings + ML model)

Balance minority classes (neg/neu)

Blog post chunking & source mix control

Modeling & Recommendations

TF-IDF + Logistic Regression sentiment model with calibration & threshold tuning

Topic mining (NMF) to uncover recurring themes

Priority scoring to highlight tourism sites most in need of improvement

System-level recommendations (themes like accessibility, signage, shade, digital info, pricing)

Outputs & Visualizations

reco_places_priority.csv – priority list of NT attractions with issues & opportunities

place_topic_counts.csv – per-place theme/topic distribution

reco_system_themes.csv – aggregated system-level recommendations
