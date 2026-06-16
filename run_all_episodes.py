"""
Batch eval runner — all episodes for show 2bcdfe58.
Saves to speech_cache.json after each episode. Resumable.
Run: python3 /Users/srini/Desktop/evals/run_all_episodes.py
"""
import sys, json, os, unicodedata
sys.path.insert(0, '/Users/srini/Desktop/evals')

import numpy as np
from azure.storage.blob import BlobServiceClient
from eval_metrics import (
    compute_utmos, text_naturalness,
    pause_alignment_score, blaser_qe_text, SONAR_LANG_CODES,
)

ACCOUNT_URL = "https://dashprodstore.blob.core.windows.net"
SAS_TOKEN   = ("sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2030-06-30T21:36:20Z"
               "&st=2024-06-30T13:36:20Z&spr=https,http"
               "&sig=ewQCKuZEeC7A6vnlFxSDxDwVU7zunyCwB4tfE6880HA%3D")
CACHE_FILE  = "/Users/srini/Desktop/evals/speech_cache.json"
SHOW_ID     = "2bcdfe58"

client = BlobServiceClient(account_url=ACCOUNT_URL, credential=SAS_TOKEN)\
         .get_container_client("auto-dubbing")

def fetch(p):       return json.loads(client.get_blob_client(p).download_blob().readall())
def fetch_bytes(p): return client.get_blob_client(p).download_blob().readall()

# ── language detection (no Streamlit import) ──────────────────────────────────
SCRIPT_BLOCKS = [
    (0x0C00, 0x0C7F, "Telugu",    "tel"),
    (0x0900, 0x097F, "Hindi",     "hin"),
    (0x0B80, 0x0BFF, "Tamil",     "tam"),
    (0x0980, 0x09FF, "Bengali",   "ben"),
    (0x0D00, 0x0D7F, "Malayalam", "mal"),
    (0x0C80, 0x0CFF, "Kannada",   "kan"),
    (0x0A80, 0x0AFF, "Gujarati",  "guj"),
    (0x0A00, 0x0A7F, "Punjabi",   "pan"),
    (0x0600, 0x06FF, "Urdu",      "urd"),
    (0x4E00, 0x9FFF, "Chinese",   "zho"),
]

def detect_lang(texts):
    counts = {}
    for t in texts:
        for ch in t:
            cp = ord(ch)
            for lo, hi, lang, code in SCRIPT_BLOCKS:
                if lo <= cp <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "Telugu", "tel"
    best = max(counts, key=counts.get)
    code = next(c for _, _, l, c in SCRIPT_BLOCKS if l == best)
    return best, code

def fmt_f(v):
    return f"{v:.2f}" if v is not None else "N/A"

# ── load cache ────────────────────────────────────────────────────────────────
try:    cache = json.load(open(CACHE_FILE))
except: cache = {}

# ── list all episodes ─────────────────────────────────────────────────────────
episodes = sorted({
    b.name.split("/episodes/")[1].split("/")[0]
    for b in client.list_blobs(name_starts_with=f"shows/dubbing/{SHOW_ID}/episodes/")
    if "/episodes/" in b.name
})
print(f"Show {SHOW_ID} — {len(episodes)} episodes: {episodes}\n", flush=True)

# ── run per episode ───────────────────────────────────────────────────────────
for ep_idx, ep in enumerate(episodes, 1):
    key  = f"{SHOW_ID}:{ep}"
    base = f"shows/dubbing/{SHOW_ID}/episodes/{ep}/intermediate"
    results = cache.get(key, {})

    try:
        fit = fetch(f"{base}/tts/final_fit.json")["dialogs"]
    except Exception as e:
        print(f"[{ep_idx}/{len(episodes)}] Episode {ep}: no pipeline data — {e}", flush=True)
        continue

    try:
        src_dialogs = fetch(f"{base}/transcripts/cleaned.json")["dialogs"]
        tgt_dialogs = fetch(f"{base}/translation/translated.json")["dialogs"]
    except Exception as e:
        print(f"[{ep_idx}/{len(episodes)}] Episode {ep}: missing transcript/translation — {e}", flush=True)
        continue

    src_map = {d["index"]: d for d in src_dialogs}
    tgt_map = {d["index"]: d for d in tgt_dialogs}

    # list WAV files
    wavs = {}
    for b in client.list_blobs(name_starts_with=f"{base}/tts/per_dialog/"):
        if b.name.endswith(".wav"):
            idx = int(b.name.split("/")[-1].replace(".wav", ""))
            wavs[idx] = b.name

    # detect target language
    tgt_texts = [d.get("text", "") for d in fit if d.get("text")]
    _, lang_code = detect_lang(tgt_texts)
    src_lang = "hin"  # default; update if Chinese shows are added

    # load vocals once per episode for pause alignment
    vocals = None
    try:    vocals = fetch_bytes(f"{base}/audio/vocals.mp3")
    except: pass

    # count how many dialogs still need computing
    pending = sum(1 for d in fit
                  if wavs.get(d["index"]) and
                  results.get(str(d["index"]), {}).get("utmos") is None)

    print(f"[{ep_idx}/{len(episodes)}] Episode {ep} — {len(fit)} dialogs, "
          f"{pending} pending, lang={lang_code}", flush=True)

    if pending == 0:
        print(f"  All cached, skipping.", flush=True)
        continue

    for d in fit:
        idx = d["index"]
        did = str(idx)
        if not wavs.get(idx):
            continue

        entry = results.get(did, {})

        # skip if already fully computed
        if all(k in entry for k in ["utmos", "grammar", "blaser", "pause_align"]):
            continue

        try:
            wav_bytes = fetch_bytes(wavs[idx])
        except Exception as e:
            print(f"  Dialog {idx}: could not fetch WAV — {e}", flush=True)
            continue

        text    = d.get("text", "")
        src_d   = src_map.get(idx, {})
        tgt_d   = tgt_map.get(idx, {})
        start_s = tgt_d.get("start_time") or src_d.get("start_time") or 0
        end_s   = tgt_d.get("end_time")   or src_d.get("end_time")   or (start_s + d["target_ms"] / 1000)

        if "utmos" not in entry:
            try:    entry["utmos"] = compute_utmos(wav_bytes)
            except Exception as e: entry["utmos_error"] = str(e)

        if "grammar" not in entry:
            try:    entry["grammar"] = text_naturalness(text)
            except Exception as e: entry["grammar_error"] = str(e)

        if "blaser" not in entry and src_d.get("text"):
            try:
                entry["blaser"] = blaser_qe_text(
                    src_d["text"], text,
                    SONAR_LANG_CODES.get(src_lang, "hin_Deva"),
                    SONAR_LANG_CODES.get(lang_code, "tel_Telu"),
                )
            except Exception as e: entry["blaser_error"] = str(e)

        if "pause_align" not in entry and vocals and start_s < end_s:
            try:
                entry["pause_align"] = pause_alignment_score(
                    vocals, wav_bytes, float(start_s), float(end_s))
            except Exception as e: entry["pause_align_error"] = str(e)

        results[did] = entry
        print(f"  {ep}/{idx:>2}  utmos={fmt_f(entry.get('utmos'))}  "
              f"blaser={fmt_f(entry.get('blaser'))}  "
              f"muril={fmt_f(entry.get('grammar'))}  "
              f"pause={fmt_f(entry.get('pause_align'))}", flush=True)

    # save after every episode
    cache[key] = results
    json.dump(cache, open(CACHE_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"  Saved episode {ep}.\n", flush=True)

print("All episodes done.", flush=True)
