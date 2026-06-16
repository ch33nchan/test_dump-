import streamlit as st
import json, os, re, unicodedata, tempfile, sys
sys.path.insert(0, os.path.dirname(__file__))
from eval_metrics import (
    isochrone_score, pause_alignment_score,
    compute_utmos, blaser_qe_text, text_naturalness,
    verify_voice_assignment, voice_consistency_score,
    extract_emotion_tags, emotion_tag_coverage,
    honorific_check, named_entity_consistency,
    find_script_errors as _find_script_errors_eval,
    WRONG_SCRIPT_BLOCKS_EVAL,
    compute_7cat_score, SONAR_LANG_CODES,
    expected_expansion, flag_expansion, EXPANSION_EXPECTATIONS,
)
from azure.storage.blob import BlobServiceClient
import plotly.graph_objects as go
import pandas as pd
import numpy as np

ACCOUNT_URL    = st.secrets.get("AZURE_ACCOUNT_URL", "https://dashprodstore.blob.core.windows.net")
SAS_TOKEN      = st.secrets.get("AZURE_SAS_TOKEN",   "")
CONTAINER      = st.secrets.get("AZURE_CONTAINER",   "auto-dubbing")
QC_STORE       = os.path.join(os.path.dirname(__file__), "qc_scores.json")
MMS_MODEL      = "facebook/mms-1b-fl102"
PIPELINE_CACHE = os.path.join(os.path.dirname(__file__), "pipeline_cache.json")

# Load pre-baked pipeline data (no Azure needed for display)
@st.cache_resource
def load_pipeline_cache():
    try:
        return json.load(open(PIPELINE_CACHE, encoding="utf-8"))
    except:
        return {}

AZURE_AVAILABLE = bool(SAS_TOKEN)

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
    (0x4E00, 0x9FFF, "Chinese",   "zho"),   # CJK Unified Ideographs
]

# For each target language: which Unicode blocks are WRONG (leak = pipeline bug)
# Keys = tgt_lang_code, values = list of (lo, hi, script_name) blocks to flag
WRONG_SCRIPT_BLOCKS = {
    "tel": [(0x0900,0x097F,"Devanagari"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK Chinese")],
    "tam": [(0x0C00,0x0C7F,"Telugu"),(0x0900,0x097F,"Devanagari"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK Chinese")],
    "ben": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK Chinese")],
    "hin": [(0x0C00,0x0C7F,"Telugu"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK Chinese")],
    "kan": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK Chinese")],
    "mal": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK Chinese")],
    "mar": [(0x0C00,0x0C7F,"Telugu"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK Chinese")],
}

CORRECTION_WEIGHTS = {
    "speaker": 1.0, "translation_major": 0.8,
    "tts_regen": 0.5, "translation_minor": 0.3, "timing_only": 0.2,
}
EQI_WEIGHTS = {"translation": 0.20, "grammar": 0.15, "voice": 0.20,
               "naturalness": 0.15, "timing": 0.20, "clarity": 0.10}

def score_label(s):
    if s is None:  return "N/A",   "#555"
    if s >= 80:    return "GOOD",   "#4caf50"
    if s >= 55:    return "MODERATE","#ff9800"
    if s >= 30:    return "POOR",   "#f44336"
    return                "CRITICAL","#b71c1c"

def fmt(v):
    """Format a score float cleanly: 80.0 not 80.00000000001, integers as int."""
    if v is None: return "—"
    r = round(float(v), 1)
    return str(int(r)) if r == int(r) else str(r)

st.set_page_config(page_title="Dubbing Evals", layout="wide")

# ── azure ─────────────────────────────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def get_container():
    if not AZURE_AVAILABLE:
        return None
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=SAS_TOKEN) \
           .get_container_client(CONTAINER)

def list_episodes(show_id):
    # use local pipeline cache first
    pc = load_pipeline_cache()
    if pc:
        return sorted(pc.keys())
    if not AZURE_AVAILABLE:
        return []
    seen = set()
    prefix = f"shows/dubbing/{show_id}/episodes/"
    for b in get_container().list_blobs(name_starts_with=prefix):
        ep = b.name[len(prefix):].split("/")[0]
        if ep: seen.add(ep)
    return sorted(seen)

@st.cache_data(ttl=300)
def fetch_bytes(path):
    if not AZURE_AVAILABLE:
        return b""
    return get_container().get_blob_client(path).download_blob().readall()

@st.cache_data(ttl=300)
def fetch_json(path):
    return json.loads(fetch_bytes(path))

# ── voice qc ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_voice_qc(voice_id):
    try:    return fetch_json(f"voice_qc/{voice_id}.json")
    except: return {}

# ── show-level overview ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def episode_full_score(show_id, episode):
    """Full EQI from pipeline_cache.json + speech_cache.json. No Azure needed."""
    pc = load_pipeline_cache()
    ep_data = pc.get(episode, {})
    fit_raw  = ep_data.get("fit")
    attempts = ep_data.get("attempts") or {}
    if not fit_raw: return None
    dialogs = fit_raw.get("dialogs", [])
    N = len(dialogs)
    if not N: return None
    if not fit: return None
    dialogs = fit.get("dialogs", [])
    N = len(dialogs)
    if not N: return None

    # load speech cache for this episode
    try:    all_cache = json.load(open(SPEECH_CACHE))
    except: all_cache = {}
    ep_cache = all_cache.get(f"{show_id}:{episode}", {})

    # timing
    timing_pass = sum(1 for d in dialogs
                      if d.get("overflow_ms", 0) <= 0 and d.get("speed_factor", 1) <= 1.4)
    n_overflow  = sum(1 for d in dialogs if d.get("overflow_ms", 0) > 0)
    n_max_rep   = sum(1 for d in dialogs if len(attempts.get(str(d["index"]), [])) >= 4)
    timing_score = round(timing_pass / N * 100, 1)

    # isochronometer
    iso_scores = [max(0, 1 - abs(d["tts_duration_ms"] - d["target_ms"]) / d["target_ms"])
                  for d in dialogs if d.get("target_ms")]
    iso_score = round(np.mean(iso_scores) * 100, 1) if iso_scores else timing_score

    # from speech cache
    utmos_v  = [ep_cache[did]["utmos"]       for did in ep_cache if ep_cache[did].get("utmos")]
    blaser_v = [ep_cache[did]["blaser"]      for did in ep_cache if ep_cache[did].get("blaser")]
    muril_v  = [ep_cache[did]["grammar"]     for did in ep_cache if ep_cache[did].get("grammar")]
    pause_v  = [ep_cache[did]["pause_align"] for did in ep_cache if ep_cache[did].get("pause_align") is not None]

    utmos_score  = round((np.mean(utmos_v) - 1) / 4 * 100, 1) if utmos_v else None
    blaser_score = round((np.mean(blaser_v) - 1) / 4 * 100, 1) if blaser_v else None
    muril_score  = round(float(np.mean(muril_v)), 1) if muril_v else None
    pause_score  = round(float(np.mean(pause_v)), 1) if pause_v else None

    # blend timing: isochron (65%) + pause (35%) if available
    if pause_score is not None:
        full_timing = round(iso_score * 0.65 + pause_score * 100 * 0.35, 1)
    else:
        full_timing = iso_score

    # weighted EQI using available dimensions
    dims = {"timing": (full_timing, 0.20), "naturalness": (utmos_score, 0.15),
            "translation": (blaser_score, 0.20), "grammar": (muril_score, 0.15)}
    avail = {k: (v, w) for k, (v, w) in dims.items() if v is not None}
    total_w = sum(w for _, w in avail.values())
    eqi = round(sum(v * w for v, w in avail.values()) / total_w, 1) if avail else None

    editor_ep  = safe(f"shows/dubbing/{show_id}/editor/episodes/{episode}/transcript.json")
    n_reviewed = len((editor_ep or {}).get("dialogs", {}))
    n_cached   = sum(1 for v in ep_cache.values() if v.get("utmos") and v.get("blaser"))

    return {
        "episode":      episode,
        "N":            N,
        "eqi":          eqi,
        "translation":  blaser_score,
        "naturalness":  utmos_score,
        "grammar":      muril_score,
        "timing":       full_timing,
        "n_overflow":   n_overflow,
        "n_max_rephrase": n_max_rep,
        "n_reviewed":   n_reviewed,
        "n_cached":     n_cached,
    }

@st.cache_data(ttl=60)
def load_show_overview(show_id, episodes):
    rows = []
    for ep in episodes:
        r = episode_full_score(show_id, ep)
        if r: rows.append(r)
    return rows

# ── load full pipeline data ───────────────────────────────────────────────────
@st.cache_data(ttl=120)
def load_pipeline(show_id, episode):
    # read from local pipeline_cache.json first — no Azure needed
    pc       = load_pipeline_cache()
    ep_data  = pc.get(episode, {})
    wav_paths = {}

    if AZURE_AVAILABLE:
        # also list WAV paths from blob for audio playback
        base = f"shows/dubbing/{show_id}/episodes/{episode}/intermediate"
        try:
            for b in get_container().list_blobs(name_starts_with=f"{base}/tts/per_dialog/"):
                fname = b.name.split("/")[-1]
                if fname.endswith(".wav"):
                    wav_paths[fname.replace(".wav", "")] = b.name
        except:
            pass

    return {
        "fit":       ep_data.get("fit"),
        "source":    ep_data.get("source"),
        "target":    ep_data.get("target"),
        "attempts":  ep_data.get("attempts"),
        "editor":    ep_data.get("editor"),
        "speakers":  ep_data.get("speakers", []),
        "wav_paths": wav_paths,
    }

# ── audio quality ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def audio_quality(path):
    if not AZURE_AVAILABLE:
        return None
    import librosa
    ab = fetch_bytes(path)
    if not ab:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(ab); tmp = f.name
    try:
        y, sr  = librosa.load(tmp, sr=None, mono=True)
        rms    = librosa.feature.rms(y=y)[0]
        rms_db = float(20 * np.log10(np.mean(rms) + 1e-8))
        clip   = float(np.mean(np.abs(y) > 0.98) * 100)
        dur    = float(len(y) / sr)
        s      = np.sort(rms)
        noise  = np.mean(s[:max(1, len(s)//4)]) + 1e-8
        speech = np.mean(s[3*len(s)//4:]) + 1e-8
        snr    = float(20 * np.log10(speech / noise))
        return {"dur": round(dur,2), "rms": round(rms_db,1),
                "clip": round(clip,3), "snr": round(snr,1)}
    finally:
        os.unlink(tmp)

# ── ASR ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mms():
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    with st.spinner(f"Loading {MMS_MODEL} (~1.5GB, first run only)…"):
        proc  = AutoProcessor.from_pretrained(MMS_MODEL)
        model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL)
    return proc, model

@st.cache_data(ttl=3600, show_spinner=False)
def run_wer(wav_path, target_text, lang_code):
    import torch, librosa, jiwer
    proc, model = load_mms()
    proc.tokenizer.set_target_lang(lang_code)
    model.load_adapter(lang_code)
    ab = fetch_bytes(wav_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(ab); tmp = f.name
    try:
        y, _  = librosa.load(tmp, sr=16000, mono=True)
        inp   = proc(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inp).logits
        asr  = proc.decode(torch.argmax(logits, dim=-1)[0])
        ref  = re.sub(r'\[.*?\]', '', target_text).strip()
        try:    wer = round(jiwer.wer(ref, asr), 3) if ref and asr else None
        except: wer = None
        return {"asr": asr, "wer": wer}
    finally:
        os.unlink(tmp)

# ── helpers ───────────────────────────────────────────────────────────────────
def detect_lang(texts):
    counts = {}
    for t in texts:
        for ch in t:
            cp = ord(ch)
            for lo, hi, lang, code in SCRIPT_BLOCKS:
                if lo <= cp <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
    if not counts: return "Telugu", "tel"
    best = max(counts, key=counts.get)
    code = next(c for _, _, l, c in SCRIPT_BLOCKS if l == best)
    return best, code

def find_script_errors(text, tgt_lang_code="tel"):
    return _find_script_errors_eval(text, tgt_lang_code)

def hl(text, tgt_lang_code="tel"):
    """Highlight wrong-script characters in red for the given target language."""
    wrong_blocks = WRONG_SCRIPT_BLOCKS.get(tgt_lang_code[:3], [])
    out = []
    for ch in text:
        cp = ord(ch)
        is_wrong = cp >= 0x0080 and any(lo <= cp <= hi for lo, hi, _ in wrong_blocks)
        if is_wrong:
            script = next((n for lo, hi, n in wrong_blocks if lo <= cp <= hi), "wrong script")
            out.append(f'<span style="background:#c62828;color:#fff;border-radius:2px;'
                       f'padding:0 2px" title="{script} U+{cp:04X}">{ch}</span>')
        else:
            out.append(ch)
    return "".join(out)

def badge(val, ok="PASS", warn="WARN", fail="FAIL"):
    c = {ok:"#4caf50", warn:"#ff9800", fail:"#f44336"}.get(val, "#555")
    return f'<span style="background:{c};color:#000;border-radius:3px;padding:1px 7px;font-size:11px;font-weight:700">{val}</span>'

# ── build full episode dialogs ────────────────────────────────────────────────
def build_dialogs(raw, tgt_lang_code="tel"):
    fit_data  = {d["index"]: d for d in (raw.get("fit") or {}).get("dialogs", [])}
    src_data  = {d["index"]: d for d in (raw.get("source") or {}).get("dialogs", [])}
    tgt_data  = {d["index"]: d for d in (raw.get("target") or {}).get("dialogs", [])}
    attempts  = raw.get("attempts") or {}
    editor    = (raw.get("editor") or {}).get("dialogs", {})
    wav_paths = raw.get("wav_paths") or {}

    dialogs = []
    for idx in sorted(fit_data.keys()):
        f   = fit_data[idx]
        s   = src_data.get(idx, {})
        t   = tgt_data.get(idx, {})
        did = str(idx)

        pipeline_text = f.get("text", "")
        editor_text   = editor.get(did, {}).get("current_text")
        was_edited    = editor_text is not None and editor_text != pipeline_text
        editor_info   = editor.get(did, {})

        # timing
        target_ms  = f.get("target_ms", 0)
        tts_ms     = f.get("tts_duration_ms", 0)
        overflow   = f.get("overflow_ms", 0)
        speed      = f.get("speed_factor", 1.0)

        # rephrase count
        n_attempts = len(attempts.get(did, []))

        # timing status
        if overflow > 500:    timing_status = "FAIL"
        elif overflow > 0:    timing_status = "WARN"
        elif speed > 1.4:     timing_status = "WARN"   # maxed out speed
        else:                 timing_status = "PASS"

        # script errors on pipeline output
        script_errors = find_script_errors(pipeline_text, tgt_lang_code)

        dialogs.append({
            "id":              idx,
            "source_text":     s.get("text", ""),
            "pipeline_text":   pipeline_text,
            "editor_text":     editor_text,
            "was_edited":      was_edited,
            "editor_info":     editor_info,
            "target_ms":       target_ms,
            "tts_ms":          tts_ms,
            "overflow_ms":     overflow,
            "speed_factor":    speed,
            "timing_status":   timing_status,
            "n_attempts":      n_attempts,
            "script_errors":   script_errors,
            "wav":             wav_paths.get(did),
            "speaker":         t.get("speaker") or s.get("speaker"),
            "voice_id":        t.get("voice_id"),
            "scene":           t.get("scene"),
        })
    return dialogs

# ── compute EQI ──────────────────────────────────────────────────────────────
def compute_scores(dialogs, aq_results, speech_results,
                   src_lang_code, tgt_lang_code):
    """
    Research-aligned 7-category scorecard.
    speech_results: per-dialog dict with keys:
      utmos, blaser, grammar, voice_check, pause_align
    """
    N = len(dialogs)
    if not N: return {}

    # ── Timing: IsoChronoMeter + pause alignment ───────────────────────────
    iso_ep, iso_per_dialog = isochrone_score(dialogs)
    iso_map = {s["id"]: s for s in iso_per_dialog}

    overflow_dialogs = [d for d in dialogs if d["overflow_ms"] > 0]
    speed_dialogs    = [d for d in dialogs if d["speed_factor"] > 1.4
                        and d["overflow_ms"] <= 0]
    timing_pass      = sum(1 for d in dialogs if d["timing_status"] == "PASS")

    pause_scores = [speech_results.get(str(d["id"]), {}).get("pause_align")
                    for d in dialogs]
    avg_pause    = float(np.mean([p for p in pause_scores if p is not None])) \
                   if any(p is not None for p in pause_scores) else None

    # ── Grammar: MuRIL perplexity + script integrity ───────────────────────
    script_pass   = sum(1 for d in dialogs if not d["script_errors"])
    script_score  = round(script_pass / N * 100, 1)
    heavy_rep     = [d for d in dialogs if d["n_attempts"] >= 4]

    grammar_perp  = [speech_results.get(str(d["id"]), {}).get("grammar")
                     for d in dialogs]
    grammar_vals  = [v for v in grammar_perp if v is not None]
    # blend script integrity (40%) with MuRIL naturalness (60%) if available
    if grammar_vals:
        grammar_score = round(0.4 * script_score + 0.6 * np.mean(grammar_vals), 1)
    else:
        grammar_score = script_score

    # ── Voice: pipeline assignment quality + consistency ───────────────────
    va_scores = [speech_results.get(str(d["id"]), {}).get("voice_check", {})
                 for d in dialogs]
    va_quality = [v.get("quality") for v in va_scores
                  if isinstance(v, dict) and v.get("quality") is not None]
    vc_score, vc_issues = voice_consistency_score(dialogs)
    n_speakers = len({d["speaker"] for d in dialogs if d["speaker"]})

    # ── Naturalness: UTMOS22 ───────────────────────────────────────────────
    utmos_vals = [speech_results.get(str(d["id"]), {}).get("utmos")
                  for d in dialogs]
    utmos_vals = [v for v in utmos_vals if v is not None]

    # ── Emotion tags (informational — not scored, pending Indic TTS model) ──
    _tag_coverage, _tag_per_dialog, _tag_freq = emotion_tag_coverage(dialogs)

    # ── Translation: BLASER-2.0-QE + editor correction blend ──────────────
    reviewed    = [d for d in dialogs if d["editor_info"]]
    n_reviewed  = len(reviewed)
    blaser_vals = [speech_results.get(str(d["id"]), {}).get("blaser")
                   for d in dialogs]
    blaser_vals = [v for v in blaser_vals if v is not None]

    # Per-type editor correction rates (only meaningful over reviewed subset)
    def editor_rate(correction_type):
        if not n_reviewed: return None
        n = sum(1 for d in reviewed
                if correction_type in (d.get("ctypes") or []))
        return n / n_reviewed

    ed_translation = editor_rate("translation")
    ed_timing      = editor_rate("timing")
    ed_speaker     = editor_rate("speaker")

    # ── Indic-specific: honorific consistency ──────────────────────────────
    honor_score, honor_detail = honorific_check(dialogs, tgt_lang_code)

    # ── Indic-specific: named entity consistency ───────────────────────────
    char_names = list({d.get("speaker","").replace("speaker_","") for d in dialogs
                       if d.get("speaker") and "crowd" not in d.get("speaker","")})
    # use Telugu/Indic character name strings from pipeline text
    ne_score, ne_issues = named_entity_consistency(dialogs, char_names)

    # ── Build scorecard ────────────────────────────────────────────────────
    cat = compute_7cat_score(
        iso_score              = iso_ep,
        pause_score            = avg_pause,
        utmos_results          = utmos_vals if utmos_vals else None,
        voice_assignment_scores= va_quality if va_quality else None,
        grammar_scores         = [grammar_score],
        blaser_scores          = blaser_vals if blaser_vals else None,
        emotion_scores         = None,  # TODO: Indic TTS emotion model
        honorific_score        = honor_score,
        editor_translation_rate= ed_translation,
        editor_timing_rate     = ed_timing,
        editor_speaker_rate    = ed_speaker,
        voice_consistency      = vc_score,
        n_speakers             = n_speakers,
    )

    # ── Issues ──────────────────────────────────────────────────────────────
    issues = []
    lo, hi = expected_expansion(src_lang_code, tgt_lang_code)
    for d in sorted(overflow_dialogs, key=lambda x: -x["overflow_ms"]):
        sev = "HIGH" if d["overflow_ms"] > 500 else "MEDIUM"
        issues.append({"sev": sev, "dim": "Timing",
            "msg": f"Dialog {d['id']}: overflows {d['overflow_ms']}ms after max rephrasing. "
                   f"Expected expansion for {src_lang_code}→{tgt_lang_code}: {lo:.1f}–{hi:.1f}x."})
    for d in speed_dialogs:
        flag, reason = flag_expansion(d["tts_ms"] / max(d["target_ms"],1),
                                      src_lang_code, tgt_lang_code)
        issues.append({"sev": "MEDIUM", "dim": "Timing",
            "msg": f"Dialog {d['id']}: speed {d['speed_factor']:.2f}x. {reason}."})
    if heavy_rep:
        issues.append({"sev": "MEDIUM", "dim": "Grammar",
            "msg": f"{len(heavy_rep)}/{N} dialogs hit max rephrase (4 attempts) — "
                   f"translation consistently too long for {src_lang_code}→{tgt_lang_code} expansion."})
    for d in dialogs:
        if d["script_errors"]:
            issues.append({"sev": "HIGH", "dim": "Grammar",
                "msg": f"Dialog {d['id']}: wrong-script chars — {', '.join(d['script_errors'][:2])}. "
                       f"Source ({src_lang_code}) bleeding into target ({tgt_lang_code}) output."})
    if vc_issues:
        issues.append({"sev": "MEDIUM", "dim": "Voice",
            "msg": f"{len(vc_issues)} speaker(s) use inconsistent voices across dialogs — "
                   f"may break character identity: {', '.join(sp for sp,_ in vc_issues[:3])}."})
    if n_reviewed < N:
        issues.append({"sev": "INFO", "dim": "Coverage",
            "msg": f"Editor corrections available for {n_reviewed}/{N} dialogs. "
                   f"Translation score partial — {N-n_reviewed} dialogs unreviewed."})
    if not blaser_vals:
        issues.append({"sev": "INFO", "dim": "Translation",
            "msg": "BLASER-2.0-QE not computed yet. Run Speech Quality tab."})
    if not utmos_vals:
        issues.append({"sev": "INFO", "dim": "Naturalness",
            "msg": "UTMOS22 not computed yet. Run Speech Quality tab."})
    issues.sort(key=lambda x: {"HIGH":0,"MEDIUM":1,"INFO":2}[x["sev"]])

    return {
        **cat, "N": N,
        "timing_pass": timing_pass,
        "n_overflow":  len(overflow_dialogs),
        "n_heavy_rephrase": len(heavy_rep),
        "n_reviewed":  n_reviewed,
        "n_corrected": sum(1 for d in reviewed if d["was_edited"]),
        "iso_per_dialog": iso_map,
        "issues": issues,
        "src_lang": src_lang_code, "tgt_lang": tgt_lang_code,
    }

def load_qc(sid, ep):
    try:    return json.load(open(QC_STORE)).get(f"{sid}:{ep}", {})
    except: return {}

def save_qc(sid, ep, qc):
    try:
        all_d = json.load(open(QC_STORE))
    except:
        all_d = {}
    all_d[f"{sid}:{ep}"] = qc
    json.dump(all_d, open(QC_STORE, "w"), indent=2, ensure_ascii=False)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dubbing Evals")
    show_id = st.text_input("Show ID", "2bcdfe58")
    if show_id:
        with st.spinner(""):
            try:    eps = list_episodes(show_id)
            except: eps = []
        episode = st.selectbox("Episode", eps or ["001"])
    else:
        episode = "001"

if not (show_id and episode): st.stop()

# ── load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading pipeline output…"):
    raw     = load_pipeline(show_id, episode)
    # detect target language first from translated text, then build with it
    _tgt_dialogs = (raw.get("fit") or {}).get("dialogs", [])
    _tgt_texts   = [d.get("text","") for d in _tgt_dialogs if d.get("text")]
    lang_name, lang_code = detect_lang(_tgt_texts) if _tgt_texts else ("Telugu","tel")
    dialogs = build_dialogs(raw, tgt_lang_code=lang_code)

aq_results = {}
for d in dialogs:
    if d["wav"]:
        try:    aq_results[str(d["id"])] = audio_quality(d["wav"])
        except: pass

SPEECH_CACHE = os.path.join(os.path.dirname(__file__), "speech_cache.json")

def load_speech_cache(sid, ep):
    key = f"{sid}:{ep}"
    try:
        cached = json.load(open(SPEECH_CACHE)).get(key, {})
    except:
        cached = {}
    # merge with session state (session state takes priority — newer runs override cache)
    session = st.session_state.get(f"speech_{sid}_{ep}", {})
    merged = {**cached, **session}
    return merged

def save_speech_cache(sid, ep, results):
    key = f"{sid}:{ep}"
    try:    all_data = json.load(open(SPEECH_CACHE))
    except: all_data = {}
    # merge new results into existing cache entry
    existing = all_data.get(key, {})
    for did, entry in results.items():
        existing.setdefault(did, {}).update(entry)
    all_data[key] = existing
    json.dump(all_data, open(SPEECH_CACHE,"w"), indent=2)

speech_results = load_speech_cache(show_id, episode)
qc_data        = load_qc(show_id, episode)

src_lang_name, src_lang_code = detect_lang(
    [d["source_text"] for d in dialogs if d["source_text"]])
scores = compute_scores(dialogs, aq_results, speech_results,
                        src_lang_code, lang_code)

# friendly source language name from actual content
src_lang = src_lang_name

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## Show `{show_id}` — Episode `{episode}`")
st.caption(
    f"Source language: **{src_lang}** — Target language (auto-detected): **{lang_name}** (`{lang_code}`)  "
    f"·  **{scores['N']} dialogs** (full episode pipeline output)  "
    f"·  Editor reviewed {scores['n_reviewed']}/{scores['N']} dialogs"
)

# ── Overall score ─────────────────────────────────────────────────────────────
overall = scores.get("overall")
olbl, oclr = score_label(overall)
lo_exp, hi_exp = expected_expansion(src_lang_code, lang_code)
st.markdown(
    f'<div style="background:#1a1a1a;border-radius:10px;padding:20px 28px;margin-bottom:12px">'
    f'<span style="font-size:12px;color:#888;text-transform:uppercase;letter-spacing:1px">Overall Dubbing Quality</span><br>'
    f'<span style="font-size:52px;font-weight:700;color:{oclr}">{fmt(overall)}</span>'
    f'<span style="font-size:18px;color:#666">/100</span>'
    f'&nbsp;&nbsp;<span style="background:{oclr};color:#000;padding:3px 12px;border-radius:16px;'
    f'font-size:13px;font-weight:600">{olbl}</span>'
    f'<span style="color:#555;font-size:12px;margin-left:20px">'
    f'{src_lang} → {lang_name}  ·  '
    f'expected expansion {lo_exp:.1f}–{hi_exp:.1f}x  ·  '
    f'{scores["timing_pass"]}/{scores["N"]} fit timing  ·  '
    f'{scores["n_overflow"]} overflow  ·  '
    f'editor reviewed {scores["n_reviewed"]}/{scores["N"]}'
    f'</span></div>',
    unsafe_allow_html=True
)

# 7 category cards — Streamlit native st.metric with help= for proper hover tooltips
CATEGORY_HELP = {
    "translation": """\
**BLASER-2.0-QE** — Weight: 20%

Meta's translation quality estimator using the SONAR multilingual encoder.
Source and dubbed texts are embedded into the same semantic space,
then a classifier trained on professional translator judgments scores quality on a 1–5 scale.

The final score blends 65% BLASER with 35% from the editor correction signal
(how often editors had to fix translations in reviewed dialogs).

**Scale:** 1 = wrong meaning · 3 = acceptable · 5 = perfect
**Good:** above 3.5 · **Poor:** below 2.5
""",
    "grammar": """\
**MuRIL pseudo-perplexity** — Weight: 15%

Google's MuRIL model (trained on 17 Indic languages) evaluates text naturalness.
Each token in the dubbed text is masked one at a time. The model tries to predict
it back — low confidence means the text is unnatural or grammatically unusual.

Also factors in honorific/register consistency: same character should use
consistent formality (formal/informal) throughout the episode.

**Scale:** 0–100 · **Good:** above 60 · **Note:** drama dialogue naturally scores lower than formal text
""",
    "voice": """\
**Voice assignment verification** — Weight: 20%

Checks that the ElevenLabs voice assigned to each character is appropriate
for their gender, age, and role.

Data sources:
- Character metadata from `episode_refined_speakers.json` (gender, age group, role)
- Voice QC verdicts from `voice_qc/{voice_id}.json` (voice gender, age, tier)

Also checks that the same character uses exactly one voice ID
throughout the episode — voice swapping mid-episode breaks character identity.

**Scale:** 0–100 · **Good:** 100 (no mismatches)
""",
    "naturalness": """\
**UTMOS22** — Weight: 15%

UTokyo-SaruLab MOS predictor, trained specifically on human naturalness
ratings collected from TTS systems. Returns a 1–5 MOS score per dialog,
then averaged across the episode.

Chosen over DNSMOS because DNSMOS was designed for speech enhancement
(denoising), not TTS quality — it gives unreliable results on clean synthesised speech.

**Scale:** 1–5 · **Good:** above 3.5 · **Poor:** below 2.5
**Source:** Baba et al. 2024, UTokyo-SaruLab
""",
    "timing": """\
**IsoChronoMeter + pause alignment** — Weight: 20%

**IsoChronoMeter** (WMT 2024): for each dialog, measures how well the dubbed
audio fits the original shot duration. A dialog that fits perfectly scores 1.0.
A dialog that overflows or is very short scores proportionally lower.
Averaged across all dialogs in the episode.

**Pause alignment**: detects silence positions in the source vocals and the
dubbed audio, then checks how closely they match. Pauses falling in the same
places means the dubbed speech breathes naturally with the original performance.

Also blends in the editor timing correction signal from reviewed dialogs.

**Scale:** 0–100 · **Good:** above 80
""",
    "clarity": """\
**Clarity proxy** — Weight: 10%

Uses the UTMOS22 naturalness score as a proxy for clarity.
For clean TTS output (no background noise, no studio artifacts),
naturalness and clarity are highly correlated.

A dedicated clarity model exists (DNSMOS SIG/BAK from Microsoft)
but it was designed for denoising tasks and gives unreliable results
on synthesised speech — so UTMOS serves as the proxy until a better
Indic TTS clarity model is available.

**Scale:** 0–100 · mirrors the Naturalness score
""",
}

cats = [
    ("translation", "Translation",  "BLASER-2.0-QE",    "20%"),
    ("grammar",     "Grammar",      "MuRIL",             "15%"),
    ("voice",       "Voice",        "Assignment check",  "20%"),
    ("naturalness", "Naturalness",  "UTMOS22",           "15%"),
    ("timing",      "Timing",       "IsoChronoMeter",    "20%"),
    ("clarity",     "Clarity",      "UTMOS proxy",       "10%"),
]
cols = st.columns(6)
for col, (key, label, sublabel, weight) in zip(cols, cats):
    v         = scores.get(key)
    lbl, clr  = score_label(v)
    col.metric(
        label=f"{label} ({weight})",
        value=fmt(v) if v is not None else "N/A",
        delta=f"{lbl} · {sublabel}",
        delta_color="off",
        help=CATEGORY_HELP.get(key, ""),
    )

st.write("")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_issues, tab_dialogs, tab_voice, tab_speech, tab_hqc = st.tabs([
    "Show Overview", "Issues & Recommendations", "All Dialogs",
    "Voice Assignment", "Speech Quality", "Human QC"
])

# ════════════════════════════════════════════════════════════════════════════════
# SHOW OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.caption(
        "Lightweight EQI across all episodes — computed from timing, rephrase iterations, "
        "and script integrity only (no audio). Click an episode in the table to load it."
    )
    with st.spinner("Computing EQI for all episodes…"):
        overview_rows = load_show_overview(show_id, eps or [episode])

    if overview_rows:
        df_ov = pd.DataFrame(overview_rows)
        df_ov = df_ov.rename(columns={
            "episode":"Episode", "N":"Dialogs", "eqi":"EQI",
            "translation":"BLASER", "naturalness":"UTMOS",
            "grammar":"MuRIL", "timing":"Timing",
            "n_overflow":"Overflows", "n_max_rephrase":"Max Rephrase",
            "n_reviewed":"Editor Reviewed", "n_cached":"Computed",
        })

        def color_eqi(v):
            try:    v = float(v)
            except: return ""
            if v >= 80:  return "background-color:#1a3a1a;color:#4caf50"
            if v >= 55:  return "background-color:#3a2a00;color:#ff9800"
            if v >= 30:  return "background-color:#3a1a1a;color:#f44336"
            return              "background-color:#2a0a0a;color:#b71c1c"

        score_cols = ["EQI", "BLASER", "UTMOS", "MuRIL", "Timing"]
        display_df = df_ov[["Episode","Dialogs","EQI","BLASER","UTMOS","MuRIL","Timing",
                             "Overflows","Max Rephrase","Editor Reviewed","Computed"]].copy()
        for col in score_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: fmt(x) if pd.notnull(x) else "—")

        col_cfg = {
            "Episode": st.column_config.TextColumn(
                "Episode",
                help="Episode number within the show."),
            "Dialogs": st.column_config.NumberColumn(
                "Dialogs",
                help="Total dialog segments in this episode (from the pipeline's full output)."),
            "EQI": st.column_config.TextColumn(
                "EQI",
                help="Episode Quality Index — weighted average of BLASER (20%), "
                     "UTMOS (20%), MuRIL (15%), Timing (20%). "
                     "Voice and Clarity excluded here as they require per-dialog computation. "
                     "Scale: 0–100. Good: above 70."),
            "BLASER": st.column_config.TextColumn(
                "BLASER",
                help="BLASER-2.0-QE translation quality (0–100). "
                     "Meta SONAR encoder embeds source Hindi/Chinese text and dubbed "
                     "Indic text into a shared semantic space — a trained classifier "
                     "scores quality calibrated against professional translator judgments. "
                     "Raw score is 1–5, converted to 0–100 here."),
            "UTMOS": st.column_config.TextColumn(
                "UTMOS",
                help="UTMOS22 naturalness MOS (0–100). "
                     "UTokyo-SaruLab neural MOS predictor trained specifically on "
                     "human naturalness ratings for TTS output. "
                     "Raw score is 1–5, converted to 0–100 here. "
                     "Good: above 62 (raw 3.5). Poor: below 37 (raw 2.5)."),
            "MuRIL": st.column_config.TextColumn(
                "MuRIL",
                help="MuRIL pseudo-perplexity naturalness score (0–100). "
                     "Google MuRIL model (17 Indic languages) — masks each token in "
                     "the dubbed text and measures prediction confidence. "
                     "Low confidence = unnatural text. Drama dialogue naturally scores "
                     "lower than formal text, so use this for comparison across episodes "
                     "rather than absolute judgement."),
            "Timing": st.column_config.TextColumn(
                "Timing",
                help="IsoChronoMeter + pause alignment (0–100). "
                     "IsoChronoMeter (WMT 2024): measures how well each dubbed dialog "
                     "fits the original shot duration. Blended with pause alignment — "
                     "how closely silence positions in the dubbed audio match the source. "
                     "Good: above 80."),
            "Overflows": st.column_config.NumberColumn(
                "Overflows",
                help="Number of dialogs where the dubbed audio still exceeds the source "
                     "window after all rephrasing and speed adjustment. "
                     "These dialogs will bleed into the next scene in the final video."),
            "Max Rephrase": st.column_config.NumberColumn(
                "Max Rephrase",
                help="Dialogs that hit the maximum 4 rephrase attempts. "
                     "When translated text is too long for the source window, the pipeline "
                     "shortens it and retries TTS — up to 4 times. Hitting the limit "
                     "means the translation model is consistently producing text that is "
                     "too verbose for this language pair's expansion ratio."),
            "Editor Reviewed": st.column_config.NumberColumn(
                "Editor Reviewed",
                help="Number of dialogs a human editor opened in the Dub Sync tool. "
                     "Editors typically review the most problematic dialogs, so this is "
                     "a biased sample — not a random coverage metric."),
            "Computed": st.column_config.NumberColumn(
                "Computed",
                help="Dialogs with all 4 speech metrics computed (UTMOS + BLASER + "
                     "MuRIL + pause alignment). Scores for episodes with partial "
                     "coverage are averages over the computed subset."),
        }
        st.dataframe(
            display_df.style.map(color_eqi, subset=["EQI"]),
            column_config=col_cfg,
            use_container_width=True, hide_index=True, height=700
        )

        # EQI trend chart
        eps_sorted = [r["episode"] for r in overview_rows]
        eqi_vals   = [r["eqi"] for r in overview_rows]
        clrs       = [score_label(v)[1] for v in eqi_vals]
        fig_ov = go.Figure()
        fig_ov.add_scatter(x=eps_sorted, y=eqi_vals, mode="lines+markers",
                           line=dict(color="#4caf50", width=2),
                           marker=dict(color=clrs, size=10))
        fig_ov.add_hline(y=70, line_dash="dash", line_color="#555",
                         annotation_text="Good (70)")
        fig_ov.add_hline(y=55, line_dash="dot", line_color="#ff9800",
                         annotation_text="Moderate (55)")
        fig_ov.update_layout(
            title=f"EQI trend — all {len(eps_sorted)} episodes",
            xaxis_title="Episode", yaxis=dict(range=[0,105], title="EQI"),
            height=340, margin=dict(t=40,b=0),
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
        )
        st.plotly_chart(fig_ov, use_container_width=True)

        # Overflow + rephrase heatmap across episodes
        col_ov, col_rep = st.columns(2)
        with col_ov:
            fig_of = go.Figure(go.Bar(
                x=eps_sorted,
                y=[r["n_overflow"] for r in overview_rows],
                marker_color=["#f44336" if r["n_overflow"]>0 else "#4caf50"
                              for r in overview_rows],
                text=[r["n_overflow"] for r in overview_rows],
                textposition="outside",
            ))
            fig_of.update_layout(
                title="Overflow dialogs per episode",
                xaxis_title="Episode", yaxis_title="Count",
                height=280, margin=dict(t=40,b=0),
                plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
            )
            st.plotly_chart(fig_of, use_container_width=True)
        with col_rep:
            fig_rp = go.Figure(go.Bar(
                x=eps_sorted,
                y=[r["n_max_rephrase"] for r in overview_rows],
                marker_color=["#f44336" if r["n_max_rephrase"] > r["N"]*0.5 else "#ff9800"
                              for r in overview_rows],
                text=[r["n_max_rephrase"] for r in overview_rows],
                textposition="outside",
            ))
            fig_rp.update_layout(
                title="Dialogs hitting max rephrase (4 attempts) per episode",
                xaxis_title="Episode", yaxis_title="Count",
                height=280, margin=dict(t=40,b=0),
                plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
            )
            st.plotly_chart(fig_rp, use_container_width=True)
    else:
        st.warning("Could not load episode data for show overview.")

# ════════════════════════════════════════════════════════════════════════════════
# ISSUES
# ════════════════════════════════════════════════════════════════════════════════
with tab_issues:
    with st.expander("Scoring reference"):
        st.markdown("""
**7-Category scorecard** (Dubbing Rubric–aligned)

| Category | Weight | Metric |
|---|---|---|
| Translation | 20% | BLASER-2.0-QE / editor correction rate |
| Grammar | 15% | MuRIL pseudo-perplexity + script integrity |
| Voice | 20% | Pipeline assignment quality + consistency |
| Naturalness | 15% | UTMOS22 MOS (1–5) |
| Timing | 20% | IsoChronoMeter + pause alignment |
| Clarity | 10% | UTMOS clarity component |
| Multispeaker | 10% | Voice consistency across speakers |

**Indic expansion expectations:** Hindi→Telugu 1.1–1.5x · Chinese→Telugu 1.5–2.5x · Hindi→Tamil 1.15–1.55x
""")

    for sev_label, sev_key in [("Critical issues","HIGH"),("Warnings","MEDIUM"),("Info","INFO")]:
        items = [i for i in scores["issues"] if i["sev"] == sev_key]
        if not items: continue
        st.markdown(f"#### {sev_label}")
        for i in items:
            fn = st.error if sev_key=="HIGH" else (st.warning if sev_key=="MEDIUM" else st.info)
            fn(f"**[{i['dim']}]** {i['msg']}")

    st.divider()
    st.markdown("#### Score breakdown")
    col_bar, col_donut = st.columns(2)

    with col_bar:
        dim_keys = ["translation","grammar","voice","naturalness","timing","clarity"]
        dim_labels = ["Translation","Grammar","Voice","Naturalness","Timing","Clarity"]
        dim_vals = [scores.get(k) for k in dim_keys]
        dim_vals_plot = [v if v is not None else 0 for v in dim_vals]
        clrs  = [score_label(v)[1] for v in dim_vals]
        text  = [fmt(v) if v is not None else "N/A" for v in dim_vals]
        fig = go.Figure(go.Bar(x=dim_labels, y=dim_vals_plot, marker_color=clrs,
                               text=text, textposition="outside"))
        fig.add_hline(y=70, line_dash="dash", line_color="#555", annotation_text="Good (70)")
        fig.update_layout(title="7-category scores",
                          yaxis=dict(range=[0,115]), height=300,
                          margin=dict(t=40,b=0), plot_bgcolor="#111",
                          paper_bgcolor="#111", font_color="#ccc")
        st.plotly_chart(fig, use_container_width=True)

    with col_donut:
        t_pass = scores["timing_pass"]
        t_warn = sum(1 for d in dialogs if d["speed_factor"]>1.4 and d["overflow_ms"]<=0)
        t_fail = scores["n_overflow"]
        fig2 = go.Figure(go.Pie(
            labels=["Fits timing","Sped up >1.4x","Overflows"],
            values=[t_pass, t_warn, t_fail], hole=0.55,
            marker_colors=["#4caf50","#ff9800","#f44336"]
        ))
        fig2.update_layout(title="Timing breakdown — all 20 dialogs",
                           height=300, margin=dict(t=40,b=0),
                           plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc")
        st.plotly_chart(fig2, use_container_width=True)

    # timing detail chart
    st.markdown("#### Timing — all 20 dialogs")
    ids      = [str(d["id"]) for d in dialogs]
    windows  = [d["target_ms"] for d in dialogs]
    tts_durs = [d["tts_ms"] for d in dialogs]
    clrs2    = ["#f44336" if d["overflow_ms"]>0 else ("#ff9800" if d["speed_factor"]>1.4 else "#4caf50")
                for d in dialogs]
    fig3 = go.Figure()
    fig3.add_bar(name="Source window (ms)", x=ids, y=windows, marker_color="#4caf50")
    fig3.add_bar(name="TTS raw (ms)",       x=ids, y=tts_durs, marker_color=clrs2)
    fig3.update_layout(barmode="group", title="Source window vs raw TTS duration per dialog",
                       xaxis_title="Dialog", yaxis_title="ms", height=320,
                       margin=dict(t=40,b=0), plot_bgcolor="#111",
                       paper_bgcolor="#111", font_color="#ccc")
    st.plotly_chart(fig3, use_container_width=True)

    # rephrase heatmap
    st.markdown("#### Rephrase attempts — dialogs that needed multiple iterations to fit")
    rep_ids  = [str(d["id"]) for d in dialogs]
    rep_vals = [d["n_attempts"] for d in dialogs]
    rep_clrs = ["#f44336" if v>=4 else ("#ff9800" if v>=2 else "#4caf50") for v in rep_vals]
    fig4 = go.Figure(go.Bar(x=rep_ids, y=rep_vals, marker_color=rep_clrs,
                             text=rep_vals, textposition="outside"))
    fig4.add_hline(y=4, line_dash="dash", line_color="#f44336", annotation_text="Max attempts (4)")
    fig4.update_layout(title="Rephrase iterations per dialog (0 = fit first try, 4 = max)",
                       xaxis_title="Dialog", yaxis_title="Attempts", height=280,
                       margin=dict(t=40,b=0), plot_bgcolor="#111",
                       paper_bgcolor="#111", font_color="#ccc")
    st.plotly_chart(fig4, use_container_width=True)

    if aq_results:
        st.markdown("#### Audio quality — all dialogs")
        aq_rows = []
        for d in dialogs:
            aq = aq_results.get(str(d["id"]))
            if not aq: continue
            status = "FAIL" if aq["snr"]<8 or aq["clip"]>0.5 else ("WARN" if aq["snr"]<15 else "OK")
            aq_rows.append({"Dialog":d["id"],"Duration(s)":aq["dur"],
                            "Loudness(dBFS)":aq["rms"],"Clipping(%)":aq["clip"],
                            "SNR(dB)":aq["snr"],"Status":status})
        df_aq = pd.DataFrame(aq_rows)
        for col in ["Duration(s)","Loudness(dBFS)","Clipping(%)","SNR(dB)"]:
            df_aq[col] = df_aq[col].apply(lambda x: fmt(x) if pd.notnull(x) else "—")
        def caq(v):
            if str(v)=="FAIL": return "background:#3a1a1a;color:#f44336"
            if str(v)=="WARN": return "background:#3a2a00;color:#ff9800"
            if str(v)=="OK":   return "background:#1a3a1a;color:#4caf50"
            return ""
        st.dataframe(df_aq.style.map(caq, subset=["Status"]),
                     use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# ALL DIALOGS
# ════════════════════════════════════════════════════════════════════════════════
with tab_dialogs:
    st.caption(
        f"All {scores['N']} dialogs from the full pipeline output. "
        f"Dialogs marked [edited] were reviewed and corrected by the editor. "
        f"The other {scores['N'] - scores['n_reviewed']} have not been reviewed — "
        f"pipeline text shown as-is."
    )

    for d in dialogs:
        aq  = aq_results.get(str(d["id"]))
        wr  = speech_results.get(str(d["id"]), {})

        t_badge = badge(d["timing_status"])
        a_badge = badge(
            "FAIL" if aq and (aq["snr"]<8 or aq["clip"]>0.5) else
            ("WARN" if aq and (aq["snr"]<15) else ("OK" if aq else "N/A"))
        )
        s_badge = badge("FAIL" if d["script_errors"] else "OK")
        r_badge = badge(
            "FAIL" if d["n_attempts"]>=4 else ("WARN" if d["n_attempts"]>=2 else "OK")
        )
        ed_note = "  [edited]" if d["was_edited"] else ""
        ov_note = f"  overflow +{d['overflow_ms']}ms" if d["overflow_ms"]>0 else ""
        sp_note = f"  speed {d['speed_factor']:.2f}x" if d["speed_factor"]>1.3 else ""

        with st.expander(
            f"Dialog {d['id']}{ed_note}{ov_note}{sp_note}",
            expanded=(d["overflow_ms"]>0 or d["script_errors"] or d["was_edited"])
        ):
            st.markdown(
                f"Timing: {t_badge}&nbsp;&nbsp;Audio: {a_badge}&nbsp;&nbsp;"
                f"Script: {s_badge}&nbsp;&nbsp;TTS Fit: {r_badge}",
                unsafe_allow_html=True
            )
            st.write("")

            cl, cr = st.columns([3,2])
            with cl:
                if d["source_text"]:
                    st.markdown("**Source text**")
                    st.markdown(
                        f'<div style="background:#0d1117;border-radius:6px;padding:10px;'
                        f'font-size:13px;color:#8b949e;line-height:1.6">{d["source_text"]}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("**Pipeline output**")
                st.markdown(
                    f'<div style="background:#111;border-radius:6px;padding:10px;'
                    f'font-size:14px;line-height:1.7">{hl(d["pipeline_text"])}</div>',
                    unsafe_allow_html=True
                )

                if d["was_edited"] and d["editor_text"]:
                    st.markdown("**Editor correction**")
                    st.markdown(
                        f'<div style="background:#0a1f0a;border:1px solid #2a4a2a;border-radius:6px;'
                        f'padding:10px;font-size:14px;line-height:1.7">{hl(d["editor_text"])}</div>',
                        unsafe_allow_html=True
                    )

                if d["script_errors"]:
                    for err in d["script_errors"]:
                        st.error(f"Script error in pipeline output: {err}")

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Source window", f"{d['target_ms']}ms",
                           help="The original shot duration in milliseconds — "
                                "how long the source actor spoke in the video.")
                tc2.metric("TTS raw",       f"{d['tts_ms']}ms",
                           help="Raw TTS audio duration before any speed adjustment. "
                                "If longer than the source window, the pipeline rephrases and retries.")
                tc3.metric("Overflow",      f"{d['overflow_ms']}ms",
                           delta_color="inverse" if d["overflow_ms"]>0 else "normal",
                           help="How much the final TTS exceeds the source window after all "
                                "rephrasing and speed adjustment. Positive = bleeds into next scene. "
                                "Negative = fits comfortably.")
                tc4.metric("Speed",         f"{d['speed_factor']:.2f}x",
                           delta_color="inverse" if d["speed_factor"]>1.3 else "normal",
                           help="Playback speed applied to the TTS audio to fit the window. "
                                "1.0 = no change. 1.5 = maximum allowed (sounds rushed). "
                                "Expected for Hindi→Telugu: 1.1–1.5x.")

                if d["n_attempts"] > 0:
                    st.caption(f"Needed {d['n_attempts']} rephrase attempt(s) to fit timing.")

                if aq:
                    ac1, ac2, ac3, ac4 = st.columns(4)
                    ac1.metric("Duration",  f"{aq['dur']}s",
                               help="TTS audio clip length in seconds.")
                    ac2.metric("Loudness",  f"{aq['rms']} dBFS",
                               help="RMS loudness in dBFS (decibels relative to full scale). "
                                    "Typical broadcast level is −20 to −25 dBFS. "
                                    "Very low values (< −35) indicate the clip is too quiet.")
                    ac3.metric("Clipping",  f"{aq['clip']}%",
                               help="% of audio samples that exceed 0.98 amplitude. "
                                    "Any clipping causes audible distortion. Should be 0%.")
                    ac4.metric("SNR",       f"{aq['snr']} dB",
                               help="Signal-to-Noise Ratio estimated from RMS distribution "
                                    "(top 25% frames = speech, bottom 25% = noise). "
                                    "Good: > 15 dB. Poor: < 8 dB.")

                if wr:
                    sc1, sc2, sc3 = st.columns(3)
                    if wr.get("utmos") is not None:
                        sc1.metric("UTMOS22", fmt(wr["utmos"]),
                                   help="UTokyo-SaruLab MOS Prediction (1–5). "
                                        "Trained on human naturalness ratings for TTS output — "
                                        "much more accurate than DNSMOS for synthesised speech. "
                                        "Good: > 3.5. Poor: < 2.5.")
                    if wr.get("blaser") is not None:
                        sc2.metric("BLASER-2.0-QE", fmt(wr["blaser"]),
                                   help="Meta's translation quality estimator (1–5). "
                                        "Uses SONAR multilingual encoder to embed source and dubbed text "
                                        "into a shared semantic space, then a trained classifier scores quality. "
                                        "Trained on professional translator judgments. No ASR step — "
                                        "compares meaning directly. Good: > 3.5.")
                    if wr.get("pause_align") is not None:
                        sc3.metric("Pause alignment", fmt(wr["pause_align"]),
                                   help="Pause structure preservation (0–1). "
                                        "Detects silence positions in source vocals and dubbed audio, "
                                        "normalises to fraction of total duration, "
                                        "then scores how closely pause positions match. "
                                        "1.0 = pauses fall in exactly the same places. "
                                        "Good: > 0.7.")

            with cr:
                if d["wav"]:
                    st.markdown("**Pipeline TTS audio**")
                    try:
                        st.audio(fetch_bytes(d["wav"]), format="audio/wav")
                    except Exception as e:
                        st.error(str(e))
                st.markdown("**Metadata**")
                st.json({k:v for k,v in {
                    "dialog_id": d["id"],
                    "speaker":   d["speaker"],
                    "voice_id":  d["voice_id"],
                    "scene":     d["scene"],
                    "speed_factor": round(d["speed_factor"],3),
                    "n_rephrase_attempts": d["n_attempts"],
                    "editor_reviewed": d["was_edited"],
                }.items() if v is not None})

# ════════════════════════════════════════════════════════════════════════════════
# VOICE ASSIGNMENT
# ════════════════════════════════════════════════════════════════════════════════
with tab_voice:
    st.caption(
        "Voice QC verdicts for every voice used in this episode, pulled from the voice library. "
        "Checks: speaker consistency (same character always uses the same voice), "
        "and whether the voice gender/age matches the character context."
    )

    # collect unique voice IDs and load their QC data
    voice_ids = list({d["voice_id"] for d in dialogs if d["voice_id"]})
    qc_map    = {}
    with st.spinner("Loading voice QC data…"):
        for vid in voice_ids:
            qc_map[vid] = load_voice_qc(vid)

    # per-dialog voice table
    st.markdown("#### Voice used per dialog")
    vrows = []
    for d in dialogs:
        vid = d["voice_id"]
        qc  = qc_map.get(vid) if vid else None
        vrows.append({
            "Dialog":      d["id"],
            "Speaker":     d["speaker"] or "—",
            "Voice ID":    vid or "—",
            "Voice name":  (qc or {}).get("name", "unknown")[:40] if qc else "—",
            "Gender":      (qc or {}).get("verdict", {}).get("gender", "—"),
            "Age":         (qc or {}).get("verdict", {}).get("age", "—"),
            "Expressiveness": (qc or {}).get("verdict", {}).get("expressiveness", "—"),
            "Tier":        (qc or {}).get("verdict", {}).get("ranking_tier", "—"),
        })
    df_v = pd.DataFrame(vrows)
    st.dataframe(df_v, use_container_width=True, hide_index=True)

    # speaker consistency check: same speaker should always use same voice
    st.markdown("#### Speaker consistency")
    speaker_voices = {}
    for d in dialogs:
        sp  = d["speaker"]
        vid = d["voice_id"]
        if sp and vid:
            speaker_voices.setdefault(sp, set()).add(vid)

    consistency_issues = []
    for sp, vids in speaker_voices.items():
        if len(vids) > 1:
            consistency_issues.append((sp, vids))

    if consistency_issues:
        for sp, vids in consistency_issues:
            names = [qc_map.get(v, {}).get("name", v) for v in vids if qc_map.get(v)]
            st.error(
                f"Speaker `{sp}` uses {len(vids)} different voices across dialogs: "
                + " / ".join(f"`{v[:12]}…` ({n[:30]})" for v, n in zip(vids, names))
            )
    else:
        st.success("All speakers use a consistent voice across dialogs.")

    # voice QC cards
    st.markdown("#### Voice library cards")
    for vid in voice_ids:
        qc = qc_map.get(vid)
        if not qc:
            st.warning(f"No QC data for voice `{vid}`")
            continue
        v = qc.get("verdict", {})
        dialogs_using = [d["id"] for d in dialogs if d["voice_id"] == vid]
        with st.expander(f"{qc.get('name','unknown')}  —  `{vid}`  —  dialogs {dialogs_using}"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Gender",        v.get("gender","—"))
            c2.metric("Age",           v.get("age","—"))
            c3.metric("Pitch",         v.get("pitch","—"))
            c4.metric("Expressiveness",v.get("expressiveness","—"))
            c5.metric("Tier",          v.get("ranking_tier","—"))

            # flag potential gender/age mismatches based on dialog content
            for did in dialogs_using:
                d = next(x for x in dialogs if x["id"] == did)
                text = d.get("source_text","") + " " + d.get("pipeline_text","")
                flags = []
                # female pronouns/markers in source while voice is male
                female_markers = ["అమ్మ","అక్క","didi","sister","she","her","akka"]
                male_markers   = ["అన్నయ్య","bhai","brother","he","his","anna"]
                text_lower = text.lower()
                if v.get("gender") == "male" and any(m in text_lower for m in female_markers):
                    flags.append("voice is male but dialog may reference female character")
                if v.get("gender") == "female" and any(m in text_lower for m in male_markers):
                    flags.append("voice is female but dialog may reference male character")
                if flags:
                    st.warning(f"Dialog {did}: {'; '.join(flags)}")

            st.json({"reviewed_at": qc.get("reviewed_at"), "initial": qc.get("initial"),
                     "verdict": v, "discard_reason": v.get("discard_reason","none")})

# ════════════════════════════════════════════════════════════════════════════════
# SPEECH QUALITY (DNSMOS + BLASER proxy + speaker similarity + pause alignment)
# ════════════════════════════════════════════════════════════════════════════════
with tab_speech:
    import os as _os
    blaser_ready = _os.path.exists("/tmp/sonar_env/bin/python")

    st.markdown(f"**Source:** {src_lang} (`{src_lang_code}`) → **Target:** {lang_name} (`{lang_code}`)")
    st.markdown(f"""
Automated speech-level evals. Run once per episode; results are cached.

| Metric | What it measures | Model | Status |
|---|---|---|---|
| UTMOS22 | TTS naturalness MOS (1–5). Trained on TTS human ratings — much better than DNSMOS for synthesised speech | SpeechMOS / UTokyo 2022 | Ready |
| BLASER-2.0-QE | Translation quality text-to-text using SONAR embeddings. Source Hindi/Chinese text vs dubbed Telugu text — no ASR, no tokenisation bias | Meta SONAR text encoder (~1GB) | {"Ready" if blaser_ready else "sonar_env not found"} |
| Pause alignment | Pauses in dubbed audio vs source audio — phrase-pause structure preservation | librosa silence detection | Ready |
| MuRIL grammar | Text naturalness via pseudo-perplexity. Trained on 17 Indic languages | google/muril-base-cased | Ready |
| Voice assignment | Gender/age match between character role and assigned ElevenLabs voice | Pipeline data + voice_qc | Ready |
""")

    wav_dialogs = [d for d in dialogs if d["wav"]]
    already     = set(speech_results.keys())
    pending     = [d for d in wav_dialogs if str(d["id"]) not in already]

    # load character metadata for voice assignment check
    char_meta = {s["speaker_id"]: s
                 for s in (raw.get("speakers") or [])}

    c1, c2, c3, c4, c5 = st.columns(5)
    run_utmos   = c1.button(f"Run UTMOS ({len(pending)})",          type="primary")
    run_blaser  = c2.button(f"Run BLASER ({len(pending)})",
                             type="primary" if blaser_ready else "secondary",
                             disabled=not blaser_ready)
    run_pause   = c3.button(f"Run pause alignment ({len(pending)})",type="secondary")
    run_grammar = c4.button(f"Run MuRIL grammar ({len(pending)})",  type="secondary")
    run_voice   = c5.button(f"Run voice check ({len(pending)})",    type="secondary")

    if any([run_utmos, run_blaser, run_pause, run_grammar, run_voice]):
        src_vocals_bytes = None
        if run_pause:
            try:
                src_vocals_bytes = fetch_bytes(
                    f"shows/dubbing/{show_id}/episodes/{episode}/intermediate/audio/vocals.mp3")
            except:
                st.warning("Could not load source vocals.mp3 — pause alignment will be skipped.")

        bar = st.progress(0)
        for i, d in enumerate(pending):
            did   = str(d["id"])
            entry = speech_results.get(did, {})
            try:
                ab = fetch_bytes(d["wav"])
            except:
                bar.progress((i+1)/len(pending)); continue

            if run_utmos:
                try:    entry["utmos"] = compute_utmos(ab)
                except Exception as e: entry["utmos_error"] = str(e)

            if run_blaser:
                # Text-based BLASER-2.0-QE — uses SONAR text encoder (~1GB vs 8GB speech encoder)
                src_text = d.get("source_text", "")
                tgt_text = d.get("pipeline_text", "")
                src_sonar = SONAR_LANG_CODES.get(src_lang_code, "hin_Deva")
                tgt_sonar = SONAR_LANG_CODES.get(lang_code, "tel_Telu")
                try:
                    entry["blaser"] = blaser_qe_text(
                        src_text, tgt_text, src_sonar, tgt_sonar)
                except Exception as e:
                    entry["blaser_error"] = str(e)

            if run_pause and src_vocals_bytes:
                src_start = d.get("start_time") or 0
                src_end   = d.get("end_time") or (src_start + (d.get("target_ms",5000)/1000))
                try:
                    entry["pause_align"] = pause_alignment_score(
                        src_vocals_bytes, ab, src_start, src_end)
                except Exception as e:
                    entry["pause_align_error"] = str(e)

            if run_grammar:
                try:    entry["grammar"] = text_naturalness(d["pipeline_text"])
                except Exception as e: entry["grammar_error"] = str(e)

            if run_voice:
                spk  = d.get("speaker")
                vid  = d.get("voice_id")
                cmeta = char_meta.get(spk)
                vqc   = load_voice_qc(vid).get("verdict") if vid else None
                if cmeta and vqc:
                    try:    entry["voice_check"] = verify_voice_assignment(cmeta, vqc)
                    except: pass

            speech_results[did] = entry
            bar.progress((i+1)/len(pending), text=f"Dialog {d['id']}…")

        bar.empty()
        st.session_state[f"speech_{show_id}_{episode}"] = speech_results
        save_speech_cache(show_id, episode, speech_results)
        scores = compute_scores(dialogs, aq_results, speech_results,
                                src_lang_code, lang_code)
        st.rerun()

    if speech_results:
        rows = []
        for d in wav_dialogs:
            sr   = speech_results.get(str(d["id"]), {})
            vc   = sr.get("voice_check") or {}
            rows.append({
                "Dialog":        d["id"],
                "UTMOS (1–5)":    sr.get("utmos"),
                "BLASER-QE":      sr.get("blaser"),
                "Emotion tags":   ", ".join(extract_emotion_tags(d.get("pipeline_text",""))),
                "Pause align":    sr.get("pause_align"),
                "MuRIL (0–100)":  sr.get("grammar"),
                "Voice quality":  vc.get("quality"),
                "Voice flags":    " | ".join(vc.get("flags",[])) or "—",
            })
        df_sp = pd.DataFrame(rows)
        for col in ["UTMOS (1–5)","BLASER-QE","Pause align","MuRIL (0–100)","Voice quality"]:
            if col in df_sp.columns:
                df_sp[col] = df_sp[col].apply(lambda x: fmt(x) if pd.notnull(x) else "—")
        st.dataframe(df_sp, use_container_width=True, hide_index=True)

        # UTMOS chart
        utmos_rows = [(str(r["Dialog"]), r["UTMOS (1–5)"]) for r in rows
                      if r["UTMOS (1–5)"] is not None]
        if utmos_rows:
            ids, vals = zip(*utmos_rows)
            fig_u = go.Figure(go.Bar(
                x=list(ids), y=list(vals),
                marker_color=["#f44336" if v<2.5 else ("#ff9800" if v<3.5 else "#4caf50")
                              for v in vals],
                text=[f"{v:.2f}" for v in vals], textposition="outside",
            ))
            fig_u.add_hline(y=3.5, line_dash="dash", line_color="#4caf50",
                            annotation_text="Good (3.5)")
            fig_u.add_hline(y=2.5, line_dash="dash", line_color="#f44336",
                            annotation_text="Poor (2.5)")
            fig_u.update_layout(
                title="UTMOS22 naturalness per dialog (1–5, higher = more natural TTS)",
                xaxis_title="Dialog", yaxis=dict(range=[1,5.5]),
                height=300, margin=dict(t=40,b=0),
                plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
            )
            st.plotly_chart(fig_u, use_container_width=True)

        blaser_rows = [(str(r["Dialog"]), r["BLASER-QE"]) for r in rows
                       if r["BLASER-QE"] is not None]
        if blaser_rows:
            ids, vals = zip(*blaser_rows)
            fig_b = go.Figure(go.Bar(
                x=list(ids), y=list(vals),
                marker_color=["#f44336" if v<2.5 else ("#ff9800" if v<3.5 else "#4caf50")
                              for v in vals],
                text=[f"{v:.2f}" for v in vals], textposition="outside",
            ))
            fig_b.add_hline(y=3.5, line_dash="dash", line_color="#ff9800",
                            annotation_text="Good (3.5)")
            fig_b.update_layout(
                title="BLASER-2.0-QE translation quality per dialog (1–5 scale)",
                xaxis_title="Dialog", yaxis=dict(range=[1,5.5]),
                height=300, margin=dict(t=40,b=0),
                plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
            )
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption("BLASER-2.0-QE from Meta SONAR. Evaluates translation quality directly "
                       "audio-to-audio using the SONAR speech encoder — no ASR step, no tokenisation.")
    else:
        st.info("Click a Run button above to compute speech quality metrics.")

# ════════════════════════════════════════════════════════════════════════════════
# HUMAN QC
# ════════════════════════════════════════════════════════════════════════════════
with tab_hqc:
    st.caption("1=unusable · 2=major issue · 3=awkward · 4=minor issue · 5=broadcast-ready")
    st.caption("Adequacy → bilingual source+target rater.  Fluency → native target-language rater.  Emotion/Audio → audio-visual rater.")
    GATES = [
        ("critical_mistranslation","Meaning reversal / wrong facts"),
        ("hallucination",          "Content added not in source"),
        ("unintelligible_speech",  "Native listener cannot understand"),
        ("wrong_speaker",          "Wrong gender / age impression"),
        ("unsafe_content",         "Slurs / offense / distortion"),
        ("named_entity_error",     "Main character/place unrecognizable"),
    ]
    for d in dialogs:
        did = str(d["id"])
        dqc = qc_data.get(did, {})
        with st.expander(
            f"Dialog {d['id']}"
            + ("  [edited]" if d["was_edited"] else "")
            + ("  [script error]" if d["script_errors"] else "")
            + ("  [scored]" if dqc else "")
        ):
            cl, cr = st.columns([2,1])
            with cl:
                st.markdown(
                    f'<div style="background:#111;border-radius:6px;padding:10px;'
                    f'font-size:14px;line-height:1.7">{hl(d["pipeline_text"])}</div>',
                    unsafe_allow_html=True
                )
                if d["was_edited"] and d["editor_text"]:
                    st.markdown("Editor correction:")
                    st.markdown(
                        f'<div style="background:#0a1f0a;border:1px solid #2a4a2a;border-radius:6px;'
                        f'padding:8px;font-size:13px">{d["editor_text"]}</div>',
                        unsafe_allow_html=True
                    )
            with cr:
                if d["wav"]:
                    try:    st.audio(fetch_bytes(d["wav"]), format="audio/wav")
                    except: pass

            with st.form(f"qc_{did}"):
                c1,c2,c3,c4 = st.columns(4)
                adeq = c1.slider("Adequacy",1,5,dqc.get("adequacy",3),key=f"a_{did}")
                flu  = c2.slider("Fluency",1,5,dqc.get("fluency",3), key=f"f_{did}")
                emot = c3.slider("Emotion",1,5,dqc.get("emotion",3), key=f"e_{did}")
                aud  = c4.slider("Audio",1,5,dqc.get("audio",3),     key=f"au_{did}")
                gc   = st.columns(3)
                gates = {gk: gc[i%3].checkbox(gl, dqc.get(f"gate_{gk}",False), key=f"g_{did}_{gk}")
                         for i,(gk,gl) in enumerate(GATES)}
                notes = st.text_area("Notes", dqc.get("notes",""), key=f"n_{did}")
                if st.form_submit_button("Save", type="primary"):
                    qc_data[did] = {"adequacy":adeq,"fluency":flu,"emotion":emot,"audio":aud,
                                    "notes":notes,**{f"gate_{k}":v for k,v in gates.items()}}
                    save_qc(show_id, episode, qc_data)
                    st.success("Saved.")

    if qc_data:
        st.divider()
        rows = []
        for did, v in qc_data.items():
            fails = [gl for gk,gl in GATES if v.get(f"gate_{gk}")]
            rows.append({"Dialog":did,"Adequacy":v.get("adequacy"),"Fluency":v.get("fluency"),
                         "Emotion":v.get("emotion"),"Audio":v.get("audio"),
                         "Hard-fail":"FAIL: "+" | ".join(fails) if fails else "OK"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
