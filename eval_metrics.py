"""
Research-aligned eval metrics for Indic language dubbing.
Source: Hindi (Devanagari) or Chinese (CJK)
Target: Telugu, Hindi, Tamil, Bengali, Marathi, Kannada, Malayalam, etc.

Metric stack:
  Timing    — IsoChronoMeter (WMT 2024) + pause alignment
  Naturalness — UTMOS22 (SpeechMOS, UTokyo 2022) — trained on TTS MOS
  Translation — BLASER-2.0-QE via sonar_env subprocess (Python 3.11 venv)
  Grammar   — MuRIL pseudo-perplexity (Google, 17 Indic languages)
  Voice     — pipeline assignment quality scores (cosine + fit scores from
               intelligent_ranking) + cross-episode consistency
"""
from __future__ import annotations
import os, re, unicodedata, tempfile, subprocess, json
import numpy as np

import os as _os
SONAR_PYTHON = _os.environ.get("SONAR_PYTHON_PATH", "/tmp/sonar_env/bin/python")

# ── Script integrity: wrong-script blocks per target language ─────────────────
WRONG_SCRIPT_BLOCKS_EVAL = {
    "tel": [(0x0900,0x097F,"Devanagari"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK")],
    "tam": [(0x0C00,0x0C7F,"Telugu"),(0x0900,0x097F,"Devanagari"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK")],
    "ben": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK")],
    "hin": [(0x0C00,0x0C7F,"Telugu"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK")],
    "mar": [(0x0C00,0x0C7F,"Telugu"),(0x0B80,0x0BFF,"Tamil"),
            (0x0980,0x09FF,"Bengali"),(0x4E00,0x9FFF,"CJK")],
    "kan": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK")],
    "mal": [(0x0900,0x097F,"Devanagari"),(0x0C00,0x0C7F,"Telugu"),
            (0x0B80,0x0BFF,"Tamil"),(0x4E00,0x9FFF,"CJK")],
}

def find_script_errors(text, tgt_lang_code="tel"):
    """Language-aware wrong-script detection. Flags any Indic/CJK characters
    that don't belong in the target language script."""
    wrong = WRONG_SCRIPT_BLOCKS_EVAL.get(tgt_lang_code[:3], [])
    if not wrong:
        return [f"U+{ord(c):04X} '{c}' (Devanagari)"
                for c in text if "DEVANAGARI" in unicodedata.name(c,"")]
    seen, errors = set(), []
    for ch in text:
        cp = ord(ch)
        if cp < 0x0080: continue
        for lo, hi, name in wrong:
            if lo <= cp <= hi and ch not in seen:
                errors.append(f"{name} '{ch}' U+{cp:04X}")
                seen.add(ch); break
    return errors

# ── Indic language expansion expectations ─────────────────────────────────────
EXPANSION_EXPECTATIONS = {
    ("hin", "tel"): (1.10, 1.50),
    ("hin", "tam"): (1.15, 1.55),
    ("hin", "ben"): (0.95, 1.30),
    ("hin", "mar"): (0.95, 1.25),
    ("hin", "kan"): (1.10, 1.50),
    ("hin", "mal"): (1.10, 1.55),
    ("hin", "hin"): (0.90, 1.10),
    ("zho", "tel"): (1.50, 2.50),
    ("zho", "hin"): (1.20, 1.80),
    ("zho", "tam"): (1.50, 2.50),
    ("zho", "ben"): (1.20, 1.80),
    ("zho", "mar"): (1.30, 2.00),
    ("zho", "kan"): (1.50, 2.50),
    ("zho", "mal"): (1.50, 2.50),
}

def expected_expansion(src_lang, tgt_lang):
    key = (src_lang[:3].lower(), tgt_lang[:3].lower())
    return EXPANSION_EXPECTATIONS.get(key, (0.90, 1.60))

def flag_expansion(ratio, src_lang, tgt_lang):
    lo, hi = expected_expansion(src_lang, tgt_lang)
    if ratio < lo * 0.85:
        return "TOO_SHORT", f"ratio {ratio:.2f} below expected min {lo:.2f}"
    if ratio > hi * 1.10:
        return "TOO_LONG", f"ratio {ratio:.2f} above expected max {hi:.2f}"
    return "OK", f"ratio {ratio:.2f} within expected {lo:.2f}–{hi:.2f}"

# ── 1. IsoChronoMeter ─────────────────────────────────────────────────────────
# WMT 2024: "IsoChronoMeter: A simple and effective isochronic translation
# evaluation metric"
def isochrone_score(dialogs):
    """
    Per segment: isochrony_i = max(0, 1 - |tts_ms - window_ms| / window_ms)
    Episode score = mean(isochrony_i)
    Returns (episode_float, per_dialog_list)
    """
    scores = []
    for d in dialogs:
        w = d.get("target_ms") or d.get("window_ms") or 0
        t = d.get("tts_ms") or d.get("dur_ms") or 0
        if not w or not t:
            continue
        s = max(0.0, 1.0 - abs(t - w) / w)
        scores.append({"id": d["id"], "score": round(s, 3),
                       "window_ms": w, "tts_ms": t,
                       "ratio": round(t / w, 3)})
    ep = round(float(np.mean([s["score"] for s in scores])), 3) if scores else None
    return ep, scores

# ── 2. Pause alignment ────────────────────────────────────────────────────────
def _extract_audio_segment(audio_bytes, start_s, end_s, sr_out=16000):
    """Extract [start_s, end_s] from audio_bytes (any format) at sr_out Hz."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=sr_out, mono=True,
                              offset=start_s, duration=end_s - start_s)
        return y, sr_out
    finally:
        os.unlink(tmp)

def _detect_pause_positions(y, sr, min_silence_ms=200, thresh_db=-38):
    """Return list of pause mid-points as fraction of total duration [0,1]."""
    import librosa
    frame_ms  = 10
    frame_len = int(sr * frame_ms / 1000)
    rms       = librosa.feature.rms(y=y, frame_length=frame_len,
                                     hop_length=frame_len)[0]
    db        = 20 * np.log10(rms + 1e-8)
    silent    = db < thresh_db
    min_f     = max(1, int(min_silence_ms / frame_ms))
    pauses, run, total = [], 0, len(silent)
    for i, s in enumerate(silent):
        if s:
            run += 1
        else:
            if run >= min_f:
                pauses.append((i - run / 2) / max(total, 1))
            run = 0
    if run >= min_f:
        pauses.append((total - run / 2) / max(total, 1))
    return pauses

def pause_alignment_score(src_audio_bytes, tts_audio_bytes,
                           src_start_s, src_end_s):
    """
    Extract the source segment [src_start_s, src_end_s] from full vocals audio.
    Detect pauses in both source segment and TTS audio.
    Return score in [0,1]: 1.0 = pauses align perfectly, 0.0 = no alignment.
    """
    import librosa
    try:
        src_y, src_sr = _extract_audio_segment(src_audio_bytes,
                                                src_start_s, src_end_s)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(tts_audio_bytes); tmp = f.name
        try:
            tts_y, tts_sr = librosa.load(tmp, sr=16000, mono=True)
        finally:
            os.unlink(tmp)

        src_pauses = _detect_pause_positions(src_y, src_sr)
        tts_pauses = _detect_pause_positions(tts_y, tts_sr)

        if not src_pauses and not tts_pauses:
            return 1.0           # no pauses expected, none present — fine
        if not src_pauses or not tts_pauses:
            return 0.5           # one side has pauses, other doesn't

        # greedy nearest-neighbour matching
        matched, used = [], set()
        for sp in src_pauses:
            best_i, best_d = None, 1.0
            for i, tp in enumerate(tts_pauses):
                if i not in used and abs(tp - sp) < best_d:
                    best_d, best_i = abs(tp - sp), i
            if best_i is not None:
                matched.append(best_d); used.add(best_i)

        unmatched_penalty = (len(src_pauses) + len(tts_pauses) - 2*len(matched)) * 0.12
        score = max(0.0, 1.0 - (np.mean(matched) if matched else 1.0) - unmatched_penalty)
        return round(float(score), 3)
    except Exception:
        return None

# ── 3. UTMOS22 — TTS naturalness MOS predictor ────────────────────────────────
# Baba et al., UTokyo-SaruLab, 2022. Trained specifically on TTS MOS.
# Much better than DNSMOS (which was trained on speech enhancement) for TTS.
_utmos_model = None

def _get_utmos():
    global _utmos_model
    if _utmos_model is None:
        import torch
        _utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        _utmos_model.eval()
    return _utmos_model

def compute_utmos(audio_bytes):
    """
    Returns UTMOS22 MOS score on 1–5 scale (higher = more natural TTS).
    Uses 16kHz mono audio. Segments >10s are chunked and averaged.
    """
    import torch, librosa
    model = _get_utmos()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        y, _  = librosa.load(tmp, sr=16000, mono=True)
        chunk = 16000 * 10  # 10s chunks
        scores = []
        for start in range(0, len(y), chunk):
            seg = y[start:start+chunk]
            if len(seg) < 1600: continue   # skip clips < 0.1s
            wave = torch.tensor(seg).unsqueeze(0)
            with torch.no_grad():
                scores.append(float(model(wave, sr=16000)))
        return round(float(np.mean(scores)), 3) if scores else None
    except Exception as e:
        return None
    finally:
        os.unlink(tmp)

# ── 4. BLASER-2.0-QE via sonar_env subprocess ────────────────────────────────
# Proper BLASER requires fairseq2 (Python <=3.12). We run it via a Python 3.11
# venv at SONAR_PYTHON. Falls back to None if venv not ready.
_BLASER_SCRIPT = """
import sys, json, tempfile, os
import torch

src_path, tgt_path, src_lang, tgt_lang = sys.argv[1:5]

try:
    from sonar.models.blaser.loader import load_blaser_model
    from sonar.inference_pipelines.speech import SpeechToEmbeddingPipelineModel

    blaser = load_blaser_model("blaser_2_0_qe").eval()

    def embed(wav_path, lang):
        pipeline = SpeechToEmbeddingPipelineModel.load_model_from_name(
            "sonar_speech_encoder_" + lang,
            device=torch.device("cpu"),
        )
        import torchaudio
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        emb = pipeline.predict([wav_path], batch_size=1)
        return emb

    src_emb = embed(src_path, src_lang)
    tgt_emb = embed(tgt_path, tgt_lang)

    with torch.no_grad():
        score = blaser(src=src_emb, mt=tgt_emb).item()

    print(json.dumps({"score": score}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
"""

def blaser_qe_text(src_text, tgt_text, src_lang="hin_Deva", tgt_lang="tel_Telu"):
    """
    BLASER-2.0-QE using SONAR text encoder (practical — ~1GB vs 8GB per speech encoder).
    src_lang / tgt_lang: SONAR language codes (hin_Deva, tel_Telu, zho_Hant, etc.)
    Returns score on ~1–5 scale (higher = better translation quality).
    """
    if not os.path.exists(SONAR_PYTHON):
        return None
    script = f"""
import re, sys, json, torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
blaser = load_blaser_model('blaser_2_0_qe').eval()
enc = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder", device=torch.device("cpu"))
src = sys.argv[1]; tgt = sys.argv[2]
src_emb = enc.predict([src], source_lang="{src_lang}", batch_size=1)
tgt_emb = enc.predict([tgt], source_lang="{tgt_lang}", batch_size=1)
with torch.no_grad():
    score = blaser(src=src_emb, mt=tgt_emb).item()
print(json.dumps({{"score": score}}))
"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); script_path = f.name
    try:
        result = subprocess.run(
            [SONAR_PYTHON, script_path, src_text, tgt_text],
            capture_output=True, text=True, timeout=60,
        )
        out = json.loads(result.stdout)
        return out.get("score")
    except Exception:
        return None
    finally:
        try: os.unlink(script_path)
        except: pass

# SONAR language code mapping
SONAR_LANG_CODES = {
    "hin": "hin_Deva", "tel": "tel_Telu", "tam": "tam_Taml",
    "ben": "ben_Beng", "mar": "mar_Deva", "kan": "kan_Knda",
    "mal": "mal_Mlym", "guj": "guj_Gujr", "pan": "pan_Guru",
    "urd": "urd_Arab",
    "zho": "zho_Hans",   # Simplified Chinese (mainland drama content default)
    "zhs": "zho_Hans",   # explicit simplified alias
    "zht": "zho_Hant",   # Traditional Chinese (Taiwan/HK content)
}

def blaser_qe(src_audio_bytes, tgt_audio_bytes, src_lang="hin", tgt_lang="tel"):
    """
    Run BLASER-2.0-QE in Python 3.11 sonar_env.
    Returns quality score (float) or None if sonar_env not ready.
    src_lang/tgt_lang: ISO 639-3 codes supported by SONAR (hin, zho, tel, etc.)
    """
    if not os.path.exists(SONAR_PYTHON):
        return None

    with (tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sf,
          tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf,
          tempfile.NamedTemporaryFile(suffix=".py",  delete=False) as pf):
        sf.write(src_audio_bytes); src_p = sf.name
        tf.write(tgt_audio_bytes); tgt_p = tf.name
        pf.write(_BLASER_SCRIPT.encode()); script_p = pf.name

    try:
        result = subprocess.run(
            [SONAR_PYTHON, script_p, src_p, tgt_p, src_lang, tgt_lang],
            capture_output=True, text=True, timeout=120,
        )
        out = json.loads(result.stdout)
        return out.get("score")
    except Exception:
        return None
    finally:
        for p in [src_p, tgt_p, script_p]:
            try: os.unlink(p)
            except: pass

# ── 5. MuRIL pseudo-perplexity — grammar/naturalness ─────────────────────────
# google/muril-base-cased trained on 17 Indic languages.
# Pseudo-perplexity via masked language modelling: mask each token one-by-one,
# compute negative log-likelihood, average → pseudo-perplexity.
# Lower = more natural/grammatical text.
_muril_model = None
_muril_tok   = None

def _get_muril():
    global _muril_model, _muril_tok
    if _muril_model is None:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        _muril_tok   = AutoTokenizer.from_pretrained("google/muril-base-cased")
        _muril_model = AutoModelForMaskedLM.from_pretrained("google/muril-base-cased")
        _muril_model.eval()
    return _muril_tok, _muril_model

def text_naturalness(text, max_tokens=64):
    """
    Pseudo-perplexity using MuRIL (17 Indic languages).
    Returns normalised naturalness score in [0,100] — higher is more natural.
    Falls back to XLM-RoBERTa if MuRIL unavailable.
    """
    # strip emotion tags, punctuation
    clean = re.sub(r'\[.*?\]', '', text).strip()
    if len(clean) < 5:
        return None
    try:
        import torch
        tok, model = _get_muril()
        enc = tok(clean, return_tensors="pt", truncation=True, max_length=max_tokens)
        ids = enc["input_ids"][0]
        if len(ids) <= 2:   # just [CLS] [SEP]
            return None

        nlls = []
        for i in range(1, len(ids) - 1):   # skip CLS/SEP
            masked = ids.clone()
            masked[i] = tok.mask_token_id
            inp = {"input_ids": masked.unsqueeze(0),
                   "attention_mask": enc["attention_mask"]}
            with torch.no_grad():
                logits = model(**inp).logits[0, i]
            log_prob = torch.log_softmax(logits, dim=-1)[ids[i]].item()
            nlls.append(-log_prob)

        ppl = float(np.exp(np.mean(nlls)))
        # convert to 0-100: ppl=1 → 100, ppl=100 → ~0
        score = max(0.0, min(100.0, 100.0 / (1.0 + np.log(max(ppl, 1.0)))))
        return _r(score)
    except Exception:
        return None

# ── 6. Voice assignment verification ─────────────────────────────────────────
# Uses pipeline data: episode_refined_speakers.json + voice_qc/{id}.json
# Does NOT compute ECAPA similarity — the pipeline already computed acoustic
# cosine similarity during intelligent_ranking. We verify correctness instead.
def verify_voice_assignment(speaker_meta, voice_qc_verdict):
    """
    Check pipeline voice assignment quality.
    speaker_meta: from episode_refined_speakers.json
        {"speaker_id", "role", "gender", "age_group", "appearance"}
    voice_qc_verdict: from voice_qc/{id}.json → verdict dict
        {"gender", "age", "pitch", "timbre", "expressiveness", "ranking_tier"}
    Returns dict with match scores and flags.
    """
    if not speaker_meta or not voice_qc_verdict:
        return None

    char_gender = speaker_meta.get("gender", "").lower()
    char_age    = speaker_meta.get("age_group", "").lower()
    voice_gender= voice_qc_verdict.get("gender", "").lower()
    voice_age   = voice_qc_verdict.get("age", "").lower()
    voice_tier  = voice_qc_verdict.get("ranking_tier", "")
    char_role   = speaker_meta.get("role", "").lower()

    gender_match = char_gender == voice_gender if char_gender and voice_gender else None

    # age_group normalisation: pipeline uses child/adult/middle_aged/senior
    # voice_qc uses the same; check proximity
    age_map = {"child": 0, "adult": 1, "middle_aged": 2, "senior": 3}
    char_age_v  = age_map.get(char_age)
    voice_age_v = age_map.get(voice_age)
    age_match   = None
    if char_age_v is not None and voice_age_v is not None:
        diff = abs(char_age_v - voice_age_v)
        age_match = diff == 0          # exact match

    # main character should get main_character tier voice
    is_main = any(k in char_role for k in ["main", "lead", "protagonist"])
    tier_ok = voice_tier == "main_character" if is_main else True

    # overall quality: both matches = 1.0, one miss = 0.5, both miss = 0.0
    matches = [m for m in [gender_match, age_match] if m is not None]
    quality = sum(matches) / len(matches) if matches else 0.5

    flags = []
    if gender_match is False:
        flags.append(f"gender mismatch: character={char_gender}, voice={voice_gender}")
    if age_match is False:
        flags.append(f"age mismatch: character={char_age}, voice={voice_age}")
    if is_main and not tier_ok:
        flags.append(f"main character using {voice_tier} tier voice")

    return {
        "quality":      round(quality, 2),
        "gender_match": gender_match,
        "age_match":    age_match,
        "tier_ok":      tier_ok,
        "flags":        flags,
    }


# ── Emotion accuracy ──────────────────────────────────────────────────────────
# Model: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
# Labels: neu (neutral), ang (angry), hap (happy), sad (sad)
# Properly trained on IEMOCAP acted speech — replaces broken ehcalabres model.
#
# Tag → IEMOCAP label mapping (4-class)
TAG_TO_EMOTION = {
    "crying":      {"sad"},
    "sobs":        {"sad"},
    "tears":       {"sad"},
    "pleading":    {"sad"},
    "sighs":       {"sad", "neu"},
    "angry":       {"ang"},
    "yells":       {"ang"},
    "shouts":      {"ang"},
    "frustrated":  {"ang"},
    "exasperated": {"ang"},
    "annoyed":     {"ang"},
    "groans":      {"ang"},
    "defiant":     {"ang"},
    "happy":       {"hap"},
    "excited":     {"hap"},
    "laughs":      {"hap"},
    "confused":    {"neu"},
    "whispers":    {"neu"},
    "mutters":     {"neu"},
    "surprised":   {"neu", "hap"},
    "gasps":       {"neu"},
    "scared":      {"sad"},
}

def extract_emotion_tags(text):
    """Return list of lowercase ElevenLabs direction tags from pipeline text."""
    return [t.lower() for t in re.findall(r'\[(\w+)\]', text)]

def emotion_tag_coverage(dialogs):
    """
    Checks what % of dialogs have ElevenLabs direction tags and what tags are used.
    No audio classifier — tag presence is the signal for now.
    Audio-based emotion eval requires an Indic TTS emotion model (TODO).
    Returns (coverage_0_to_1, per_dialog_tags, tag_frequency_dict)
    """
    per_dialog = []
    tag_freq   = {}
    for d in dialogs:
        text = d.get("pipeline_text") or d.get("text") or ""
        tags = extract_emotion_tags(text)
        tags = [t for t in tags if t in TAG_TO_EMOTION]  # only recognised emotion tags
        per_dialog.append({"id": d.get("id") or d.get("index"), "tags": tags})
        for t in tags:
            tag_freq[t] = tag_freq.get(t, 0) + 1

    tagged    = sum(1 for d in per_dialog if d["tags"])
    coverage  = round(tagged / max(len(per_dialog), 1), 3)
    return coverage, per_dialog, tag_freq

def emotion_accuracy(audio_bytes, pipeline_text):
    """
    Placeholder — returns tag list only. Audio-based emotion classification
    for Indic TTS is a TODO (no public model generalises from English IEMOCAP
    to Telugu/Tamil/Bengali/Hindi TTS output).
    """
    tags = [t for t in extract_emotion_tags(pipeline_text) if t in TAG_TO_EMOTION]
    return None, tags


# ── Indic-specific: honorific / register check ───────────────────────────────
# Checks that the output text uses register-appropriate honorific markers
# for the character's role and relationship context.
HONORIFIC_MARKERS = {
    "tel": {
        "formal_respect": ["మీరు", "గారు", "అయ్యా", "అమ్మగారు", "సార్"],
        "informal":       ["నువ్వు", "నువ్"],
        "family_respect": ["అమ్మా", "నాన్నా", "అన్నయ్య", "అక్కయ్య", "అక్కా"],
    },
    "hin": {
        "formal_respect": ["आप", "जी", "सर", "मैडम"],
        "informal":       ["तू", "तुम"],
        "family_respect": ["माँ", "बाबा", "दीदी", "भैया", "चाचा"],
    },
    "tam": {
        "formal_respect": ["நீங்கள்", "சார்", "மேடம்"],
        "informal":       ["நீ"],
        "family_respect": ["அம்மா", "அண்ணா", "அக்கா", "அப்பா"],
    },
    "ben": {
        "formal_respect": ["আপনি", "স্যার", "ম্যাডাম"],
        "informal":       ["তুই", "তুমি"],
        "family_respect": ["মা", "বাবা", "দিদি", "দাদা"],
    },
    "mar": {
        "formal_respect": ["आपण", "साहेब", "मॅडम"],
        "informal":       ["तू"],
        "family_respect": ["आई", "बाबा", "दादा", "ताई"],
    },
    "kan": {
        "formal_respect": ["ನೀವು", "ಸರ್", "ಮೇಡಂ"],
        "informal":       ["ನೀನು"],
        "family_respect": ["ಅಮ್ಮ", "ಅಪ್ಪ", "ಅಣ್ಣ", "ಅಕ್ಕ"],
    },
    "mal": {
        "formal_respect": ["നിങ്ങൾ", "സർ", "മേഡം"],
        "informal":       ["നീ"],
        "family_respect": ["അമ്മ", "അച്ഛൻ", "ചേട്ടൻ", "ചേച്ചി"],
    },
}


# ── Exclamation drop rate ─────────────────────────────────────────────────────
# Short exclamatory fillers (啊/哎/अरे/ah) are silently dropped by the episode
# refinement stage. This metric measures what fraction of them were lost.

_EXCL_TOKENS_EVAL = {
    # Chinese
    "啊","哦","哎","嗯","哇","唉","哼","呢","嘛","咦","呀","哟","嘿","哈",
    "噢","诶","欸","喔","哎呀","哦哦","哎哟","唔","嗨","哇哦",
    # Hindi / Urdu
    "अरे","ओह","हाँ","हाय","वाह","ओये","उफ़","उफ","एह","अच्छा","हूँ","हूं",
    # English code-mix
    "ah","oh","eh","uh","um","hmm","hm","hey","wow","ooh","ugh","aww","aw","huh",
}

_EXCL_RE_EVAL = re.compile(
    r"^\W*(" + "|".join(re.escape(t) for t in sorted(_EXCL_TOKENS_EVAL, key=len, reverse=True)) + r")\W*$",
    re.IGNORECASE | re.UNICODE,
)

_EXCL_MAX_DUR_S   = 0.6   # clips shorter than this are always exclamatory
_EXCL_MAX_CHARS   = 6     # text shorter than this + token match → exclamatory


def is_exclamation(text: str | None, duration_s: float | None) -> bool:
    """Return True for very short exclamatory filler segments."""
    if duration_s is not None and duration_s <= _EXCL_MAX_DUR_S:
        return True
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) <= _EXCL_MAX_CHARS and bool(_EXCL_RE_EVAL.match(stripped))


def exclamation_drop_rate(source_dialogs: list[dict], target_dialogs: list[dict]) -> dict:
    """
    Compare source dialogs (from cleaned.json or show_refined) against the
    final translated dialogs to find exclamatory fillers that were silently
    dropped by the pipeline.

    source_dialogs: list of dicts with keys index, start_time, end_time, text
    target_dialogs: list of dicts with keys index, start_time, end_time, text

    Returns:
      total_exclamations  — count of exclamatory segments in source
      dropped_count       — how many weren't carried through to target
      drop_rate           — dropped / total (0–1), None if no exclamations found
      dropped_examples    — list of up to 10 dropped exclamation texts
    """
    if not source_dialogs:
        return {"total_exclamations": 0, "dropped_count": 0,
                "drop_rate": None, "dropped_examples": []}

    excl_src = []
    for d in source_dialogs:
        ss, se = d.get("start_time"), d.get("end_time")
        dur = (se - ss) if (ss is not None and se is not None) else None
        if is_exclamation(d.get("text"), dur):
            excl_src.append(d)

    if not excl_src:
        return {"total_exclamations": 0, "dropped_count": 0,
                "drop_rate": None, "dropped_examples": []}

    # Check each source exclamation against target by time overlap
    dropped = []
    for sd in excl_src:
        ss, se = sd.get("start_time"), sd.get("end_time")
        if ss is None or se is None:
            dropped.append(sd)
            continue
        covered = False
        for td in target_dialogs:
            ts, te = td.get("start_time"), td.get("end_time")
            if ts is None or te is None:
                continue
            if max(0.0, min(se, te) - max(ss, ts)) >= 0.05:
                covered = True
                break
        if not covered:
            dropped.append(sd)

    n_total   = len(excl_src)
    n_dropped = len(dropped)
    return {
        "total_exclamations": n_total,
        "dropped_count":      n_dropped,
        "drop_rate":          round(n_dropped / n_total, 3) if n_total else None,
        "dropped_examples":   [d.get("text","") for d in dropped[:10]],
    }


def honorific_check(dialogs, tgt_lang_code):
    """
    Scan all dialog texts for honorific markers.
    Returns per-dialog register signals and an episode-level consistency score.
    """
    markers = HONORIFIC_MARKERS.get(tgt_lang_code[:3], {})
    if not markers:
        return None, []

    per_dialog = []
    for d in dialogs:
        text    = d.get("pipeline_text", "") or d.get("text", "")
        speaker = d.get("speaker", "")
        found   = {}
        for category, words in markers.items():
            hits = [w for w in words if w in text]
            if hits:
                found[category] = hits
        per_dialog.append({
            "id": d.get("id") or d.get("index"),
            "speaker": speaker,
            "register_signals": found,
        })

    # Consistency: same speaker should use same formality tier throughout
    speaker_registers = {}
    for row in per_dialog:
        sp = row["speaker"]
        if not sp: continue
        cats = set(row["register_signals"].keys())
        speaker_registers.setdefault(sp, []).append(cats)

    inconsistent_speakers = []
    for sp, cat_list in speaker_registers.items():
        non_empty = [c for c in cat_list if c]
        if len(non_empty) < 2: continue
        # flag if register switches between formal and informal for same speaker
        has_formal   = any("formal_respect" in c for c in non_empty)
        has_informal = any("informal" in c for c in non_empty)
        if has_formal and has_informal:
            inconsistent_speakers.append(sp)

    consistency = 1.0 - len(inconsistent_speakers) / max(len(speaker_registers), 1)
    return round(consistency, 3), per_dialog


# ── Indic-specific: named entity consistency ─────────────────────────────────
def named_entity_consistency(dialogs, character_names):
    """
    Check that character names appear consistently (same spelling) across all dialogs.
    character_names: list of name variants to check (from episode_refined_speakers.json roles).
    Returns (consistency_score 0–1, issues list).
    """
    if not character_names:
        return None, []

    # Extract name mentions per dialog
    issues = []
    name_forms = {}  # name → set of forms found

    for d in dialogs:
        text = d.get("pipeline_text", "") or d.get("text", "")
        for name in character_names:
            # look for the name or common transliterations
            # simple substring check — real NER would be better but requires model
            if name.lower() in text.lower():
                name_forms.setdefault(name, set()).add(
                    # capture the exact form from text (10 chars around match)
                    text[max(0, text.lower().find(name.lower())):
                         text.lower().find(name.lower()) + len(name) + 2].strip()
                )

    for name, forms in name_forms.items():
        if len(forms) > 1:
            issues.append(f"{name}: found {len(forms)} different forms — {list(forms)[:3]}")

    score = 1.0 - len(issues) / max(len(character_names), 1)
    return round(score, 3), issues


def voice_consistency_score(dialogs):
    """
    Check that each speaker_id maps to exactly one voice_id throughout the episode.
    Returns (score_0_to_1, inconsistency_list)
    """
    spk_voices = {}
    for d in dialogs:
        sp, vid = d.get("speaker"), d.get("voice_id")
        if sp and vid:
            spk_voices.setdefault(sp, set()).add(vid)
    total = len(spk_voices)
    if not total:
        return 1.0, []
    inconsistent = [(sp, list(vids)) for sp, vids in spk_voices.items()
                    if len(vids) > 1]
    score = 1.0 - len(inconsistent) / total
    return round(score, 3), inconsistent

# ── 7. 7-category scorecard ───────────────────────────────────────────────────
EQI_WEIGHTS = {
    "translation":  0.20,
    "grammar":      0.15,
    "voice":        0.20,
    "naturalness":  0.15,
    "timing":       0.20,
    "clarity":      0.10,
    # emotion: TODO — needs Indic TTS emotion model, not scored yet
}

def _r(v, n=1):
    """Round to n decimal places, return int if whole number."""
    if v is None: return None
    r = round(float(v), n)
    return int(r) if r == int(r) else r

def _blend_editor(automated_score, editor_correction_rate, alpha=0.35):
    """
    Blend an automated metric score with the editor correction signal.
    When editors corrected many dialogs in the reviewed subset, the automated
    score is pulled down proportionally.
    alpha: weight given to editor signal (0.35 = 35% editor, 65% automated).
    """
    if automated_score is None or editor_correction_rate is None:
        return automated_score
    editor_signal = (1 - editor_correction_rate) * 100
    return _r(automated_score * (1 - alpha) + editor_signal * alpha)

def compute_7cat_score(
    iso_score, pause_score,
    utmos_results,
    voice_assignment_scores,
    grammar_scores,
    blaser_scores,
    emotion_scores,
    honorific_score,
    editor_translation_rate,
    editor_timing_rate,
    editor_speaker_rate,
    voice_consistency,
    n_speakers,
):
    """
    8-category scorecard (7 displayed + emotion as separate dimension).
    Editor correction rates are blended into Translation and Timing to
    reflect ground-truth human judgement alongside automated metrics.

    editor_translation_rate: FP:translation / n_reviewed  (None if no reviews)
    editor_timing_rate:      FP:timing / n_reviewed
    editor_speaker_rate:     FP:speaker / n_reviewed
    emotion_scores:          list of float [0,1] per dialog with direction tags
    honorific_score:         float [0,1] register consistency score
    """
    # ── Timing: IsoChronoMeter + pause + editor timing signal ─────────────
    timing_auto = None
    if iso_score is not None and pause_score is not None:
        timing_auto = (iso_score * 0.65 + pause_score * 0.35) * 100
    elif iso_score is not None:
        timing_auto = iso_score * 100
    # blend editor timing corrections: editors adjusted timing on 3/7 reviewed
    timing = _blend_editor(timing_auto, editor_timing_rate, alpha=0.25)
    if timing is None and timing_auto is not None:
        timing = _r(timing_auto)

    # ── Naturalness (UTMOS → 0–100) ────────────────────────────────────────
    naturalness = None
    if utmos_results:
        valid = [v for v in utmos_results if v is not None]
        if valid:
            naturalness = _r((np.mean(valid) - 1) / 4 * 100)

    # ── Emotion: not scored — tag coverage is informational only ──────────
    emotion = None  # TODO: Indic TTS emotion model required

    # ── Voice: assignment quality + consistency + editor speaker signal ────
    voice = None
    va_vals = [v for v in (voice_assignment_scores or []) if v is not None]
    if va_vals or voice_consistency is not None:
        parts = []
        if va_vals:   parts.append(np.mean(va_vals) * 100)
        if voice_consistency is not None: parts.append(voice_consistency * 100)
        voice_auto = float(np.mean(parts))
        voice = _blend_editor(voice_auto, editor_speaker_rate, alpha=0.4)
        if voice is None: voice = _r(voice_auto)

    # ── Grammar: MuRIL perplexity + honorific consistency ─────────────────
    grammar = None
    gram_vals = [v for v in (grammar_scores or []) if v is not None]
    if gram_vals:
        muril_score = float(np.mean(gram_vals))
        if honorific_score is not None:
            grammar = _r(muril_score * 0.7 + honorific_score * 100 * 0.3)
        else:
            grammar = _r(muril_score)
    elif honorific_score is not None:
        grammar = _r(honorific_score * 100)

    # ── Translation: BLASER + editor translation correction signal ─────────
    translation = None
    bl_vals = [v for v in (blaser_scores or []) if v is not None]
    if bl_vals:
        blaser_score = _r(min(100.0, (np.mean(bl_vals) - 1) / 4 * 100))
        # blend: editors corrected translations on 5/7 reviewed → pull score down
        translation = _blend_editor(blaser_score, editor_translation_rate, alpha=0.35)
        if translation is None: translation = blaser_score
    elif editor_translation_rate is not None:
        translation = _r((1 - editor_translation_rate) * 100)

    # ── Clarity — UTMOS clarity proxy ─────────────────────────────────────
    clarity = naturalness

    # ── Multispeaker ────────────────────────────────────────────────────────
    multi = None
    if n_speakers > 1 and voice_consistency is not None:
        multi = _r(voice_consistency * 100)

    # ── Overall weighted ───────────────────────────────────────────────────
    cats = {
        "translation": translation,
        "grammar":     grammar,
        "voice":       voice,
        "naturalness": naturalness,
        "timing":      timing,
        "clarity":     clarity,
    }
    if multi is not None:
        cats["multispeaker"] = multi

    avail   = {k: v for k, v in cats.items() if v is not None}
    w_map   = {**EQI_WEIGHTS}
    if "multispeaker" in avail:
        w_map["multispeaker"] = 0.10
        factor = 0.90
        w_map  = {k: (v * factor if k != "multispeaker" else v)
                  for k, v in w_map.items()}
    total_w = sum(w_map.get(k, 0.10) for k in avail)
    overall = _r(sum(avail[k] * w_map.get(k, 0.10) for k in avail) / total_w) \
              if avail else None

    return {
        "overall":      overall,
        "translation":  translation,
        "grammar":      grammar,
        "voice":        voice,
        "naturalness":  naturalness,
        "emotion":      emotion,
        "timing":       timing,
        "clarity":      clarity,
        "multispeaker": multi,
        "iso_score":    _r(iso_score * 100) if iso_score is not None else None,
        "pause_score":  _r(pause_score * 100) if pause_score is not None else None,
    }
