#!/usr/bin/env python3
"""video_agent.py — CLI video-editing toolkit for Samir's content pipeline.

Pure stdlib + ffmpeg/ffprobe subprocesses. The only optional dependency is
`faster-whisper` (for `transcribe`); everything else works with a stock ffmpeg.

Usage:
    python3 video_agent.py <command> [options]
    python3 video_agent.py reel raw.mp4 reel_final.mp4 --captions on
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class VideoAgentError(Exception):
    pass


def run(cmd, capture=True, check_desc="ffmpeg"):
    """Run a subprocess; raise VideoAgentError with the tail of stderr on failure."""
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None, text=True)
    except FileNotFoundError:
        raise VideoAgentError(f"required binary not found: {cmd[0]}")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise VideoAgentError(
            f"{check_desc} failed (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc


def require_input(path):
    if not path or not os.path.isfile(path):
        raise VideoAgentError(f"input not found: {path}")
    return path


def prepare_output(path, force):
    if os.path.exists(path) and not force:
        raise VideoAgentError(
            f"output exists (use --force to overwrite): {path}")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


def ffprobe_json(path):
    proc = run([FFPROBE, "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", path], check_desc="ffprobe")
    return json.loads(proc.stdout)


def video_stream(info):
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return None


def audio_stream(info):
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def duration_of(path):
    info = ffprobe_json(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        raise VideoAgentError(f"could not determine duration of {path}")


def parse_ts(value):
    """Parse seconds or H:M:S[.ms] into float seconds."""
    value = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)
    parts = value.split(":")
    if len(parts) > 3:
        raise VideoAgentError(f"bad timestamp: {value}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise VideoAgentError(f"bad timestamp: {value}")
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total


def fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def find_font():
    for p in (FONT_BOLD, FONT_REGULAR,
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.isfile(p):
            return p
    raise VideoAgentError("no usable TTF font found on this system")


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

def cmd_probe(args):
    path = require_input(args.input)
    info = ffprobe_json(path)
    v = video_stream(info)
    a = audio_stream(info)
    dur = info.get("format", {}).get("duration", "?")
    print(f"file:     {path}")
    print(f"duration: {dur}s")
    if v:
        fps = v.get("r_frame_rate", "?")
        try:
            n, d = fps.split("/")
            fps = f"{float(n) / float(d):.2f}"
        except Exception:
            pass
        print(f"video:    {v.get('codec_name')} {v.get('width')}x{v.get('height')} "
              f"@{fps}fps pix_fmt={v.get('pix_fmt')}")
    else:
        print("video:    (none)")
    if a:
        print(f"audio:    {a.get('codec_name')} {a.get('sample_rate')}Hz "
              f"{a.get('channels')}ch")
    else:
        print("audio:    (none)")
    print(f"size:     {os.path.getsize(path) / 1e6:.1f} MB")


# --------------------------------------------------------------------------
# reframe
# --------------------------------------------------------------------------

REFRAME_TARGETS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


def cmd_reframe(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    if args.to not in REFRAME_TARGETS:
        raise VideoAgentError(f"bad --to (choose from {list(REFRAME_TARGETS)})")
    tw, th = REFRAME_TARGETS[args.to]
    bg = getattr(args, "bg", "crop")
    if bg not in ("crop", "blur"):
        raise VideoAgentError("--bg must be crop|blur")
    if bg == "crop":
        print(f"reframing to {args.to} ({tw}x{th}, center crop) ...")
        vf = (f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
              f"crop={tw}:{th},format=yuv420p")
        cmd = [FFMPEG, "-y", "-i", src, "-vf", vf,
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", out]
        run(cmd, check_desc="reframe")
    else:
        print(f"reframing to {args.to} ({tw}x{th}, blurred background) ...")
        fg_scale = f"scale={tw}:-2" if tw <= th else f"scale=-2:{th}"
        fc = (f"[0:v]split[a][b];"
              f"[a]scale={tw}:{th}:force_original_aspect_ratio=increase,"
              f"crop={tw}:{th},gblur=sigma=40,eq=brightness=-0.15[bg];"
              f"[b]{fg_scale}[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]")
        cmd = [FFMPEG, "-y", "-i", src, "-filter_complex", fc,
               "-map", "[v]", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", out]
        run(cmd, check_desc="reframe")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# SRT <-> ASS captions
# --------------------------------------------------------------------------

SRT_CUE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.S)


def srt_to_ts(ts):
    ts = ts.replace(",", ".")
    h, m, rest = ts.split(":")
    s = float(rest)
    return int(h) * 3600 + int(m) * 60 + s


def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    cues = []
    for m in SRT_CUE.finditer(text):
        start, end = srt_to_ts(m.group(1)), srt_to_ts(m.group(2))
        body = m.group(3).strip().replace("\r", "")
        body = re.sub(r"<[^>]+>", "", body)  # strip html-ish tags
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines and end > start:
            cues.append((start, end, " ".join(lines)))
    if not cues:
        raise VideoAgentError(f"no cues parsed from {path}")
    return cues


def ass_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(cues, ass_path, width, height, style):
    if style == "reel":
        size = max(28, int(height * 0.052))
        outline, marginv = 3, int(height * 0.09)
    else:  # simple
        size = max(20, int(height * 0.035))
        outline, marginv = 2, int(height * 0.05)
    # ASS colours are &HAABBGGRR
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,{size},&H00FFFFFF,&H000019FF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,24,24,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def esc(t):
        return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    lines = [header]
    for start, end, text in cues:
        lines.append(f"Dialogue: 0,{ass_ts(start)},{ass_ts(end)},Cap,,0,0,0,,{esc(text)}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_karaoke_ass(words, ass_path, width, height):
    """Word-by-word highlight captions using ASS karaoke timing.

    words: list of {"word", "start", "end"} dicts.
    Groups words into cues (max 6 words or 2.8s); each cue is ONE Dialogue
    line with a {\\k<centiseconds>} tag per word. libass flips each word from
    the style's SecondaryColour (white, waiting) to PrimaryColour (yellow,
    sung) as its time arrives — the classic viral-caption look.
    Returns the number of words written.
    """
    size = max(40, int(64 * width / 1080))
    marginv = int(height * 0.09)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,DejaVu Sans,{size},&H0000FFFF,&H00FFFFFF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,24,24,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def esc(t):
        return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    clean = [w for w in words
             if str(w.get("word", "")).strip()
             and float(w.get("end", 0)) > float(w.get("start", 0))]
    cues, cur = [], []
    for w in clean:
        cur.append(w)
        span = float(cur[-1]["end"]) - float(cur[0]["start"])
        if len(cur) >= 6 or span >= 2.8:
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    lines = [header]
    for cue in cues:
        c0, c1 = float(cue[0]["start"]), float(cue[-1]["end"])
        parts = []
        for w in cue:
            cs = max(1, int(round((float(w["end"]) - float(w["start"])) * 100)))
            parts.append(f"{{\\k{cs}}}{esc(str(w['word']).strip())}")
        text = " ".join(parts)
        lines.append(f"Dialogue: 0,{ass_ts(c0)},{ass_ts(c1)},Karaoke,,0,0,0,,{text}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(clean)


def cmd_caption(args):
    # Karaoke can be called with 2 positionals (srt omitted):
    #   caption INPUT OUTPUT --style karaoke --words w.words.json
    srt, src_arg, out_arg = args.srt, args.input, args.output
    if args.style == "karaoke" and out_arg is None and args.words:
        srt, src_arg, out_arg = None, srt, src_arg
    src = require_input(src_arg)
    if not out_arg:
        raise VideoAgentError("caption needs input and output paths")
    out = prepare_output(out_arg, args.force)
    style = args.style
    if style == "karaoke":
        if not args.words:
            raise VideoAgentError(
                "karaoke style needs --words <words.json> "
                "(run: transcribe --word-timestamps)")
        with open(require_input(args.words), encoding="utf-8") as f:
            words = json.load(f)

        def make_ass(ass_path, w, h):
            n = write_karaoke_ass(words, ass_path, w, h)
            print(f"burning karaoke captions ({n} words) ...")
    else:
        if not args.srt:
            raise VideoAgentError(
                "caption needs an .srt file (or use --style karaoke --words)")
        cues = parse_srt(require_input(args.srt))

        def make_ass(ass_path, w, h):
            write_ass(cues, ass_path, w, h, style)
            print(f"burning {len(cues)} cues ({style} style) ...")
    info = ffprobe_json(src)
    v = video_stream(info)
    if not v:
        raise VideoAgentError("input has no video stream")
    w, h = int(v["width"]), int(v["height"])
    with tempfile.TemporaryDirectory(prefix="va_cap_") as td:
        ass = os.path.join(td, "subs.ass")
        make_ass(ass, w, h)
        # libass needs the path escaped for the filter parser
        ass_esc = ass.replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")
        cmd = [FFMPEG, "-y", "-i", src, "-vf", f"ass='{ass_esc}'",
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-c:a", "copy", "-movflags", "+faststart", out]
        run(cmd, check_desc="caption")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# transcribe (optional faster-whisper)
# --------------------------------------------------------------------------

def srt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


def cmd_transcribe(args):
    src = require_input(args.input)
    out = args.output or os.path.splitext(src)[0] + ".srt"
    if os.path.exists(out) and not args.force:
        raise VideoAgentError(f"output exists (use --force to overwrite): {out}")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise VideoAgentError(
            "faster-whisper is not installed. Install it with:\n"
            "  TMPDIR=~/workspace/.tmp pip install --break-system-packages faster-whisper\n"
            "then re-run transcribe.")
    print(f"loading faster-whisper model '{args.model}' (first run downloads it) ...")
    try:
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    except Exception as e:
        raise VideoAgentError(f"could not load whisper model '{args.model}': {e}")
    print("transcribing (word timestamps) ...")
    try:
        segments, _info = model.transcribe(src, language=args.lang or None,
                                           word_timestamps=True, vad_filter=True)
        words = []
        for seg in segments:
            if seg.words:
                words.extend(seg.words)
            elif seg.text.strip():
                # fallback: no word timings, split evenly across the segment
                toks = seg.text.split()
                dur = max(0.01, seg.end - seg.start)
                for i, tok in enumerate(toks):
                    t0 = seg.start + dur * i / len(toks)
                    t1 = seg.start + dur * (i + 1) / len(toks)
                    words.append(type("W", (), {"word": tok, "start": t0, "end": t1}))
    except Exception as e:
        raise VideoAgentError(f"transcription failed: {e}")
    # collapse words into phrase cues: max 8 words or 3.0s per cue
    cues, cur = [], []
    for w in words:
        txt = w.word.strip()
        if not txt:
            continue
        cur.append(w)
        span = cur[-1].end - cur[0].start
        if len(cur) >= 8 or span >= 3.0:
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            text = " ".join(w.word.strip() for w in cue)
            f.write(f"{i}\n{srt_ts(cue[0].start)} --> {srt_ts(cue[-1].end)}\n{text}\n\n")
    print(f"wrote {out} ({len(cues)} cues)")
    if getattr(args, "word_timestamps", False):
        wj = os.path.splitext(out)[0] + ".words.json"
        wdata = [{"word": w.word.strip(),
                  "start": round(float(w.start), 3),
                  "end": round(float(w.end), 3)}
                 for w in words if w.word.strip()]
        with open(wj, "w", encoding="utf-8") as f:
            json.dump(wdata, f, ensure_ascii=False)
        print(f"wrote {wj} ({len(wdata)} words)")
    if not cues:
        print("note: no speech detected in this clip")


# --------------------------------------------------------------------------
# cut-silence (silencedetect -> select/aselect, keeps A/V in sync)
# --------------------------------------------------------------------------

def detect_silences(path, threshold_db, min_dur):
    proc = run([FFMPEG, "-hide_banner", "-i", path,
                "-af", f"silencedetect=noise={threshold_db}dB:d={min_dur}",
                "-f", "null", "-"], check_desc="silence detect")
    starts, ends = [], []
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m:
            starts.append(float(m.group(1)))
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m:
            ends.append(float(m.group(1)))
    # pair them; a trailing start without end extends to EOF
    dur = duration_of(path)
    silences = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else dur
        if e > s:
            silences.append((s, e))
    return silences, dur


def _keep_segments(src, out, keeps, desc):
    """Keep only keeps=[(start,end)...] of src (complement was cut).

    Re-encodes with select/aselect so A/V stay in sync."""
    if not keeps:
        raise VideoAgentError(f"{desc}: nothing left to keep")
    sel = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keeps)
    cmd = [FFMPEG, "-y", "-i", src,
           "-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB",
           "-af", f"aselect='{sel}',asetpts=N/SR/TB",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", out]
    run(cmd, check_desc=desc)


def cmd_cut_silence(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    print(f"detecting silence (< {args.threshold}dB, >= {args.min}s) ...")
    silences, dur = detect_silences(src, args.threshold, args.min)
    if not silences:
        print("no silence found — copying input unchanged")
        shutil.copyfile(src, out)
        return
    # keep-segments = complement of silences
    keeps = []
    cursor = 0.0
    for s, e in sorted(silences):
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < dur:
        keeps.append((cursor, dur))
    total_kept = sum(e - s for s, e in keeps)
    print(f"cutting {len(silences)} silent spans, keeping {total_kept:.1f}s of {dur:.1f}s ...")
    _keep_segments(src, out, keeps, "cut-silence")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# defiller — cut filler words (um/uh/...) using word-level timings
# --------------------------------------------------------------------------

DEFAULT_FILLERS = "um,uh,ah,er,erm,hmm,uhm"

PUNCT_STRIP = ".,!?;:'\"()[]\u2026\u2014-"


def load_words_json(path):
    with open(require_input(path), encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for w in data:
        try:
            words.append({"word": str(w["word"]),
                          "start": float(w["start"]),
                          "end": float(w["end"])})
        except (KeyError, ValueError, TypeError):
            continue
    return [w for w in words if w["word"].strip() and w["end"] > w["start"]]


def cmd_defiller(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    fillers = {f.strip().lower() for f in args.fillers.split(",") if f.strip()}
    if not fillers:
        raise VideoAgentError("--fillers is empty")
    if args.words:
        words = load_words_json(args.words)
    else:
        print("no --words given, transcribing first (tiny model) ...")
        with tempfile.TemporaryDirectory(prefix="va_defill_") as td:
            srt = os.path.join(td, "w.srt")
            ns = argparse.Namespace(input=src, output=srt, model="tiny",
                                    lang="en", force=True, word_timestamps=True)
            try:
                cmd_transcribe(ns)
            except VideoAgentError as e:
                raise VideoAgentError(f"defiller needs transcription: {e}")
            words = load_words_json(os.path.splitext(srt)[0] + ".words.json")
    if not words:
        print("no words found — copying input unchanged")
        shutil.copyfile(src, out)
        return
    dur = duration_of(src)
    cuts = []
    for w in words:
        txt = w["word"].strip().lower().strip(PUNCT_STRIP)
        if txt in fillers:
            s = max(0.0, w["start"] - 0.08)
            e = min(dur, w["end"] + 0.15)
            if e > s:
                cuts.append((s, e))
    cuts.sort()
    merged = []
    for s, e in cuts:
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    if not merged:
        print("no filler words found — copying input unchanged")
        shutil.copyfile(src, out)
        return
    keeps = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < dur:
        keeps.append((cursor, dur))
    total_cut = sum(e - s for s, e in merged)
    print(f"cutting {len(merged)} filler spans ({total_cut:.1f}s) ...")
    _keep_segments(src, out, keeps, "defiller")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# trim
# --------------------------------------------------------------------------

def cmd_trim(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    start = parse_ts(args.start)
    end = parse_ts(args.end)
    dur = duration_of(src)
    if not (0 <= start < end <= dur + 0.05):
        raise VideoAgentError(
            f"bad range: start={start:.2f} end={end:.2f} (duration {dur:.2f}s)")
    print(f"trimming {fmt_ts(start)} -> {fmt_ts(end)} ...")
    cmd = [FFMPEG, "-y", "-i", src, "-ss", str(start), "-to", str(end),
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", out]
    run(cmd, check_desc="trim")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# concat (normalize each input, then concat demuxer with -c copy)
# --------------------------------------------------------------------------

def cmd_concat(args):
    if len(args.inputs) < 2:
        raise VideoAgentError("concat needs at least 2 inputs")
    out = prepare_output(args.output, args.force)
    for p in args.inputs:
        require_input(p)
    print(f"normalizing {len(args.inputs)} inputs ...")
    with tempfile.TemporaryDirectory(prefix="va_concat_") as td:
        normed = []
        for i, src in enumerate(args.inputs):
            dst = os.path.join(td, f"n{i}.mp4")
            info = ffprobe_json(src)
            has_audio = audio_stream(info) is not None
            vf = "scale=1080:-2:force_original_aspect_ratio=decrease,fps=30,format=yuv420p"
            cmd = [FFMPEG, "-y", "-i", src, "-vf", vf,
                   "-c:v", "libx264", "-preset", "medium", "-crf", "20"]
            if has_audio:
                cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
            else:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-c:a", "aac", "-b:a", "128k", "-shortest",
                        "-map", "0:v", "-map", "1:a"]
            cmd += ["-movflags", "+faststart", dst]
            run(cmd, check_desc=f"concat normalize ({src})")
            normed.append(dst)
        lst = os.path.join(td, "list.txt")
        with open(lst, "w") as f:
            for p in normed:
                f.write(f"file '{p}'\n")
        print("concatenating ...")
        run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst,
             "-c", "copy", "-movflags", "+faststart", out],
            check_desc="concat")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# text overlay
# --------------------------------------------------------------------------

def drawtext_escape(s):
    return (s.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
             .replace("%", "\\%").replace(",", "\\,"))


def cmd_text(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    font = find_font()
    y_map = {"top": "80", "center": "(h-text_h)/2", "bottom": "h-text_h-80"}
    if args.pos not in y_map:
        raise VideoAgentError("--pos must be top|center|bottom")
    filt = (f"drawtext=fontfile='{font}':text='{drawtext_escape(args.text)}':"
            f"fontsize={args.size}:fontcolor={args.color}:"
            f"borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y={y_map[args.pos]}")
    if args.start is not None or args.end is not None:
        s = parse_ts(args.start) if args.start is not None else 0
        e = parse_ts(args.end) if args.end is not None else duration_of(src)
        filt += f":enable='between(t,{s},{e})'"
    print(f"overlaying text ({args.pos}) ...")
    cmd = [FFMPEG, "-y", "-i", src, "-vf", filt,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "copy", "-movflags", "+faststart", out]
    run(cmd, check_desc="text")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# thumbnail
# --------------------------------------------------------------------------

def cmd_thumbnail(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    at = parse_ts(args.at)
    print(f"grabbing frame at {at:.1f}s ...")
    run([FFMPEG, "-y", "-ss", str(at), "-i", src, "-frames:v", "1",
         "-vf", "scale=1280:-2", "-q:v", "3", out], check_desc="thumbnail")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# compress
# --------------------------------------------------------------------------

def cmd_compress(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    dur = duration_of(src)
    if dur <= 0:
        raise VideoAgentError("zero-duration input")
    audio_br = 128  # kbps
    total_bits = args.max_mb * 1024 * 1024 * 8 * 0.90
    video_br = int(total_bits / dur / 1000) - audio_br
    if video_br < 200:
        raise VideoAgentError(
            f"target {args.max_mb}MB too small for {dur:.0f}s of video")
    maxrate = int(video_br * 1.5)
    bufsize = int(video_br * 3)
    info = ffprobe_json(src)
    # cap the largest dimension at 1920 (leaves 1080x1920 reels untouched,
    # shrinks 4K); force_original_aspect_ratio=decrease never upscales
    vf = "scale=1920:1920:force_original_aspect_ratio=decrease,format=yuv420p"
    print(f"compressing to ~{args.max_mb}MB "
          f"(video {video_br}k, audio {audio_br}k, {dur:.0f}s) ...")
    cmd = [FFMPEG, "-y", "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "medium",
           "-b:v", f"{video_br}k", "-maxrate", f"{maxrate}k",
           "-bufsize", f"{bufsize}k",
           "-c:a", "aac", "-b:a", f"{audio_br}k",
           "-movflags", "+faststart", out]
    run(cmd, check_desc="compress")
    got_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out} ({got_mb:.1f} MB)")


# --------------------------------------------------------------------------
# progressbar — thin progress bar along the bottom
# --------------------------------------------------------------------------

PROGRESS_COLORS = {"gold": "0xB07D28", "white": "white", "red": "red",
                   "yellow": "yellow"}


def parse_bar_color(c):
    c = c.strip()
    if c.startswith("#"):
        return "0x" + c[1:]
    return PROGRESS_COLORS.get(c.lower(), c)


def cmd_progressbar(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    info = ffprobe_json(src)
    v = video_stream(info)
    if not v:
        raise VideoAgentError("input has no video stream")
    w, h = int(v["width"]), int(v["height"])
    try:
        dur = float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        raise VideoAgentError(f"could not determine duration of {src}")
    if dur <= 0:
        raise VideoAgentError("zero-duration input")
    col = parse_bar_color(args.color)
    bh = args.height
    # NOTE: this ffmpeg's drawbox does not accept W/H constants in
    # expressions, so bake the probed dimensions in as plain numbers.
    # `t` (frame timestamp) does work.
    bx, bw, by = 20, w - 40, h - 30
    vf = (f"drawbox=x={bx}:y={by}:w={bw}:h={bh}:color=black@0.35:t=fill,"
          f"drawbox=x={bx}:y={by}:w='min({bw},{bw}*t/{dur:.3f})':h={bh}:"
          f"color={col}:t=fill")
    print(f"adding progress bar ({args.color}) ...")
    cmd = [FFMPEG, "-y", "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "copy", "-movflags", "+faststart", out]
    run(cmd, check_desc="progressbar")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# mix — background music ducked under the voice
# --------------------------------------------------------------------------

def cmd_mix(args):
    src = require_input(args.input)
    mus = require_input(args.music)
    out = prepare_output(args.output, args.force)
    info = ffprobe_json(src)
    if audio_stream(info) is None:
        raise VideoAgentError("input video has no audio track to duck against")
    dur = duration_of(src)
    ratio = min(20.0, max(1.5, 1.0 / max(args.duck, 0.05)))
    fc = (
        f"[1:a]volume={args.music_db}dB,aresample=44100,"
        f"aloop=loop=-1:size=2000000000,atrim=0:{dur:.3f},"
        f"asetpts=PTS-STARTPTS,aformat=channel_layouts=stereo[mus];"
        f"[0:a]aresample=44100,aformat=channel_layouts=stereo,asplit[v1][v2];"
        f"[v1]acompressor=threshold=-22dB:ratio=3:attack=200:release=800[ctl];"
        f"[mus][ctl]sidechaincompress=threshold=0.08:ratio={ratio:.1f}:"
        f"attack=200:release=1000[musd];"
        f"[v2][musd]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )
    print(f"mixing music (ducked to ~{args.duck:.0%} under voice, {args.music_db}dB) ...")
    cmd = [FFMPEG, "-y", "-i", src, "-stream_loop", "-1", "-i", mus,
           "-filter_complex", fc,
           "-map", "0:v:0", "-map", "[aout]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", out]
    run(cmd, check_desc="mix")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# reel: defiller -> reframe -> transcribe -> captions -> progress -> music -> compress
# --------------------------------------------------------------------------

def cmd_reel(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    if args.captions not in ("on", "off"):
        raise VideoAgentError("--captions must be on|off")
    karaoke = args.karaoke
    step = 0

    def nxt(label):
        nonlocal step
        step += 1
        print(f"[{step}] {label} ...")

    with tempfile.TemporaryDirectory(prefix="va_reel_") as td:
        def stage(name):
            return os.path.join(td, name + ".mp4")

        cur = src
        if args.defiller:
            nxt("removing filler words")
            d1 = stage("defilled")
            cmd_defiller(argparse.Namespace(
                input=cur, output=d1, words=None,
                fillers=DEFAULT_FILLERS, force=True))
            cur = d1

        nxt("reframing to 9:16" + (" (blurred background)" if args.bg_blur else ""))
        r1 = stage("reframed")
        cmd_reframe(argparse.Namespace(
            input=cur, output=r1, to="9:16",
            bg="blur" if args.bg_blur else "crop", force=True))
        cur = r1

        srt = os.path.join(td, "caps.srt")
        if karaoke or args.captions == "on":
            nxt("transcribing" + (" (word timestamps)" if karaoke else ""))
            try:
                cmd_transcribe(argparse.Namespace(
                    input=cur, output=srt, model="tiny", lang="en",
                    force=True, word_timestamps=karaoke))
                c1 = stage("captioned")
                if karaoke:
                    nxt("burning karaoke captions")
                    cmd_caption(argparse.Namespace(
                        input=cur, output=c1, srt=None, style="karaoke",
                        words=os.path.splitext(srt)[0] + ".words.json",
                        force=True))
                else:
                    nxt("burning captions")
                    cmd_caption(argparse.Namespace(
                        input=cur, output=c1, srt=srt, style="reel",
                        words=None, force=True))
                cur = c1
            except VideoAgentError as e:
                print(f"WARNING: captions skipped ({e}); continuing without them")
        else:
            print("(captions off — skipping transcribe + burn)")

        if args.progress:
            nxt("adding progress bar")
            p1 = stage("progress")
            cmd_progressbar(argparse.Namespace(
                input=cur, output=p1, color="gold", height=6, force=True))
            cur = p1

        if args.music:
            nxt("mixing background music")
            m1 = stage("mixed")
            cmd_mix(argparse.Namespace(
                input=cur, music=args.music, output=m1,
                duck=0.25, music_db=-18, force=True))
            cur = m1

        nxt("compressing to <=90MB")
        cmd_compress(argparse.Namespace(
            input=cur, output=out, max_mb=90, force=True))
    print(f"DONE -> {out}")


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--force", action="store_true",
                   help="overwrite output if it exists")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="video_agent.py",
        description="CLI video-editing toolkit (ffmpeg-powered).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="inspect a video file")
    p.add_argument("input")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("reframe", help="crop+scale to an aspect ratio")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--to", default="9:16", help="9:16|1:1|16:9 (default 9:16)")
    p.add_argument("--bg", default="crop",
                   help="crop = center crop (default) | blur = blurred-background "
                        "reframe, keeps full frame")
    add_common(p); p.set_defaults(func=cmd_reframe)

    p = sub.add_parser("caption", help="burn subtitles into video")
    p.add_argument("srt", nargs="?", default=None,
                   help="subtitle file (omit for --style karaoke: "
                        "caption INPUT OUTPUT --style karaoke --words w.json)")
    p.add_argument("input", nargs="?", default=None)
    p.add_argument("output", nargs="?", default=None)
    p.add_argument("--style", default="reel", help="reel|simple|karaoke")
    p.add_argument("--words", default=None,
                   help="words.json from 'transcribe --word-timestamps' "
                        "(required for karaoke)")
    add_common(p); p.set_defaults(func=cmd_caption)

    p = sub.add_parser("transcribe", help="speech -> SRT (needs faster-whisper)")
    p.add_argument("input")
    p.add_argument("--model", default="tiny", help="tiny|base")
    p.add_argument("--lang", default="en", help="language code (default en)")
    p.add_argument("--word-timestamps", action="store_true",
                   help="also write <output>.words.json with per-word timings "
                        "(needed for karaoke captions and defiller)")
    p.add_argument("-o", "--output", default=None, help="output .srt path")
    add_common(p); p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("cut-silence", help="remove silent spans")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--threshold", type=float, default=-35, help="dB (default -35)")
    p.add_argument("--min", type=float, default=0.4, help="min silence seconds")
    add_common(p); p.set_defaults(func=cmd_cut_silence)

    p = sub.add_parser("defiller", help="cut filler words (um/uh/...)")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--words", default=None,
                   help="words.json (from 'transcribe --word-timestamps'); "
                        "if omitted, transcribes with the tiny model")
    p.add_argument("--fillers", default=DEFAULT_FILLERS,
                   help=f"comma-separated filler words (default: {DEFAULT_FILLERS})")
    add_common(p); p.set_defaults(func=cmd_defiller)

    p = sub.add_parser("trim", help="cut a time range")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    add_common(p); p.set_defaults(func=cmd_trim)

    p = sub.add_parser("concat", help="join clips")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", required=True)
    add_common(p); p.set_defaults(func=cmd_concat)

    p = sub.add_parser("text", help="text overlay")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--text", required=True)
    p.add_argument("--pos", default="bottom", help="top|center|bottom")
    p.add_argument("--start", default=None); p.add_argument("--end", default=None)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--color", default="white")
    add_common(p); p.set_defaults(func=cmd_text)

    p = sub.add_parser("thumbnail", help="extract a still")
    p.add_argument("input"); p.add_argument("-o", "--output", required=True)
    p.add_argument("--at", default="3", help="timestamp (default 3s)")
    add_common(p); p.set_defaults(func=cmd_thumbnail)

    p = sub.add_parser("compress", help="shrink to a target size")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--max-mb", type=float, default=90)
    add_common(p); p.set_defaults(func=cmd_compress)

    p = sub.add_parser("progressbar", help="thin progress bar along the bottom")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--color", default="gold",
                   help="gold|white|red|yellow or #RRGGBB (default gold)")
    p.add_argument("--height", type=int, default=6, help="bar height px")
    add_common(p); p.set_defaults(func=cmd_progressbar)

    p = sub.add_parser("mix", help="background music ducked under voice")
    p.add_argument("input"); p.add_argument("music"); p.add_argument("output")
    p.add_argument("--duck", type=float, default=0.25,
                   help="music level kept while voice is active (default 0.25)")
    p.add_argument("--music-db", type=float, default=-18,
                   help="music base volume in dB (default -18)")
    add_common(p); p.set_defaults(func=cmd_mix)

    p = sub.add_parser("reel", help="raw clip -> captioned 9:16 reel")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--captions", default="on", help="on|off")
    p.add_argument("--karaoke", action="store_true",
                   help="word-by-word highlight captions (needs faster-whisper)")
    p.add_argument("--defiller", action="store_true",
                   help="cut filler words first (needs faster-whisper)")
    p.add_argument("--bg-blur", action="store_true",
                   help="blurred-background reframe instead of center crop")
    p.add_argument("--progress", action="store_true",
                   help="add gold progress bar")
    p.add_argument("--music", default=None,
                   help="background music file (auto-ducked under voice)")
    add_common(p); p.set_defaults(func=cmd_reel)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except VideoAgentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
