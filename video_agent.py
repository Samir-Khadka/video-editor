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
    vf = (f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
          f"crop={tw}:{th},format=yuv420p")
    print(f"reframing to {args.to} ({tw}x{th}) ...")
    cmd = [FFMPEG, "-y", "-i", src, "-vf", vf,
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


def cmd_caption(args):
    src = require_input(args.input)
    srt = require_input(args.srt)
    out = prepare_output(args.output, args.force)
    cues = parse_srt(srt)
    info = ffprobe_json(src)
    v = video_stream(info)
    if not v:
        raise VideoAgentError("input has no video stream")
    w, h = int(v["width"]), int(v["height"])
    print(f"burning {len(cues)} cues ({args.style} style) ...")
    with tempfile.TemporaryDirectory(prefix="va_cap_") as td:
        ass = os.path.join(td, "subs.ass")
        write_ass(cues, ass, w, h, args.style)
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
    if not keeps:
        raise VideoAgentError("whole clip is silence — nothing to keep")
    total_kept = sum(e - s for s, e in keeps)
    sel = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keeps)
    print(f"cutting {len(silences)} silent spans, keeping {total_kept:.1f}s of {dur:.1f}s ...")
    cmd = [FFMPEG, "-y", "-i", src,
           "-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB",
           "-af", f"aselect='{sel}',asetpts=N/SR/TB",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", out]
    run(cmd, check_desc="cut-silence")
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
# reel: reframe -> transcribe -> captions -> compress
# --------------------------------------------------------------------------

def cmd_reel(args):
    src = require_input(args.input)
    out = prepare_output(args.output, args.force)
    if args.captions not in ("on", "off"):
        raise VideoAgentError("--captions must be on|off")
    with tempfile.TemporaryDirectory(prefix="va_reel_") as td:
        stage1 = os.path.join(td, "reframed.mp4")
        print("[1/4] reframing to 9:16 ...")
        ns = argparse.Namespace(input=src, output=stage1, to="9:16", force=True)
        cmd_reframe(ns)

        srt = os.path.join(td, "caps.srt")
        captioned = os.path.join(td, "captioned.mp4")
        if args.captions == "on":
            print("[2/4] transcribing ...")
            try:
                nt = argparse.Namespace(input=stage1, output=srt, model="tiny",
                                        lang="en", force=True)
                cmd_transcribe(nt)
                print("[3/4] burning captions ...")
                nc = argparse.Namespace(input=stage1, output=captioned,
                                        srt=srt, style="reel", force=True)
                cmd_caption(nc)
            except VideoAgentError as e:
                print(f"WARNING: captions skipped ({e}); continuing without them")
                captioned = stage1
        else:
            print("[2/4] captions off — skipping")
            print("[3/4] captions off — skipping")
            captioned = stage1

        print("[4/4] compressing to <=90MB ...")
        nk = argparse.Namespace(input=captioned, output=out, max_mb=90, force=True)
        cmd_compress(nk)
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
    add_common(p); p.set_defaults(func=cmd_reframe)

    p = sub.add_parser("caption", help="burn subtitles into video")
    p.add_argument("srt"); p.add_argument("input"); p.add_argument("output")
    p.add_argument("--style", default="reel", help="reel|simple")
    add_common(p); p.set_defaults(func=cmd_caption)

    p = sub.add_parser("transcribe", help="speech -> SRT (needs faster-whisper)")
    p.add_argument("input")
    p.add_argument("--model", default="tiny", help="tiny|base")
    p.add_argument("--lang", default="en", help="language code (default en)")
    p.add_argument("-o", "--output", default=None, help="output .srt path")
    add_common(p); p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("cut-silence", help="remove silent spans")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--threshold", type=float, default=-35, help="dB (default -35)")
    p.add_argument("--min", type=float, default=0.4, help="min silence seconds")
    add_common(p); p.set_defaults(func=cmd_cut_silence)

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

    p = sub.add_parser("reel", help="raw clip -> captioned 9:16 reel")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--captions", default="on", help="on|off")
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
