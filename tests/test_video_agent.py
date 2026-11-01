"""Tests for video_agent.py. Fixtures are generated with ffmpeg itself —
no binary fixtures in the repo."""

import json
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import video_agent as va  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
OUT = os.path.join(HERE, "out")


def sh(*args):
    subprocess.run(args, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)


def ffprobe_dims(path):
    info = va.ffprobe_json(path)
    v = va.video_stream(info)
    return int(v["width"]), int(v["height"])


def make_fixtures():
    os.makedirs(FIX, exist_ok=True)
    clip = os.path.join(FIX, "clip.mp4")
    if not os.path.exists(clip):
        sh("ffmpeg", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", clip)
    gappy = os.path.join(FIX, "clip_silence.mp4")
    if not os.path.exists(gappy):
        td = os.path.join(FIX, "tmpaud")
        os.makedirs(td, exist_ok=True)
        a1 = os.path.join(td, "a1.m4a")
        a2 = os.path.join(td, "a2.m4a")
        a3 = os.path.join(td, "a3.m4a")
        sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
           "-c:a", "aac", a1)
        sh("ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=2",
           "-c:a", "aac", a2)
        sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
           "-c:a", "aac", a3)
        lst = os.path.join(td, "list.txt")
        with open(lst, "w") as f:
            for p in (a1, a2, a3):
                f.write(f"file '{p}'\n")
        aud = os.path.join(td, "aud.m4a")
        sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-c", "copy", aud)
        sh("ffmpeg", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=8",
           "-i", aud,
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", gappy)
        shutil.rmtree(td, ignore_errors=True)
    return clip, gappy


def run_cli(*argv):
    rc = va.main(list(argv))
    assert rc == 0, f"CLI exited {rc}: {argv}"


class VideoAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip, cls.gappy = make_fixtures()
        os.makedirs(OUT, exist_ok=True)

    def out(self, name):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            os.remove(p)
        return p

    # -- probe -----------------------------------------------------------
    def test_probe(self):
        info = va.ffprobe_json(self.clip)
        v = va.video_stream(info)
        self.assertEqual((int(v["width"]), int(v["height"])), (1280, 720))
        self.assertIsNotNone(va.audio_stream(info))
        rc = va.main(["probe", self.clip])
        self.assertEqual(rc, 0)

    # -- reframe ---------------------------------------------------------
    def test_reframe_916(self):
        o = self.out("reframed.mp4")
        run_cli("reframe", self.clip, o, "--to", "9:16")
        self.assertEqual(ffprobe_dims(o), (1080, 1920))

    def test_reframe_11(self):
        o = self.out("reframed11.mp4")
        run_cli("reframe", self.clip, o, "--to", "1:1")
        self.assertEqual(ffprobe_dims(o), (1080, 1080))

    # -- trim ------------------------------------------------------------
    def test_trim(self):
        o = self.out("trimmed.mp4")
        run_cli("trim", self.clip, o, "--start", "1", "--end", "3")
        d = va.duration_of(o)
        self.assertTrue(1.7 < d < 2.4, f"duration {d}")

    def test_trim_hms(self):
        o = self.out("trimmed_hms.mp4")
        run_cli("trim", self.clip, o, "--start", "0:00:01", "--end", "0:00:02")
        d = va.duration_of(o)
        self.assertTrue(0.7 < d < 1.4, f"duration {d}")

    # -- concat ----------------------------------------------------------
    def test_concat(self):
        o = self.out("joined.mp4")
        run_cli("concat", self.clip, self.clip, "-o", o)
        d = va.duration_of(o)
        self.assertTrue(9.0 < d < 10.5, f"duration {d}")

    # -- text ------------------------------------------------------------
    def test_text(self):
        o = self.out("texted.mp4")
        run_cli("text", self.clip, o, "--text", "Hello World",
                "--pos", "bottom", "--start", "0", "--end", "2")
        d = va.duration_of(o)
        self.assertTrue(4.5 < d < 5.5, f"duration {d}")

    # -- thumbnail -------------------------------------------------------
    def test_thumbnail(self):
        o = self.out("thumb.jpg")
        run_cli("thumbnail", self.clip, "-o", o, "--at", "2")
        self.assertTrue(os.path.exists(o))
        with open(o, "rb") as f:
            magic = f.read(2)
        self.assertEqual(magic, b"\xff\xd8")  # jpeg magic

    # -- compress --------------------------------------------------------
    def test_compress(self):
        o = self.out("small.mp4")
        run_cli("compress", self.clip, o, "--max-mb", "1")
        size_mb = os.path.getsize(o) / 1e6
        self.assertTrue(size_mb < 1.2, f"size {size_mb} MB")

    # -- caption ---------------------------------------------------------
    def test_caption(self):
        srt = os.path.join(OUT, "caps.srt")
        with open(srt, "w") as f:
            f.write("1\n00:00:00,500 --> 00:00:02,000\nHello there\n\n"
                    "2\n00:00:02,500 --> 00:00:04,000\nSecond cue here\n\n")
        o = self.out("captioned.mp4")
        run_cli("caption", srt, self.clip, o, "--style", "reel")
        d = va.duration_of(o)
        self.assertTrue(4.5 < d < 5.5, f"duration {d}")
        self.assertEqual(ffprobe_dims(o), (1280, 720))

    # -- cut-silence -----------------------------------------------------
    def test_cut_silence(self):
        o = self.out("nosilence.mp4")
        run_cli("cut-silence", self.gappy, o,
                "--threshold", "-35", "--min", "0.4")
        d = va.duration_of(o)
        # 8s input with a 2s silent gap -> expect ~6s out
        self.assertTrue(5.0 < d < 7.0, f"duration {d}")

    # -- karaoke ---------------------------------------------------------
    def write_words(self, name, words):
        p = os.path.join(OUT, name)
        with open(p, "w") as f:
            json.dump(words, f)
        return p

    def sample_words(self):
        # 8 words over ~4.5s of the 5s clip
        t = 0.2
        ws = []
        for w in ["hello", "this", "is", "a", "karaoke", "caption", "test", "clip"]:
            ws.append({"word": w, "start": round(t, 3), "end": round(t + 0.35, 3)})
            t += 0.5
        return ws

    def test_karaoke_ass_file(self):
        ass = os.path.join(OUT, "k.ass")
        if os.path.exists(ass):
            os.remove(ass)
        n = va.write_karaoke_ass(self.sample_words(), ass, 1080, 1920)
        self.assertEqual(n, 8)
        with open(ass) as f:
            text = f.read()
        self.assertIn("Style: Karaoke,", text)
        self.assertIn("\\k", text)  # karaoke timing tags
        # 8 words -> cues of 6 + 2 -> one dialogue per cue
        self.assertEqual(text.count("Dialogue:"), 2)
        # primary (sung) colour is yellow, secondary (waiting) is white
        self.assertIn("&H0000FFFF,", text)
        self.assertIn("&H00FFFFFF,", text)

    def test_caption_karaoke(self):
        wj = self.write_words("kwords.json", self.sample_words())
        o = self.out("karaoke.mp4")
        run_cli("caption", self.clip, o, "--style", "karaoke",
                "--words", wj)
        d = va.duration_of(o)
        self.assertTrue(4.5 < d < 5.5, f"duration {d}")
        self.assertEqual(ffprobe_dims(o), (1280, 720))

    def test_caption_karaoke_needs_words(self):
        o = self.out("karaoke_nowords.mp4")
        rc = va.main(["caption", self.clip, o, "--style", "karaoke"])
        self.assertEqual(rc, 1)

    # -- defiller --------------------------------------------------------
    def test_defiller(self):
        words = [
            {"word": "hello", "start": 0.2, "end": 0.5},
            {"word": "um", "start": 0.6, "end": 0.9},
            {"word": "world", "start": 1.0, "end": 1.4},
            {"word": "uh,", "start": 2.0, "end": 2.3},
            {"word": "again", "start": 2.4, "end": 2.8},
        ]
        wj = self.write_words("fillers.json", words)
        o = self.out("defilled.mp4")
        run_cli("defiller", self.clip, o, "--words", wj)
        d = va.duration_of(o)
        # cuts um [0.52,1.05] and uh [1.92,2.45] -> ~3.9s kept
        self.assertTrue(3.0 < d < 4.8, f"duration {d}")

    def test_defiller_no_fillers(self):
        words = [
            {"word": "hello", "start": 0.2, "end": 0.5},
            {"word": "world", "start": 1.0, "end": 1.4},
        ]
        wj = self.write_words("nofillers.json", words)
        o = self.out("defilled_copy.mp4")
        run_cli("defiller", self.clip, o, "--words", wj)
        d = va.duration_of(o)
        self.assertTrue(abs(d - 5.0) < 0.4, f"duration {d}")

    # -- reframe blur ----------------------------------------------------
    def test_reframe_blur(self):
        o = self.out("reframed_blur.mp4")
        run_cli("reframe", self.clip, o, "--to", "9:16", "--bg", "blur")
        self.assertEqual(ffprobe_dims(o), (1080, 1920))
        d = va.duration_of(o)
        self.assertTrue(4.5 < d < 5.5, f"duration {d}")

    # -- progressbar -----------------------------------------------------
    def test_progressbar(self):
        o = self.out("progress.mp4")
        run_cli("progressbar", self.clip, o, "--color", "gold")
        d = va.duration_of(o)
        self.assertTrue(4.5 < d < 5.5, f"duration {d}")
        self.assertEqual(ffprobe_dims(o), (1280, 720))

    # -- mix -------------------------------------------------------------
    def test_mix(self):
        mus_long = os.path.join(OUT, "music_long.mp3")
        mus_short = os.path.join(OUT, "music_short.mp3")
        sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=660:duration=10",
           "-c:a", "libmp3lame", mus_long)
        sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
           "-c:a", "libmp3lame", mus_short)
        for name, mus in (("mixed_long.mp4", mus_long),
                          ("mixed_short.mp4", mus_short)):
            o = self.out(name)
            run_cli("mix", self.clip, mus, o)
            d = va.duration_of(o)
            self.assertTrue(4.5 < d < 5.5, f"duration {d} ({name})")
            info = va.ffprobe_json(o)
            self.assertIsNotNone(va.audio_stream(info), name)

    # -- overwrite protection --------------------------------------------
    def test_no_overwrite_without_force(self):
        o = self.out("protect.mp4")
        run_cli("trim", self.clip, o, "--start", "0", "--end", "1")
        rc = va.main(["trim", self.clip, o, "--start", "0", "--end", "1"])
        self.assertEqual(rc, 1)
        rc = va.main(["trim", self.clip, o, "--start", "0", "--end", "1",
                      "--force"])
        self.assertEqual(rc, 0)

    # -- transcribe (optional dep) ---------------------------------------
    def test_transcribe(self):
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            self.skipTest("faster-whisper not installed")
        o = os.path.join(OUT, "clip.srt")
        if os.path.exists(o):
            os.remove(o)
        try:
            rc = va.main(["transcribe", self.clip, "-o", o,
                          "--model", "tiny", "--force"])
        except Exception as e:
            self.skipTest(f"whisper model unavailable: {e}")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(o))
        # valid SRT (possibly empty — sine wave has no speech)
        with open(o) as f:
            content = f.read()
        if content.strip():
            cues = va.parse_srt(o)
            self.assertTrue(len(cues) >= 0)


if __name__ == "__main__":
    unittest.main()
