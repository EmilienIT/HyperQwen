#!/usr/bin/env python3
"""Check that a Vision-enabled server reads pixels, rather than answering from text alone.

Send two locally generated PNGs with different random six-digit numbers and the same
question. The expected numbers appear only in the pixels, never in the prompt text.
No image library, external image host or model files are needed on the client.

  VISION=1 bash single-user/start_qwen.sh   # server, or batch/alternative.sh
  python bench/vision_smoke.py             # client, using bench/harness.py's URL/key

Exit 0: both numbers read correctly; 1: an incorrect/error response; 2: no server answered.
"""
import base64
import random
import re
import struct
import urllib.error
import zlib

import harness

# Five-by-seven glyphs, scaled up for an unambiguous OCR challenge.
DIGITS = (
    "01110/10001/10001/10001/10001/10001/01110",
    "00100/01100/00100/00100/00100/00100/01110",
    "01110/10001/00001/00010/00100/01000/11111",
    "11110/00001/00001/01110/00001/00001/11110",
    "00010/00110/01010/10010/11111/00010/00010",
    "11111/10000/10000/11110/00001/00001/11110",
    "01110/10000/10000/11110/10001/10001/01110",
    "11111/00001/00010/00100/01000/01000/01000",
    "01110/10001/10001/01110/10001/10001/01110",
    "01110/10001/10001/01111/00001/00001/01110",
)


def marker_png(marker):
    """A grayscale PNG containing only the six-digit marker, with no text metadata."""
    if not re.fullmatch(r"[0-9]{6}", marker):
        raise ValueError("marker must contain six ASCII digits")
    width, height, scale = 512, 192, 12
    pixels = bytearray(b"\xff" * width * height)
    for i, digit in enumerate(marker):
        for row, bits in enumerate(DIGITS[int(digit)].split("/")):
            for col, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        offset = (54 + row * scale + dy) * width + 40 + (i * 6 + col) * scale
                        pixels[offset:offset + scale] = b"\0" * scale

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    raw = b"".join(b"\0" + pixels[y * width:(y + 1) * width] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def new_markers():
    return [str(n) for n in random.SystemRandom().sample(range(100000, 1000000), 2)]


def main():
    reached, failures = 0, 0
    for i, marker in enumerate(new_markers(), 1):
        image = "data:image/png;base64," + base64.b64encode(marker_png(marker)).decode("ascii")
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Read the number in this image. Reply with only the six digits."},
                {"type": "image_url", "image_url": {"url": image}},
            ]}],
            "temperature": 0, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            reply = harness.post("/v1/chat/completions", payload, timeout=120)
        except urllib.error.HTTPError as exc:
            reached += 1
            failures += 1
            print(f"FAIL image {i}: HTTP {exc.code}")
            continue
        except OSError as exc:
            failures += 1
            print(f"FAIL image {i}: {type(exc).__name__}")
            continue
        except ValueError:
            reached += 1
            failures += 1
            print(f"FAIL image {i}: invalid JSON response")
            continue
        reached += 1
        try:
            answer = reply["choices"][0]["message"]["content"] or ""
            numbers = re.findall(r"(?<!\d)[0-9]{6}(?!\d)", answer)
        except (KeyError, IndexError, TypeError):
            numbers = []
        if numbers == [marker]:
            print(f"PASS image {i}: number read from pixels")
        else:
            failures += 1
            print(f"FAIL image {i}: image number not read correctly (check VISION=1 and the processor/tower)")
    if not reached:
        print("INVALID: no server answered, so there is no Vision verdict")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
