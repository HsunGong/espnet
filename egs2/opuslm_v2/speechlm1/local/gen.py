#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render dialogue JSON/JSONL to a chat-style HTML.
- User/Assistant emoji avatar
- Convert audio messages into spectrogram PNG (scipy + matplotlib)
- Color <think>...</think> segments differently
- Responsive font sizes; customizable font-family
- Container width: 90%, bubble width: 85%
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
        break
    return items


def to_float_mono(sr: int, y: np.ndarray) -> np.ndarray:
    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Convert to float
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    elif y.dtype.kind == "u":
        y = (y.astype(np.float32) - np.iinfo(y.dtype).max / 2) / (np.iinfo(y.dtype).max / 2)
    else:
        y = y.astype(np.float32)

    # Handle NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def save_spectrogram(
    audio_path: Path,
    out_png: Path,
    *,
    figsize: Tuple[float, float] = (20, 1),
    dpi: int = 300,
    xtick_step_sec: float = 2.0,
) -> None:
    # Use soundfile for broad format support (wav, flac, ogg, etc.)
    y, sr = sf.read(str(audio_path))
    y = to_float_mono(sr, y)

    plt.figure(figsize=figsize)
    frequencies, times, spec = signal.spectrogram(y, sr)
    spec_db = 10 * np.log10(spec + 1e-10)

    plt.pcolormesh(times, frequencies / 1000.0, spec_db, shading="gouraud")
    plt.gca().yaxis.set_visible(False)

    if len(times) > 0:
        plt.xlim(0, times[-1])
        # ticks = list(np.arange(0, times[-1], xtick_step_sec))
        # labels = [f"{int(x)}s" for x in ticks]
        
        # # Only add end label if not too close to last tick
        # if not ticks or (times[-1] - ticks[-1] > 0.2):
        #     ticks.append(times[-1])
        #     labels.append(f"{times[-1]:.1f}s")
            
        plt.xticks([0, times[-1]], ["0s", f"{times[-1]:.1f}s"])

    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close()


def escape_and_preserve_newlines(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


def render_text_with_think(s: str) -> str:
    """
    Convert raw text into HTML.
    - Escape HTML
    - Wrap <think>...</think> content in span.think
    """
    # We want to preserve think boundaries before escaping,
    # but still escape think content as text.
    parts: List[str] = []
    last = 0
    for m in THINK_RE.finditer(s):
        pre = s[last : m.start()]
        think_body = m.group(1)

        if pre:
            parts.append(f"<span class='txt'>{escape_and_preserve_newlines(pre)}</span>")

        parts.append(f"<span class='think'>{escape_and_preserve_newlines(think_body)}</span>")
        last = m.end()

    tail = s[last:]
    if tail:
        parts.append(f"<span class='txt'>{escape_and_preserve_newlines(tail)}</span>")

    return "".join(parts) if parts else ""


def guess_role_avatar(role: str) -> str:
    r = role.lower()
    if r == "user":
        return "🙍🏻‍♂️"
    if r == "assistant":
        return "🤖"
    if r == "system":
        return "⚙️ SYSTEM:"
    return "💬"


def msg_to_html(
    role: str,
    mtype: str,
    content: Any,
    assets_rel: str,
    msg_index: int,
    *,
    spectro_height_px: int = 92,
) -> str:
    """
    Render one message block into HTML.
    """
    role_lower = role.lower()
    
    if role_lower == "system":
        return f'<div class="message-row system"><span class="system-msg">⚙️ {escape_and_preserve_newlines(str(content))}</span></div>'

    avatar = guess_role_avatar(role)
    
    body_html = ""
    if mtype == "text":
        body_html = render_text_with_think(str(content))
    elif mtype == "audio":
        # content is now dict {"spec": ..., "orig": ...}
        if isinstance(content, dict):
            img_src = html.escape(content.get("spec", ""))
            orig_name = html.escape(content.get("orig", "audio.wav"))
        else:
            img_src = html.escape(str(content))
            orig_name = os.path.basename(str(content))
            
        body_html = (
            f'<div class="spectrogram-container"><img src="{img_src}" class="spectrogram-img"></div>'
            # f'<span class="audio-label">🎵 {orig_name}</span>'
        )
    elif mtype == "image":
        img_src = html.escape(str(content))
        body_html = f'<img src="{img_src}" class="media-content">'
    else:
        body_html = f"<span class='txt'>{escape_and_preserve_newlines(str(content))}</span>"

    return f"""
    <div class="message-row {role_lower}">
        <div class="avatar {role_lower}">{avatar}</div>
        <div class="bubble">{body_html}</div>
    </div>
    """.strip()


def build_html(
    example_id: str,
    messages: List[List[Any]],
    *,
    title: str,
    font_family: str,
    width_px: int = 800,
    font_size: int = 16,
) -> str:
    bubble_width_pct = 70
    
    css = """
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f2f5;
        margin: 0;
        padding: 10px;
        display: flex;
        justify-content: center;
        min-width: 1330px;
    }
    .chat-container {
        width: 1280px; 
        flex-shrink: 0;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        padding: 4px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        box-sizing: border-box;
    }
    .title {
        font-size: 20px;
        font-weight: bold;
        color: #333;
    }
    .meta {
        font-size: 12px;
        color: #999;
        margin-top: 2px;
    }

    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 2px;
        width: 100%;
        margin: 0 0;
    }
    .message-row.assistant { flex-direction: row-reverse; }
    .message-row.system { justify-content: center; margin: 0px 0; }
    
    /* 无背景头像 */
    .avatar {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
        user-select: none;
        /* padding-top: 5px; */
    }
    
    /* 固定宽度气泡 */
    .bubble {
        width: 90%; 
        padding: 4px 4px;
        border-radius: 12px;
        position: relative;
        font-size: 12px;
        line-height: 1.6;
        word-wrap: break-word;
        box-sizing: border-box;
    }
    .message-row.assistant .bubble {
        background-color: #f7f7f8;
        color: #1f2937;
        border: 1px solid #e5e7eb;
    }
    .message-row.user .bubble {
        background-color: #95ec69;
        color: #000;
    }
    
    .system-msg {
        background-color: #e5e7eb;
        color: #6b7280;
        font-size: 8px;
        padding: 4px 4px;
        border-radius: 12px;
        font-weight: bold;
    }
    .think {
        display: block;
        /* margin-bottom: 15px; */
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 2px solid #94a3b8;
        padding: 4px 4px;
        font-size: 10px;
        color: #64748b;
        font-style: italic;
        border-radius: 4px;
    }
    .media-content {
        width: 100%;
        border-radius: 0px;
        /* margin-top: 2px; */
        display: block;
    }
    .spectrogram-container {
        background: #fff;
        border-radius: 2px;
        overflow: hidden;
        /* margin-top: 2px; */
        width: 100%;
    }
    .spectrogram-img {
        width: 100%;
        display: block;
    }
    .audio-label {
        font-size: 12px;
        color: #666;
        margin-top: 2px;
        display: block;
        text-align: right;
    }
    /* Legacy classes mapping */
    .txt { }
    """.strip()

    blocks = []
    for i, (role, mtype, content) in enumerate(messages):
        blocks.append(msg_to_html(role, mtype, content, assets_rel="", msg_index=i))

    body = "\n".join(blocks)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="chat-container">
    <div class="top">
      <div class="title">{html.escape(title)}</div>
      <div class="meta">example_id: {html.escape(example_id)}</div>
    </div>
    {body}
  </div>
</body>
</html>
"""


def convert_audio_messages_to_spectrograms(
    example: Dict[str, Any],
    out_dir: Path,
    *,
    spectro_dirname: str = "assets",
) -> Dict[str, Any]:
    """
    Replace any message ["*", "audio", "/path/to.wav"] with ["*", "audio", "assets/xxx.png"].
    """
    messages = example.get("messages", [])
    if not isinstance(messages, list):
        return example

    assets_dir = out_dir / spectro_dirname
    ensure_dir(assets_dir)

    new_messages: List[List[Any]] = []
    audio_count = 0

    for idx, m in enumerate(messages):
        if not (isinstance(m, list) and len(m) == 3):
            continue
        role, mtype, content = m

        if mtype == "audio":
            audio_path = Path(str(content))
            out_png = assets_dir / f"spectrogram_{audio_count:04d}.png"
            audio_count += 1

            if not audio_path.exists():
                # keep a placeholder text if audio missing
                new_messages.append([role, "text", f"[Missing audio file] {audio_path}"])
                continue

            try:
                save_spectrogram(audio_path, out_png)
                rel = f"{spectro_dirname}/{out_png.name}"
                new_messages.append([role, "audio", {"spec": rel, "orig": audio_path.name}])
            except Exception as e:
                new_messages.append([role, "text", f"[Audio->spectrogram failed] {audio_path}\nError: {e}"])
        else:
            new_messages.append([role, mtype, content])

    example = dict(example)
    example["messages"] = new_messages
    return example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="/mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_sft/egs2/opuslm_v2/speechlm1/data/sft_part2/v1/stage5_filtered/filtered_imaginary.jsonl",
        help="Path to a JSON or JSONL file.",
    )
    ap.add_argument("--output_dir", type=str, default="chat_output", help="Output directory.")
    ap.add_argument(
        "--font_family",
        type=str,
        default='Arial, sans-serif',
    )
    ap.add_argument("--width_px", type=int, default=800, help="Container width in px.")
    ap.add_argument("--font_size", type=int, default=16, help="Font size in px.")
    ap.add_argument("--title", type=str, default="Dialogue Preview")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    examples = read_json_or_jsonl(in_path)
    if not examples:
        raise SystemExit(f"No examples found in: {in_path}")

    for ex in examples:
        example_id = str(ex.get("example_id", "example"))
        ex_dir = out_root / example_id
        ensure_dir(ex_dir)

        ex2 = convert_audio_messages_to_spectrograms(ex, ex_dir)

        html_path = ex_dir / "index.html"
        html_text = build_html(
            example_id=example_id,
            messages=ex2["messages"],
            title=args.title,
            font_family=args.font_family,
            width_px=args.width_px,
            font_size=args.font_size,
        )
        html_path.write_text(html_text, encoding="utf-8")

        # also dump the transformed json for inspection
        (ex_dir / "rendered.json").write_text(
            json.dumps(ex2, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"[OK] {example_id} -> {html_path}")

    print(
        f"\nDone. Open the generated index.html under: {out_root}/<example_id>/index.html"
    )


if __name__ == "__main__":
    main()
