"""Очищає YouTube auto-VTT субтитри в чистий текст.

YouTube VTT для авто-субтитрів містить багато дублювань через rolling captions
(той самий рядок з інкрементальною підсвіткою через <c> теги). Беремо тільки
"фінальні" версії рядків.
"""
import re
import sys
from pathlib import Path

CUE_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
TAG = re.compile(r"<[^>]+>")
HTML_ENTITIES = {
    "&gt;": "",   # > використовується для позначки спікера, нам не потрібно
    "&lt;": "",
    "&amp;": "&",
    "&quot;": '"',
    "&#39;": "'",
    "&nbsp;": " ",
}


def clean_vtt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_lines: list[str] = []
    in_cue = False
    cue_lines: list[str] = []

    def flush_cue():
        if not cue_lines:
            return
        # Беремо ОСТАННІЙ непустий рядок з cue — він зазвичай повний фінальний текст
        for line in reversed(cue_lines):
            line = TAG.sub("", line).strip()
            for ent, repl in HTML_ENTITIES.items():
                line = line.replace(ent, repl)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                out_lines.append(line)
                break

    for line in raw:
        if CUE_TIME.match(line):
            flush_cue()
            cue_lines = []
            in_cue = True
            continue
        if not line.strip():
            flush_cue()
            cue_lines = []
            in_cue = False
            continue
        if in_cue:
            cue_lines.append(line)
    flush_cue()

    # Дедуп послідовних дублікатів (rolling captions часто повторюють попередній рядок)
    dedup: list[str] = []
    for ln in out_lines:
        if not dedup or dedup[-1] != ln:
            dedup.append(ln)

    text = " ".join(dedup)
    text = re.sub(r"\s+", " ", text).strip()
    # Грубий розбір на речення для читабельності
    text = re.sub(r"([.!?])\s+", r"\1\n", text)
    return text


def main():
    src_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    ids_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    if ids_file:
        wanted = {ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()}
        vtt_files = [src_dir / f"{vid}.uk.vtt" for vid in sorted(wanted)]
        vtt_files = [p for p in vtt_files if p.exists()]
    else:
        vtt_files = sorted(src_dir.glob("*.vtt"))

    chunks: list[str] = []
    for vtt in vtt_files:
        cleaned = clean_vtt(vtt)
        if cleaned:
            chunks.append(cleaned)
            print(f"{vtt.name}: {len(cleaned):,} chars")
    full = "\n\n".join(chunks)
    out_path.write_text(full, encoding="utf-8")
    print(f"\nFiles: {len(chunks)} | Total: {len(full):,} chars → {out_path}")


if __name__ == "__main__":
    main()