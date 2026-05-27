from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "figures_quiz"
MANIFEST_PATH = OUT_DIR / "manifest.json"
ANSWERS_DIR = OUT_DIR / "strong_answers"
ANSWERS_JSON = OUT_DIR / "strong_answers.json"


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def build_prompt(item: dict[str, str]) -> str:
    return f"""You are creating a strong final-answer version for an MBA Deep Learning exam study card.
Use the attached image and answer the question below. Write a polished final answer, not a checklist.
Make it 3-5 concise paragraphs. Cover all subquestions explicitly, with enough depth for exam revision.
Use terminology accurately, but keep the explanation understandable for an MBA deep-learning course.
Do not mention that you are an AI or that you cannot see the image. Do not include markdown bullets.

Card:
Source: {item["source"]}
Figure: {item["figure_id"]}
Caption: {item["caption"]}

Question:
{item["question"]}
"""


def run_codex(item: dict[str, str], answer_path: Path, log_path: Path) -> None:
    image_path = OUT_DIR / item["image"]
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--cd",
        str(ROOT),
        "--image",
        str(image_path),
        "-o",
        str(answer_path),
        "-",
    ]
    result = subprocess.run(
        cmd,
        input=build_prompt(item),
        text=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed for {item['source']} {item['figure_id']}; see {log_path}")
    if not answer_path.exists() or not answer_path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"codex exec produced no answer for {item['source']} {item['figure_id']}")


def main() -> None:
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    workers = 4

    jobs = []
    cached = []
    for index, item in enumerate(manifest, start=1):
        key = f"{item['source']}|{item['figure_id']}"
        base = slug(key)
        answer_path = ANSWERS_DIR / f"{base}.txt"
        log_path = ANSWERS_DIR / f"{base}.log"
        if not answer_path.exists() or not answer_path.read_text(encoding="utf-8").strip():
            jobs.append((index, item, answer_path, log_path))
        else:
            cached.append(key)

    print(f"Cached {len(cached)} answers; generating {len(jobs)} with {workers} parallel workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_codex, item, answer_path, log_path): (index, item)
            for index, item, answer_path, log_path in jobs
        }
        for future in as_completed(futures):
            index, item = futures[future]
            key = f"{item['source']}|{item['figure_id']}"
            future.result()
            print(f"[{index:02d}/{len(manifest)}] generated {key}", flush=True)

    generated: list[dict[str, str]] = []
    for item in manifest:
        key = f"{item['source']}|{item['figure_id']}"
        answer_path = ANSWERS_DIR / f"{slug(key)}.txt"
        generated.append(
            {
                "key": key,
                "source": item["source"],
                "figure_id": item["figure_id"],
                "answer": answer_path.read_text(encoding="utf-8").strip(),
            }
        )

    ANSWERS_JSON.write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(generated)} strong answers to {ANSWERS_JSON}")


if __name__ == "__main__":
    main()
