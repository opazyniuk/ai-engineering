"""
Evaluator для DVC homework.
Перевіряє dvc_workspace/ після того як студент виконав завдання.

Використання:
    python evaluate.py
"""

import subprocess
import sys
from pathlib import Path

import yaml


WORK_DIR = Path("dvc_workspace")
TOTAL = 0
EARNED = 0


def check(name: str, points: int, condition: bool, detail: str = ""):
    global TOTAL, EARNED
    TOTAL += points
    if condition:
        EARNED += points
        print(f"  [PASS] {name} (+{points})")
    else:
        msg = f" — {detail}" if detail else ""
        print(f"  [FAIL] {name} (0/{points}){msg}")


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, cwd=str(WORK_DIR),
                          capture_output=True, text=True)


if not WORK_DIR.exists():
    print("ERROR: dvc_workspace/ not found.")
    print("Create it and set up your DVC repo there.")
    sys.exit(1)


# ===========================================================================
# Check 1: Repo initialized
# ===========================================================================
print("=" * 50)
print("CHECK 1: Repo + DVC init")
print("=" * 50)

check("git repo exists", 3, (WORK_DIR / ".git").is_dir())
check(".dvc/ exists", 3, (WORK_DIR / ".dvc").is_dir())
check(".dvcignore exists", 2, (WORK_DIR / ".dvcignore").exists())

r = run("dvc version")
check("dvc works", 2, r.returncode == 0)


# ===========================================================================
# Check 2: Dataset tracked by DVC
# ===========================================================================
print(f"\n{'=' * 50}")
print("CHECK 2: DVC tracking")
print("=" * 50)

dvc_file = WORK_DIR / "dataset.csv.dvc"
check("dataset.csv.dvc exists", 4, dvc_file.exists())

if dvc_file.exists():
    try:
        dvc_content = yaml.safe_load(dvc_file.read_text())
        outs = dvc_content.get("outs", [])
        check(".dvc has md5 hash", 3, len(outs) > 0 and "md5" in outs[0])
        check(".dvc tracks dataset.csv", 2, len(outs) > 0 and outs[0].get("path") == "dataset.csv")
    except Exception as e:
        check(".dvc valid YAML", 5, False, str(e))
else:
    TOTAL += 5

gitignore = WORK_DIR / ".gitignore"
check("dataset.csv in .gitignore", 3, gitignore.exists() and "dataset.csv" in gitignore.read_text())


# ===========================================================================
# Check 3: Version commits
# ===========================================================================
print(f"\n{'=' * 50}")
print("CHECK 3: Commits (v1 + v2)")
print("=" * 50)

r = run("git log --oneline")
commits = [line for line in r.stdout.strip().split("\n") if line.strip()]
check("At least 3 commits", 3, len(commits) >= 3, f"found {len(commits)}")
check("Commit with 'v1'", 3, any("v1" in c.lower() for c in commits))
check("Commit with 'v2'", 3, any("v2" in c.lower() for c in commits))


# ===========================================================================
# Check 4: Clean data (v2 requirements)
# ===========================================================================
print(f"\n{'=' * 50}")
print("CHECK 4: Data quality (v2)")
print("=" * 50)

dataset = WORK_DIR / "dataset.csv"
if dataset.exists():
    content = dataset.read_text()
    lines = content.strip().split("\n")
    header = lines[0]
    data_lines = lines[1:]

    check("Has header", 1, "id" in header and "name" in header)
    check("10 data rows", 4, len(data_lines) == 10, f"found {len(data_lines)}")

    # No duplicates (unique ids)
    ids = [line.split(",")[0] for line in data_lines]
    check("No duplicate ids", 4, len(ids) == len(set(ids)),
          f"ids={ids}")

    # No empty values
    has_empty = any(",," in line or line.endswith(",") for line in data_lines)
    check("No empty values", 4, not has_empty)

    # Consistent labels (all lowercase)
    categories = [line.split(",")[2] for line in data_lines if len(line.split(",")) > 2]
    all_lower = all(c == c.lower() for c in categories)
    check("All categories lowercase", 3, all_lower,
          f"found: {[c for c in categories if c != c.lower()]}")

    # Bob = enterprise
    check("Bob = enterprise", 3, "Bob,enterprise" in content,
          "Bob should have category 'enterprise'")

    # Hank = 4800
    check("Hank value = 4800", 3, "Hank,smb,4800" in content,
          "Hank should have value 4800")
else:
    check("dataset.csv exists", 22, False)


# ===========================================================================
# Check 5: DVC cache (2 versions)
# ===========================================================================
print(f"\n{'=' * 50}")
print("CHECK 5: DVC cache")
print("=" * 50)

cache_dir = WORK_DIR / ".dvc" / "cache"
if cache_dir.exists():
    cache_files = [f for f in cache_dir.rglob("*") if f.is_file()]
    check("Cache has files", 3, len(cache_files) >= 1)
    check("2+ versions cached", 4, len(cache_files) >= 2,
          f"found {len(cache_files)}")
else:
    check("DVC cache exists", 7, False)


# ===========================================================================
# Check 6: Clean state
# ===========================================================================
print(f"\n{'=' * 50}")
print("CHECK 6: Final state")
print("=" * 50)

r = run("dvc status")
is_clean = r.returncode == 0 and ("up to date" in r.stdout.lower() or r.stdout.strip() == "")
check("dvc status clean", 4, is_clean, f"stdout={r.stdout[:80]}")


# ===========================================================================
# RESULT
# ===========================================================================
print(f"\n{'=' * 50}")
pct = (EARNED / TOTAL * 100) if TOTAL > 0 else 0
print(f"РЕЗУЛЬТАТ: {EARNED}/{TOTAL} балів ({pct:.0f}%)")
print("=" * 50)

if pct >= 90:
    print("Відмінно!")
elif pct >= 70:
    print("Добре!")
elif pct >= 50:
    print("Задовільно.")
else:
    print("Потрібно доопрацювати.")
