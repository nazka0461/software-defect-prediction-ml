import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.commit_metrics import extract_file_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict defect risk for changed files.")
    parser.add_argument("--output", required=True, help="Path to the JSON report to write.")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base-ref", default=None)
    parser.add_argument("--head-ref", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def _run_git_diff(repo_root: Path, base_ref: str = None, head_ref: str = None) -> List[str]:
    if base_ref and head_ref:
        cmd = ["git", "diff", "--name-only", base_ref, head_ref]
    elif os.environ.get("GITHUB_BASE_REF") and os.environ.get("GITHUB_SHA"):
        cmd = [
            "git",
            "diff",
            "--name-only",
            f"origin/{os.environ['GITHUB_BASE_REF']}",
            os.environ["GITHUB_SHA"],
        ]
    else:
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().lower().endswith((".py", ".java"))
    ]


def _wait_for_api(api_url: str, timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    health_url = f"{api_url.rstrip('/')}/health"
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(f"API did not become healthy: {last_error}")


def _write_step_summary(report: Dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "# Defect Risk Check",
        "",
        f"- Model: `{report['model']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Files analyzed: `{report['file_count']}`",
        "",
        "| File | Probability | Label |",
        "| --- | ---: | --- |",
    ]
    for item in report["predictions"]:
        lines.append(
            f"| `{item['file']}` | {item['probability']:.3f} | {item['label']} |"
        )

    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    health = _wait_for_api(args.api_url, timeout=args.timeout)
    raw_feature_order = health.get("raw_feature_order", [])

    changed_files = _run_git_diff(
        repo_root=repo_root,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
    )

    predictions = []
    for relative_path in changed_files:
        file_path = repo_root / relative_path
        if not file_path.exists():
            continue

        metrics = extract_file_metrics(file_path)
        payload_metrics = {
            feature_name: metrics[feature_name]
            for feature_name in raw_feature_order
            if feature_name in metrics
        }
        response = requests.post(
            f"{args.api_url.rstrip('/')}/predict",
            json={"metrics": payload_metrics},
            timeout=args.timeout,
        )
        response.raise_for_status()
        prediction = response.json()
        predictions.append(
            {
                "file": relative_path.replace("\\", "/"),
                "probability": float(prediction["probability"]),
                "label": prediction["label"],
                "top_features": prediction.get("top_features", []),
                "metrics": payload_metrics,
            }
        )

    predictions.sort(key=lambda item: item["probability"], reverse=True)

    report = {
        "model": health.get("model"),
        "dataset": health.get("dataset"),
        "file_count": len(predictions),
        "predictions": predictions,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_step_summary(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
