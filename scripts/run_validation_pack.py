"""Запустить набор валидаций без видео и записать отчёт для защиты."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    stdout_path: Path
    stderr_path: Path

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _rel_from(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return _rel(path)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_check(
    name: str,
    command: list[str],
    logs_dir: Path,
    continue_on_fail: bool,
    timeout_s: int,
) -> CheckResult:
    print(f"[validation] {name}")
    print("  $ " + " ".join(command))
    t0 = time.time()
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    stdout_path = logs_dir / f"{safe_name}.stdout.txt"
    stderr_path = logs_dir / f"{safe_name}.stderr.txt"
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        stderr = f"{stderr}\nTimed out after {timeout_s}s\n"
        returncode = 124
    except OSError as exc:
        stdout = ""
        stderr = f"Failed to start command: {exc}\n"
        returncode = 127

    duration_s = time.time() - t0
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    status = "PASS" if returncode == 0 else "FAIL"
    print(f"  -> {status} ({duration_s:.1f}s)")
    if stdout:
        tail = stdout.strip().splitlines()[-5:]
        for line in tail:
            print(f"     {line}")
    if returncode != 0 and not continue_on_fail:
        print(f"[validation] stopping after failed check: {name}", file=sys.stderr)
    return CheckResult(
        name=name,
        command=command,
        returncode=returncode,
        duration_s=duration_s,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _structural_metrics(path: Path) -> dict:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _format_number(value: object, precision: int) -> str:
    if isinstance(value, int | float):
        return f"{value:.{precision}f}"
    return "N/A"


def _write_json_report(
    output: Path,
    results: list[CheckResult],
    structural_output: Path,
) -> None:
    payload = {
        "passed": all(r.passed for r in results),
        "checks": [
            {
                "name": r.name,
                "command": r.command,
                "returncode": r.returncode,
                "passed": r.passed,
                "duration_s": round(r.duration_s, 3),
                "stdout": _rel(r.stdout_path),
                "stderr": _rel(r.stderr_path),
            }
            for r in results
        ],
        "structural_summary": _structural_metrics(structural_output),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown_report(
    output: Path,
    results: list[CheckResult],
    structural_output: Path,
) -> None:
    structural = _structural_metrics(structural_output)
    lines = [
        "# Validation pack report",
        "",
        f"Overall status: {'PASS' if all(r.passed for r in results) else 'FAIL'}",
        "",
        "## Checks",
        "",
        "| Check | Status | Time, s | Logs |",
        "|---|---:|---:|---|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.name} | {status} | {r.duration_s:.1f} | "
            f"[stdout]({_rel_from(r.stdout_path, output)}) / "
            f"[stderr]({_rel_from(r.stderr_path, output)}) |"
        )

    lines.extend([
        "",
        "## Structural benchmark",
        "",
    ])
    if structural:
        position_error = _format_number(structural.get("position_error_m"), 2)
        peak_score = _format_number(structural.get("peak_score"), 3)
        sigma_xy = _format_number(structural.get("sigma_xy_m"), 1)
        lines.extend([
            f"- accepted: `{structural.get('accepted')}`",
            f"- position_error_m: `{position_error}`",
            f"- peak_score: `{peak_score}`",
            f"- sigma_xy_m: `{sigma_xy}`",
            f"- synthetic_bev: `{_rel_from(structural_output / 'synthetic_bev.png', output)}`",
            f"- score_map: `{_rel_from(structural_output / 'score_map.png', output)}`",
        ])
    else:
        lines.append("- Structural summary not found.")

    lines.extend([
        "",
        "## How to use this report",
        "",
        "Use `summary.md` as a compact defense artifact: it shows which reproducible",
        "checks passed without requiring new flight video. It does not claim final",
        "flight ATE/RPE; that still requires a real route with telemetry or ground truth.",
        "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/validation_pack"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--video", type=Path, default=None,
                   help="optional local flight video for the VO advisory stage")
    p.add_argument("--skip-slow", action="store_true",
                   help="skip slower EKF/scale/map-measurement checks")
    p.add_argument("--continue-on-fail", action="store_true",
                   help="run all checks even if one fails")
    p.add_argument("--check-timeout-s", type=int, default=1200,
                   help="timeout per check in seconds")
    args = p.parse_args()
    if args.check_timeout_s <= 0:
        p.error("--check-timeout-s must be a positive integer")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    logs_dir = output / "logs"
    structural_output = output / "structural"
    dem_output = output / "dem"
    ekf_output = output / "ekf"
    scale_output = output / "scale"
    map_output = output / "map_measurement"
    vo_video = args.video if args.video is not None else output / "no_video_available.mp4"
    output.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    py = args.python
    checks = [
        ("compileall", [py, "-m", "compileall", "-q", "src", "scripts"]),
        ("dem lookup", [
            py, "scripts/verify_dem_lookup.py",
            "--dem", "data/dem/test_synthetic.tif",
            "--output", str(dem_output),
        ]),
        ("optical flow vo", [
            py, "scripts/verify_optical_flow_vo.py",
            "--camera-config", "configs/camera_gopro_hx.yaml",
            "--video", str(vo_video),
        ]),
        ("structural matching", [
            py, "scripts/verify_structural_matching.py",
            "--dataset-root", "data/overture_ru_dataset_starter",
            "--region-id", "moscow_city_small",
            "--output", str(structural_output),
        ]),
    ]
    if not args.skip_slow:
        checks.extend([
            ("ekf", [py, "scripts/verify_ekf.py", "--output", str(ekf_output)]),
            ("scale correction", [
                py, "scripts/verify_scale_correction.py",
                "--output", str(scale_output),
            ]),
            ("map measurement", [
                py, "scripts/verify_map_measurement.py",
                "--output", str(map_output),
            ]),
        ])

    results = []
    for name, command in checks:
        result = _run_check(
            name,
            command,
            logs_dir,
            args.continue_on_fail,
            args.check_timeout_s,
        )
        results.append(result)
        if not result.passed and not args.continue_on_fail:
            break

    _write_json_report(output, results, structural_output)
    _write_markdown_report(output, results, structural_output)
    print(f"[validation] summary -> {_rel(output / 'summary.md')}")
    print(f"[validation] json    -> {_rel(output / 'summary.json')}")

    if all(r.passed for r in results):
        print("[validation] VALIDATION PACK PASSED")
        return
    print("[validation] VALIDATION PACK FAILED")
    sys.exit(2)


if __name__ == "__main__":
    main()
