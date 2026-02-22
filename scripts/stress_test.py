#!/usr/bin/env python3
"""WSL2 CUDA stability stress test for ALS-LM training environment.

Runs a micro training loop on CUDA to verify GPU stability under sustained
load. Collects GPU temperature, memory, utilization, and step timing metrics
via pynvml. Supports quick (10,000 steps) and extended (8+ hours) tiers.

Usage:
    python scripts/stress_test.py                    # Quick: 10,000 steps
    python scripts/stress_test.py --extended         # Extended: 8+ hours
    python scripts/stress_test.py --steps 50000      # Custom step count
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pynvml
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WSL2 CUDA stability stress test"
    )
    parser.add_argument(
        "--steps", type=int, default=10_000,
        help="Number of training steps (default: 10000)"
    )
    parser.add_argument(
        "--extended", action="store_true",
        help="Run for 8+ hours (calculates steps from measured timing)"
    )
    parser.add_argument(
        "--report-every", type=int, default=100,
        help="Metrics reporting interval in steps (default: 100)"
    )
    parser.add_argument(
        "--temp-limit", type=int, default=85,
        help="GPU temperature auto-abort threshold in Celsius (default: 85)"
    )
    parser.add_argument(
        "--output", type=str, default="reports",
        help="Output directory for JSON report (default: reports/)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for micro training loop (default: 32)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=2048,
        help="Hidden dimension controlling model size (default: 2048)"
    )
    return parser.parse_args()


def get_gpu_metrics(handle: int) -> dict:
    """Sample GPU metrics via pynvml."""
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "temp_c": temp,
        "mem_used_mb": round(mem_info.used / (1024 * 1024)),
        "mem_total_mb": round(mem_info.total / (1024 * 1024)),
        "utilization_pct": util.gpu,
    }


def build_model(
    input_dim: int, hidden_dim: int, output_dim: int
) -> nn.Module:
    """Build a small sequential model for realistic memory allocation."""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_header(gpu_name: str, vram_mb: int) -> None:
    """Print system information header."""
    print("=" * 70)
    print("WSL2 CUDA Stress Test")
    print("=" * 70)
    print(f"GPU:            {gpu_name}")
    print(f"VRAM:           {vram_mb} MB")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA:           {torch.version.cuda}")
    print(f"Python:         {sys.version.split()[0]}")
    print(f"Time:           {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)


def save_report(
    report: dict, output_dir: str, extended: bool
) -> str:
    """Save JSON report to output directory. Returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = "stress_test_long.json" if extended else "stress_test_quick.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    return filepath


def run_stress_test(args: argparse.Namespace) -> None:
    """Main stress test loop."""
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot run stress test.")
        print("Run: python -c \"import torch; print(torch.cuda.is_available())\"")
        sys.exit(1)

    # Initialize pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_mb = round(mem_info.total / (1024 * 1024))

    print_header(gpu_name, vram_mb)

    # Build model and move to CUDA
    input_dim = 512
    output_dim = 512
    model = build_model(input_dim, args.hidden_dim, output_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    model_params = count_parameters(model)

    print(f"Model params:   {model_params:,}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Hidden dim:     {args.hidden_dim}")
    print(f"Temp limit:     {args.temp_limit}C")

    # Determine total steps
    total_steps = args.steps

    # For extended mode, measure step timing from first 100 steps
    if args.extended:
        print("\nMeasuring step timing from first 100 steps...")
        warmup_times = []
        for i in range(100):
            x = torch.randn(args.batch_size, input_dim, device="cuda")
            target = torch.randn(args.batch_size, output_dim, device="cuda")

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            warmup_times.append(t1 - t0)

        avg_step_time = statistics.mean(warmup_times)
        total_steps = int(8 * 3600 / avg_step_time)
        projected_hours = (total_steps * avg_step_time) / 3600
        print(f"Average step time: {avg_step_time * 1000:.2f} ms")
        print(f"Steps for 8 hours: {total_steps:,}")
        print(f"Projected duration: {projected_hours:.1f} hours")

    print(f"\nRunning {total_steps:,} steps...")
    print("-" * 70)

    # Initialize tracking
    start_time = datetime.now(timezone.utc)
    step_times = []
    gpu_samples = []
    warnings = []
    status = "completed"
    completed_steps = 0
    rolling_window = 50  # steps for rolling mean

    try:
        for step in range(total_steps):
            # Generate random batch
            x = torch.randn(args.batch_size, input_dim, device="cuda")
            target = torch.randn(args.batch_size, output_dim, device="cuda")

            # Timed training step
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_ms = (t1 - t0) * 1000
            step_times.append(step_ms)
            completed_steps = step + 1

            # NaN detection
            if torch.isnan(loss):
                print(f"\nERROR: NaN loss detected at step {step}")
                status = "aborted_nan"
                break

            # Hang detection: check if step took >10x rolling mean
            if len(step_times) > rolling_window:
                recent = step_times[-rolling_window:]
                rolling_mean = statistics.mean(recent)
                if step_ms > 10 * rolling_mean:
                    msg = (
                        f"Hang detected at step {step}: "
                        f"{step_ms:.1f}ms (rolling mean: {rolling_mean:.1f}ms)"
                    )
                    print(f"\nWARNING: {msg}")
                    warnings.append(msg)

            # Periodic reporting
            if (step + 1) % args.report_every == 0 or step == 0:
                metrics = get_gpu_metrics(handle)
                sample = {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    **metrics,
                }
                gpu_samples.append(sample)

                # Temperature auto-abort
                if metrics["temp_c"] >= args.temp_limit:
                    msg = (
                        f"GPU temperature {metrics['temp_c']}C >= "
                        f"limit {args.temp_limit}C"
                    )
                    print(f"\nWARNING: {msg} -- aborting for safety")
                    warnings.append(msg)
                    status = "aborted_temp"
                    break

                # Terminal progress
                mean_ms = statistics.mean(step_times[-args.report_every:])
                print(
                    f"Step {step + 1:>{len(str(total_steps))}}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {mean_ms:.1f}ms/step | "
                    f"GPU: {metrics['temp_c']}C, "
                    f"{metrics['mem_used_mb']}MB/{metrics['mem_total_mb']}MB, "
                    f"{metrics['utilization_pct']}%"
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving partial results...")
        status = "aborted_interrupt"
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n\nCUDA error: {e}")
            print("Saving partial results...")
            status = "aborted_cuda_error"
            warnings.append(f"CUDA error: {e}")
        else:
            raise

    end_time = datetime.now(timezone.utc)
    wall_clock = (end_time - start_time).total_seconds()

    # Compute timing statistics
    timing = {}
    if step_times:
        sorted_times = sorted(step_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        timing = {
            "mean_ms": round(statistics.mean(step_times), 3),
            "median_ms": round(statistics.median(step_times), 3),
            "p95_ms": round(sorted_times[min(p95_idx, len(sorted_times) - 1)], 3),
            "p99_ms": round(sorted_times[min(p99_idx, len(sorted_times) - 1)], 3),
            "max_ms": round(max(step_times), 3),
            "min_ms": round(min(step_times), 3),
        }

    # Build report
    report = {
        "metadata": {
            "gpu_name": gpu_name,
            "gpu_vram_mb": vram_mb,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": sys.version.split()[0],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "wall_clock_seconds": round(wall_clock, 2),
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "status": status,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "model_params": model_params,
        },
        "timing": timing,
        "gpu_samples": gpu_samples,
        "warnings": warnings,
    }

    # Print final summary
    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)
    print(f"Status:         {status}")
    print(f"Steps:          {completed_steps:,}/{total_steps:,}")
    print(f"Wall clock:     {wall_clock:.1f}s")
    if timing:
        print(f"Step timing:    mean={timing['mean_ms']:.1f}ms, "
              f"median={timing['median_ms']:.1f}ms, "
              f"p95={timing['p95_ms']:.1f}ms, "
              f"max={timing['max_ms']:.1f}ms")
    if warnings:
        print(f"Warnings:       {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")

    # Save report
    filepath = save_report(report, args.output, args.extended)
    print(f"Report saved:   {filepath}")
    print("=" * 70)

    # Cleanup
    pynvml.nvmlShutdown()

    # Exit with appropriate code
    if status in ("aborted_nan", "aborted_cuda_error"):
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    run_stress_test(args)
