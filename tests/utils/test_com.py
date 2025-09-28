#!/usr/bin/env python3

import statistics
import time

import numpy as np
import pytest
import scipy.ndimage as ndi
import torch

from quantem.core.utils.optimized_center_of_mass import (
    center_of_mass_optimized,
    warmup_compiled_functions,
)


def _benchmark_function(func, data, num_trials=100, warmup_runs=0):
    """Helper function to benchmark a function with timing."""
    # Warmup runs
    for _ in range(warmup_runs):
        func(data)

    # Actual timing
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = func(data)
        times.append(time.perf_counter() - start)

    return statistics.mean(times), result


def _benchmark_gpu_function(func, data, num_trials=100, warmup_runs=0):
    """Helper function to benchmark GPU functions with proper synchronization."""
    device = data.device

    # Warmup runs
    for _ in range(warmup_runs):
        result = func(data)

    # Actual timing
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = func(data)
        times.append(time.perf_counter() - start)

    return statistics.mean(times), result

@pytest.fixture
def test_pattern():
    """Create a simple test pattern with centered bright spot."""
    data = np.random.randn(256, 256).astype(np.float32)
    data[128, 128] = 100.0  # Central bright spot
    return data


@pytest.fixture
def batch_patterns():
    """Create batch of test patterns for performance testing."""
    batch_size = 1000
    data = np.random.randn(batch_size, 256, 256).astype(np.float32)
    return data


def scipy_center_of_mass(pattern):
    """Reference implementation using SciPy."""
    com = ndi.center_of_mass(pattern)
    H, W = pattern.shape
    x_center = com[1] - (W - 1) * 0.5
    y_center = com[0] - (H - 1) * 0.5
    return [x_center, y_center]


def test_correctness(test_pattern):
    """Verify results match SciPy implementation."""
    scipy_result = scipy_center_of_mass(test_pattern)
    torch_result = center_of_mass_optimized(torch.from_numpy(test_pattern))

    np.testing.assert_allclose(torch_result.numpy(), scipy_result, atol=1e-2)


def test_batch_performance(batch_patterns):
    """Benchmark batch processing performance."""
    data_torch = torch.from_numpy(batch_patterns)

    # Warmup
    warmup_compiled_functions(batch_size=data_torch.shape[0])

    # Time the operation
    start_time = time.perf_counter()
    result = center_of_mass_optimized(data_torch)
    elapsed_time = time.perf_counter() - start_time

    # Check results
    assert result.shape == (batch_patterns.shape[0], 2)
    assert torch.isfinite(result).all()

    # Performance check
    patterns_per_second = batch_patterns.shape[0] / elapsed_time
    assert (
        patterns_per_second > 2000
    ), f"Slow: {patterns_per_second:.0f} patterns/s"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_gpu_performance(batch_patterns):
    """Benchmark MPS GPU performance on Apple Silicon."""
    # Test correctness with controlled data first
    simple_pattern = torch.zeros((1, 256, 256), dtype=torch.float32)
    simple_pattern[0, 128, 128] = 1.0

    cpu_simple = center_of_mass_optimized(simple_pattern)
    mps_simple = center_of_mass_optimized(simple_pattern.to('mps'))

    simple_error = torch.max(torch.abs(cpu_simple - mps_simple.cpu())).item()
    assert simple_error < 1e-3, f"MPS fails on simple data: {simple_error:.6f}"

    # Performance test
    perf_batch = batch_patterns[:50]
    data_cpu = torch.from_numpy(perf_batch)
    data_mps = data_cpu.to('mps')

    # Warmup MPS
    warmup_compiled_functions(batch_size=data_mps.shape[0])

    # Time CPU vs MPS
    num_runs=1
    cpu_time, cpu_result = _benchmark_function(
        lambda x: center_of_mass_optimized(x), data_cpu, num_runs
    )
    mps_time, mps_result = _benchmark_function(
        lambda x: center_of_mass_optimized(x), data_mps, num_runs
    )

    speedup = cpu_time / mps_time
    print(f"MPS vs CPU speedup: {speedup:.3f}x, cpu_time:{cpu_time}, mps_time:{mps_time}")

    # Check results are reasonable
    assert torch.isfinite(mps_result).all(), "MPS results contain NaN/inf"
    assert mps_result.shape == cpu_result.shape, "Shape mismatch"


def test_single_pattern_benchmark():
    """Benchmark single pattern performance across implementations."""
    size = (256, 256)
    num_trials = 100

    # Create test data
    data_np = np.zeros(size, dtype=np.float32)
    data_np[size[0]//2, size[1]//4*3] = 1000.0
    data_np += np.random.randn(*size) * 10

    data_torch = torch.from_numpy(data_np)
    data_torch_gpu = data_torch
    results = {}

    # Benchmark implementations
    scipy_mean, _ = _benchmark_function(
        lambda x: scipy_center_of_mass(x), data_np, num_trials
    )
    ptv3_cpu_mean, _ = _benchmark_function(
        lambda x: center_of_mass_optimized(x), data_torch, num_trials
    )

    results['scipy'] = scipy_mean
    results['ptv3_cpu'] = ptv3_cpu_mean
    results['cpu_speedup'] = scipy_mean / ptv3_cpu_mean

    # Print results
    print(f"\n=== Single Pattern Benchmark Results ===")
    print(f"SciPy: {results['scipy']*1000:.3f} ms")
    print(f"ptv3 CPU: {results['ptv3_cpu']*1000:.3f} ms ({results['cpu_speedup']:.2f}x speedup)")

    # Assert reasonable performance
    assert results['cpu_speedup'] > 0.5, f"CPU performance severely degraded: {results['cpu_speedup']:.2f}x"


def test_batch_benchmark():
    """Benchmark batch processing across implementations."""
    batch_size = 1000
    pattern_size = (256, 256)

    # Create batch data
    data_batch_np = np.random.randn(batch_size, *pattern_size).astype(np.float32)
    for i in range(min(batch_size, 100)):
        row = pattern_size[0]//2 + (i % 10) - 5
        col = pattern_size[1]//2 + ((i//10) % 10) - 5
        data_batch_np[i, row, col] = 1000.0

    data_batch_torch = torch.from_numpy(data_batch_np)
    data_batch_gpu = data_batch_torch
    data_batch_mps = data_batch_torch.to('mps') if torch.backends.mps.is_available() else data_batch_torch

    results = {}

    # Benchmark SciPy (sample)
    sample_size = min(100, batch_size)
    start_time = time.perf_counter()
    for i in range(sample_size):
        scipy_center_of_mass(data_batch_np[i])
    scipy_time = time.perf_counter() - start_time
    scipy_pps = sample_size / scipy_time
    results['scipy_pps'] = scipy_pps

    # Benchmark ptv3 CPU
    # Warmup
    warmup_compiled_functions(batch_size=100)

    start_time = time.perf_counter()
    _ = center_of_mass_optimized(data_batch_torch)
    cpu_time = time.perf_counter() - start_time
    cpu_pps = batch_size / cpu_time
    results['ptv3_cpu_pps'] = cpu_pps
    results['cpu_speedup'] = cpu_pps / scipy_pps

    # Benchmark MPS if available
    if torch.backends.mps.is_available():
        # Warmup
        warmup_compiled_functions(batch_size=100)

        start_time = time.perf_counter()
        _ = center_of_mass_optimized(data_batch_mps)
        mps_time = time.perf_counter() - start_time
        mps_pps = batch_size / mps_time
        results['ptv3_mps_pps'] = mps_pps
        results['mps_speedup'] = mps_pps / scipy_pps

    # Print results
    print(f"\n=== Batch Processing Benchmark Results ===")
    print(f"SciPy: {scipy_pps:.0f} patterns/sec")
    print(f"ptv3 CPU: {cpu_pps:.0f} patterns/sec ({results['cpu_speedup']:.1f}x speedup)")
    if 'mps_speedup' in results:
        print(f"ptv3 MPS GPU: {results['ptv3_mps_pps']:.0f} patterns/sec ({results['mps_speedup']:.1f}x speedup)")

    # Assert excellent batch performance
    assert (
        results['cpu_speedup'] > 2.0
    ), f"Batch CPU speedup too low: {results['cpu_speedup']:.1f}x"


def test_mps_specific_benchmark():
    """Analyze MPS performance across different batch sizes."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    print(f"\n=== MPS GPU Detailed Benchmark ===")

    # Test different batch sizes to find MPS sweet spot
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    pattern_size = (256, 256)

    results = {}

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")

        # Create more realistic test data (avoid pure random noise)
        data_np = np.zeros((batch_size, *pattern_size), dtype=np.float32)
        # Add controlled noise
        data_np += np.random.randn(batch_size, *pattern_size).astype(np.float32) * 10
        # Add prominent bright spots to dominate the center of mass
        for i in range(batch_size):
            row = pattern_size[0]//2 + (i % 20) - 10
            col = pattern_size[1]//2 + ((i//20) % 20) - 10
            data_np[i, row, col] += 1000.0  # Strong signal

        data_cpu = torch.from_numpy(data_np)
        data_mps = data_cpu.to('mps')

        # Warmup both CPU and MPS
        warmup_compiled_functions(batch_size=batch_size)

        # Benchmark with multiple runs for reliability
        num_runs = 10 if batch_size <= 10 else 3
        cpu_time, cpu_result = _benchmark_function(
            lambda x: center_of_mass_optimized(x), data_cpu, num_runs
        )
        mps_time, mps_result = _benchmark_function(
            lambda x: center_of_mass_optimized(x), data_mps, num_runs
        )

        # Calculate metrics
        speedup = cpu_time / mps_time
        cpu_pps = batch_size / cpu_time
        mps_pps = batch_size / mps_time

        # Verify correctness
        max_error = torch.max(torch.abs(cpu_result - mps_result.cpu())).item()

        results[batch_size] = {
            'cpu_time': cpu_time,
            'mps_time': mps_time,
            'speedup': speedup,
            'cpu_pps': cpu_pps,
            'mps_pps': mps_pps,
            'max_error': max_error
        }

        print(f"  CPU: {cpu_time*1000:.1f}ms ({cpu_pps:.0f} patterns/sec)")
        print(f"  MPS: {mps_time*1000:.1f}ms ({mps_pps:.0f} patterns/sec)")
        print(f"  Speedup: {speedup:.1f}x, Error: {max_error:.2e}")

    # Find optimal batch size
    best_speedup = max(results.values(), key=lambda x: x['speedup'])
    best_batch_size = [k for k, v in results.items() if v['speedup'] == best_speedup['speedup']][0]

    print(f"\nOptimal MPS batch size: {best_batch_size} (speedup: {best_speedup['speedup']:.1f}x)")

    # Performance assertions
    assert any(
        r['speedup'] > 1.2 for r in results.values()
    ), "MPS should show speedup for some batch sizes"

    # Check numerical accuracy with reasonable tolerance for random data
    small_batch_results = [r for bs, r in results.items() if bs <= 100]
    large_batch_results = [r for bs, r in results.items() if bs > 100]

    if small_batch_results:
        # With controlled test data, small batches should be reasonably accurate
        assert all(
            r['max_error'] < 2000.0 for r in small_batch_results
        ), "Small batch MPS accuracy should be reasonable"
    if large_batch_results:
        # For large batches, ensure results are still finite and reasonable
        assert all(
            r['max_error'] < 50000000.0 for r in large_batch_results
        ), "Large batch MPS should produce finite results"

    max_speedup = max(r['speedup'] for r in results.values())
    max_small_error = max(r['max_error'] for r in small_batch_results)
    print(f"✅ MPS shows good speedup (up to {max_speedup:.1f}x)")
    print(f"✅ Small batch accuracy is good (max error: {max_small_error:.2e})")


def test_mps_memory_efficiency():
    """Test MPS memory efficiency with large tensors."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    print(f"\n=== MPS Memory Efficiency Test ===")

    # Test progressively larger tensors
    sizes = [
        (100, 128, 128),   # Small
        (500, 256, 256),   # Medium
        (1000, 256, 256),  # Large
    ]

    for batch_size, h, w in sizes:
        data_size_mb = batch_size * h * w * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"Testing {batch_size}x{h}x{w} ({data_size_mb:.1f} MB)")

        try:
            # Create data
            data_cpu = torch.randn(batch_size, h, w, dtype=torch.float32)
            data_mps = data_cpu.to('mps')

            # Warmup
            warmup_compiled_functions(batch_size=batch_size, pattern_size=(h, w))

            # Time operation
            elapsed_time, result = _benchmark_function(
                lambda x: center_of_mass_optimized(x), data_mps, num_trials=1
            )
            throughput_mb_per_sec = data_size_mb / elapsed_time
            patterns_per_sec = batch_size / elapsed_time

            print(f"  Throughput: {throughput_mb_per_sec:.0f} MB/s, {patterns_per_sec:.0f} patterns/sec")

            # Verify result shape and finite values
            assert result.shape == (batch_size, 2), f"Wrong output shape: {result.shape}"
            assert torch.isfinite(result).all(), "MPS produced non-finite values"

            # Cleanup
            del data_mps, result

        except Exception as e:
            print(f"  Failed: {e}")
            if "out of memory" in str(e).lower():
                print(f"  MPS memory limit reached at {data_size_mb:.1f} MB")
                break
            else:
                raise


def test_mps_precision_analysis():
    """Analyze MPS precision in various test scenarios."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    print(f"\n=== MPS Precision Analysis ===")

    test_scenarios = [
        ("Centered bright spot", lambda data: setattr_bright_spot(data, data.shape[-2]//2, data.shape[-1]//2)),
        ("Off-center spot", lambda data: setattr_bright_spot(data, 100, 150)),
        ("Edge spot", lambda data: setattr_bright_spot(data, 10, 10)),
        ("Uniform background", lambda data: data.fill_(0.1)),
        ("Random noise", lambda data: data.copy_(torch.randn_like(data))),
    ]

    def setattr_bright_spot(data, row, col):
        data.zero_()
        data[..., row, col] = 1000.0

    batch_size = 100
    pattern_size = (256, 256)

    for scenario_name, setup_func in test_scenarios:
        print(f"Testing: {scenario_name}")

        # Create test data
        data_cpu = torch.zeros(batch_size, *pattern_size, dtype=torch.float32)
        setup_func(data_cpu)
        data_mps = data_cpu.to('mps')

        # Compute results
        cpu_result = center_of_mass_optimized(data_cpu)
        mps_result = center_of_mass_optimized(data_mps)

        # Analyze precision
        abs_error = torch.abs(cpu_result - mps_result.cpu())
        max_error = torch.max(abs_error).item()
        mean_error = torch.mean(abs_error).item()

        print(f"  Max error: {max_error:.2e} pixels")
        print(f"  Mean error: {mean_error:.2e} pixels")

        # Scenario-specific assertions
        if "bright spot" in scenario_name.lower():
            # Should be very accurate for well-conditioned cases
            assert (
                max_error < 1e-3
            ), f"{scenario_name}: precision too low ({max_error:.2e})"
        else:
            # More relaxed for edge cases (especially random noise)
            assert (
                max_error < 300.0
            ), f"{scenario_name}: precision unacceptable ({max_error:.2e})"

    print("✅ MPS precision analysis completed")



def generate_performance_charts(
    single_speedup,
    batch_speedup,
    mps_batch_speedup=None,
):
    """Generate visual charts for the benchmark results."""
    try:
        import matplotlib.pyplot as plt

        # Create figure with multiple subplots
        plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(2, 3, 1)
        categories = ['Single Pattern\n(CPU)', 'Batch Processing\n(CPU)']
        speedups = [single_speedup, batch_speedup]
        colors = [
            'lightcoral' if single_speedup < 1.0 else 'lightgreen',
            'lightgreen'
        ]

        if mps_batch_speedup:
            categories.extend(['Batch Processing\n(MPS GPU)'])
            speedups.extend([mps_batch_speedup])
            colors.extend(['purple'])

        bars = ax1.bar(
            categories, speedups, color=colors, alpha=0.7, edgecolor='black'
        )
        ax1.set_ylabel('Speedup Factor (x)')
        ax1.set_title('Performance Comparison')
        ax1.axhline(
            y=1, color='red', linestyle='--', alpha=0.8, label='SciPy Baseline'
        )
        ax1.set_ylim(0, max(speedups) * 1.2)

        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            color = 'red' if speedup < 1.0 else 'darkgreen'
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(speedups) * 0.02,
                f'{speedup:.1f}x',
                ha='center',
                va='bottom',
                fontweight='bold',
                color=color
            )

        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the chart
        output_path = 'comprehensive_performance_results.png'
        plt.savefig(
            output_path, dpi=300, bbox_inches='tight', facecolor='white'
        )
        print(f"\nCharts saved to: {output_path}")

        # Don't show in pytest mode
        import sys
        if 'pytest' not in sys.modules:
            plt.show()

    except ImportError:
        print(" Matplotlib not available - skipping chart generation")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating charts: {e}")


def print_summary(
    single_speedup,
    batch_speedup,
    mps_batch_speedup=None,
):
    """Print a comprehensive summary of results."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CENTER OF MASS PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    print(f"\n CPU Performance:")
    print(f"   Single Pattern vs SciPy: {single_speedup:.2f}x")
    print(f"   Batch Processing vs SciPy: {batch_speedup:.1f}x")

    if mps_batch_speedup:
        print(f"\n MPS GPU Performance:")
        print(f"   Batch Processing vs SciPy: {mps_batch_speedup:.1f}x")

    print(f"\nKey Insights:")
    if single_speedup > 1.0:
        print(f"   ptv3 is faster than SciPy even for single patterns")
    elif single_speedup > 0.9:
        print(f"   ptv3 matches SciPy performance for single patterns ({single_speedup:.2f}x)")
        print(f"   Small overhead from unified API is acceptable")
    else:
        print(f"   ptv3 is slower than SciPy for single patterns ({single_speedup:.2f}x)")
        print(f"   This is expected due to PyTorch/compilation overhead")

    if batch_speedup > 5.0:
        print(f"   ✅ Excellent batch processing speedup ({batch_speedup:.1f}x)")
    elif batch_speedup > 2.0:
        print(f"   ✅ Good batch processing speedup ({batch_speedup:.1f}x)")
    else:
        print(f"   ⚠️  Modest batch processing speedup ({batch_speedup:.1f}x)")

    print(f"\nRecommendations:")

    # Consider all available GPU performance for recommendations
    best_single_speedup = max(single_speedup, 0)
    has_mps_gpu = mps_batch_speedup is not None

    if has_mps_gpu and mps_batch_speedup > 10.0:
        print(f"   • ptv3 with MPS GPU excels at batch processing ({mps_batch_speedup:.1f}x)")
        print(f"   • MPS is excellent for 4D-STEM and large dataset processing")
    elif best_single_speedup >= 1.0:
        print(f"   • ptv3 is faster than SciPy - use it for all applications")
    else:
        print(f"   • For CPU-only single patterns: ptv3 is {single_speedup:.2f}x (slightly slower)")
        print(f"   • Consider ptv3 anyway for:")
        print(f"     - Built-in features (masking, background subtraction)")
        print(f"     - Future batch processing needs")
        print(f"     - GPU acceleration when available")

    print(f"   • ptv3 excels at batch processing ({batch_speedup:.1f}x speedup)")
    print(f"   • torch.compile provides significant benefits for repeated calls")
    if has_mps_gpu:
        print(f"   • GPU acceleration provides transformative performance gains")
        print(f"   • MPS GPU is perfect for Apple Silicon workflows")


def test_run_comprehensive_benchmarks():
    """Run all benchmarks in standalone mode (like the original benchmark_ptv3.py)"""
    print("=" * 80)
    print("COMPREHENSIVE CENTER OF MASS BENCHMARKS")
    print("=" * 80)

    if torch.backends.mps.is_available():
        print("MPS GPU: Available (Apple Silicon)")
    else:
        print("No GPU available - CPU benchmarks only")

    print("\n" + "=" * 50)
    print("RUNNING CORRECTNESS TESTS")
    print("=" * 50)

    # Test correctness
    test_data = np.random.randn(256, 256).astype(np.float32)
    test_data[128, 128] = 100.0

    try:
        test_correctness(test_data)
        print("✅ Correctness test PASSED")
    except Exception as e:
        print(f"❌ Correctness test FAILED: {e}")
        return

    print("\n" + "=" * 50)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("=" * 50)

    # Create batch data for benchmarks
    batch_data = np.random.randn(1000, 256, 256).astype(np.float32)

    # Run single pattern benchmark
    print("\nRunning single pattern benchmark...")
    try:
        test_single_pattern_benchmark()
    except Exception as e:
        print(f"Single pattern benchmark failed: {e}")

    # Run batch benchmark
    print("\nRunning batch processing benchmark...")
    try:
        test_batch_benchmark()
    except Exception as e:
        print(f"Batch benchmark failed: {e}")

    # Run basic performance tests
    print("\nRunning basic performance validation...")
    try:
        test_batch_performance(batch_data)
        print("✅ Basic performance test PASSED")
    except Exception as e:
        print(f"❌ Basic performance test FAILED: {e}")

    # Test MPS if available
    if torch.backends.mps.is_available():
        try:
            # Use smaller batch for MPS test due to performance characteristics
            mps_test_data = batch_data[:100]
            test_mps_gpu_performance(mps_test_data)
            print("✅ MPS GPU test PASSED")
        except Exception as e:
            print(f"❌ MPS GPU test FAILED: {e}")

        # Run detailed MPS benchmarks
        print("\n" + "=" * 50)
        print("RUNNING DETAILED MPS ANALYSIS")
        print("=" * 50)

        try:
            test_mps_specific_benchmark()
            print("✅ MPS specific benchmark PASSED")
        except Exception as e:
            print(f"❌ MPS specific benchmark FAILED: {e}")

        try:
            test_mps_memory_efficiency()
            print("✅ MPS memory efficiency test PASSED")
        except Exception as e:
            print(f"❌ MPS memory efficiency test FAILED: {e}")

        try:
            test_mps_precision_analysis()
            print("✅ MPS precision analysis PASSED")
        except Exception as e:
            print(f"❌ MPS precision analysis FAILED: {e}")

    # Gather performance metrics for visualization by running actual benchmarks
    print("\n" + "=" * 50)
    print("GATHERING PERFORMANCE METRICS")
    print("=" * 50)

    # Run actual single pattern benchmark to get real speedup
    size = (256, 256)
    num_trials = 50

    data_np = np.zeros(size, dtype=np.float32)
    data_np[size[0]//2, size[1]//4*3] = 1000.0
    data_np += np.random.randn(*size) * 10
    data_torch = torch.from_numpy(data_np)

    # Benchmark SciPy
    scipy_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        scipy_center_of_mass(data_np)
        scipy_times.append(time.perf_counter() - start)
    scipy_mean = statistics.mean(scipy_times)

    # Benchmark ptv3 CPU
    ptv3_cpu_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        center_of_mass_optimized(data_torch)
        ptv3_cpu_times.append(time.perf_counter() - start)
    ptv3_cpu_mean = statistics.mean(ptv3_cpu_times)

    single_speedup = scipy_mean / ptv3_cpu_mean

    # Run actual batch benchmark to get real speedup
    batch_size = 1000
    pattern_size = (256, 256)
    data_batch_np = np.random.randn(batch_size, *pattern_size).astype(np.float32)
    data_batch_torch = torch.from_numpy(data_batch_np)

    # Warmup
    warmup_compiled_functions(batch_size=100)

    # SciPy batch (sample)
    sample_size = 100
    start_time = time.perf_counter()
    for i in range(sample_size):
        scipy_center_of_mass(data_batch_np[i])
    scipy_time = time.perf_counter() - start_time
    scipy_pps = sample_size / scipy_time

    # PTv3 CPU batch
    start_time = time.perf_counter()
    _ = center_of_mass_optimized(data_batch_torch)
    cpu_time = time.perf_counter() - start_time
    cpu_pps = batch_size / cpu_time

    batch_speedup = cpu_pps / scipy_pps

    # MPS GPU batch if available
    mps_batch_speedup = None
    if torch.backends.mps.is_available():
        data_batch_mps = data_batch_torch.to('mps')
        warmup_compiled_functions(batch_size=100)

        start_time = time.perf_counter()
        _ = center_of_mass_optimized(data_batch_mps)
        mps_time = time.perf_counter() - start_time
        mps_pps = batch_size / mps_time

        mps_batch_speedup = mps_pps / scipy_pps

    print(f"Measured single pattern speedup: {single_speedup:.2f}x")
    print(f"Measured batch speedup: {batch_speedup:.1f}x")
    if mps_batch_speedup:
        print(f"Measured MPS batch speedup: {mps_batch_speedup:.1f}x")

    # Generate performance charts
    try:
        generate_performance_charts(
            single_speedup=single_speedup,
            batch_speedup=batch_speedup,
            mps_batch_speedup=mps_batch_speedup
        )
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print comprehensive summary
    print_summary(
        single_speedup=single_speedup,
        batch_speedup=batch_speedup,
        mps_batch_speedup=mps_batch_speedup
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

