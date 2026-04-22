#!/usr/bin/env python3

"""
CSA Performance Profiler
Comprehensive profiling tool to identify speed bottlenecks in CSA implementation
"""

import time
import torch
import psutil
import GPUtil
from contextlib import contextmanager
from typing import Dict, List, Any
from collections import defaultdict
import json
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    component: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    gpu_memory_before: float = 0.0
    gpu_memory_after: float = 0.0
    gpu_memory_delta: float = 0.0
    metadata: Dict[str, Any] = None

class CSAPerformanceProfiler:
    """Comprehensive performance profiler for CSA components"""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.gpu_available = torch.cuda.is_available()
        self._start_time = None
        self._total_start_memory = None

    def start_profiling(self):
        """Start overall profiling session"""
        self.metrics = []
        self._start_time = time.time()
        self._total_start_memory = self._get_memory_usage()

    def end_profiling(self) -> Dict[str, Any]:
        """End profiling session and return summary"""
        total_time = time.time() - self._start_time
        total_memory_delta = self._get_memory_usage() - self._total_start_memory

        # Aggregate metrics by component
        component_summary = defaultdict(list)
        for metric in self.metrics:
            component_summary[metric.component].append(metric.duration)

        summary = {
            "total_time": total_time,
            "total_memory_delta": total_memory_delta,
            "component_breakdown": {},
            "bottlenecks": [],
            "recommendations": []
        }

        # Analyze each component
        for component, durations in component_summary.items():
            avg_duration = sum(durations) / len(durations)
            total_component_time = sum(durations)
            percentage = (total_component_time / total_time) * 100

            summary["component_breakdown"][component] = {
                "avg_duration": avg_duration,
                "total_duration": total_component_time,
                "percentage": percentage,
                "call_count": len(durations)
            }

            # Identify bottlenecks (>20% of total time)
            if percentage > 20:
                summary["bottlenecks"].append({
                    "component": component,
                    "percentage": percentage,
                    "recommendation": self._get_optimization_recommendation(component)
                })

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(summary)

        return summary

    @contextmanager
    def profile_component(self, component_name: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling individual components"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        gpu_memory_before = self._get_gpu_memory_usage() if self.gpu_available else 0.0

        try:
            yield
        finally:
            end_time = time.time()
            memory_after = self._get_memory_usage()
            gpu_memory_after = self._get_gpu_memory_usage() if self.gpu_available else 0.0

            metric = PerformanceMetrics(
                component=component_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_after - memory_before,
                gpu_memory_before=gpu_memory_before,
                gpu_memory_after=gpu_memory_after,
                gpu_memory_delta=gpu_memory_after - gpu_memory_before,
                metadata=metadata or {}
            )

            self.metrics.append(metric)

    def _get_memory_usage(self) -> float:
        """Get current system memory usage in MB"""
        return psutil.virtual_memory().used / 1024 / 1024

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not self.gpu_available:
            return 0.0
        try:
            gpu = GPUtil.getGPUs()[0]
            return gpu.memoryUsed
        except:
            return 0.0

    def _get_optimization_recommendation(self, component: str) -> str:
        """Get optimization recommendation for a component"""
        recommendations = {
            "attention_matching": "Consider reducing compression frequency or making it lazy",
            "turbo_quant": "Optimize quantization kernels or use faster precision fallback",
            "ssd_speculation": "Implement CUDA streams for true async execution",
            "background_recovery": "Ensure non-blocking execution, reduce frequency",
            "token_generation": "Profile model forward passes, optimize batching"
        }
        return recommendations.get(component, "Profile component for specific optimizations")

    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate overall optimization recommendations"""
        recommendations = []

        # Check for major bottlenecks
        bottlenecks = summary.get("bottlenecks", [])
        if bottlenecks:
            recommendations.append(f"🚨 Address {len(bottlenecks)} major bottlenecks consuming >20% of time each")

        # Memory efficiency
        total_memory = summary.get("total_memory_delta", 0)
        if total_memory > 100:  # >100MB increase
            recommendations.append("💾 High memory usage detected - consider memory pooling or lazy loading")

        # Component-specific advice
        components = summary.get("component_breakdown", {})
        if "attention_matching" in components and components["attention_matching"]["percentage"] > 30:
            recommendations.append("🧠 Attention Matching is slow - consider per-token compression or skip for short contexts")

        if "turbo_quant" in components and components["turbo_quant"]["percentage"] > 25:
            recommendations.append("🔢 TurboQuant overhead high - try 4-bit precision or cached quantization")

        if "ssd_speculation" in components and components["ssd_speculation"]["percentage"] < 10:
            recommendations.append("⚡ SSD speculation is fast - ensure it's running asynchronously")

        # General recommendations
        recommendations.extend([
            "🔍 Use torch.profiler for detailed GPU kernel analysis",
            "📊 Monitor GPU utilization with nvidia-smi during generation",
            "⚡ Consider model quantization (GPTQ/AWQ) for baseline speed",
            "🔄 Implement batching for multiple concurrent requests"
        ])

        return recommendations

    def export_metrics(self, filename: str = "csa_performance_profile.json"):
        """Export detailed metrics to JSON file"""
        export_data = {
            "summary": self.end_profiling(),
            "detailed_metrics": [asdict(m) for m in self.metrics],
            "export_time": time.time(),
            "total_measurements": len(self.metrics)
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"📊 Performance profile exported to {filename}")
        return export_data

# Global profiler instance
profiler = CSAPerformanceProfiler()

def get_profiler() -> CSAPerformanceProfiler:
    """Get the global profiler instance"""
    return profiler

@contextmanager
def profile_component(component_name: str, metadata: Dict[str, Any] = None):
    """Convenience function for profiling components"""
    with profiler.profile_component(component_name, metadata):
        yield

def start_csa_profiling():
    """Start CSA performance profiling"""
    profiler.start_profiling()
    print("🎯 CSA Performance Profiling Started")

def end_csa_profiling(save_to_file: bool = True) -> Dict[str, Any]:
    """End CSA performance profiling and return summary"""
    summary = profiler.end_profiling()

    print("📊 CSA Performance Profiling Complete")
    print(f"⏱️  Total Time: {summary['total_time']:.3f}s")
    print(f"💾 Memory Delta: {summary['total_memory_delta']:+.1f}MB")

    if summary['bottlenecks']:
        print(f"🚨 Found {len(summary['bottlenecks'])} bottlenecks:")
        for bottleneck in summary['bottlenecks']:
            print(f"   • {bottleneck['component']}: {bottleneck['percentage']:.1f}%")

    if save_to_file:
        profiler.export_metrics()

    return summary