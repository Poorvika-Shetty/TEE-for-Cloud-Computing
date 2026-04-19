from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass(frozen=True)
class WorkloadProfile:
    base_latency_ms: float
    base_throughput_rps: float
    base_memory_mb: float
    io_sensitivity: float
    cpu_sensitivity: float
    memory_sensitivity: float


WORKLOADS = {
    "DNA Alignment (Short Reads)": WorkloadProfile(
        base_latency_ms=210.0,
        base_throughput_rps=92000.0,
        base_memory_mb=2100.0,
        io_sensitivity=0.72,
        cpu_sensitivity=0.60,
        memory_sensitivity=0.58,
    ),
    "DNA Alignment (Long Reads)": WorkloadProfile(
        base_latency_ms=395.0,
        base_throughput_rps=47000.0,
        base_memory_mb=3300.0,
        io_sensitivity=0.56,
        cpu_sensitivity=0.74,
        memory_sensitivity=0.70,
    ),
    "Variant Calling Pre-processing": WorkloadProfile(
        base_latency_ms=175.0,
        base_throughput_rps=115000.0,
        base_memory_mb=1850.0,
        io_sensitivity=0.64,
        cpu_sensitivity=0.54,
        memory_sensitivity=0.49,
    ),
    "Metagenomics Screening": WorkloadProfile(
        base_latency_ms=260.0,
        base_throughput_rps=76000.0,
        base_memory_mb=2900.0,
        io_sensitivity=0.68,
        cpu_sensitivity=0.67,
        memory_sensitivity=0.72,
    ),
}


TEE_PROFILES = {
    "SGX": {
        "latency_mult": 1.58,
        "throughput_mult": 0.64,
        "memory_mult": 1.42,
        "security_score": 9.5,
        "io_penalty": 0.18,
        "cpu_penalty": 0.16,
        "mem_penalty": 0.12,
    },
    "TDX": {
        "latency_mult": 1.26,
        "throughput_mult": 0.82,
        "memory_mult": 1.18,
        "security_score": 8.9,
        "io_penalty": 0.10,
        "cpu_penalty": 0.08,
        "mem_penalty": 0.07,
    },
    "SEV-SNP": {
        "latency_mult": 1.17,
        "throughput_mult": 0.88,
        "memory_mult": 1.12,
        "security_score": 8.5,
        "io_penalty": 0.08,
        "cpu_penalty": 0.06,
        "mem_penalty": 0.05,
    },
}


SGX_RUNTIMES = {
    "Native Enclave": {
        "latency_mult": 1.00,
        "throughput_mult": 1.00,
        "memory_mult": 1.00,
        "security_delta": 0.00,
        "description": "Direct SGX programming model with lowest software stack overhead.",
    },
    "Gramine-SGX": {
        "latency_mult": 1.18,
        "throughput_mult": 0.86,
        "memory_mult": 1.11,
        "security_delta": -0.10,
        "description": "Library-OS compatibility layer with easier migration and extra overhead.",
    },
    "Occlum-SGX": {
        "latency_mult": 1.12,
        "throughput_mult": 0.90,
        "memory_mult": 1.07,
        "security_delta": -0.05,
        "description": "LibOS approach with moderate overhead and stronger portability.",
    },
}


def calculate_baseline(
    workload: WorkloadProfile,
    data_size_gb: int,
    concurrency_vcpus: int,
    memory_pressure: float,
) -> dict[str, float]:
    size_factor = math.log2(data_size_gb + 1.0)
    parallel_gain = 1.0 + 0.30 * math.log2(concurrency_vcpus + 1.0)
    contention = 1.0 + 0.012 * max(concurrency_vcpus - 24, 0)

    latency_ms = workload.base_latency_ms * (1.0 + 0.055 * size_factor) * contention
    throughput_rps = workload.base_throughput_rps * parallel_gain / (1.0 + 0.025 * size_factor)
    memory_mb = workload.base_memory_mb * (1.0 + 0.20 * (memory_pressure - 1.0))

    return {
        "latency_ms": latency_ms,
        "throughput_rps": throughput_rps,
        "memory_mb": memory_mb,
    }


def simulate_tee(
    tee_name: str,
    workload: WorkloadProfile,
    baseline: dict[str, float],
    data_size_gb: int,
    concurrency_vcpus: int,
    memory_pressure: float,
    sgx_runtime: str,
) -> dict[str, float | str]:
    profile = TEE_PROFILES[tee_name]
    size_factor = math.log2(data_size_gb + 1.0)

    latency_ms = baseline["latency_ms"] * profile["latency_mult"]
    latency_ms *= 1.0 + workload.io_sensitivity * profile["io_penalty"] * (0.75 + 0.25 * size_factor)
    latency_ms *= 1.0 + max(concurrency_vcpus - 16, 0) * 0.009 * (1.0 + profile["cpu_penalty"])

    throughput_rps = baseline["throughput_rps"] * profile["throughput_mult"]
    throughput_rps /= (
        1.0
        + workload.cpu_sensitivity
        * profile["cpu_penalty"]
        * math.log2(concurrency_vcpus + 1.0)
        / 10.0
    )
    throughput_rps = max(throughput_rps, 1.0)

    memory_mb = baseline["memory_mb"] * profile["memory_mult"]
    memory_mb *= 1.0 + workload.memory_sensitivity * profile["mem_penalty"] * (memory_pressure - 0.8)

    security_score = profile["security_score"]

    if tee_name == "SGX":
        runtime = SGX_RUNTIMES[sgx_runtime]
        latency_ms *= runtime["latency_mult"]
        throughput_rps *= runtime["throughput_mult"]
        memory_mb *= runtime["memory_mult"]
        security_score += runtime["security_delta"]

    latency_overhead_pct = ((latency_ms / baseline["latency_ms"]) - 1.0) * 100.0
    throughput_loss_pct = (1.0 - (throughput_rps / baseline["throughput_rps"])) * 100.0
    memory_overhead_pct = ((memory_mb / baseline["memory_mb"]) - 1.0) * 100.0

    latency_component = baseline["latency_ms"] / latency_ms
    throughput_component = throughput_rps / baseline["throughput_rps"]
    memory_component = baseline["memory_mb"] / memory_mb

    performance_index = 100.0 * (
        0.45 * latency_component + 0.40 * throughput_component + 0.15 * memory_component
    )

    return {
        "TEE": tee_name,
        "Latency (ms)": latency_ms,
        "Throughput (reads/s)": throughput_rps,
        "Memory (MB)": memory_mb,
        "Latency overhead (%)": latency_overhead_pct,
        "Throughput loss (%)": throughput_loss_pct,
        "Memory overhead (%)": memory_overhead_pct,
        "Performance Index": performance_index,
        "Security Score (0-10)": max(0.0, min(10.0, security_score)),
    }


def build_scalability_frame(
    workload: WorkloadProfile,
    data_size_gb: int,
    memory_pressure: float,
    sgx_runtime: str,
) -> pd.DataFrame:
    points = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    rows: list[dict[str, float | str | int]] = []

    for concurrency_vcpus in points:
        baseline = calculate_baseline(workload, data_size_gb, concurrency_vcpus, memory_pressure)
        for tee_name in TEE_PROFILES:
            result = simulate_tee(
                tee_name=tee_name,
                workload=workload,
                baseline=baseline,
                data_size_gb=data_size_gb,
                concurrency_vcpus=concurrency_vcpus,
                memory_pressure=memory_pressure,
                sgx_runtime=sgx_runtime,
            )
            rows.append(
                {
                    "Concurrency (vCPUs)": concurrency_vcpus,
                    "TEE": tee_name,
                    "Throughput (reads/s)": result["Throughput (reads/s)"],
                    "Latency (ms)": result["Latency (ms)"],
                }
            )

    return pd.DataFrame(rows)


st.set_page_config(page_title="TEE Performance Simulation Dashboard", layout="wide")
st.title("Trusted Execution Environments (TEE) Performance Simulator")
st.caption(
    "Simulation dashboard for SGX, TDX, and SEV-SNP under confidential bioinformatics workloads."
)

with st.sidebar:
    st.header("Simulation Inputs")
    workload_name = st.selectbox("Workload", list(WORKLOADS.keys()), index=0)
    data_size_gb = st.slider("Input dataset size (GB)", min_value=10, max_value=500, value=120, step=10)
    concurrency_vcpus = st.slider("Concurrent compute units (vCPUs)", min_value=1, max_value=64, value=16)
    memory_pressure = st.slider("Memory pressure factor", min_value=0.8, max_value=2.0, value=1.1, step=0.1)
    sgx_runtime = st.selectbox("SGX runtime mode", list(SGX_RUNTIMES.keys()), index=1)
    security_weight_pct = st.slider(
        "Security weight in overall score (%)", min_value=0, max_value=100, value=60, step=5
    )

    st.markdown("---")
    st.markdown("**SGX runtime note**")
    st.caption(SGX_RUNTIMES[sgx_runtime]["description"])

workload = WORKLOADS[workload_name]
security_weight = security_weight_pct / 100.0

baseline = calculate_baseline(workload, data_size_gb, concurrency_vcpus, memory_pressure)
result_rows = [
    simulate_tee(
        tee_name=tee_name,
        workload=workload,
        baseline=baseline,
        data_size_gb=data_size_gb,
        concurrency_vcpus=concurrency_vcpus,
        memory_pressure=memory_pressure,
        sgx_runtime=sgx_runtime,
    )
    for tee_name in TEE_PROFILES
]
results_df = pd.DataFrame(result_rows)
results_df["Trade-off Score"] = (
    security_weight * (results_df["Security Score (0-10)"] * 10.0)
    + (1.0 - security_weight) * results_df["Performance Index"]
)

st.subheader("Baseline (Non-TEE) Reference")
base_col1, base_col2, base_col3 = st.columns(3)
base_col1.metric("Baseline latency", f"{baseline['latency_ms']:.1f} ms")
base_col2.metric("Baseline throughput", f"{baseline['throughput_rps']:,.0f} reads/s")
base_col3.metric("Baseline memory", f"{baseline['memory_mb']:,.0f} MB")

st.subheader("TEE Benchmark Table")
display_df = results_df.sort_values("Trade-off Score", ascending=False).copy()
st.dataframe(
    display_df.style.format(
        {
            "Latency (ms)": "{:.2f}",
            "Throughput (reads/s)": "{:,.0f}",
            "Memory (MB)": "{:,.0f}",
            "Latency overhead (%)": "{:.1f}",
            "Throughput loss (%)": "{:.1f}",
            "Memory overhead (%)": "{:.1f}",
            "Performance Index": "{:.1f}",
            "Security Score (0-10)": "{:.1f}",
            "Trade-off Score": "{:.1f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_latency = px.bar(
        results_df,
        x="TEE",
        y="Latency (ms)",
        color="TEE",
        text_auto=".1f",
        title="Latency under Confidential Execution",
    )
    fig_latency.update_layout(showlegend=False)
    st.plotly_chart(fig_latency, use_container_width=True)

    fig_memory = px.bar(
        results_df,
        x="TEE",
        y="Memory overhead (%)",
        color="TEE",
        text_auto=".1f",
        title="Memory Overhead vs Baseline",
    )
    fig_memory.update_layout(showlegend=False)
    st.plotly_chart(fig_memory, use_container_width=True)

with chart_col2:
    fig_throughput = px.bar(
        results_df,
        x="TEE",
        y="Throughput (reads/s)",
        color="TEE",
        text_auto=".0f",
        title="Throughput under Confidential Execution",
    )
    fig_throughput.update_layout(showlegend=False)
    st.plotly_chart(fig_throughput, use_container_width=True)

    fig_tradeoff = px.scatter(
        results_df,
        x="Performance Index",
        y="Security Score (0-10)",
        size="Trade-off Score",
        color="TEE",
        text="TEE",
        title="Security vs Performance Trade-off",
    )
    fig_tradeoff.update_traces(textposition="top center")
    st.plotly_chart(fig_tradeoff, use_container_width=True)

st.subheader("Scalability View")
scalability_df = build_scalability_frame(workload, data_size_gb, memory_pressure, sgx_runtime)
fig_scalability = px.line(
    scalability_df,
    x="Concurrency (vCPUs)",
    y="Throughput (reads/s)",
    color="TEE",
    markers=True,
    title="Throughput Scalability Across Enclave Types",
)
st.plotly_chart(fig_scalability, use_container_width=True)

fastest_tee = results_df.loc[results_df["Latency (ms)"].idxmin(), "TEE"]
highest_throughput_tee = results_df.loc[results_df["Throughput (reads/s)"].idxmax(), "TEE"]
best_security_tee = results_df.loc[results_df["Security Score (0-10)"].idxmax(), "TEE"]
best_tradeoff_tee = results_df.loc[results_df["Trade-off Score"].idxmax(), "TEE"]


