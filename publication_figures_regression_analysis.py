"""
Publication Figure Generator & Linear Thermal Regression Script
Reads the CSV telemetry outputs produced by phase1_scaling_benchmark.py and
phase2_thermopace4b_benchmark.py, computes OLS thermal regressions (Eqs. 2 & 5),
and exports 300 DPI publication-ready figures with bottom-aligned panel captions.

Required Input Files:
  - green_4bit_2048_metrics.csv
  - emissions.csv
  - thermopace_results.csv

Generated Output Figures:
  - fig1_node_demand.png        (IEA global demand + CodeCarbon node energy split)
  - fig2_scaling_profile.png    (Power, Energy/Token vs Temp, VRAM, and PPW curves)
  - fig5_thermopace_eval.png    (Baseline vs. ThermoPace-4B 4-panel comparison)
  - fig6_batch_projection.png   (Analytical multi-batch VRAM & weight amortization)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.edgecolor": "#111111",
            "axes.linewidth": 0.9,
        }
    )


def compute_regressions(df_phase1: pd.DataFrame):
    temps = df_phase1["Peak_Temp_C"].to_numpy(dtype=float)
    avg_w = df_phase1["Avg_Power_W"].to_numpy(dtype=float)
    j_tok = df_phase1["Joules_Per_Token"].to_numpy(dtype=float)

    c1_p, c0_p = np.polyfit(temps, avg_w, 1)
    c1_e, c0_e = np.polyfit(temps, j_tok, 1)
    r_temp_energy = np.corrcoef(temps, j_tok)[0, 1]

    print("=== OLS THERMAL REGRESSION LAWS ===")
    print(f"Eq. (2) Power Law : P_avg(T) = {c0_p:.3f} + {c1_p:.4f} * T  (W)")
    print(
        f"Eq. (5) Energy Law: E_tok(T) = {c0_e:.3f} + {c1_e:.4f} * T  "
        f"(J/tok, Pearson r = {r_temp_energy:.3f})\n"
    )
    return r_temp_energy


def plot_fig1(df_emissions: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Panel (a): IEA Electricity 2024 Global Data Center Projections
    labels_a = [
        "2022\n(Actual)",
        "2024\n(Est.)",
        "2026\n(Low Case)",
        "2026\n(Base Case)",
        "2026\n(High Case)",
    ]
    twh = [460, 620, 650, 800, 1050]
    colors_a = ["#4472c4", "#2b7bba", "#8faadc", "#1f4e78", "#c55a11"]
    bars_a = ax1.bar(labels_a, twh, color=colors_a, width=0.52)
    ax1.set_ylabel("Global Electricity Demand (TWh/yr)")
    ax1.set_ylim(0, 1220)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v in zip(bars_a, twh):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 28,
            f"{v} TWh",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.text(
        0.5,
        -0.25,
        "(a) Global Data Center Power Demand (IEA)",
        transform=ax1.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel (b): Measured Node Energy Split from CodeCarbon emissions.csv
    row = df_emissions.iloc[0]
    gpu_wh = float(row["gpu_energy"]) * 1000.0
    ram_wh = float(row["ram_energy"]) * 1000.0
    cpu_wh = float(row["cpu_energy"]) * 1000.0
    tot_wh = gpu_wh + ram_wh + cpu_wh

    subs = [
        "NVIDIA Tesla T4\nGPU (4-Bit NF4)",
        "Host DDR4 RAM\n(12.67 GB)",
        "Intel Xeon CPU\n(2 vCPUs)",
    ]
    wh_vals = [gpu_wh, ram_wh, cpu_wh]
    pcts = [100.0 * v / tot_wh for v in wh_vals]
    bars_b = ax2.bar(subs, wh_vals, color=["#1f4e78", "#4472c4", "#8faadc"], width=0.45)
    ax2.set_ylabel("Measured Energy (Wh / 4,032 Tokens)")
    ax2.set_ylim(0, max(wh_vals) * 1.28)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v, p in zip(bars_b, wh_vals, pcts):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.16,
            f"{v:.2f} Wh\n({p:.1f}%)",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.text(
        0.5,
        -0.25,
        "(b) Measured Node Energy Split (CodeCarbon)",
        transform=ax2.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig("fig1_node_demand.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fig2(df_phase1: pd.DataFrame, r_val: float):
    toks = [str(t) for t in df_phase1["Target_Tokens"]]
    x = np.arange(len(toks))
    avg_w = df_phase1["Avg_Power_W"].to_numpy()
    peak_w = df_phase1["Peak_Power_W"].to_numpy()
    j_tok = df_phase1["Joules_Per_Token"].to_numpy()
    temp = df_phase1["Peak_Temp_C"].to_numpy()
    vram = df_phase1["Peak_VRAM_GB"].to_numpy()
    ppw = df_phase1["Tokens_Per_Sec"].to_numpy() / avg_w

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))

    # (a) GPU Power Draw
    ax = axes[0, 0]
    ax.bar(x - 0.14, avg_w, width=0.28, color="#1f4e78", label="Avg Power (W)")
    ax.plot(
        x + 0.14,
        peak_w,
        color="#c55a11",
        marker="o",
        linewidth=2.0,
        label="Peak Power (W)",
    )
    for i in range(len(x)):
        ax.text(
            x[i] - 0.14,
            avg_w[i] + 1.5,
            f"{avg_w[i]:.1f}",
            ha="center",
            fontsize=7.5,
            color="#1f4e78",
            fontweight="bold",
        )
        ax.text(
            x[i] + 0.14,
            peak_w[i] + 2.2,
            f"{peak_w[i]:.1f}",
            ha="center",
            fontsize=7.5,
            color="#c55a11",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(toks)
    ax.set_ylabel("Board Power (W)")
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.text(
        0.5,
        -0.20,
        "(a) GPU Power Draw vs. Output Length",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (b) Energy Cost & Thermal Rise
    ax = axes[0, 1]
    l1 = ax.plot(
        x,
        j_tok,
        color="#2b7bba",
        marker="s",
        linewidth=2.2,
        label="Energy/Token (J/tok)",
    )
    ax.fill_between(x, j_tok, 3.0, color="#2b7bba", alpha=0.12)
    l3 = ax.axvline(
        3,
        color="#38761d",
        linestyle="--",
        linewidth=1.3,
        label="512-Tok Sweet Spot",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(toks)
    ax.set_ylabel("Energy (J / Token)", color="#1f4e78")
    ax.set_ylim(3.0, 5.2)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    ax_t = ax.twinx()
    l2 = ax_t.plot(
        x,
        temp,
        color="#a61c1c",
        marker="^",
        linestyle="-.",
        linewidth=2.0,
        label="Peak Die Temp (°C)",
    )
    ax_t.set_ylabel("Peak Temp (°C)", color="#a61c1c")
    ax_t.set_ylim(40, 90)
    ax_t.tick_params(axis="y", colors="#a61c1c")
    lines = l1 + [l3] + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax.text(
        0.5,
        -0.20,
        f"(b) Energy Cost & Thermal Rise (r = {r_val:.2f})",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (c) Peak VRAM Scaling
    ax = axes[1, 0]
    ax.plot(x, vram, color="#5b4b8a", marker="D", linewidth=2.2)
    for i, v in enumerate(vram):
        ax.text(x[i], v + 0.007, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(toks)
    ax.set_xlabel("Generated Output Tokens")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_ylim(1.92, 2.12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    vram_pct = 100.0 * (vram[-1] - vram[0]) / vram[0]
    ax.text(
        0.5,
        -0.25,
        f"(c) 4-Bit NF4 Peak VRAM Scaling (+{vram_pct:.2f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (d) Performance per Watt Drop
    ax = axes[1, 1]
    ax.plot(x, ppw, color="#38761d", marker="o", linewidth=2.2)
    ax.fill_between(x, ppw, 0.18, color="#38761d", alpha=0.12)
    for i, v in enumerate(ppw):
        offset = -0.014 if i == len(ppw) - 1 else 0.008
        ax.text(
            x[i],
            v + offset,
            f"{v:.3f}",
            ha="center",
            fontsize=8,
            color="#274e13",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(toks)
    ax.set_xlabel("Generated Output Tokens")
    ax.set_ylabel("Performance/Watt (Tok/s/W)")
    ax.set_ylim(0.18, 0.34)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ppw_drop = 100.0 * (ppw[0] - ppw[-1]) / ppw[0]
    ax.text(
        0.5,
        -0.25,
        f"(d) Performance per Watt Drop (-{ppw_drop:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.38, bottom=0.12)
    plt.savefig("fig2_scaling_profile.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fig5(df_phase2: pd.DataFrame, grid_intensity: float = 165.54):
    base = df_phase2.iloc[0]
    prop = df_phase2.iloc[1]

    systems = ["Uncontrolled\nBaseline", "ThermoPace-4B\n(Proposed)"]
    tec = [float(base["Total_Energy_J"]), float(prop["Total_Energy_J"])]
    avg_p = [float(base["Avg_Power_W"]), float(prop["Avg_Power_W"])]
    peak_p = [float(base["Peak_Power_W"]), float(prop["Peak_Power_W"])]
    ce = [(e / 3.6e6) * grid_intensity for e in tec]
    ppw = [
        float(base["Tokens_Per_Sec"]) / avg_p[0],
        float(prop["Tokens_Per_Sec"]) / avg_p[1],
    ]
    j_tok = [float(base["Joules_Per_Token"]), float(prop["Joules_Per_Token"])]

    tec_red = 100.0 * (tec[0] - tec[1]) / tec[0]
    peak_red = 100.0 * (peak_p[0] - peak_p[1]) / peak_p[0]
    ppw_gain = 100.0 * (ppw[1] - ppw[0]) / ppw[0]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    # (a) TEC
    ax = axes[0, 0]
    b1 = ax.bar(systems, tec, color=["#7f8c8d", "#1f4e78"], width=0.45)
    ax.set_ylabel("Total Energy (TEC, J)")
    ax.set_ylim(0, max(tec) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v in zip(b1, tec):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 260,
            f"{v:,.1f} J",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.annotate(
        f"-{tec_red:.2f}% TEC\n(-{tec[0] - tec[1]:,.0f} J)",
        xy=(0.82, tec[1] + 150),
        xytext=(0.45, tec[0] * 0.74),
        arrowprops=dict(arrowstyle="->", color="#38761d", lw=1.8),
        fontsize=8.5,
        fontweight="bold",
        color="#38761d",
        ha="center",
    )
    ax.text(
        0.5,
        -0.22,
        "(a) Total Energy Consumption (TEC)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (b) Power Draw
    ax = axes[0, 1]
    x = np.arange(2)
    b_a = ax.bar(x - 0.17, avg_p, width=0.32, color="#2b7bba", label="Avg Power (W)")
    b_p = ax.bar(x + 0.17, peak_p, width=0.32, color="#c55a11", label="Peak Power (W)")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Board Power (W)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b in b_a:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2,
            f"{b.get_height():.1f} W",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1f4e78",
        )
    for b in b_p:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2,
            f"{b.get_height():.1f} W",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#c55a11",
        )
    ax.text(
        0.5,
        -0.22,
        f"(b) Board Power (-{peak_red:.2f}% Peak Spike)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (c) Carbon Footprint
    ax = axes[1, 0]
    b3 = ax.bar(systems, ce, color=["#95a5a6", "#38761d"], width=0.45)
    ax.set_ylabel("Carbon Emissions (CE, gCO2eq)")
    ax.set_ylim(0, max(ce) * 1.24)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v in zip(b3, ce):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.015,
            f"{v:.3f} gCO2eq",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.text(
        0.5,
        -0.22,
        f"(c) Operational Carbon Footprint (-{tec_red:.2f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    # (d) PPW Gain
    ax = axes[1, 1]
    b4 = ax.bar(systems, ppw, color=["#7f8c8d", "#2b7bba"], width=0.45)
    ax.set_ylabel("Perf / Watt (PPW, tok/s/W)")
    ax.set_ylim(0, 0.30)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v, jt in zip(b4, ppw, j_tok):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.010,
            f"{v:.3f} tok/s/W\n({jt:.3f} J/tok)",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
        )
    ax.text(
        0.5,
        -0.22,
        f"(d) Performance per Watt (+{ppw_gain:.2f}% Gain)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.38, bottom=0.12)
    plt.savefig("fig5_thermopace_eval.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fig6():
    batches = [
        "B=1\n(Measured)",
        "B=2\n(Proj.)",
        "B=4\n(Proj.)",
        "B=8\n(Proj.)",
        "B=16\n(Proj.)",
    ]
    vram_proj = [1.991, 2.047, 2.159, 2.383, 2.831]
    j_tok_proj = [4.177, 2.506, 1.671, 1.253, 1.044]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    bars = ax1.bar(
        batches,
        vram_proj,
        color=["#1f4e78", "#4472c4", "#4472c4", "#4472c4", "#4472c4"],
        width=0.46,
    )
    ax1.axhline(
        16.0, color="#a61c1c", linestyle="--", linewidth=1.4, label="T4 16 GB VRAM Limit"
    )
    ax1.set_ylabel("Total VRAM Footprint (GB)")
    ax1.set_ylim(0, 18.2)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b, v in zip(bars, vram_proj):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.45,
            f"{v:.2f} GB\n({100.0 * v / 16.0:.1f}%)",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
        )
    ax1.text(
        0.5,
        -0.25,
        "(a) VRAM Capacity Scaling (Analytical)",
        transform=ax1.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    ax2.plot(batches, j_tok_proj, color="#38761d", marker="s", linewidth=2.3, markersize=7)
    ax2.fill_between(batches, j_tok_proj, 0.8, color="#38761d", alpha=0.12)
    ax2.set_ylabel("Amortized Energy (J / Token)")
    ax2.set_ylim(0.8, 4.9)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(j_tok_proj):
        ax2.text(
            i,
            v + 0.16,
            f"{v:.2f} J/tok",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.text(
        0.5,
        -0.25,
        "(b) Projected Batch Weight-Read Amortization",
        transform=ax2.transAxes,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig("fig6_batch_projection.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    required_files = [
        "green_4bit_2048_metrics.csv",
        "emissions.csv",
        "thermopace_results.csv",
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing telemetry CSV file(s): {missing}. "
            "Run phase1_scaling_benchmark.py and phase2_thermopace4b_benchmark.py first."
        )

    setup_style()
    df_phase1 = pd.read_csv("green_4bit_2048_metrics.csv")
    df_emissions = pd.read_csv("emissions.csv")
    df_phase2 = pd.read_csv("thermopace_results.csv")

    r_val = compute_regressions(df_phase1)
    plot_fig1(df_emissions)
    plot_fig2(df_phase1, r_val)
    plot_fig5(df_phase2)
    plot_fig6()
    print(
        "Successfully generated 300 DPI figures: fig1_node_demand.png, "
        "fig2_scaling_profile.png, fig5_thermopace_eval.png, fig6_batch_projection.png"
    )


if __name__ == "__main__":
    main()