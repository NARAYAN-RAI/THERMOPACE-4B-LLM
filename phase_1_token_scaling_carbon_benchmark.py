"""
Phase I: Empirical Thermal, Power, VRAM, and Carbon Scaling Benchmark
Model: Qwen/Qwen2.5-3B-Instruct (4-Bit NF4 Double Quantization)
Hardware: Entry-Level Cloud GPU (e.g., NVIDIA Tesla T4, 16 GB VRAM)

Generates:
  1. green_4bit_2048_metrics.csv  - Per-horizon GPU power, temperature, speed, energy, and VRAM
  2. emissions.csv               - Full-node (GPU + RAM + CPU) electricity and CO2eq log via CodeCarbon
  3. green_4bit_2048_profile.png - 3-panel summary chart of power, energy/token, and VRAM scaling
"""

import time
import threading
import torch
import pynvml
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class GPUPowerMonitor:
    """Background daemon thread polling NVIDIA GPU board power (W) and die temp (C) at 10 Hz."""

    def __init__(self, handle, interval: float = 0.1):
        self.handle = handle
        self.interval = interval
        self.watts = []
        self.temps = []
        self._running = False
        self.thread = None

    def _poll(self):
        while self._running:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            temp_c = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self.watts.append(power_mw / 1000.0)
            self.temps.append(temp_c)
            time.sleep(self.interval)

    def start(self):
        self.watts = []
        self.temps = []
        self._running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False
        if self.thread is not None:
            self.thread.join()
        avg_w = sum(self.watts) / len(self.watts) if self.watts else 0.0
        peak_w = max(self.watts) if self.watts else 0.0
        peak_t = max(self.temps) if self.temps else 0
        return avg_w, peak_w, peak_t


def main():
    # 1. Initialize 10 Hz NVML Hardware Telemetry
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    print(f"Detected Hardware: {gpu_name}")

    # 2. Load Qwen2.5-3B-Instruct in 4-Bit NF4 Double Quantization
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_id} in 4-bit NF4 precision...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Base 4-Bit Model Footprint in VRAM: {base_vram_gb:.3f} GB\n")

    # 3. GPU Warm-Up Pass (Initializes CUDA contexts & fused SDPA kernels)
    warmup_inputs = tokenizer("Warm up GPU kernels.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=16,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    # 4. Doubling Token Scaling Sweep: 64 -> 2,048 Tokens
    token_targets = [64, 128, 256, 512, 1024, 2048]
    prompt = (
        "Provide a comprehensive, exhaustive, step-by-step technical explanation of how "
        "modern computers process instructions, manage memory hierarchies, and execute "
        "parallel workloads on graphics processing units."
    )
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    monitor = GPUPowerMonitor(gpu_handle, interval=0.1)
    tracker = EmissionsTracker(
        project_name="Green_4Bit_2048_Scaling",
        output_dir=".",
        log_level="error",
    )
    tracker.start()

    results = []
    for target_tokens in token_targets:
        print(f"Running workload: {target_tokens} tokens...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor.start()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                min_new_tokens=target_tokens,
                max_new_tokens=target_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed_sec = time.perf_counter() - t0
        avg_watts, peak_watts, peak_temp = monitor.stop()

        tokens_generated = outputs.shape[1] - input_len
        total_joules = avg_watts * elapsed_sec
        joules_per_token = total_joules / tokens_generated
        tokens_per_sec = tokens_generated / elapsed_sec
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        results.append(
            {
                "Target_Tokens": target_tokens,
                "Actual_Tokens": tokens_generated,
                "Time_Sec": round(elapsed_sec, 2),
                "Tokens_Per_Sec": round(tokens_per_sec, 2),
                "Avg_Power_W": round(avg_watts, 2),
                "Peak_Power_W": round(peak_watts, 2),
                "Peak_Temp_C": peak_temp,
                "Total_Energy_J": round(total_joules, 2),
                "Joules_Per_Token": round(joules_per_token, 3),
                "Peak_VRAM_GB": round(peak_vram_gb, 3),
            }
        )

    total_co2_kg = tracker.stop()
    pynvml.nvmlShutdown()

    # 5. Export Metrics CSV & Quick Summary Figure
    df = pd.DataFrame(results)
    csv_path = "green_4bit_2048_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== 64-TO-2048 TOKEN SCALING RESULTS ===")
    print(df.to_string(index=False))
    if total_co2_kg is not None:
        print(f"\nTotal Operational Carbon Footprint (CodeCarbon): {total_co2_kg * 1000:.5f} gCO2eq")

    x = list(range(len(df)))
    x_labels = [str(t) for t in df["Target_Tokens"]]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    ax1.bar(x, df["Avg_Power_W"], color="#2b7bba", width=0.55, label="Avg Power (W)")
    ax1.plot(x, df["Peak_Power_W"], color="#d95f02", marker="o", linewidth=2, label="Peak Power (W)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Generated Tokens")
    ax1.set_ylabel("Power (Watts)")
    ax1.set_title("(a) GPU Power Draw vs. Output Length", y=-0.25)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.plot(x, df["Joules_Per_Token"], color="#1b9e77", marker="s", linewidth=2.5)
    ax2.fill_between(x, df["Joules_Per_Token"], color="#1b9e77", alpha=0.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Generated Tokens")
    ax2.set_ylabel("Energy (Joules / Token)")
    ax2.set_title("(b) Energy Cost per Token (Lower = Greener)", y=-0.25)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    ax3.plot(x, df["Peak_VRAM_GB"], color="#7570b3", marker="^", linewidth=2.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels)
    ax3.set_xlabel("Generated Tokens")
    ax3.set_ylabel("Peak GPU Memory (GB)")
    ax3.set_title("(c) 4-Bit Peak VRAM Scaling (KV-Cache)", y=-0.25)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("green_4bit_2048_profile.png", dpi=300, bbox_inches="tight")
    print("Saved green_4bit_2048_metrics.csv, emissions.csv, and green_4bit_2048_profile.png")

    # Automatically trigger browser download if running inside Google Colab
    try:
        from google.colab import files
        files.download(csv_path)
        files.download("emissions.csv")
    except ImportError:
        pass


if __name__ == "__main__":
    main()