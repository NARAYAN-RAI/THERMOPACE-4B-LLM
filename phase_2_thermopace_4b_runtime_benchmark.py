"""
Phase II: ThermoPace-4B Zero-Privilege User-Space Thermodynamic Runtime
Modules:
  1. Closed-Loop RC-PID Thermodynamic Token Pacer (stride-16 NVML polling, T_set = 58.0 C)
  2. GreenKV Shannon Logit-Entropy Tracker (SDPA / FlashAttention-2 safe, zero extra forward pass)
  3. Confidence-Gated Speculative Self-Exit (EMA top-1 certainty >= 0.90 at clause boundaries)

Generates:
  - thermopace_results.csv - Controlled side-by-side comparison vs. Uncontrolled Baseline
"""

import time
import threading
import torch
import pynvml
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)


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


class ThermoPaceLogitsProcessor(LogitsProcessor):
    """
    Module 1 & Module 2:
      - Computes next-token Shannon entropy H_k directly from logits (SDPA-safe).
      - Updates EMA top-1 token certainty p_bar_max.
      - Polls GPU die temperature every `poll_stride` steps and injects bounded
        PD micro-sleep pulses when T_curr > t_set to arrest thermal leakage.
    """

    def __init__(
        self,
        handle,
        t_set: float = 58.0,
        kp: float = 0.0035,
        kd: float = 0.002,
        poll_stride: int = 16,
        max_sleep: float = 0.045,
    ):
        self.handle = handle
        self.t_set = t_set
        self.kp = kp
        self.kd = kd
        self.poll_stride = poll_stride
        self.max_sleep = max_sleep

        self.step = 0
        self.sleep_events = 0
        self.confidence_ema = 0.0
        self.entropy_scores = []

        self.prev_temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        self.prev_time = time.perf_counter()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.step += 1

        # Module 2: Shannon entropy H_k & EMA confidence from output logits
        probs = torch.softmax(scores[0].float(), dim=-1)
        top_p = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
        self.entropy_scores.append(entropy)
        self.confidence_ema = 0.85 * self.confidence_ema + 0.15 * top_p

        # Module 1: Stride-16 closed-loop RC-PID thermodynamic duty cycling
        if self.step % self.poll_stride == 0:
            now = time.perf_counter()
            curr_temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            dt = max(now - self.prev_time, 1e-3)
            dtemp_dt = (curr_temp - self.prev_temp) / dt

            if curr_temp > self.t_set:
                sleep_sec = self.kp * (curr_temp - self.t_set) + self.kd * max(
                    0.0, dtemp_dt
                )
                sleep_sec = min(max(sleep_sec, 0.0), self.max_sleep)
                if sleep_sec > 0.0:
                    torch.cuda.synchronize()
                    time.sleep(sleep_sec)
                    self.sleep_events += 1

            self.prev_temp = curr_temp
            self.prev_time = time.perf_counter()

        return scores


class ConfidenceSelfExitCriteria(StoppingCriteria):
    """
    Module 3: Terminates autoregressive generation cleanly at clause boundaries
    ('.' or '\\n') once minimum length is reached and smoothed certainty >= conf_thresh.
    """

    def __init__(
        self,
        processor: ThermoPaceLogitsProcessor,
        tokenizer,
        prompt_len: int,
        min_new_tokens: int = 512,
        conf_thresh: float = 0.90,
    ):
        self.processor = processor
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens
        self.conf_thresh = conf_thresh
        self.boundary_ids = set()
        for ch in [".", "\n"]:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            if ids:
                self.boundary_ids.add(ids[-1])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < self.min_new_tokens:
            return False
        last_tok = input_ids[0, -1].item()
        return (last_tok in self.boundary_ids) and (
            self.processor.confidence_ema >= self.conf_thresh
        )


def main():
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

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

    prompt = (
        "Provide an exhaustive technical explanation of how modern GPUs execute "
        "parallel workloads and manage memory hierarchies."
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    in_len = inputs["input_ids"].shape[1]

    # Warm-up pass
    with torch.inference_mode():
        _ = model.generate(
            **inputs, max_new_tokens=16, pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()

    monitor = GPUPowerMonitor(gpu_handle, interval=0.1)
    comparison = []

    for idx, mode in enumerate(
        ["Baseline_Uncontrolled_2048", "ThermoPace_4B_Controlled"]
    ):
        if idx == 1:
            print("Passive cooldown for 30 seconds before ThermoPace-4B run...")
            time.sleep(30)

        print(f"Running {mode}...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor.start()
        t0 = time.perf_counter()

        with torch.inference_mode():
            if mode == "Baseline_Uncontrolled_2048":
                out = model.generate(
                    **inputs,
                    min_new_tokens=2048,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                pacer_pauses = 0
                mean_entropy = None
            else:
                pacer = ThermoPaceLogitsProcessor(gpu_handle, t_set=58.0)
                stopper = ConfidenceSelfExitCriteria(
                    pacer,
                    tokenizer,
                    prompt_len=in_len,
                    min_new_tokens=512,
                    conf_thresh=0.90,
                )
                out = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([pacer]),
                    stopping_criteria=StoppingCriteriaList([stopper]),
                    pad_token_id=tokenizer.eos_token_id,
                )
                pacer_pauses = pacer.sleep_events
                mean_entropy = (
                    round(
                        sum(pacer.entropy_scores) / len(pacer.entropy_scores), 3
                    )
                    if pacer.entropy_scores
                    else 0.0
                )

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        avg_w, peak_w, peak_t = monitor.stop()

        gen_toks = out.shape[1] - in_len
        total_j = avg_w * dt
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        comparison.append(
            {
                "System": mode,
                "Tokens_Generated": gen_toks,
                "Time_Sec": round(dt, 2),
                "Tokens_Per_Sec": round(gen_toks / dt, 2),
                "Avg_Power_W": round(avg_w, 2),
                "Peak_Power_W": round(peak_w, 2),
                "Peak_Temp_C": peak_t,
                "Total_Energy_J": round(total_j, 2),
                "Joules_Per_Token": round(total_j / gen_toks, 3),
                "Peak_VRAM_GB": round(peak_vram, 3),
                "PID_Thermal_Pulses": pacer_pauses,
                "Mean_Logit_Entropy": mean_entropy,
            }
        )

    pynvml.nvmlShutdown()

    df_out = pd.DataFrame(comparison)
    csv_path = "thermopace_results.csv"
    df_out.to_csv(csv_path, index=False)

    print("\n=== BASELINE VS. THERMOPACE-4B RESULTS ===")
    print(df_out.to_string(index=False))

    # Automatically trigger browser download if running inside Google Colab
    try:
        from google.colab import files
        files.download(csv_path)
    except ImportError:
        pass


if __name__ == "__main__":
    main()