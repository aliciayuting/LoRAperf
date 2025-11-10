"""
Simplified LoRA Performance Benchmark
Easy to configure and understand
"""

import torch
import time
import gc
import csv
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import random
from typing import List, Dict, Optional, Union

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

BASE_MODEL = "huggyllama/llama-7b"  # Base model
ADAPTER_MODEL = "timdettmers/qlora-alpaca-7b"  # LoRA adapter

NUM_WARMUP_RUNS = 3  # Warmup runs (not counted in statistics)
NUM_BENCHMARK_RUNS = 20  # Number of runs for benchmarking

MACHINE_NAME = "1a6000"  # Identifier for the machine/environment

OUTPUT_CSV = f"./lora_benchmark_{MACHINE_NAME}.csv"
SUMMARY_CSV = f"./lora_benchmark_summary_{MACHINE_NAME}.csv" 

# Dataset configuration - Choose evaluation dataset from QLoRA paper
DATASET_NAME = "oasst1"  # Options: "vicuna", "oasst1", "mixed", "alpaca"
NUM_TEST_SAMPLES = 1000    # Number of prompts to test on
DATASET_SEED = 42        # Random seed for reproducible dataset sampling

# Test prompts will be loaded from dataset (see below)

# Generation parameters
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
DO_SAMPLE = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clear_memory():
    """Clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def reset_peak_memory():
    """Reset peak memory tracking"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory_mb():
    """Get peak GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_adapter_separate(base_model_name, adapter_name, test_prompts):
    """Benchmark adapter running separately (Wx + ABx)"""
    print("\n" + "="*70)
    print("BENCHMARKING: ADAPTER (SEPARATE) - Wx + ABx")
    print("="*70)
    
    clear_memory()
    reset_peak_memory()
    
    # Load base model
    print("\n1. Loading base model...")
    start_time = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_load_time = time.time() - start_time
    base_memory = get_gpu_memory_mb()
    print(f"   ✓ Base model loaded in {base_load_time:.2f}s")
    print(f"   ✓ Base model memory: {base_memory:.2f} MB")
    
    # Load adapter
    print("\n2. Loading adapter...")
    adapter_start = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_name)
    adapter_load_time = time.time() - adapter_start
    total_memory = get_gpu_memory_mb()
    adapter_memory = total_memory - base_memory
    peak_load_memory = get_peak_memory_mb()
    
    print(f"   ✓ Adapter loaded in {adapter_load_time:.2f}s")
    print(f"   ✓ Adapter memory: {adapter_memory:.2f} MB")
    print(f"   ✓ Total memory: {total_memory:.2f} MB")
    print(f"   ✓ Peak load memory: {peak_load_memory:.2f} MB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warmup
    print(f"\n3. Running {NUM_WARMUP_RUNS} warmup iterations...")
    model.eval()
    for i in range(NUM_WARMUP_RUNS):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"   Warmup {i+1}/{NUM_WARMUP_RUNS} complete")
    
    # Benchmark inference
    num_test_runs = len(test_prompts)
    print(f"\n4. Running {num_test_runs} benchmark iterations...")
    inference_times = []
    peak_memories = []
    
    for i in range(NUM_BENCHMARK_RUNS):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
  
        reset_peak_memory()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start
        peak_mem = get_peak_memory_mb()
        
        inference_times.append(inference_time)
        peak_memories.append(peak_mem)
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/{NUM_BENCHMARK_RUNS} runs")
    
    results = {
        'base_load_time': base_load_time,
        'adapter_load_time': adapter_load_time,
        'total_load_time': base_load_time + adapter_load_time,
        'base_memory': base_memory,
        'adapter_memory': adapter_memory,
        'total_memory': total_memory,
        'peak_load_memory': peak_load_memory,
        'inference_times': inference_times,
        'peak_memories': peak_memories,
        'mean_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'median_inference_time': np.median(inference_times),
        'mean_peak_memory': np.mean(peak_memories),
        'max_peak_memory': np.max(peak_memories)
    }
    
    # Cleanup
    del model, tokenizer, base_model
    clear_memory()
    
    return results


def benchmark_merged_model(base_model_name, adapter_name, test_prompts):
    """Benchmark merged model"""
    print("\n" + "="*70)
    print("BENCHMARKING: MERGED MODEL")
    print("="*70)
    
    clear_memory()
    reset_peak_memory()
    
    # Load and merge
    print("\n1. Loading base model...")
    start_time = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("2. Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    
    print("3. Merging adapter into base model...")
    merge_start = time.time()
    model = model.merge_and_unload()
    merge_time = time.time() - merge_start
    
    total_load_time = time.time() - start_time
    total_memory = get_gpu_memory_mb()
    peak_load_memory = get_peak_memory_mb()
    
    print(f"   ✓ Model loaded and merged in {total_load_time:.2f}s")
    print(f"   ✓ Merge time: {merge_time:.2f}s")
    print(f"   ✓ Total memory: {total_memory:.2f} MB")
    print(f"   ✓ Peak load memory: {peak_load_memory:.2f} MB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warmup
    print(f"\n4. Running {NUM_WARMUP_RUNS} warmup iterations...")
    model.eval()
    for i in range(NUM_WARMUP_RUNS):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"   Warmup {i+1}/{NUM_WARMUP_RUNS} complete")
    
    # Benchmark inference
    print(f"\n5. Running {NUM_BENCHMARK_RUNS} benchmark iterations...")
    inference_times = []
    peak_memories = []
    
    for i in range(NUM_BENCHMARK_RUNS):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        reset_peak_memory()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start
        peak_mem = get_peak_memory_mb()
        
        inference_times.append(inference_time)
        peak_memories.append(peak_mem)
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i+1}/{NUM_BENCHMARK_RUNS} runs")
    
    results = {
        'load_time': total_load_time,
        'merge_time': merge_time,
        'total_memory': total_memory,
        'peak_load_memory': peak_load_memory,
        'inference_times': inference_times,
        'peak_memories': peak_memories,
        'mean_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'median_inference_time': np.median(inference_times),
        'mean_peak_memory': np.mean(peak_memories),
        'max_peak_memory': np.max(peak_memories)
    }
    
    # Cleanup
    del model, tokenizer
    clear_memory()
    
    return results


def save_results(adapter_results, merged_results):
    """Save results to CSV files"""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save detailed run-by-run results
    print(f"\n1. Saving detailed results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run', 
            'adapter_inference_time_s', 
            'adapter_peak_memory_mb',
            'merged_inference_time_s',
            'merged_peak_memory_mb'
        ])
        
        for i in range(len(adapter_results['inference_times'])):
            writer.writerow([
                i + 1,
                adapter_results['inference_times'][i],
                adapter_results['peak_memories'][i],
                merged_results['inference_times'][i],
                merged_results['peak_memories'][i]
            ])
    
    print("   ✓ Detailed results saved")
    
    # Save summary statistics
    print(f"\n2. Saving summary to {SUMMARY_CSV}...")
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Adapter (Separate)', 'Merged', 'Difference'])
        
        # Loading metrics
        writer.writerow(['=== LOADING METRICS ===', '', '', ''])
        writer.writerow([
            'Total Load Time (s)',
            f"{adapter_results['total_load_time']:.2f}",
            f"{merged_results['load_time']:.2f}",
            f"{adapter_results['total_load_time'] - merged_results['load_time']:.2f}"
        ])
        writer.writerow([
            'Base Load Time (s)',
            f"{adapter_results['base_load_time']:.2f}",
            'N/A',
            ''
        ])
        writer.writerow([
            'Adapter Load Time (s)',
            f"{adapter_results['adapter_load_time']:.2f}",
            'N/A',
            ''
        ])
        writer.writerow([
            'Merge Time (s)',
            'N/A',
            f"{merged_results['merge_time']:.2f}",
            ''
        ])
        writer.writerow([
            'Total Memory (MB)',
            f"{adapter_results['total_memory']:.2f}",
            f"{merged_results['total_memory']:.2f}",
            f"{adapter_results['total_memory'] - merged_results['total_memory']:.2f}"
        ])
        writer.writerow([
            'Base Memory (MB)',
            f"{adapter_results['base_memory']:.2f}",
            'N/A',
            ''
        ])
        writer.writerow([
            'Adapter Memory (MB)',
            f"{adapter_results['adapter_memory']:.2f}",
            'N/A (merged)',
            ''
        ])
        writer.writerow([
            'Peak Load Memory (MB)',
            f"{adapter_results['peak_load_memory']:.2f}",
            f"{merged_results['peak_load_memory']:.2f}",
            f"{adapter_results['peak_load_memory'] - merged_results['peak_load_memory']:.2f}"
        ])
        
        # Inference metrics
        writer.writerow(['', '', '', ''])
        writer.writerow(['=== INFERENCE METRICS ===', '', '', ''])
        writer.writerow([
            'Mean Inference Time (s)',
            f"{adapter_results['mean_inference_time']:.4f}",
            f"{merged_results['mean_inference_time']:.4f}",
            f"{adapter_results['mean_inference_time'] - merged_results['mean_inference_time']:.4f}"
        ])
        writer.writerow([
            'Std Inference Time (s)',
            f"{adapter_results['std_inference_time']:.4f}",
            f"{merged_results['std_inference_time']:.4f}",
            f"{adapter_results['std_inference_time'] - merged_results['std_inference_time']:.4f}"
        ])
        writer.writerow([
            'Median Inference Time (s)',
            f"{adapter_results['median_inference_time']:.4f}",
            f"{merged_results['median_inference_time']:.4f}",
            f"{adapter_results['median_inference_time'] - merged_results['median_inference_time']:.4f}"
        ])
        writer.writerow([
            'Min Inference Time (s)',
            f"{adapter_results['min_inference_time']:.4f}",
            f"{merged_results['min_inference_time']:.4f}",
            f"{adapter_results['min_inference_time'] - merged_results['min_inference_time']:.4f}"
        ])
        writer.writerow([
            'Max Inference Time (s)',
            f"{adapter_results['max_inference_time']:.4f}",
            f"{merged_results['max_inference_time']:.4f}",
            f"{adapter_results['max_inference_time'] - merged_results['max_inference_time']:.4f}"
        ])
        writer.writerow([
            'Mean Peak Memory (MB)',
            f"{adapter_results['mean_peak_memory']:.2f}",
            f"{merged_results['mean_peak_memory']:.2f}",
            f"{adapter_results['mean_peak_memory'] - merged_results['mean_peak_memory']:.2f}"
        ])
        writer.writerow([
            'Max Peak Memory (MB)',
            f"{adapter_results['max_peak_memory']:.2f}",
            f"{merged_results['max_peak_memory']:.2f}",
            f"{adapter_results['max_peak_memory'] - merged_results['max_peak_memory']:.2f}"
        ])
        
        # Performance comparison
        writer.writerow(['', '', '', ''])
        writer.writerow(['=== PERFORMANCE COMPARISON ===', '', '', ''])
        speedup = adapter_results['mean_inference_time'] / merged_results['mean_inference_time']
        writer.writerow([
            'Speed Ratio (Adapter/Merged)',
            f"{speedup:.2f}x",
            '',
            f"Merged is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        ])
        memory_ratio = adapter_results['total_memory'] / merged_results['total_memory']
        writer.writerow([
            'Memory Ratio (Adapter/Merged)',
            f"{memory_ratio:.2f}x",
            '',
            f"Adapter uses {memory_ratio:.2f}x {'more' if memory_ratio > 1 else 'less'} memory"
        ])
    
    print("   ✓ Summary saved")
    

def print_summary(adapter_results, merged_results):
    """Print summary to console"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print("\n📊 LOADING METRICS:")
    print(f"{'Metric':<40} {'Adapter':<15} {'Merged':<15}")
    print("-" * 70)
    print(f"{'Total Load Time (s)':<40} {adapter_results['total_load_time']:<15.2f} {merged_results['load_time']:<15.2f}")
    print(f"{'Total Memory (MB)':<40} {adapter_results['total_memory']:<15.2f} {merged_results['total_memory']:<15.2f}")
    print(f"{'Base Memory (MB)':<40} {adapter_results['base_memory']:<15.2f} {'N/A':<15}")
    print(f"{'Adapter Memory (MB)':<40} {adapter_results['adapter_memory']:<15.2f} {'N/A':<15}")
    print(f"{'Peak Load Memory (MB)':<40} {adapter_results['peak_load_memory']:<15.2f} {merged_results['peak_load_memory']:<15.2f}")
    
    print("\n⚡ INFERENCE METRICS:")
    print(f"{'Metric':<40} {'Adapter':<15} {'Merged':<15}")
    print("-" * 70)
    print(f"{'Mean Inference Time (s)':<40} {adapter_results['mean_inference_time']:<15.4f} {merged_results['mean_inference_time']:<15.4f}")
    print(f"{'Std Inference Time (s)':<40} {adapter_results['std_inference_time']:<15.4f} {merged_results['std_inference_time']:<15.4f}")
    print(f"{'Median Inference Time (s)':<40} {adapter_results['median_inference_time']:<15.4f} {merged_results['median_inference_time']:<15.4f}")
    print(f"{'Min/Max Time (s)':<40} {adapter_results['min_inference_time']:.4f}/{adapter_results['max_inference_time']:.4f} {merged_results['min_inference_time']:.4f}/{merged_results['max_inference_time']:.4f}")
    print(f"{'Mean Peak Memory (MB)':<40} {adapter_results['mean_peak_memory']:<15.2f} {merged_results['mean_peak_memory']:<15.2f}")
    
    speedup = adapter_results['mean_inference_time'] / merged_results['mean_inference_time']
    memory_ratio = adapter_results['total_memory'] / merged_results['total_memory']
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"  • Merged model is {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'} than adapter")
    print(f"  • Adapter uses {memory_ratio:.2f}x {'MORE' if memory_ratio > 1 else 'LESS'} memory than merged")
    print(f"  • Memory overhead of adapter: {adapter_results['adapter_memory']:.2f} MB")




# ============================================================================
# DATASET LOADER
# ============================================================================

def load_oasst1_validation(num_samples: Optional[int] = None) -> List[str]:
    """
    Load OASST1 (OpenAssistant) validation prompts
    
    Args:
        num_samples: Number of samples to return. If None, return all available.
    
    Returns:
        List of prompt strings
    """
    print("Loading OASST1 validation dataset...")
    try:
        # load from Hugging Face datasets
        dataset = load_dataset("OpenAssistant/oasst1", split="validation")
        
        # Extract user prompts (initial messages in conversations)
        prompts = []
        seen = set()  # Avoid duplicates
        
        for item in dataset:
            if item['role'] == 'prompter':  # User messages
                text = item['text'].strip()
                if text and text not in seen:
                    prompts.append(text)
                    seen.add(text)
        
        oasst1_data = prompts
        print(f"✓ Loaded {len(prompts)} prompts from OASST1 validation set")
        
    except ImportError:
        print("⚠️  'datasets' library not installed. Using fallback prompts.")
        print("   Install with: pip install datasets")
        return []
    except Exception as e:
        print(f"⚠️  Could not load OASST1 from Hugging Face: {e}")
        print("   Using fallback prompts...")
        return []
    
    if num_samples is None:
        return oasst1_data
    else:
        return random.sample(oasst1_data, min(num_samples, len(oasst1_data)))
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("LoRA PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"\nBase Model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_MODEL}")
    print(f"Evaluation Dataset: {DATASET_NAME}")
    print(f"Number of Test Samples: {NUM_TEST_SAMPLES}")
    print(f"Benchmark Runs: {NUM_BENCHMARK_RUNS} (+ {NUM_WARMUP_RUNS} warmup)")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Running on CPU will be very slow!")
        print("Consider running on a machine with GPU support.\n")
    
    # Load evaluation dataset
    print(f"\n{'='*70}")
    print("LOADING EVALUATION DATASET")
    print(f"{'='*70}")
    
    try:
        test_prompts = load_oasst1_validation( num_samples=NUM_TEST_SAMPLES )
        print(f"\n✓ Loaded {len(test_prompts)} test prompts from {DATASET_NAME} dataset")
        print(f"\nSample prompts:")
        for i, prompt in enumerate(test_prompts[:3], 1):
            preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
            print(f"  {i}. {preview}")
    except Exception as e:
        print(f"\n⚠️  Could not load dataset: {e}")
        print("Using fallback test prompts...")
        test_prompts = [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "List three benefits of exercise.",
            "Write a haiku about coding.",
        ] * (NUM_TEST_SAMPLES // 4 + 1)
        test_prompts = test_prompts[:NUM_TEST_SAMPLES]
    
    # Run benchmarks using test_prompts
    adapter_results = benchmark_adapter_separate(BASE_MODEL, ADAPTER_MODEL, test_prompts)
    merged_results = benchmark_merged_model(BASE_MODEL, ADAPTER_MODEL, test_prompts)
    
    # Print and save results
    print_summary(adapter_results, merged_results)
    save_results(adapter_results, merged_results)
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETED!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  • {OUTPUT_CSV}")
    print(f"  • {SUMMARY_CSV}")
    print("\n")


if __name__ == "__main__":
    main()