import os
import sys
import json
import pickle
import shutil
import subprocess
import re
import traceback
from datetime import datetime
from threading import Lock

PYTHON_EXECUTABLE = sys.executable or "python3"

BENCHMARK_DISPLAY_NAMES = {
    "sudoku": "Sudoku",
    "sudoku_gt": "Sudoku-GT",
    "jsudoku": "JSudoku",
    "latin_square": "Latin Square",
    "graph_coloring_register": "Graph Coloring",
    "examtt_v1": "ExamTT-V1",
    "examtt_v2": "ExamTT-V2",
    "nurse": "Nurse"
}

TARGET_CONSTRAINTS = {
    "sudoku": 27,
    "sudoku_gt": 27,
    "jsudoku": 27,
    "latin_square": 18,
    "graph_coloring_register": 5,
    "examtt_v1": 7,
    "examtt_v2": 9,
    "nurse": 13
}

CONFIG_PATTERN = re.compile(r"^(?P<benchmark>.+)_sol(?P<solutions>\d+)_overfitted(?P<overfitted>\d+)$")


def run_command(cmd, description):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print("Command: " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print("\n[ERROR] Command failed with return code " + str(result.returncode))
        return False, None
    return True, result.stdout


def run_phase2(experiment, phase1_pickle, *, approach="cop", max_queries=5000, timeout=1200, config_tag=None):
    script_map = {
        "cop": "main_alldiff_cop.py",
        "lion": "main_alldiff_lion19.py"
    }
    filename_map = {
        "cop": f"{experiment}_phase2.pkl",
        "lion": f"{experiment}_lion19_phase2.pkl"
    }
    script = script_map.get(approach.lower(), "main_alldiff_cop.py")
    cmd = [
        PYTHON_EXECUTABLE,
        script,
        "--experiment",
        experiment,
        "--phase1_pickle",
        phase1_pickle,
        "--max_queries",
        str(max_queries),
        "--timeout",
        str(timeout)
    ]
    success, _ = run_command(cmd, f"Phase 2 ({approach.upper()}): {experiment}")
    if not success:
        return False, None
    default_output = "phase2_output"
    source_pickle = os.path.join(default_output, filename_map[approach.lower()])
    base_output_dir = f"phase2_output_{approach.lower()}"
    if config_tag:
        target_dir = os.path.join(base_output_dir, experiment)
        file_suffix_map = {
            "cop": "phase2.pkl",
            "lion": "lion19_phase2.pkl"
        }
        dest_filename = f"{experiment}_{config_tag}_{file_suffix_map[approach.lower()]}"
    else:
        target_dir = base_output_dir
        dest_filename = filename_map[approach.lower()]
    os.makedirs(target_dir, exist_ok=True)
    target_pickle = os.path.join(target_dir, dest_filename)
    if os.path.exists(target_pickle):
        os.remove(target_pickle)
    if os.path.exists(source_pickle):
        shutil.move(source_pickle, target_pickle)
        print("\n[INFO] Moved " + source_pickle + " to " + target_pickle)
        return True, target_pickle
    print("\n[WARNING] Expected output file not found: " + source_pickle)
    return True, source_pickle


def run_phase3(experiment, phase2_pickle, *, approach="cop", config_tag=None):
    cmd = [
        PYTHON_EXECUTABLE,
        "run_phase3_simple.py",
        "--experiment",
        experiment,
        "--phase2_pickle",
        phase2_pickle
    ]
    try:
        success, _ = run_command(cmd, f"Phase 3 ({approach.upper()}): {experiment}")
    except Exception as exc:
        print("\n[ERROR] Phase 3 command execution failed: " + str(exc))
        return False
    if not success:
        return False
    default_output = "phase3_output"
    base_output_dir = f"phase3_output_{approach.lower()}"
    if config_tag:
        target_dir = os.path.join(base_output_dir, experiment)
        results_json_name = f"{experiment}_{config_tag}_phase3_results.json"
        final_model_name = f"{experiment}_{config_tag}_final_model.pkl"
    else:
        target_dir = base_output_dir
        results_json_name = f"{experiment}_phase3_results.json"
        final_model_name = f"{experiment}_final_model.pkl"
    os.makedirs(target_dir, exist_ok=True)
    file_mapping = {
        f"{experiment}_phase3_results.json": results_json_name,
        f"{experiment}_final_model.pkl": final_model_name
    }
    for source_name, dest_name in file_mapping.items():
        source_path = os.path.join(default_output, source_name)
        dest_path = os.path.join(target_dir, dest_name)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print("[INFO] Moved " + source_path + " to " + dest_path)
        else:
            print("[WARNING] Expected file not found: " + source_path)
    return True


def load_phase1_pickle(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        print("[ERROR] Failed to load Phase 1 pickle: " + str(exc))
        return None


def load_phase3_results(benchmark_name, approach="cop", config_tag=None):
    output_dir = f"phase3_output_{approach.lower()}"
    if config_tag:
        json_path = os.path.join(output_dir, benchmark_name, f"{benchmark_name}_{config_tag}_phase3_results.json")
    else:
        json_path = os.path.join(output_dir, f"{benchmark_name}_phase3_results.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as handle:
            return json.load(handle)
    except Exception as exc:
        print("[ERROR] Failed to load Phase 3 results: " + str(exc))
        return None


def load_phase2_pickle(path):
    if not path or not os.path.exists(path):
        print("[WARNING] Phase 2 pickle not found: " + str(path))
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        print("[ERROR] Failed to load Phase 2 pickle (" + str(path) + "): " + str(exc))
        return None


def parse_int(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            if "." in value:
                return int(float(value))
            return int(value)
        except ValueError:
            return 0
    return 0


def parse_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def extract_metrics(benchmark_name, num_solutions, phase1_pickle_path, *, approach="cop", config_tag=None, phase2_pickle_path=None):
    phase1_data = load_phase1_pickle(phase1_pickle_path)
    if phase1_data is None:
        return None
    phase2_data = load_phase2_pickle(phase2_pickle_path) if phase2_pickle_path else None
    if phase2_data is None:
        print("[WARNING] Skipping metrics extraction due to missing Phase 2 data: " + str(phase2_pickle_path))
        return None
    phase2_stats = phase2_data.get("phase2_stats", {})
    if not isinstance(phase2_stats, dict):
        phase2_stats = {}
    validated_globals = phase2_stats.get("validated", len(phase2_data.get("C_validated", [])))
    cp_implication = phase2_stats.get("cp_implication", {}) if isinstance(phase2_stats, dict) else {}
    implied_constraints = None
    if isinstance(cp_implication, dict):
        implied_constraints = cp_implication.get("implied_count")
    phase3_results = load_phase3_results(benchmark_name, approach=approach, config_tag=config_tag)
    phase3_available = phase3_results is not None
    if not phase3_available:
        print("[WARNING] Phase 3 results missing for " + benchmark_name + " (" + approach + ", " + str(config_tag) + "). Using Phase 2-only metrics.")
    metrics = {}
    metrics["Prob."] = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name)
    metrics["Approach"] = approach.upper()
    metrics["Sols"] = num_solutions
    start_c = len(phase1_data.get("CG", []))
    metrics["StartC"] = start_c
    metrics["Implied"] = implied_constraints if implied_constraints is not None else "N/A"
    metrics["CT"] = TARGET_CONSTRAINTS.get(benchmark_name, "N/A")
    if phase3_available:
        metrics["Bias"] = phase3_results.get("phase1", {}).get("B_fixed_size", len(phase2_data.get("B_fixed", [])))
    else:
        metrics["Bias"] = len(phase2_data.get("B_fixed", []))
    viol_queries = 0
    if phase3_available:
        viol_queries = phase3_results.get("phase2", {}).get("queries", phase2_stats.get("queries", 0))
    else:
        viol_queries = phase2_stats.get("queries", 0)
    metrics["ViolQ"] = parse_int(viol_queries)
    metrics["InvC"] = start_c - parse_int(phase3_results.get("phase2", {}).get("validated_globals", validated_globals) if phase3_available else validated_globals)
    metrics["MQuQ"] = parse_int(phase3_results.get("phase3", {}).get("queries", 0) if phase3_available else 0)
    metrics["TQ"] = metrics["ViolQ"] + metrics["MQuQ"]
    p1_time = 0.0
    metrics["P1T(s)"] = round(p1_time, 2)
    phase2_time_value = parse_float(phase3_results.get("phase2", {}).get("time", phase2_stats.get("time", 0)) if phase3_available else phase2_stats.get("time", 0))
    metrics["VT(s)"] = round(phase2_time_value, 2)
    phase3_time_value = parse_float(phase3_results.get("phase3", {}).get("time", 0) if phase3_available else 0)
    metrics["MQuT(s)"] = round(phase3_time_value, 2)
    metrics["TT(s)"] = round(metrics["P1T(s)"] + metrics["VT(s)"] + metrics["MQuT(s)"], 2)
    metrics["ALQ"] = "N/A"
    metrics["PAQ"] = "N/A"
    metrics["ALT(s)"] = "N/A"
    metrics["PAT(s)"] = "N/A"
    if phase3_available:
        eval_data = phase3_results.get("evaluation", {})
        constraint_level = eval_data.get("constraint_level", {})
        solution_level = eval_data.get("solution_level", {})
        metrics["precision"] = round(parse_float(constraint_level.get("precision", 0)) * 100, 2)
        metrics["recall"] = round(parse_float(constraint_level.get("recall", 0)) * 100, 2)
        metrics["s_precision"] = round(parse_float(solution_level.get("s_precision", 0)) * 100, 2)
        metrics["s_recall"] = round(parse_float(solution_level.get("s_recall", 0)) * 100, 2)
    else:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["s_precision"] = 0.0
        metrics["s_recall"] = 0.0
    return metrics


def append_metrics_to_csv(metrics_list, csv_path, metrics_lock):
    if not metrics_list:
        return
    with metrics_lock:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a") as handle:
            if not file_exists:
                handle.write("Prob.,Approach,Sols,StartC,Implied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),precision,recall,s_precision,s_recall\n")
            for m in metrics_list:
                handle.write(
                    f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['Implied']},{m['InvC']},{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},{m['ALQ']},{m['PAQ']},"
                    f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},{m['ALT(s)']},{m['PAT(s)']},{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n"
                )


def update_progress_file(completed, total, progress_path, metrics_lock):
    with metrics_lock:
        with open(progress_path, "w") as handle:
            handle.write("Experiment Progress\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"Completed: {completed}/{total} tasks\n")
            if total > 0:
                handle.write(f"Progress: {100 * completed / total:.1f}%\n")
            else:
                handle.write("Progress: 0.0%\n")
            handle.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write("=" * 60 + "\n")


def discover_cached_configs(base_dir):
    configs = []
    if not os.path.isdir(base_dir):
        return configs
    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        match = CONFIG_PATTERN.match(entry)
        if not match:
            continue
        benchmark = match.group("benchmark")
        num_solutions = int(match.group("solutions"))
        num_overfitted = int(match.group("overfitted"))
        pickle_path = os.path.join(entry_path, f"{benchmark}_phase1.pkl")
        if os.path.exists(pickle_path):
            configs.append(
                {
                    "benchmark": benchmark,
                    "num_solutions": num_solutions,
                    "num_overfitted": num_overfitted,
                    "phase1_pickle": pickle_path,
                    "config_tag": f"sol{num_solutions}_of{num_overfitted}"
                }
            )
    configs.sort(key=lambda item: (item["benchmark"], item["num_solutions"], item["num_overfitted"]))
    return configs


def group_configs_for_display(configs):
    grouped = {}
    for cfg in configs:
        key = cfg["benchmark"]
        grouped.setdefault(key, []).append((cfg["num_solutions"], cfg["num_overfitted"]))
    for key in grouped:
        grouped[key].sort()
    return grouped


def get_phase2_pickle_path(benchmark, approach, config_tag):
    """Construct the expected Phase 2 pickle path for a given config."""
    file_suffix_map = {
        "cop": "phase2.pkl",
        "lion": "lion19_phase2.pkl"
    }
    base_output_dir = f"phase2_output_{approach.lower()}"
    if config_tag:
        target_dir = os.path.join(base_output_dir, benchmark)
        dest_filename = f"{benchmark}_{config_tag}_{file_suffix_map[approach.lower()]}"
    else:
        filename_map = {
            "cop": f"{benchmark}_phase2.pkl",
            "lion": f"{benchmark}_lion19_phase2.pkl"
        }
        target_dir = base_output_dir
        dest_filename = filename_map[approach.lower()]
    return os.path.join(target_dir, dest_filename)


def process_cached_config(config, approaches):
    benchmark = config["benchmark"]
    num_solutions = config["num_solutions"]
    config_tag = config["config_tag"]
    phase1_pickle = config["phase1_pickle"]
    metrics = []
    for approach in approaches:
        print("\n" + "-" * 60)
        print(f"[TASK] Running {approach.upper()} approach for {benchmark} ({config_tag})")
        print("-" * 60 + "\n")
        
        # Check if Phase 2 pickle already exists
        expected_phase2_pickle = get_phase2_pickle_path(benchmark, approach, config_tag)
        if os.path.exists(expected_phase2_pickle):
            print(f"[INFO] Phase 2 pickle already exists: {expected_phase2_pickle}")
            print("[INFO] Skipping Phase 2, running only Phase 3...")
            phase2_pickle = expected_phase2_pickle
            phase2_success = True
        else:
            phase2_success, phase2_pickle = run_phase2(
                benchmark,
                phase1_pickle,
                approach=approach,
                config_tag=config_tag
            )
        if not phase2_success:
            print(f"\n[TASK ERROR] Phase 2 ({approach.upper()}) failed for {benchmark}")
            continue
        phase3_success = False
        try:
            phase3_success = run_phase3(
                benchmark,
                phase2_pickle,
                approach=approach,
                config_tag=config_tag
            )
            if not phase3_success:
                print(f"\n[TASK WARNING] Phase 3 ({approach.upper()}) failed for {benchmark}; proceeding with Phase 2 metrics only.")
        except Exception as exc:
            print(f"\n[TASK EXCEPTION] Phase 3 ({approach.upper()}) crashed for {benchmark}: {exc}")
            traceback.print_exc()
            phase3_success = False
        try:
            metrics_row = extract_metrics(
                benchmark,
                num_solutions,
                phase1_pickle,
                approach=approach,
                config_tag=config_tag,
                phase2_pickle_path=phase2_pickle
            )
            if metrics_row:
                metrics.append(metrics_row)
                if phase3_success:
                    print(f"\n[TASK SUCCESS] Extracted metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
                else:
                    print(f"\n[TASK PARTIAL] Recorded Phase 2 metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()} (Phase 3 unavailable)")
            else:
                print(f"\n[TASK WARNING] Could not extract metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}")
        except Exception as exc:
            print(f"\n[TASK ERROR] Failed to extract metrics for {benchmark} | Solutions={num_solutions} | {approach.upper()}: {exc}")
            traceback.print_exc()
    return metrics


def main():
    print("\n" + "=" * 80)
    print("SOLUTION VARIANCE EXPERIMENTS - CACHED PHASE 2/3")
    print("=" * 80)
    print("Starting at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80 + "\n")
    output_dir = "solution_variance_output"
    configs = discover_cached_configs(output_dir)
    if not configs:
        print("[ERROR] No cached Phase 1 configurations found in " + output_dir)
        return
    grouped = group_configs_for_display(configs)
    print("Cached Phase 1 configurations:")
    for benchmark in sorted(grouped.keys()):
        display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        mapped = {sol: over for sol, over in grouped[benchmark]}
        print("  - " + display_name + ": " + str(mapped))
    print("=" * 80 + "\n")
    approaches = ["cop"]
    all_metrics = []
    metrics_lock = Lock()
    intermediate_csv_path = os.path.join(output_dir, "intermediate_results.csv")
    progress_path = os.path.join(output_dir, "progress.txt")
    with open(intermediate_csv_path, "w") as handle:
        handle.write("Prob.,Approach,Sols,StartC,Implied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),precision,recall,s_precision,s_recall\n")
    print("[INFO] Intermediate results will be saved to: " + intermediate_csv_path)
    print("[INFO] Progress tracking file: " + progress_path + "\n")
    total_tasks = len(configs)
    print("Total tasks to process: " + str(total_tasks))
    print("Each task reuses cached Phase 1 data and runs COP (Phase 2+3) and LION (Phase 2+3) if listed")
    print("Processing sequentially...\n")
    update_progress_file(0, total_tasks, progress_path, metrics_lock)
    for index, config in enumerate(configs, start=1):
        benchmark = config["benchmark"]
        num_solutions = config["num_solutions"]
        config_tag = config["config_tag"]
        print("\n" + "=" * 80)
        print(f"[TASK {index}/{total_tasks}] Processing: {benchmark} | Solutions={num_solutions} | {config_tag}")
        print("=" * 80 + "\n")
        try:
            config_metrics = process_cached_config(config, approaches)
            with metrics_lock:
                all_metrics.extend(config_metrics)
            append_metrics_to_csv(config_metrics, intermediate_csv_path, metrics_lock)
            update_progress_file(index, total_tasks, progress_path, metrics_lock)
            print("\n" + "=" * 80)
            print(f"[PROGRESS] Completed {index}/{total_tasks}: {benchmark} with {num_solutions} solutions ({config_tag})")
            print(f"[PROGRESS] Collected {len(config_metrics)} metric sets from this task")
            print(f"[PROGRESS] Results appended to: {intermediate_csv_path}")
            print("=" * 80 + "\n")
        except Exception as exc:
            print(f"\n[ERROR] Task failed for {benchmark} with {num_solutions} solutions ({config_tag}): {exc}")
            traceback.print_exc()
            update_progress_file(index, total_tasks, progress_path, metrics_lock)
    print("\n" + "=" * 80)
    print("ALL TASKS COMPLETED")
    print("=" * 80)
    print("Total metrics collected: " + str(len(all_metrics)))
    print("=" * 80 + "\n")
    print("\n\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80 + "\n")
    if not all_metrics:
        print("[WARNING] No metrics collected. Check for errors in pipeline execution.")
    print("\n[INFO] Intermediate results during execution: " + intermediate_csv_path)
    print("[INFO] Generating final summary reports...\n")
    report_path = os.path.join(output_dir, "variance_results.txt")
    with open(report_path, "w") as handle:
        handle.write("Solution Variance Experiment Results (Cached Phase 2/3)\n")
        handle.write("=" * 140 + "\n\n")
        handle.write(f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'StartC':<8} {'Implied':<9} {'InvC':<6} {'CT':<5} {'Bias':<6} {'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'ALQ':<5} {'PAQ':<5} {'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8} {'ALT(s)':<7} {'PAT(s)':<7}\n")
        for m in all_metrics:
            handle.write(
                f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['StartC']:<8} {str(m['Implied']):<9} {m['InvC']:<6} {str(m['CT']):<5} {m['Bias']:<6} "
                f"{m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} {m['ALQ']:<5} {m['PAQ']:<5} {m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8} {m['ALT(s)']:<7} {m['PAT(s)']:<7}\n"
            )
        handle.write("\n" + "=" * 140 + "\n")
        handle.write("Legend:\n")
        handle.write("  Approach: COP or LION methodology\n")
        handle.write("  Sols: Number of given solutions (positive examples)\n")
        handle.write("  StartC: Number of candidate constraints from passive learning\n")
        handle.write("  InvC: Number of constraints invalidated by refinement\n")
        handle.write("  CT: Number of AllDifferent constraints in target model\n")
        handle.write("  Bias: Size of generated bias\n")
        handle.write("  ViolQ: Violation queries (Phase 2)\n")
        handle.write("  MQuQ: Active learning queries (Phase 3)\n")
        handle.write("  TQ: Total queries (ViolQ + MQuQ)\n")
        handle.write("  ALQ: Queries for purely Active Learning baseline\n")
        handle.write("  PAQ: Queries for Passive+Active baseline\n")
        handle.write("  P1T(s): Phase 1 passive learning time\n")
        handle.write("  VT(s): Duration of violation phase\n")
        handle.write("  MQuT(s): Duration of active learning phase\n")
        handle.write("  TT(s): Overall runtime (P1T + VT + MQuT)\n")
        handle.write("  ALT(s): Runtime for Active Learning baseline\n")
        handle.write("  PAT(s): Runtime for Passive+Active baseline\n")
    print("[SAVED] Formatted results saved to: " + report_path)
    print("\n" + f"{'Prob.':<15} {'Approach':<9} {'Sols':<6} {'StartC':<8} {'Implied':<9} {'InvC':<6} {'CT':<5} {'Bias':<6} {'ViolQ':<7} {'MQuQ':<7} {'TQ':<6} {'P1T(s)':<8} {'VT(s)':<8} {'MQuT(s)':<9} {'TT(s)':<8}")
    print("=" * 120)
    for m in all_metrics:
        print(
            f"{m['Prob.']:<15} {m['Approach']:<9} {m['Sols']:<6} {m['StartC']:<8} {str(m['Implied']):<9} {m['InvC']:<6} "
            f"{str(m['CT']):<5} {m['Bias']:<6} {m['ViolQ']:<7} {m['MQuQ']:<7} {m['TQ']:<6} {m['P1T(s)']:<8} {m['VT(s)']:<8} {m['MQuT(s)']:<9} {m['TT(s)']:<8}"
        )
    csv_path = os.path.join(output_dir, "variance_results.csv")
    with open(csv_path, "w") as handle:
        handle.write("Prob.,Approach,Sols,StartC,Implied,InvC,CT,Bias,ViolQ,MQuQ,TQ,ALQ,PAQ,P1T(s),VT(s),MQuT(s),TT(s),ALT(s),PAT(s),precision,recall,s_precision,s_recall\n")
        for m in all_metrics:
            handle.write(
                f"{m['Prob.']},{m['Approach']},{m['Sols']},{m['StartC']},{m['Implied']},{m['InvC']},{m['CT']},{m['Bias']},{m['ViolQ']},{m['MQuQ']},{m['TQ']},{m['ALQ']},{m['PAQ']},"
                f"{m['P1T(s)']},{m['VT(s)']},{m['MQuT(s)']},{m['TT(s)']},{m['ALT(s)']},{m['PAT(s)']},{m['precision']},{m['recall']},{m['s_precision']},{m['s_recall']}\n"
            )
    print("[SAVED] CSV results saved to: " + csv_path)
    comparison_path = os.path.join(output_dir, "cop_vs_lion_comparison.txt")
    with open(comparison_path, "w") as handle:
        handle.write("COP vs LION Comparison Summary\n")
        handle.write("=" * 120 + "\n\n")
        grouped_metrics = {}
        for m in all_metrics:
            key = (m["Prob."], m["Sols"])
            if key not in grouped_metrics:
                grouped_metrics[key] = {}
            grouped_metrics[key][m["Approach"]] = m
        for (benchmark_name, sols), approaches_map in sorted(grouped_metrics.items()):
            handle.write("\n" + f"{benchmark_name} - {sols} solutions:\n")
            handle.write("-" * 100 + "\n")
            cop = approaches_map.get("COP", {})
            lion = approaches_map.get("LION", {})
            if cop and lion:
                handle.write(f"{'Metric':<20} {'COP':<15} {'LION':<15} {'Difference':<15}\n")
                handle.write("-" * 100 + "\n")
                handle.write(f"{'ViolQ':<20} {cop.get('ViolQ', 0):<15} {lion.get('ViolQ', 0):<15} {lion.get('ViolQ', 0) - cop.get('ViolQ', 0):<15}\n")
                handle.write(f"{'MQuQ':<20} {cop.get('MQuQ', 0):<15} {lion.get('MQuQ', 0):<15} {lion.get('MQuQ', 0) - cop.get('MQuQ', 0):<15}\n")
                handle.write(f"{'Total Queries':<20} {cop.get('TQ', 0):<15} {lion.get('TQ', 0):<15} {lion.get('TQ', 0) - cop.get('TQ', 0):<15}\n")
                handle.write(f"{'VT(s)':<20} {cop.get('VT(s)', 0):<15} {lion.get('VT(s)', 0):<15} {lion.get('VT(s)', 0) - cop.get('VT(s)', 0):<15.2f}\n")
                handle.write(f"{'MQuT(s)':<20} {cop.get('MQuT(s)', 0):<15} {lion.get('MQuT(s)', 0):<15} {lion.get('MQuT(s)', 0) - cop.get('MQuT(s)', 0):<15.2f}\n")
                handle.write(f"{'Total Time (s)':<20} {cop.get('TT(s)', 0):<15} {lion.get('TT(s)', 0):<15} {lion.get('TT(s)', 0) - cop.get('TT(s)', 0):<15.2f}\n")
                handle.write(f"{'Precision (%)':<20} {cop.get('precision', 0):<15} {lion.get('precision', 0):<15} {lion.get('precision', 0) - cop.get('precision', 0):<15.2f}\n")
                handle.write(f"{'Recall (%)':<20} {cop.get('recall', 0):<15} {lion.get('recall', 0):<15} {lion.get('recall', 0) - cop.get('recall', 0):<15.2f}\n")
                cop_implied = cop.get("Implied", "N/A")
                lion_implied = lion.get("Implied", "N/A")
                if isinstance(cop_implied, (int, float)) and isinstance(lion_implied, (int, float)):
                    implied_diff = lion_implied - cop_implied
                    diff_str = f"{implied_diff:<15}"
                else:
                    diff_str = f"{'N/A':<15}"
                handle.write(f"{'Implied':<20} {str(cop_implied):<15} {str(lion_implied):<15} {diff_str}\n")
                handle.write("\nWinner: ")
                if cop.get("TQ", 0) < lion.get("TQ", 0):
                    handle.write("COP (fewer queries)\n")
                elif lion.get("TQ", 0) < cop.get("TQ", 0):
                    handle.write("LION (fewer queries)\n")
                else:
                    handle.write("TIE (same queries)\n")
            else:
                if cop:
                    handle.write("  COP: Available\n")
                if lion:
                    handle.write("  LION: Available\n")
                if not cop:
                    handle.write("  COP: Missing\n")
                if not lion:
                    handle.write("  LION: Missing\n")
    print("[SAVED] COP vs LION comparison saved to: " + comparison_path)
    json_path = os.path.join(output_dir, "variance_experiment_detailed.json")
    summary_overfitted_mapping = {}
    for benchmark in grouped:
        summary_overfitted_mapping[benchmark] = {}
        for sol, over in grouped[benchmark]:
            summary_overfitted_mapping[benchmark][str(sol)] = over
    summary = {
        "timestamp": datetime.now().isoformat(),
        "metrics": all_metrics,
        "total_benchmarks": len(grouped),
        "solution_configurations": {k: [sol for sol, _ in v] for k, v in grouped.items()},
        "overfitted_constraints_mapping": summary_overfitted_mapping,
        "parallel_execution": {
            "max_workers": 1,
            "total_tasks": len(configs)
        }
    }
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[SAVED] Detailed JSON saved to: " + json_path)
    total_expected_configs = len(configs) * len(approaches)
    successful_configs = len(all_metrics)
    cop_results = [m for m in all_metrics if m["Approach"] == "COP"]
    lion_results = [m for m in all_metrics if m["Approach"] == "LION"]
    print("\n" + "=" * 80)
    print("FINAL STATISTICS (CACHED EXECUTION)")
    print("=" * 80)
    print("Total configurations expected: " + str(total_expected_configs))
    print(f"Successful completions: {successful_configs}/{total_expected_configs}")
    print("  - COP approach: " + str(len(cop_results)) + " successful")
    print("  - LION approach: " + str(len(lion_results)) + " successful")
    if total_expected_configs > 0:
        print(f"Success rate: {100 * successful_configs / total_expected_configs:.1f}%")
    print("\n" + "=" * 80)
    print("OUTPUT FILES GENERATED")
    print("=" * 80)
    print("Real-time monitoring (updated during execution):")
    print("  - " + intermediate_csv_path)
    print("  - " + progress_path)
    print("\nFinal summary reports:")
    print("  - " + report_path)
    print("  - " + csv_path)
    print("  - " + comparison_path)
    print("  - " + json_path)
    print("\nCompleted at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

