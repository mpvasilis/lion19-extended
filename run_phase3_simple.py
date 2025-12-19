#!/usr/bin/env python3
"""
Simplified Phase 3: GrowAcq Active Learning

This script runs GrowAcq using:
- Bias from Phase 2 (B_fixed)
- Decomposed learned AllDifferent constraints as initial CL

No validation checks, no debug output - just straightforward GrowAcq learning.
"""

import os
import sys
import pickle
import time
import json
import argparse
from datetime import datetime

from cpmpy import Model, cpm_array
from cpmpy.transformations.get_variables import get_variables
from pycona import GrowAcq, MQuAcq2, ProblemInstance, ConstraintOracle
from pycona.ca_environment import ActiveCAEnv
from pycona.query_generation import PQGen
from pycona.find_constraint.findc import FindC

# Benchmark constructors (global version for oracle)
from benchmarks_global import (
    construct_sudoku, construct_jsudoku, construct_latin_square,
    construct_graph_coloring_register, construct_graph_coloring_scheduling,
    construct_examtt_simple, construct_sudoku_4x4_gt
)
from benchmarks_global.sudoku_greater_than import construct_sudoku_greater_than


def load_phase2_pickle(pickle_path):
    """Load phase 2 pickle and extract key data."""
    if not os.path.exists(pickle_path):
        print(f"[ERROR] Phase 2 pickle not found: {pickle_path}")
        sys.exit(1)
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract validated constraints
    C_validated = data.get('C_validated', [])
    
    # Extract phase1 data (contains B_fixed)
    phase1_data = data.get('phase1_data', {})
    B_fixed = phase1_data.get('B_fixed', []) if phase1_data else []
    
    # Extract phase2 stats
    phase2_stats = data.get('phase2_stats', {'queries': 0, 'time': 0})
    
    print(f"Loaded Phase 2 pickle: {pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")
    print(f"  - Bias (B_fixed): {len(B_fixed)}")
    print(f"  - Phase 2 queries: {phase2_stats.get('queries', 0)}")
    
    return C_validated, B_fixed, phase2_stats


def construct_benchmark(experiment_name):
    """Construct the benchmark instance and oracle."""
    exp_lower = experiment_name.lower()
    
    if 'graph_coloring_register' in exp_lower or exp_lower == 'register':
        result = construct_graph_coloring_register()
    elif 'graph_coloring_scheduling' in exp_lower or exp_lower == 'scheduling':
        result = construct_graph_coloring_scheduling()
    elif 'latin_square' in exp_lower or 'latin' in exp_lower:
        result = construct_latin_square(n=9)
    elif 'jsudoku' in exp_lower:
        result = construct_jsudoku(grid_size=9)
    elif 'sudoku_4x4_gt' in exp_lower:
        result = construct_sudoku_4x4_gt(2, 2, 4)
    elif 'sudoku_gt' in exp_lower or 'sudoku_greater' in exp_lower:
        result = construct_sudoku_greater_than(3, 3, 9)
    elif 'sudoku' in exp_lower:
        result = construct_sudoku(3, 3, 9)
    elif 'examtt' in exp_lower:
        result = construct_examtt_simple(
            nsemesters=30, courses_per_semester=20, 
            slots_per_day=18, days_for_exams=35
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Handle different return formats (some return 2, some return 3 values)
    if len(result) == 3:
        instance, oracle, _ = result
    else:
        instance, oracle = result
    
    return instance, oracle


def decompose_alldifferent(constraints):
    """
    Decompose AllDifferent constraints to binary != constraints.
    Keep other constraints as-is.
    """
    binary_constraints = []
    
    for c in constraints:
        c_str = str(c)
        
        # Check if it's an AllDifferent constraint
        if hasattr(c, 'name') and c.name == "alldifferent":
            # Skip if contains arithmetic operations
            if any(op in c_str for op in ['//', '/', '*', '+', '%']):
                continue
            
            # Decompose to binary !=
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                binary_constraints.extend(decomposed[0])
        else:
            # Keep non-AllDifferent constraints as-is
            binary_constraints.append(c)
    
    # Remove duplicates
    unique = {}
    for c in binary_constraints:
        unique[str(c)] = c
    
    return list(unique.values())


def create_decomposed_oracle(oracle):
    """Create an oracle with decomposed binary constraints."""
    decomposed_constraints = []
    non_global = []
    
    for c in oracle.constraints:
        if hasattr(c, 'name') and c.name == 'alldifferent':
            decomposed = c.decompose()
            if decomposed and len(decomposed) > 0:
                decomposed_constraints.extend(decomposed[0])
        else:
            non_global.append(c)
    
    # Deduplicate
    unique_binary = list({str(c): c for c in decomposed_constraints}.values())
    all_constraints = unique_binary + non_global
    
    return ConstraintOracle(all_constraints)


def get_scope(constraint):
    """Extract variables from a constraint."""
    import cpmpy
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


def prune_bias_to_oracle_scopes(bias, oracle_constraints):
    oracle_scopes = set()
    for c in oracle_constraints:
        scope = get_scope(c)
        if len(scope) >= 2:
            scope_key = frozenset(v.name for v in scope)
            oracle_scopes.add(scope_key)
    
    print(f"  Oracle has constraints on {len(oracle_scopes)} unique variable pair scopes")
    
    pruned_bias = []
    removed_count = 0
    
    for c in bias:
        scope = get_scope(c)
        if len(scope) >= 2:
            scope_key = frozenset(v.name for v in scope)
            if scope_key in oracle_scopes:
                pruned_bias.append(c)
            else:
                removed_count += 1
        else:
            pruned_bias.append(c)
    
    print(f"  Removed {removed_count} bias constraints on non-oracle scopes")
    print(f"  Pruned bias size: {len(pruned_bias)}")
    
    return pruned_bias


def run_growacq_simple(
    experiment_name,
    phase2_pickle_path,
    max_queries=1000,
    verbose=1
):
    """
    Simple GrowAcq runner.
    
    Args:
        experiment_name: Name of the benchmark
        phase2_pickle_path: Path to phase2 pickle file
        max_queries: Maximum queries for GrowAcq
        verbose: Verbosity level (0-3)
    
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Phase 3: GrowAcq Active Learning (Simple)")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*70}\n")
    
    # 1. Load Phase 2 data
    C_validated, B_fixed, phase2_stats = load_phase2_pickle(phase2_pickle_path)
    
    # 2. Construct benchmark
    instance, oracle = construct_benchmark(experiment_name)
    
    print(f"\nBenchmark: {experiment_name}")
    print(f"  - Variables: {len(instance.X)}")
    print(f"  - Oracle constraints: {len(oracle.constraints)}")
    
    # 3. Decompose validated constraints (AllDifferent -> binary !=)
    CL_init = decompose_alldifferent(C_validated)
    print(f"\nInitial CL (decomposed from validated globals): {len(CL_init)}")
    
    # 4. Create decomposed oracle for active learning
    oracle_decomposed = create_decomposed_oracle(oracle)
    oracle_decomposed.variables_list = cpm_array(instance.X)
    
    print(f"Oracle (decomposed): {len(oracle_decomposed.constraints)} constraints")
    
    # 5. CRITICAL: Prune bias to only include scopes that exist in oracle
    # This prevents FindC collapse when querying scopes without oracle constraints
    print(f"\nPruning bias to oracle scopes...")
    print(f"  Original bias size: {len(B_fixed)}")
    B_pruned = prune_bias_to_oracle_scopes(B_fixed, oracle_decomposed.constraints)
    
    # 6. Set up GrowAcq
    variables = get_variables(CL_init + B_pruned) if (CL_init or B_pruned) else list(instance.X.flat)
    
    ca_instance = ProblemInstance(
        variables=cpm_array(variables),
        init_cl=CL_init,
        name=f"{experiment_name}_phase3",
        bias=B_pruned  # Use pruned bias to prevent FindC collapse
    )
    
    print(f"\nGrowAcq Setup:")
    print(f"  - Variables: {len(ca_instance.variables)}")
    print(f"  - Initial CL: {len(ca_instance.cl)}")
    print(f"  - Bias (pruned): {len(ca_instance.bias)}")
    
    # Create GrowAcq with MQuAcq2 as inner algorithm
    findc = FindC(time_limit=1)
    qgen = PQGen(time_limit=2)
    ca_env = ActiveCAEnv(qgen=qgen, findc=findc)
    
    inner_mquacq2 = MQuAcq2(ca_env=ca_env)
    growacq = GrowAcq(ca_env=ca_env, inner_algorithm=inner_mquacq2)
    
    # 6. Run GrowAcq
    print(f"\n{'='*70}")
    print(f"Starting GrowAcq...")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    learned_instance = growacq.learn(
        ca_instance,
        oracle=oracle_decomposed,
        verbose=verbose
    )
    
    phase3_time = time.time() - start_time
    phase3_queries = growacq.env.metrics.total_queries
    
    # 7. Results
    learned_constraints = learned_instance.cl
    
    print(f"\n{'='*70}")
    print(f"GrowAcq Complete")
    print(f"{'='*70}")
    print(f"  - Queries: {phase3_queries}")
    print(f"  - Time: {phase3_time:.2f}s")
    print(f"  - Learned constraints: {len(learned_constraints)}")
    
    # 8. Evaluate against oracle
    target_strs = set(str(c) for c in oracle_decomposed.constraints)
    learned_strs = set(str(c) for c in learned_constraints)
    
    correct = len(target_strs & learned_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    precision = correct / len(learned_constraints) if learned_constraints else 0
    recall = correct / len(oracle_decomposed.constraints) if oracle_decomposed.constraints else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEvaluation:")
    print(f"  - Target constraints: {len(oracle_decomposed.constraints)}")
    print(f"  - Learned constraints: {len(learned_constraints)}")
    print(f"  - Correct: {correct}")
    print(f"  - Missing: {missing}")
    print(f"  - Spurious: {spurious}")
    print(f"  - Precision: {precision:.2%}")
    print(f"  - Recall: {recall:.2%}")
    print(f"  - F1: {f1:.2%}")
    
    # Total queries and time
    total_queries = phase2_stats.get('queries', 0) + phase3_queries
    total_time = phase2_stats.get('time', 0) + phase3_time
    
    print(f"\n{'='*70}")
    print(f"Total (Phase 2 + Phase 3)")
    print(f"{'='*70}")
    print(f"  - Total queries: {total_queries}")
    print(f"  - Total time: {total_time:.2f}s")
    
    # 9. Save results
    results = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'phase2': {
            'queries': phase2_stats.get('queries', 0),
            'time': phase2_stats.get('time', 0),
            'validated_globals': len(C_validated)
        },
        'phase3': {
            'queries': phase3_queries,
            'time': phase3_time,
            'initial_cl': len(CL_init),
            'bias': len(B_pruned),
            'bias_original': len(B_fixed),
            'learned_constraints': len(learned_constraints)
        },
        'total': {
            'queries': total_queries,
            'time': total_time
        },
        'evaluation': {
            'target_size': len(oracle_decomposed.constraints),
            'learned_size': len(learned_constraints),
            'correct': correct,
            'missing': missing,
            'spurious': spurious,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    output_dir = "phase3_simple_output"
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results: {results_path}")
    
    # Save learned model
    # Handle both numpy arrays and lists for instance.X
    if hasattr(instance.X, 'flat'):
        variables_list = list(instance.X.flat)
    else:
        variables_list = list(instance.X) if not isinstance(instance.X, list) else instance.X
    
    model_data = {
        'experiment': experiment_name,
        'learned_constraints': learned_constraints,
        'variables': variables_list,
        'phase3_queries': phase3_queries,
        'phase3_time': phase3_time
    }
    
    model_path = os.path.join(output_dir, f"{experiment_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[SAVED] Model: {model_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple Phase 3: GrowAcq with decomposed AllDifferent CL'
    )
    parser.add_argument(
        '--experiment', type=str, default='sudoku',
        help='Experiment name (sudoku, jsudoku, latin_square, etc.)'
    )
    parser.add_argument(
        '--phase2_pickle', type=str, default=None,
        help='Path to Phase 2 pickle (default: phase2_output/{experiment}_phase2.pkl)'
    )
    parser.add_argument(
        '--max_queries', type=int, default=1000,
        help='Max queries for GrowAcq'
    )
    parser.add_argument(
        '--verbose', type=int, default=1,
        help='Verbosity level (0-3)'
    )
    
    args = parser.parse_args()
    
    # Default phase2 pickle path
    if args.phase2_pickle is None:
        args.phase2_pickle = f"phase2_output/{args.experiment}_phase2.pkl"
    
    run_growacq_simple(
        experiment_name=args.experiment,
        phase2_pickle_path=args.phase2_pickle,
        max_queries=args.max_queries,
        verbose=args.verbose
    )

