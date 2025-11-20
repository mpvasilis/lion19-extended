import argparse
import os
import pickle
import random
import sys
import time
from itertools import combinations

import numpy as np
import cpmpy as cp
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables

from cpmpy.expressions.globalconstraints import AllDifferent

from main_alldiff_cop import (  
    construct_instance,
    load_phase1_data,
    extract_alldifferent_constraints,
    initialize_probabilities,
    update_supporting_evidence,
    build_constraint_violation,
    variables_to_assignment,
)


def flatten_variables(variables):
    if isinstance(variables, np.ndarray):
        return [var for var in variables.flat]
    if isinstance(variables, (list, tuple, set)):
        flat = []
        for item in variables:
            flat.extend(flatten_variables(item))
        return flat
    return [variables]


def get_domain_values(var, max_domain_size=1024):
    values = set()

    domain_attr = getattr(var, "domain", None)
    if domain_attr is not None:
        try:
            values = set(list(domain_attr))
        except TypeError:
            if hasattr(domain_attr, "values"):
                try:
                    values = set(domain_attr.values())
                except TypeError:
                    values = set()

    if not values:
        lb = getattr(var, "lb", None)
        ub = getattr(var, "ub", None)
        if lb is not None and ub is not None:
            lb = int(lb)
            ub = int(ub)
            size = ub - lb + 1
            if size <= max_domain_size:
                values = set(range(lb, ub + 1))
            else:
                step = max(1, size // max_domain_size)
                values = set(range(lb, ub + 1, step))

    return values


def synchronise_assignments(model_vars, oracle_vars):
    value_map = {}

    for var in model_vars:
        name = getattr(var, "name", None)
        if name is None:
            continue

        value = None
        if callable(getattr(var, "value", None)):
            value = var.value()
        if value is None and hasattr(var, "_value"):
            value = getattr(var, "_value")

        if value is not None:
            value_map[str(name)] = value

    for ovar in oracle_vars:
        name = getattr(ovar, "name", None)
        if name is None:
            continue

        value = value_map.get(str(name))
        if value is None:
            continue

        if hasattr(ovar, "_value"):
            ovar._value = value


def interpret_oracle_response(response):
    if isinstance(response, bool):
        return response
    if isinstance(response, str):
        return response.strip().lower() in {"yes", "y", "true", "1"}
    return bool(response)


def manual_sudoku_oracle_check(assignment, oracle, oracle_variables):

    try:
  
        
        check_model = cp.Model()
        
        
        for c in oracle.constraints:
            check_model += c
        
        print(f"    [ORACLE CHECK] Created model with {len(oracle.constraints)} TRUE constraints")
        
        
        var_map = {}
        if oracle_variables is not None:
            for var in oracle_variables:
                var_name = str(getattr(var, 'name', ''))
                if var_name:
                    var_map[var_name] = var
        
        
        assignments_added = 0
        for var_name, value in assignment.items():
            if value is not None and not isinstance(value, bool):
                if var_name in var_map:
                    check_model += (var_map[var_name] == value)
                    assignments_added += 1
        
        print(f"    [ORACLE CHECK] Added {assignments_added} assignment constraints")
        print(f"    [ORACLE CHECK] Assignment: {assignment}")
        
        
        result = check_model.solve(time_limit=5)
        
        if result:
            print(f"    [ORACLE CHECK] Model is SAT - Assignment is VALID")
            return True
        else:
            print(f"    [ORACLE CHECK] Model is UNSAT - Assignment is INVALID")
            return False
            
    except Exception as e:
        print(f"    [ORACLE CHECK] Error during check: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_variable_pairs(scope_vars):
    pairs = []

    sorted_scope = sorted(scope_vars, key=lambda v: str(getattr(v, "name", v)))

    for xi, xj in combinations(sorted_scope, 2):
        domain_i = get_domain_values(xi)
        domain_j = get_domain_values(xj)
        intersection = domain_i & domain_j

        if not intersection:
            continue

        score = len(intersection)
        pairs.append((score, xi, xj, tuple(sorted(intersection))))

    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


def query_driven_refinement(
    experiment_name,
    candidate_constraints,
    solver_variables,
    oracle_variables,
    oracle,
    probabilities,
    *,
    alpha=0.42,
    theta_max=0.98,
    max_queries=500,
    timeout=600,
    solver_timeout=30,
    additional_constraints=None,
    random_seed=42,
):

    rng = random.Random(random_seed)
    start_time = time.time()

    solver_vars = list(solver_variables)
    oracle_vars = list(oracle_variables)

    remaining_constraints = list(candidate_constraints)
    removed_constraints = set()
    validated_constraints = set()  

    probability_map = {c: probabilities.get(c, 0.3) for c in remaining_constraints}
    
    
    remaining_constraints.sort(key=lambda c: probability_map.get(c, 0.5))
    print(f"\n[OPTIMIZATION] Sorted {len(remaining_constraints)} constraints by ascending probability")

    total_queries = 0
    solver_calls = 0
    solver_time_acc = 0.0
    pairs_considered = 0
    
    
    query_cache = {}  

    for idx, constraint in enumerate(remaining_constraints, start=1):
        if constraint in removed_constraints:
            continue

        if max_queries is not None and total_queries >= max_queries:
            print(f"\n[STOP] Query budget ({max_queries}) exhausted before processing remaining constraints.")
            break

        elapsed = time.time() - start_time
        if timeout is not None and elapsed >= timeout:
            print(f"\n[STOP] Timeout ({timeout}s) reached before processing remaining constraints.")
            break

        print(f"\n{'-'*70}")
        print(f"Constraint {idx}/{len(remaining_constraints)}")
        print(constraint)

        scope_vars = list(get_variables([constraint]))
        if len(scope_vars) < 2:
            print("  [SKIP] Constraint scope too small to generate variable pairs.")
            continue

        pairs = prepare_variable_pairs(scope_vars)
        
        
        MAX_PAIRS_TO_TEST = 10
        if len(pairs) > MAX_PAIRS_TO_TEST:
            print(f"  [OPTIMIZATION] Testing top {MAX_PAIRS_TO_TEST} of {len(pairs)} pairs")
            pairs = pairs[:MAX_PAIRS_TO_TEST]
        
        pairs_considered += len(pairs)

        if not pairs:
            print("  [ACCEPT] No overlapping domains among pairs; accepting constraint.")
            validated_constraints.add(constraint)
            continue

        violation_found = False

        for score, xi, xj, intersection in pairs:
            if max_queries is not None and total_queries >= max_queries:
                break

            elapsed = time.time() - start_time
            if timeout is not None and elapsed >= timeout:
                break

            test_value = rng.choice(intersection)
            print(f"  [TRY] Pair ({xi.name}, {xj.name}) score={score} value={test_value}")

            model = cp.Model()

            if additional_constraints:
                model += list(additional_constraints)

            
            
            for other in remaining_constraints:
                if other is constraint or other in removed_constraints:
                    continue
                model += other

            model += (xi == test_value)
            model += (xj == test_value)

            solver_calls += 1

            solver_start = time.time()
            solved = model.solve(time_limit=solver_timeout)
            solver_time_acc += time.time() - solver_start

            if not solved:
                print("    -> UNSAT or timeout for this pair")
                continue

            violation_found = True

            synchronise_assignments(solver_vars, oracle_vars)
            
            
            print(f"    -> Violating assignment found:")
            assignment_dict = {}
            for var in solver_vars:
                if hasattr(var, 'value') and var.value() is not None:
                    assignment_dict[var.name] = var.value()
            
            
            if len(assignment_dict) <= 20:
                
                print(f"       {assignment_dict}")
            else:
                
                scope_names = {var.name for var in scope_vars}
                relevant_assignment = {k: v for k, v in assignment_dict.items() if k in scope_names}
                print(f"       Scope variables: {relevant_assignment}")
                print(f"       Full assignment has {len(assignment_dict)} variables")
            
            
            assignment_sig = tuple(sorted((v.name, v.value()) for v in solver_vars if v.value() is not None))
            if assignment_sig in query_cache:
                is_valid = query_cache[assignment_sig]
                print(f"    -> [CACHED] Oracle response: {'YES' if is_valid else 'NO'}")
            else:
                total_queries += 1  
                
                
                manual_result = manual_sudoku_oracle_check(assignment_dict, oracle, oracle_vars)
                
                if manual_result is not None:
                    answer = manual_result
                    print(f"    -> [MANUAL ORACLE] Result: {'YES (valid)' if answer else 'NO (invalid)'}")
                else:
                    
                    print(f"    -> [MANUAL ORACLE] Failed, using standard oracle")
                    answer = oracle.answer_membership_query(oracle_vars)
                
                is_valid = interpret_oracle_response(answer)
                query_cache[assignment_sig] = is_valid
                print(f"    -> Oracle response: {'YES' if is_valid else 'NO'}")

            if is_valid:
                
                removed_constraints.add(constraint)
                probability_map.pop(constraint, None)
                print("    -> Constraint refuted by valid counterexample. Removing from candidate set.")
                break  
            else:
                
                updated_prob = update_supporting_evidence(probability_map.get(constraint, 0.5), alpha)
                probability_map[constraint] = updated_prob
                print(f"    -> Constraint supported. Updated probability: {updated_prob:.3f}")

                if updated_prob >= theta_max:
                    print(f"    -> Probability exceeds theta_max ({theta_max}); accepting constraint.")
                    validated_constraints.add(constraint)
                    break  
                
                
                print(f"    -> Probability {updated_prob:.3f} < theta_max ({theta_max}); continuing to next pair.")

        if not violation_found:
            print("  [ACCEPT] No violating assignment found; accepting constraint.")
            validated_constraints.add(constraint)

    final_constraints = [c for c in remaining_constraints if c not in removed_constraints]
    elapsed_total = time.time() - start_time

    stats = {
        "queries": total_queries,
        "time": elapsed_total,
        "validated": len(final_constraints),
        "rejected": len(remaining_constraints) - len(final_constraints),
        "solver_calls": solver_calls,
        "pairs_considered": pairs_considered,
    }

    print(f"\n{'='*70}")
    print(f"Refinement complete for {experiment_name}")
    print(f"Validated constraints: {stats['validated']}")
    print(f"Rejected constraints: {stats['rejected']}")
    print(f"Total queries: {total_queries}")
    print(f"Total time: {elapsed_total:.2f}s")
    if elapsed_total > 0 and total_queries > 0:
        print(f"Queries per second: {total_queries/elapsed_total:.2f}")
    print(f"Solver calls: {solver_calls}")
    print(f"Pairs considered: {pairs_considered}")
    print(f"{'='*70}\n")

    return final_constraints, probability_map, stats, removed_constraints


def main():
    parser = argparse.ArgumentParser(
        description="HCAR AllDifferent Phase 2 - LION19 Query-Driven Refinement"
    )
    parser.add_argument("--experiment", type=str, default="sudoku", help="Benchmark name")
    parser.add_argument("--phase1_pickle", type=str, default=None, help="Phase 1 pickle path")
    parser.add_argument("--alpha", type=float, default=0.42, help="Bayesian update parameter")
    parser.add_argument("--theta_max", type=float, default=0.98, help="Acceptance threshold")
    parser.add_argument("--max_queries", type=int, default=500, help="Maximum membership queries")
    parser.add_argument("--timeout", type=int, default=600, help="Overall timeout in seconds")
    parser.add_argument("--solver_timeout", type=int, default=30, help="Solver timeout per query (s)")
    parser.add_argument("--prior", type=float, default=0.5, help="Default prior probability")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--use_bias",
        action="store_true",
        help="Include Phase 1 pruned bias (B_fixed) as hard constraints in queries",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("HCAR AllDifferent - LION19 Query-Driven Phase 2")
    print(f"Experiment: {args.experiment}")
    print(f"Alpha: {args.alpha}")
    print(f"Theta_max: {args.theta_max}")
    print(f"Max queries: {args.max_queries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Solver timeout per query: {args.solver_timeout}s")
    print(f"Prior: {args.prior}")
    print(f"Random seed: {args.random_seed}")
    print(f"Use bias constraints: {args.use_bias}")
    print(f"{'='*70}\n")

    instance, oracle = construct_instance(args.experiment)
    oracle.variables_list = cpm_array(instance.X)

    oracle_variables = flatten_variables(instance.X)

    phase1_data = None
    candidate_constraints = None
    solver_variables = None
    initial_probabilities = None
    additional_constraints = []

    if args.phase1_pickle:
        phase1_data = load_phase1_data(args.phase1_pickle)
        candidate_constraints = phase1_data.get("CG", [])
        solver_variables = flatten_variables(phase1_data.get("variables", []))
        initial_probs = phase1_data.get("initial_probabilities", {})
        initial_probabilities = {c: initial_probs.get(c, args.prior) for c in candidate_constraints}

        if args.use_bias:
            additional_constraints = list(phase1_data.get("B_fixed", []))

    if not candidate_constraints:
        print("\n[INFO] No Phase 1 data provided or CG empty; extracting AllDifferent constraints from oracle.")
        candidate_constraints = extract_alldifferent_constraints(oracle)
        solver_variables = oracle_variables
        initial_probabilities = initialize_probabilities(candidate_constraints, prior=args.prior)
    else:
        print(f"Loaded {len(candidate_constraints)} candidate constraints from Phase 1 data.")

    if not candidate_constraints:
        print("\n[ERROR] No candidate constraints available. Exiting.")
        return 1

    final_constraints, probability_map, stats, removed_constraints = query_driven_refinement(
        args.experiment,
        candidate_constraints,
        solver_variables,
        oracle_variables,
        oracle,
        initial_probabilities,
        alpha=args.alpha,
        theta_max=args.theta_max,
        max_queries=args.max_queries,
        timeout=args.timeout,
        solver_timeout=args.solver_timeout,
        additional_constraints=additional_constraints,
        random_seed=args.random_seed,
    )

    target_constraints = extract_alldifferent_constraints(oracle)
    target_strs = {str(c) for c in target_constraints}
    learned_strs = {str(c) for c in final_constraints}

    correct = len(target_strs & learned_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)

    precision = correct / len(final_constraints) if final_constraints else 0.0
    recall = correct / len(target_constraints) if target_constraints else 0.0

    print(f"Target AllDifferent constraints: {len(target_constraints)}")
    print(f"Learned AllDifferent constraints: {len(final_constraints)}")
    print(f"Correct: {correct}")
    print(f"Missing: {missing}")
    print(f"Spurious: {spurious}")

    if correct == len(target_constraints) and spurious == 0:
        print("\n[SUCCESS] Perfect learning!")
    else:
        if missing > 0:
            print("\n[DETAIL] Missing constraints:")
            for c in target_constraints:
                if str(c) not in learned_strs:
                    print(f"  - {c}")

        if spurious > 0:
            print("\n[DETAIL] Spurious constraints:")
            for c in final_constraints:
                if str(c) not in target_strs:
                    print(f"  - {c}")

    print(f"\n{'='*60}")
    print("Final Statistics")
    print(f"{'='*60}")
    print(f"Total queries: {stats['queries']}")
    print(f"Total time: {stats['time']:.2f}s")
    if stats['time'] > 0 and stats['queries'] > 0:
        print(f"Queries per second: {stats['queries']/stats['time']:.2f}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"CP Implication Check")
    print(f"{'='*60}")

    cp_implication_results = {}
    target_constraint_list = list(getattr(oracle, 'constraints', []))

    if not target_constraint_list:
        print("Oracle exposes no constraints; skipping implication check.")
        cp_implication_results = {
            'skipped': True,
            'reason': 'no target constraints available'
        }
    elif not final_constraints:
        print("No validated constraints to check; skipping implication check.")
        cp_implication_results = {
            'skipped': True,
            'reason': 'no validated constraints'
        }
    else:
        base_constraints = list(target_constraint_list)
        implied = []
        not_implied = []
        counterexamples = []

        print(f"Target constraint count: {len(base_constraints)}")
        print(f"Validated constraints to check: {len(final_constraints)}")

        for idx, constraint in enumerate(final_constraints, start=1):
            violation_expr = build_constraint_violation(constraint)

            test_model = cp.Model()
            test_model += base_constraints
            test_model += violation_expr

            has_counterexample = test_model.solve()

            if has_counterexample:
                assignment = variables_to_assignment(oracle_variables)
                assignment_copy = dict(assignment)

                preview_items = list(assignment_copy.items())
                preview_str = ", ".join(f"{k}={v}" for k, v in preview_items[:10])
                if len(preview_items) > 10:
                    preview_str += ", ..."

                print(f"  [FAIL] Constraint not implied: {constraint}")
                if preview_str:
                    print(f"    Counterexample: {{{preview_str}}}")

                not_implied.append(str(constraint))
                counterexamples.append({
                    'constraint': str(constraint),
                    'assignment': assignment_copy
                })
            else:
                print(f"  [OK] Constraint implied: {constraint}")
                implied.append(str(constraint))

        implied_count = len(implied)
        not_implied_count = len(not_implied)

        print(f"Number of implied constraints: {implied_count}")
        print(f"\nImplication summary: implied={implied_count}, "
              f"not_implied={not_implied_count}, checked={len(final_constraints)}")

        cp_implication_results = {
            'skipped': False,
            'checked': len(final_constraints),
            'implied': implied,
            'not_implied': counterexamples,
            'status': 'all_implied' if not not_implied else 'partial',
            'implied_count': implied_count,
            'not_implied_count': not_implied_count,
            'target_constraint_count': len(target_constraint_list)
        }

    
    stats['cp_implication'] = cp_implication_results

    phase2_output = {
        "C_validated": final_constraints,
        "C_validated_strs": [str(c) for c in final_constraints],
        "probabilities": probability_map,
        "experiment_name": args.experiment,
        "phase2_stats": stats,
        "removed_constraints": [str(c) for c in removed_constraints],
        "phase1_data": phase1_data if phase1_data is not None else None,
        "E_plus": phase1_data.get("E+", None) if phase1_data else None,
        "B_fixed": phase1_data.get("B_fixed", None) if phase1_data else None,
        "all_variables": oracle_variables,
        "metadata": {
            "approach": "lion19",
            "alpha": args.alpha,
            "theta_max": args.theta_max,
            "random_seed": args.random_seed,
            "solver_timeout": args.solver_timeout,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_queries": stats['queries'],
            "total_time": stats['time'],
            "precision": precision,
            "recall": recall,
        },
    }

    phase2_output_dir = "phase2_output"
    os.makedirs(phase2_output_dir, exist_ok=True)
    phase2_pickle_path = os.path.join(phase2_output_dir, f"{args.experiment}_lion19_phase2.pkl")

    with open(phase2_pickle_path, "wb") as f:
        pickle.dump(phase2_output, f)

    
    cp_implication_log_path = os.path.join(phase2_output_dir, f"{args.experiment}_lion19_cp_implication.log")
    if cp_implication_results.get('skipped', False):
        log_contents = [
            "CP Implication Check",
            "====================",
            f"Status: SKIPPED",
            f"Reason: {cp_implication_results.get('reason', 'unknown')}",
            ""
        ]
    else:
        log_contents = [
            "CP Implication Check",
            "====================",
            f"Target constraint count: {cp_implication_results.get('target_constraint_count', 0)}",
            f"Validated constraints checked: {cp_implication_results.get('checked', 0)}",
            f"Implied constraints: {cp_implication_results.get('implied_count', 0)}",
            f"Not implied constraints: {cp_implication_results.get('not_implied_count', 0)}",
            "",
            "Implied constraints:",
        ]
        implied_list = cp_implication_results.get('implied', [])
        if implied_list:
            log_contents.extend(f"  - {c}" for c in implied_list)
        else:
            log_contents.append("  (none)")

        log_contents.extend([
            "",
            "Counterexamples for non-implied constraints:"
        ])
        counterexamples = cp_implication_results.get('not_implied', [])
        if counterexamples:
            for counter in counterexamples:
                constraint_str = counter.get('constraint', '<unknown>')
                assignment = counter.get('assignment', {})
                assignment_preview = ", ".join(f"{k}={v}" for k, v in list(assignment.items())[:15])
                if len(assignment) > 15:
                    assignment_preview += ", ..."
                log_contents.append(f"  - {constraint_str}")
                log_contents.append(f"    Assignment: {{{assignment_preview}}}")
        else:
            log_contents.append("  (none)")
        log_contents.append("")

    with open(cp_implication_log_path, 'w') as f:
        f.write("\n".join(log_contents))

    print(f"Phase 2 outputs saved to: {phase2_pickle_path}")
    print(f"  - Validated constraints: {len(final_constraints)}")
    print(f"  - Rejected constraints: {len(removed_constraints)}")
    print(f"  - CP implication log saved to: {cp_implication_log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


