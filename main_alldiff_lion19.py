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


def extract_grid_position(var):
    """
    Extract row and column position from a variable name.
    Assumes variable names follow pattern like 'grid[i,j]' or 'x[i,j]'.
    Returns (row, col) tuple or (None, None) if pattern not found.
    """
    name = str(getattr(var, 'name', var))
    
    
    if '[' in name and ',' in name:
        try:
            
            bracket_content = name.split('[')[1].split(']')[0]
            parts = bracket_content.split(',')
            if len(parts) >= 2:
                row = int(parts[0].strip())
                col = int(parts[1].strip())
                return (row, col)
        except (ValueError, IndexError):
            pass
    
    return (None, None)


def compute_manhattan_distance(var1, var2):
    """
    Compute Manhattan distance between two variables based on their grid positions.
    d(x_i, x_j) = |r(x_i) - r(x_j)| + |c(x_i) - c(x_j)|
    Returns 0 if positions cannot be determined.
    """
    pos1 = extract_grid_position(var1)
    pos2 = extract_grid_position(var2)
    
    if pos1[0] is None or pos2[0] is None:
        return 0
    
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def compute_involvement_scores(variables, candidate_constraints):
    """
    Compute involvement score I(x) for each variable.
    I(x) = number of candidate constraints in which x appears.
    """
    involvement = {str(getattr(v, 'name', v)): 0 for v in variables}
    
    for constraint in candidate_constraints:
        scope_vars = list(get_variables([constraint]))
        for var in scope_vars:
            var_name = str(getattr(var, 'name', var))
            if var_name in involvement:
                involvement[var_name] += 1
    
    return involvement


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


def bayesian_update_lion19(P_prior, p_d_given_valid=0.95, p_d_given_invalid=0.05):
    """
    Update probability using Bayes' rule as specified in LION19 paper.
    
    Given that the oracle rejected a violating assignment (event D):
    P(c ∈ C* | D) = P(D | c ∈ C*) · P(c) / [P(D | c ∈ C*) · P(c) + P(D | c ∉ C*) · (1 - P(c))]
    
    Args:
        P_prior: Current probability P(c ∈ C*)
        p_d_given_valid: P(D | c ∈ C*) = probability oracle rejects given c is valid (default: 0.95)
        p_d_given_invalid: P(D | c ∉ C*) = probability oracle rejects given c is invalid (default: 0.05)
    
    Returns:
        Updated posterior probability P(c ∈ C* | D)
    """
    numerator = p_d_given_valid * P_prior
    denominator = p_d_given_valid * P_prior + p_d_given_invalid * (1 - P_prior)
    
    if denominator == 0:
        return P_prior
    
    return numerator / denominator


def manual_sudoku_oracle_check(assignment, oracle, oracle_variables):
    """
    Manually check if an assignment is valid by creating a CP model with TRUE oracle constraints.
    
    Args:
        assignment: Dictionary mapping variable names to values
        oracle: Oracle object containing TRUE constraints
        oracle_variables: List/array of oracle variables
        
    Returns:
        True if assignment is valid (satisfiable with true constraints)
        False if assignment is invalid (unsatisfiable with true constraints)
        None if check cannot be performed
    """
    try:
        import cpmpy as cp
        

        if not hasattr(oracle, 'constraints') or not oracle.constraints:
            print(f"    [ORACLE CHECK] Oracle has no constraints")
            return None
        

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


def prepare_variable_pairs(scope_vars, involvement_scores=None, alpha=1.0, beta=0.5):
    """
    Compute candidate variable pairs with their domain intersections and heuristic scores.
    
    Uses the LION19 paper scoring function:
    score(x_i, x_j) = α · |d(x_i, x_j)| - β · (I(x_i) + I(x_j))
    
    where:
    - d(x_i, x_j) is the Manhattan distance between variables
    - I(x) is the involvement score (number of candidate constraints containing x)
    - α, β are positive weighting parameters (default: α=1.0, β=0.5)
    
    Pairs with higher scores (greater spatial separation and lower involvement) are prioritized.
    """
    pairs = []

    sorted_scope = sorted(scope_vars, key=lambda v: str(getattr(v, "name", v)))

    for xi, xj in combinations(sorted_scope, 2):
        domain_i = get_domain_values(xi)
        domain_j = get_domain_values(xj)
        intersection = domain_i & domain_j

        if not intersection:
            continue

        
        manhattan_dist = compute_manhattan_distance(xi, xj)
        
        
        involvement_i = 0
        involvement_j = 0
        if involvement_scores:
            xi_name = str(getattr(xi, 'name', xi))
            xj_name = str(getattr(xj, 'name', xj))
            involvement_i = involvement_scores.get(xi_name, 0)
            involvement_j = involvement_scores.get(xj_name, 0)
        
        
        score = alpha * manhattan_dist - beta * (involvement_i + involvement_j)
        
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
    scoring_alpha=1.0,
    scoring_beta=0.5,
    p_d_given_valid=0.95,
    p_d_given_invalid=0.05,
    theta_max=0.98,
    max_queries=500,
    timeout=600,
    solver_timeout=30,
    additional_constraints=None,
    random_seed=42,
    use_all_candidates_in_model=True,
):
    """
    Run the LION19 Query-Driven refinement on the candidate constraints.
    
    Implements Algorithm 1 from the LION19 paper:
    - Sorts constraints by ascending prior probability (test suspicious ones first)
    - Uses scoring function: score(x_i,x_j) = α·d(x_i,x_j) - β·(I(x_i)+I(x_j))
    - Creates model M' = (C_G \\ {c}) ∪ {x_i=v, x_j=v} for violation queries
    - Updates probabilities using Bayesian inference with specified likelihoods
    
    Args:
        experiment_name: Name of the experiment
        candidate_constraints: Set of candidate AllDifferent constraints (C_G)
        solver_variables: Variables for the CP solver
        oracle_variables: Variables for oracle queries
        oracle: Oracle object for membership queries
        probabilities: Initial probability estimates for each constraint
        scoring_alpha: α parameter for scoring function (default: 1.0)
        scoring_beta: β parameter for scoring function (default: 0.5)
        p_d_given_valid: P(D | c ∈ C*) - probability oracle rejects if c is valid (default: 0.95)
        p_d_given_invalid: P(D | c ∉ C*) - probability oracle rejects if c is invalid (default: 0.05)
        theta_max: Acceptance threshold for probability (default: 0.98)
        max_queries: Maximum number of oracle queries
        timeout: Overall timeout in seconds
        solver_timeout: Timeout per solver call
        additional_constraints: Optional B_fixed constraints from Phase 1
        random_seed: Random seed for reproducibility
        use_all_candidates_in_model: If True, include all other candidates (C_G \\ {c}) in model M'
    
    Returns:
        Tuple of (final_constraints, probability_map, stats, removed_constraints)
    """

    rng = random.Random(random_seed)
    start_time = time.time()

    solver_vars = list(solver_variables)
    oracle_vars = list(oracle_variables)

    remaining_constraints = list(candidate_constraints)
    removed_constraints = set()
    validated_constraints = set()

    probability_map = {c: probabilities.get(c, 0.3) for c in remaining_constraints}

    
    involvement_scores = compute_involvement_scores(solver_vars, remaining_constraints)
    print(f"\n[LION19] Computed involvement scores for {len(involvement_scores)} variables")
    
    
    
    
    remaining_constraints.sort(key=lambda c: probability_map.get(c, 0.5), reverse=True)
    print(f"[LION19] Sorted {len(remaining_constraints)} constraints by DESCENDING probability")
    print(f"[LION19] Scoring parameters: α={scoring_alpha}, β={scoring_beta}")
    print(f"[LION19] Bayesian parameters: P(D|valid)={p_d_given_valid}, P(D|invalid)={p_d_given_invalid}")
    print(f"[LION19] Model construction: {'Include all candidates (C_G \\ {{c}})' if use_all_candidates_in_model else 'Only validated constraints'}")

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
        print(f"Constraint {idx}/{len(remaining_constraints)} (P={probability_map.get(constraint, 0.5):.3f})")
        print(constraint)

        scope_vars = list(get_variables([constraint]))
        if len(scope_vars) < 2:
            print("  [SKIP] Constraint scope too small to generate variable pairs.")
            continue

        
        pairs = prepare_variable_pairs(
            scope_vars, 
            involvement_scores=involvement_scores,
            alpha=scoring_alpha,
            beta=scoring_beta
        )

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
            print(f"  [TRY] Pair ({xi.name}, {xj.name}) score={score:.2f} value={test_value}")

            model = cp.Model()

            
            if additional_constraints:
                model += list(additional_constraints)
                print(f"    [MODEL] Including {len(additional_constraints)} B_fixed constraints")

            if use_all_candidates_in_model:
                
                
                other_candidates = [c for c in remaining_constraints 
                                   if c != constraint and c not in removed_constraints]
                for other_c in other_candidates:
                    model += other_c
                print(f"    [MODEL] Including {len(other_candidates)} other candidate constraints (C_G \\ {{c}})")
            else:
                
                for validated_c in validated_constraints:
                    model += validated_c
                if validated_constraints:
                    print(f"    [MODEL] Including {len(validated_constraints)} validated constraints")

            
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
                print("    -> Constraint REFUTED by valid counterexample. Removing from candidate set.")
                break
            else:
                
                
                old_prob = probability_map.get(constraint, 0.5)
                updated_prob = bayesian_update_lion19(
                    old_prob,
                    p_d_given_valid=p_d_given_valid,
                    p_d_given_invalid=p_d_given_invalid
                )
                probability_map[constraint] = updated_prob
                print(f"    -> Constraint SUPPORTED. Bayesian update: {old_prob:.3f} -> {updated_prob:.3f}")

                if updated_prob >= theta_max:
                    print(f"    -> Probability {updated_prob:.3f} >= theta_max ({theta_max}); ACCEPTING constraint.")
                    validated_constraints.add(constraint)
                    break

                print(f"    -> Probability {updated_prob:.3f} < theta_max ({theta_max}); continuing to next pair.")

        if not violation_found:
            print("  [ACCEPT] No violating assignment found; accepting constraint.")
            validated_constraints.add(constraint)

    
    final_constraints = [c for c in remaining_constraints if c not in removed_constraints]
    elapsed_total = time.time() - start_time
    
    print(f"\n[RESULT] Keeping {len(final_constraints)} validated constraints")

    stats = {
        "queries": total_queries,
        "time": elapsed_total,
        "validated": len(final_constraints),
        "rejected": len(remaining_constraints) - len(final_constraints),
        "solver_calls": solver_calls,
        "pairs_considered": pairs_considered,
        "scoring_alpha": scoring_alpha,
        "scoring_beta": scoring_beta,
        "p_d_given_valid": p_d_given_valid,
        "p_d_given_invalid": p_d_given_invalid,
    }

    print(f"\n{'='*70}")
    print(f"LION19 Refinement complete for {experiment_name}")
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
    
    
    parser.add_argument("--scoring_alpha", type=float, default=1.0, 
                        help="α parameter for scoring function (Manhattan distance weight)")
    parser.add_argument("--scoring_beta", type=float, default=0.5,
                        help="β parameter for scoring function (involvement score weight)")
    
    
    parser.add_argument("--p_d_given_valid", type=float, default=0.95,
                        help="P(D|c∈C*) - probability oracle rejects if constraint is valid")
    parser.add_argument("--p_d_given_invalid", type=float, default=0.05,
                        help="P(D|c∉C*) - probability oracle rejects if constraint is invalid")
    
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
    parser.add_argument(
        "--use_all_candidates",
        action="store_true",
        default=False,
        help="Include all other candidates (C_G \\ {c}) in model M' (original LION19 paper behavior)",
    )

    args = parser.parse_args()
    
    
    use_all_candidates_in_model = args.use_all_candidates

    print(f"\n{'='*70}")
    print("HCAR AllDifferent - LION19 Query-Driven Phase 2")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment}")
    print(f"\nLION19 Scoring Function: score(x_i,x_j) = α·d(x_i,x_j) - β·(I(x_i)+I(x_j))")
    print(f"  α (scoring_alpha): {args.scoring_alpha}")
    print(f"  β (scoring_beta): {args.scoring_beta}")
    print(f"\nLION19 Bayesian Update:")
    print(f"  P(D|c∈C*): {args.p_d_given_valid}")
    print(f"  P(D|c∉C*): {args.p_d_given_invalid}")
    print(f"\nOther Parameters:")
    print(f"  Theta_max: {args.theta_max}")
    print(f"  Max queries: {args.max_queries}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Solver timeout per query: {args.solver_timeout}s")
    print(f"  Prior probability: {args.prior}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Use B_fixed constraints: {args.use_bias}")
    print(f"  Model M' includes: {'All candidates (C_G \\ {{c}})' if use_all_candidates_in_model else 'Only validated constraints'}")
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
        scoring_alpha=args.scoring_alpha,
        scoring_beta=args.scoring_beta,
        p_d_given_valid=args.p_d_given_valid,
        p_d_given_invalid=args.p_d_given_invalid,
        theta_max=args.theta_max,
        max_queries=args.max_queries,
        timeout=args.timeout,
        solver_timeout=args.solver_timeout,
        additional_constraints=additional_constraints,
        random_seed=args.random_seed,
        use_all_candidates_in_model=use_all_candidates_in_model,
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
            "scoring_alpha": args.scoring_alpha,
            "scoring_beta": args.scoring_beta,
            "p_d_given_valid": args.p_d_given_valid,
            "p_d_given_invalid": args.p_d_given_invalid,
            "theta_max": args.theta_max,
            "random_seed": args.random_seed,
            "solver_timeout": args.solver_timeout,
            "use_all_candidates_in_model": use_all_candidates_in_model,
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


