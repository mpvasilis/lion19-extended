import argparse
import copy
import os
import pickle
import time
import sys
import cpmpy as cp
from cpmpy import *
from cpmpy import cpm_array
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.globalconstraints import AllDifferent
from pycona.utils import get_kappa, get_con_subset
from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global
from cpmpy.transformations.normalize import toplevel_list


def variables_to_assignment(variables):
    """Extract variable assignments as a name -> value mapping."""
    assignment = {}

    for var in variables:
        name = getattr(var, "name", None)
        if name is None:
            continue

        value = None
        if callable(getattr(var, "value", None)):
            value = var.value()
        if value is None and hasattr(var, "_value"):
            value = getattr(var, "_value")

        
        if value is not None and not isinstance(value, bool):
            assignment[str(name)] = value

    return assignment


def assignment_signature(assignment):
    """Create a canonical signature for an assignment dictionary."""
    if not assignment:
        return None
    return tuple(sorted(assignment.items()))


def load_phase1_data(pickle_path):
    
    print(f"\nLoading Phase 1 data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded Phase 1 data:")
    print(f"  CG (global constraints): {len(data['CG'])}")
    print(f"  B_fixed (pruned bias): {len(data['B_fixed'])}")
    print(f"  E+ (positive examples): {len(data['E+'])}")
    print(f"  Initial probabilities: {len(data.get('initial_probabilities', {}))}")

    return data


def display_sudoku_grid(variables, title="Sudoku Grid", debug=False):
    
    
    max_row = -1
    max_col = -1
    
    for var in variables:
        if hasattr(var, 'name') and ('[' in str(var.name)) and (',' in str(var.name)):
            try:
                var_name = str(var.name)
                parts = var_name.split('[')[1].split(']')[0].split(',')
                row = int(parts[0])
                col = int(parts[1])
                max_row = max(max_row, row)
                max_col = max(max_col, col)
            except:
                continue
    
    
    if max_row < 0 or max_col < 0:
        max_row, max_col = 8, 8
    
    rows = max_row + 1
    cols = max_col + 1
    
    print(f"\n{title} ({rows}x{cols})")
    separator_width = cols * 4 + 1
    print("  " + "-" * separator_width)

    grid = [[None for _ in range(cols)] for _ in range(rows)]

    if debug and len(variables) > 0:
        print(f"\n  DEBUG: Checking first 3 variables...")
        for i, var in enumerate(variables[:3]):
            print(f"    Var {i}: name={var.name if hasattr(var, 'name') else 'NO NAME'}, "
                  f"value={var.value() if hasattr(var, 'value') else 'NO VALUE METHOD'}, "
                  f"type={type(var)}")

    for var in variables:
        if hasattr(var, 'name') and ('[' in str(var.name)) and (',' in str(var.name)):

            try:
                var_name = str(var.name)
                parts = var_name.split('[')[1].split(']')[0].split(',')
                row = int(parts[0])
                col = int(parts[1])

                if callable(getattr(var, 'value', None)):
                    val = var.value()
                elif hasattr(var, '_value'):
                    val = var._value
                else:
                    val = None
                
                
                if val is not None:
                    
                    if isinstance(val, bool):
                        val = None
                    
                    elif isinstance(val, (int, float)):
                        try:
                            val = int(val)
                            
                            
                            if val < 0:
                                val = None
                        except (ValueError, TypeError):
                            val = None
                    else:
                        val = None
                
                if val is not None:
                    grid[row][col] = val
            except Exception as e:
                if debug:
                    print(f"  DEBUG: Error parsing {var.name}: {e}")
                continue

    
    for i in range(rows):
        row_str = "  |"
        for j in range(cols):
            val = grid[i][j]
            if val is None:
                row_str += "  . "
            else:
                
                row_str += f" {val:>2} "
        row_str += "|"
        print(row_str)
    
    print("  " + "-" * separator_width)

    filled = sum(1 for row in grid for cell in row if cell is not None)
    total_cells = rows * cols
    print(f"  Filled cells: {filled}/{total_cells}")


def synchronise_assignments(solver_vars, oracle_vars):
    """
    Synchronize variable assignments from solver to oracle.
    Maps values by variable name.
    """
    value_map = {}
    
    
    for var in solver_vars:
        name = getattr(var, "name", None)
        if name is None:
            continue
        
        value = None
        if callable(getattr(var, "value", None)):
            value = var.value()
        if value is None and hasattr(var, "_value"):
            value = getattr(var, "_value")
        
        
        if value is not None and not isinstance(value, bool):
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


def extract_alldifferent_constraints(oracle):
    
    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def build_constraint_violation(constraint):
    
    if isinstance(constraint, AllDifferent):
        variables = []
        if getattr(constraint, "args", None):
            args = constraint.args
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                variables = list(args[0])
            else:
                variables = list(args)
        if not variables:
            variables = list(get_variables(constraint))
        
        variables = list(variables)
        violation_terms = []
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                violation_terms.append(variables[i] == variables[j])
        
        if not violation_terms:
            return 0 == 1
        
        return cp.sum(violation_terms) >= 1
    
    return ~constraint


def initialize_probabilities(constraints, prior=0.5):
    
    probabilities = {}
    for c in constraints:
        probabilities[c] = prior
    return probabilities


def update_supporting_evidence(P_c, alpha):
    
    return P_c + (1 - P_c) * (1 - alpha)


def manual_sudoku_oracle_check(assignment, oracle, oracle_variables):

    try:
        
        check_model = cp.Model()
        
        print(f"    [ORACLE CHECK] Created model with {len(oracle.constraints)} TRUE constraints")
        
        
        var_map = {}
        for var in oracle_variables:
            var_name = str(getattr(var, 'name', ''))
            if var_name:
                var_map[var_name] = var
        
        
        assignments_added = 0
        assigned_vars = []
        for var_name, value in assignment.items():
            if var_name in var_map:
                check_model += (var_map[var_name] == value)
                print(var_map[var_name] == value)
                assignments_added += 1
                assigned_vars.append(var_map[var_name])

        
        con_subset = get_con_subset(oracle.constraints, set(assigned_vars))
        print(f"con_subset: {len(con_subset)} relevant constraints")
        for c in con_subset:
            check_model += c
        
        print(f"    [ORACLE CHECK] Added {assignments_added} assignment constraints")
        print(f"    [ORACLE CHECK] Assignment: {assignment}")
        
        print(check_model)
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


def interpret_oracle_response(response):
    
    if isinstance(response, bool):
        return response
    if isinstance(response, str):
        return response.strip().lower() in {"yes", "y", "true", "1"}
    return bool(response)


def generate_violation_query(CG, C_validated, probabilities, all_variables, oracle=None,
                             previous_queries=None, positive_examples=None, B_fixed=None, bias_weight=0.5):
    
    print(f"  Building COP model: {len(CG)} candidates, {len(C_validated)} validated, {len(all_variables)} variables")

    model = cp.Model()

    model_vars = get_variables(list(CG))

    C_validated_dec = toplevel_list([c.decompose()[0] for c in C_validated])
    
    # Get constraints from CL that involve Y
    Cl = get_con_subset(C_validated_dec, model_vars)

    # Add ONLY the ones that are relevant to the current CG
    for c in Cl:
        model += c

    exclusion_assignments = []
    if previous_queries:
        exclusion_assignments.extend(previous_queries)

    if exclusion_assignments:
        for idx, assignment in enumerate(exclusion_assignments):
            diff_terms = []
            flag = True # If all variables in the assignment are in the current CG, then add the constraint
            for var in model_vars:
                var_name = str(getattr(var, 'name', ''))
                if var_name in assignment:
                    diff_terms.append(var != assignment[var_name])
                else:
                    flag = False
                    break

            if flag:
                model += cp.any(diff_terms)

    gamma = {str(c): cp.boolvar(name=f"gamma_{i}") for i, c in enumerate(CG)}

    for c in CG:
        c_str = str(c)
        
        model += (gamma[c_str] == ~c)
    
    gamma_list = list(gamma.values())
    model += (cp.sum(gamma_list) >= 1)
    
    bias_violations = []
    relevant_bias = []
    
    if B_fixed is not None and len(B_fixed) > 0:
        print(f"  Processing B_fixed bias: {len(B_fixed)} constraints")
                
        relevant_bias = get_con_subset(B_fixed, model_vars) # Get constraints from B_fixed that involve Y
        
        print(f"  Relevant B_fixed constraints (overlap with CG scope): {len(relevant_bias)}/{len(B_fixed)}")
        
        for i, bias_c in enumerate(relevant_bias):
            beta_i = cp.boolvar(name=f"beta_{i}")
            model += (beta_i == ~bias_c)

            bias_violations.append(beta_i)
        
        print(f"  Added {len(bias_violations)} bias violation indicators")
    
    constraint_violation_term = cp.sum(probabilities[c]* gamma[str(c)] for c in CG)
    
    objective = 5000 * constraint_violation_term
    
    if bias_violations:
        bias_violation_term = cp.sum(bias_violations)
        objective += bias_violation_term


    model.minimize(objective)

    print(f"  Solving COP...")
    solve_start = time.time()

    result = model.solve(time_limit=5)
    solve_time = time.time() - solve_start
    
    bias_violated_flag = False
    
    if not result:
        print("UNSAT")
    else:
        violated = []
        for i, c in enumerate(CG):
            gi = gamma[str(c)].value()
            if gi is None:
                print(f"gamma_{i} has no value (solver didn’t assign).")
            elif gi:  
                violated.append((i, c))
        print(f"Violated {len(violated)}/{len(CG)} constraints:")
        for i, c in violated:
            print(f" - gamma_{i} -> VIOLATED: {c}")
            
    
    if bias_violations:
        total_bias_violations = sum(beta.value() if beta.value() is not None else 0 for beta in bias_violations)
        if total_bias_violations > 0:
            bias_violated_flag = True
            print(f"  Total violated B_fixed bias constraints: {total_bias_violations}/{len(bias_violations)}")
            
            
            violated_bias_constraints = []
            for i, beta in enumerate(bias_violations):
                if beta.value() and len(violated_bias_constraints) < 20:
                    violated_bias_constraints.append((i, relevant_bias[i]))
            
            if violated_bias_constraints:
                print(f"  First {len(violated_bias_constraints)} violated bias constraints:")
                for i, bias_c in violated_bias_constraints:
                    print(f"    - beta_{i} -> VIOLATED: {bias_c}")
        else:
            print(f"  No B_fixed bias constraints violated.")
    
    if result:
        print(f"  Solved in {solve_time:.2f}s - found violation query")
        
        Y = []
        for v in model_vars:
            var_name = str(getattr(v, 'name', ''))            
            
            if not var_name.startswith('gamma_') and not var_name.startswith('beta_'):
                Y.append(v)
        
        values_set = sum(1 for v in Y if v.value() is not None)
        print(f"  Variables with values: {values_set}/{len(Y)}")
            
        Viol_e = get_kappa(CG, Y)
        print(f"  Violating {len(Viol_e)}/{len(CG)} constraints")
            
        gamma_violations = []
        for i, c in enumerate(CG):
            gi = gamma[str(c)].value()
            if gi:
                gamma_violations.append(c)
        
        if len(gamma_violations) != len(Viol_e):
            print(f"    Gamma indicates {len(gamma_violations)} violations")
            print(f"    get_kappa found {len(Viol_e)} violations")
            print(f"  This may indicate variable synchronization issues.")
        
        assignment = variables_to_assignment(Y)
        
        return Y, Viol_e, "SAT", assignment, bias_violated_flag
    else:
        print(f"  UNSAT after {solve_time:.2f}s - cannot find violation query")
        return None, [], "UNSAT", {}, False



def cop_refinement_recursive(CG_cand, C_validated, oracle, probabilities, all_variables,
                             alpha, theta_max, theta_min, max_queries, timeout, 
                             recursion_depth=0, experiment_name="",
                             query_signature_cache=None, query_assignments=None,
                             negative_query_assignments=None,
                             query_history=None, positive_query_examples=None,
                             positive_signature_cache=None, phase1_positive_examples=None,
                             B_fixed=None, bias_weight=0.5, validation_log=None):
    """
    Recursive COP-based constraint refinement.
    
    This is the core refinement algorithm that can be called recursively for disambiguation.
    When oracle rejects a query violating multiple constraints, we recursively call this
    on the violated set to determine which constraints are correct.
    
    Algorithm Flow:
    1. Generate violation query that violates subset of CG_cand (minimizing sum of P(c))
    2. Ask oracle about the query
    3. If oracle accepts (TRUE):
         - All violated constraints are INCORRECT (counterexample found)
         - Remove them and continue
    4. If oracle rejects (FALSE):
         - At least one violated constraint is CORRECT
         - If single constraint: must be correct, validate it
         - If multiple constraints: RECURSIVELY call this function on violated set
            to disambiguate which are correct via bisection-like search
            * Recursive call receives filtered C_validated containing only constraints
              whose scope has at least 2 variables in common with violated set (via get_con_subset)
    5. Repeat until budget exhausted or all constraints classified
    
    Args:
        CG_cand: Candidate global constraints to refine
        C_validated: Already validated constraints (used as hard constraints in COP)
        oracle: Oracle for membership queries
        probabilities: Current probability estimates for each constraint
        all_variables: All problem variables
        alpha: Bayesian learning rate (decay factor for refutation)
        theta_max: Acceptance threshold
        theta_min: Rejection threshold
        max_queries: Query budget for this call (including recursive calls)
        timeout: Time budget in seconds
        recursion_depth: Current recursion depth (for logging/indentation)
        experiment_name: Name of experiment (for special handling like sudoku display)
    
    Returns:
        C_validated: List of validated constraints
        CG_remaining: Set of remaining uncertain constraints  
        probabilities: Updated probability dictionary
        queries_used: Number of queries consumed
    """
    start_time = time.time()
    queries_used = 0

    if phase1_positive_examples is None:
        phase1_positive_examples = []
    if query_signature_cache is None:
        query_signature_cache = set()
    if query_assignments is None:
        query_assignments = []
    if negative_query_assignments is None:
        negative_query_assignments = []
    if query_history is None:
        query_history = []
    if positive_query_examples is None:
        positive_query_examples = []
    if positive_signature_cache is None:
        positive_signature_cache = set()
    if validation_log is None:
        validation_log = []

    
    CG = set(CG_cand) if not isinstance(CG_cand, set) else CG_cand.copy()
    C_val = list(C_validated)  
    probs = probabilities.copy()
    
    indent = "  " * recursion_depth
    positive_examples = phase1_positive_examples 
    
    print(f"\n{indent}{'-'*50}")
    print(f"{indent}COP Refinement [Depth={recursion_depth}]")
    print(f"{indent}{'-'*50}")
    print(f"{indent}Candidates: {len(CG)}, Validated: {len(C_val)}, Budget: {max_queries}q, {timeout}s")
    
    iteration = 0
    consecutive_unsat = 0
    
    while True:
        iteration += 1
        
        
        if queries_used >= max_queries:
            print(f"{indent}[STOP] Query budget exhausted ({queries_used}/{max_queries})")
            break
        
        if time.time() - start_time > timeout:
            print(f"{indent}[STOP] Timeout reached")
            break
        
        if not CG:
            print(f"{indent}[STOP] No more candidates")
            break
        
        
        if len(CG) > 0 and min(probs[c] for c in CG) > theta_max:
            print(f"{indent}[STOP] All remaining P(c) > {theta_max}")
            for c in CG:
                C_val.append(c)
                print(f"{indent}  [ACCEPT] {c} (P={probs[c]:.3f})")
                validation_log.append({
                    'constraint': str(c),
                    'method': 'threshold_accept',
                    'probability': probs[c],
                    'depth': recursion_depth,
                    'iteration': iteration
                })
            CG = set()
            break
        
        print(f"\n{indent}[Iter {iteration}] {len(C_val)} validated, {len(CG)} candidates, {queries_used}q used")
        
        
        print(f"{indent}[QUERY] Generating violation query...")

        
        
        
        current_level_positive_examples = []
        if positive_examples:
            current_level_positive_examples.extend(positive_examples)
        if positive_query_examples:
            current_level_positive_examples.extend(positive_query_examples)

        Y, Viol_e, status, assignment, bias_violated = generate_violation_query(
            CG,
            C_val,
            probs,
            all_variables,
            oracle,
            previous_queries=negative_query_assignments,
            positive_examples=current_level_positive_examples,
            B_fixed=B_fixed,
            bias_weight=bias_weight
        )

        for c in Viol_e:
            if c not in CG:
                raise Exception(f"After query generation: Constraint {c} is in Viol_e but not in CG: {CG}")
        
        if status == "UNSAT":
            consecutive_unsat += 1
            print(f"{indent}[UNSAT] No violation query exists (consecutive: {consecutive_unsat})")
            
            
            for c in list(CG):
                if probs[c] >= 0.7:
                    C_val.append(c)
                    print(f"{indent}  [ACCEPT] {c} (P={probs[c]:.3f})")
            
            CG = {c for c in CG if probs[c] < 0.7}
            
            if not CG or consecutive_unsat >= 2:
                
                for c in list(CG):
                    if probs[c] >= 0.5:
                        C_val.append(c)
                        print(f"{indent}  [FINAL ACCEPT] {c} (P={probs[c]:.3f})")
                    else:
                        print(f"{indent}  [FINAL REJECT] {c} (P={probs[c]:.3f})")
                break
            
            continue
        
        consecutive_unsat = 0

        assignment_signature_value = assignment_signature(assignment)
        if assignment_signature_value is None:
            print(f"{indent}[WARN] Generated query has no assigned values; skipping")
            continue        

        query_signature_cache.add(assignment_signature_value)
        assignment_snapshot = assignment.copy()
        query_assignments.append(assignment_snapshot)
        
        print(f"{indent}Generated query violating {len(Viol_e)} constraints")
        for c in Viol_e:
            print(f"{indent}  - {c} (P={probs[c]:.3f})")
        
        
        try:
            display_sudoku_grid(Y, title=f"{indent}Violation Query Assignment", debug=False)
        except Exception as e:
            print(f"{indent}Error displaying grid: {e}")
        
        
        print(f"{indent}[ORACLE] Asking...")
        
        
        assignment_for_oracle = variables_to_assignment(Y)
        non_none_assignments = {k: v for k, v in assignment_for_oracle.items()}
        print(f"{indent}[DEBUG] Sending {len(non_none_assignments)} assigned variables to oracle")
        if len(non_none_assignments) <= 10:
            print(f"{indent}[DEBUG] Assignment: {non_none_assignments}")
                
        oracle_vars = getattr(oracle, 'variables_list', None)
        manual_result = manual_sudoku_oracle_check(non_none_assignments, oracle, oracle_vars)
        
        if manual_result is not None:
            answer_raw = manual_result
            print(f"{indent}[MANUAL ORACLE] Result: {'YES (valid)' if answer_raw else 'NO (invalid)'}")
        else:
            
            print(f"{indent}[MANUAL ORACLE] Failed, using standard oracle")
            
            answer_raw = oracle.answer_membership_query(Y)
        
        answer = interpret_oracle_response(answer_raw)
        
        queries_used += 1

        record = {
            'assignment': assignment,
            'assignment_signature': assignment_signature_value,
            'violated_constraints': [str(c) for c in Viol_e],
            'answer': answer,
            'depth': recursion_depth,
            'iteration': iteration,
            'timestamp': time.time()
        }
        query_history.append(record)
        
        if answer:
            
            print(f"{indent}Oracle: YES (valid) - Remove all {len(Viol_e)} violated constraints")
            for c in Viol_e:
                if c in CG:
                    prev_size = len(CG)
                    CG.remove(c)
                    if prev_size == len(CG):
                        raise Exception(f"  Constraint {c} was removed from CG_cand but is still in CG: {CG}")
                    probs[c] *= alpha 
                    print(f"{indent}  [REMOVE] {c} (P={probs[c]:.3f})")

            if assignment_signature_value not in positive_signature_cache:
                positive_signature_cache.add(assignment_signature_value)
                assignment_snapshot = assignment.copy()
                positive_query_examples.append(assignment_snapshot)
                phase1_positive_examples.append(assignment_snapshot)
        else:
            
            print(f"{indent}Oracle: NO (invalid) - Disambiguate {len(Viol_e)} violated constraints")
            negative_query_assignments.append(assignment_snapshot)
            
            if len(Viol_e) == 1 and (len(CG_cand) == 1 or not bias_violated):
                
                c = list(Viol_e)[0]
                print(f"{indent}  [SINGLE VIOLATION] Must be correct: {c}")
                                
                probs[c] = update_supporting_evidence(probs[c], alpha)
                if probs[c] >= theta_max:
                    if c in CG:
                        CG.remove(c)
                        C_val.append(c)
                    print(f"{indent}  [VALIDATE] {c} (P={probs[c]:.3f})")
                    validation_log.append({
                        'constraint': str(c),
                        'method': 'single_violation',
                        'probability': probs[c],
                        'depth': recursion_depth,
                        'iteration': iteration,
                        'query_id': len(query_history) - 1
                    })
                else:
                    print(f"{indent}  [DEFER] {c} (P={probs[c]:.3f} < {theta_max})")
            
            else:
                
                
                
                
                print(f"{indent}[DISAMBIGUATE] Recursively refining {len(Viol_e)} constraints...")

                    
                
                decomposed_viol = []
                for c in Viol_e:
                    decomposed_viol.append(c)
                
                S = get_variables(decomposed_viol)
                print(f"{indent}  Variables in violated constraints: {len(S)}")
                
                C_val_filtered = get_con_subset(C_val, S) if C_val else []
                print(f"{indent}  Relevant validated constraints: {len(C_val_filtered)}/{len(C_val)}")
                
                
                
                other_cg_constraints = [c for c in CG if c not in Viol_e]
                if other_cg_constraints:
                    relevant_other_cg = get_con_subset(other_cg_constraints, S)
                    print(f"{indent}  Relevant non-disambiguated constraints: {len(relevant_other_cg)}/{len(other_cg_constraints)}")
                    C_val_filtered.extend(relevant_other_cg)
                else:
                    print(f"{indent}  No other non-disambiguated constraints to include")
                
                recursive_budget = min(max_queries - queries_used, max(10, (max_queries - queries_used) // 2))
                recursive_timeout = max(10, (timeout - (time.time() - start_time)) / 2)
                
                print(f"{indent}  Recursive budget: {recursive_budget}q, {recursive_timeout:.0f}s")
                
                
                
                recursive_positive_examples = []
                if positive_examples:
                    recursive_positive_examples.extend(positive_examples)
                if positive_query_examples:
                    recursive_positive_examples.extend(positive_query_examples)
                
                C_val_recursive, CG_remaining_recursive, probs_recursive, queries_recursive, E_plus_sub, validation_log_sub = \
                    cop_refinement_recursive(
                        CG_cand=list(Viol_e),
                        C_validated=C_val_filtered,  
                        oracle=oracle,
                        probabilities=probs,
                        all_variables=all_variables,
                        alpha=alpha,
                        theta_max=theta_max,
                        theta_min=theta_min,
                        max_queries=recursive_budget,
                        timeout=recursive_timeout,
                        recursion_depth=recursion_depth + 1,
                        experiment_name=experiment_name,
                        query_signature_cache=query_signature_cache,
                        query_assignments=query_assignments,
                        negative_query_assignments=negative_query_assignments,
                        query_history=query_history,
                        positive_query_examples=[], 
                        positive_signature_cache=positive_signature_cache,
                        phase1_positive_examples=recursive_positive_examples, 
                        B_fixed=B_fixed,
                        bias_weight=bias_weight,
                        validation_log=validation_log  
                    )
                
                queries_used += queries_recursive
                print(f"{indent}[DISAMBIGUATE] Recursive call used {queries_recursive}q")
                
                
                if E_plus_sub:
                    positive_query_examples.extend(E_plus_sub)
                    
                    
                
                
                for c in Viol_e:
                    if c in probs_recursive:
                        probs[c] = probs_recursive[c]
                
                
                
                
                C_val_ids = {id(c) for c in C_val}
                C_val_recursive_ids = {id(c) for c in C_val_recursive}
                Viol_e_ids = {id(c) for c in Viol_e}
                
                ToValidate = [c for c in C_val_recursive if id(c) in Viol_e_ids and id(c) not in C_val_ids]
                
                ToRemove = [c for c in Viol_e if id(c) not in C_val_recursive_ids and c not in CG_remaining_recursive]
                
                print(f"{indent}[DISAMBIGUATE] Results: {len(ToValidate)} validated, {len(ToRemove)} removed")
                
                
                for c in ToValidate:
                    current_prob = probs.get(c, 0.0)
                    if current_prob >= theta_max:
                        if c in CG:
                            CG.remove(c)
                            C_val.append(c)
                        print(f"{indent}  [VALIDATE] {c} (P={current_prob:.3f})")
                        validation_log.append({
                            'constraint': str(c),
                            'method': 'disambiguation',
                            'probability': current_prob,
                            'depth': recursion_depth,
                            'iteration': iteration
                        })
                    else:
                        print(f"{indent}  [DEFER] {c} (P={current_prob:.3f} < {theta_max})")
                
                for c in ToRemove:
                    if c in CG:
                        CG.remove(c)
                    print(f"{indent}  [REMOVE] {c} (P={probs[c]:.3f})")
    duration = time.time() - start_time
    print(f"\n{indent}{'-'*50}")
    print(f"{indent}Refinement [Depth={recursion_depth}] Complete")
    print(f"{indent}Validated: {len(C_val)}, Remaining: {len(CG)}, Queries: {queries_used}, Time: {duration:.2f}s")
    print(f"{indent}{'-'*50}")
    
    return C_val, CG, probs, queries_used, positive_query_examples, validation_log



def cop_based_refinement(experiment_name, oracle, candidate_constraints, initial_probabilities,
                         variables, alpha=0.42, theta_max=0.9, theta_min=0.1, 
                         max_queries=500, timeout=600, phase1_positive_examples=None,
                         B_fixed=None, bias_weight=0.5):
    """
    Wrapper function for the recursive COP-based refinement.
    
    Implements Algorithm: COP-Based Interactive Refinement with Disambiguation.
    Uses recursive COP refinement for principled disambiguation instead of 
    hard-coded engineering approaches.
    """
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"COP-Based Refinement for {experiment_name}")
    print(f"{'='*60}")
    print(f"Initial candidate constraints: {len(candidate_constraints)}")
    print(f"Parameters: alpha={alpha}, theta_max={theta_max}, theta_min={theta_min}")
    print(f"Budget: {max_queries} queries, {timeout}s timeout")
    if B_fixed is not None:
        print(f"B_fixed bias: {len(B_fixed)} constraints, weight={bias_weight}")
    else:
        print(f"B_fixed bias: None")
    print()
    
    phase1_positive_examples = phase1_positive_examples or []
    original_phase1_positive_examples = [copy.deepcopy(example) for example in phase1_positive_examples]
    positive_examples_repository = [copy.deepcopy(example) for example in phase1_positive_examples]

    phase1_positive_signatures = set()
    for example in positive_examples_repository:
        sig = assignment_signature(example)
        if sig is not None:
            phase1_positive_signatures.add(sig)

    query_signature_cache = set(phase1_positive_signatures)
    positive_signature_cache = set(phase1_positive_signatures)
    query_assignments = []
    negative_query_assignments = []
    query_history = []
    positive_query_examples = []

    
    validation_log = []
    C_validated, CG_remaining, probabilities_final, queries_used, _, validation_log = cop_refinement_recursive(
        CG_cand=candidate_constraints,
        C_validated=[],
        oracle=oracle,
        probabilities=initial_probabilities,
        all_variables=variables,
        alpha=alpha,
        theta_max=theta_max,
        theta_min=theta_min,
        max_queries=max_queries,
        timeout=timeout,
        recursion_depth=0,
        experiment_name=experiment_name,
        query_signature_cache=query_signature_cache,
        query_assignments=query_assignments,
        negative_query_assignments=negative_query_assignments,
        query_history=query_history,
        positive_query_examples=positive_query_examples,
        positive_signature_cache=positive_signature_cache,
        phase1_positive_examples=positive_examples_repository,
        B_fixed=B_fixed,
        bias_weight=bias_weight,
        validation_log=validation_log
    )
    
    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"Refinement Complete")
    print(f"{'='*60}")
    print(f"Validated constraints: {len(C_validated)}")
    print(f"Rejected constraints: {len(candidate_constraints) - len(C_validated)}")
    print(f"Uncertain/remaining: {len(CG_remaining)}")
    print(f"Total queries: {queries_used}")
    print(f"Total time: {total_duration:.2f}s")
    print(f"\nValidated constraints:")
    for c in C_validated:
        print(f"  [OK] {c}")
    
    if CG_remaining:
        print(f"\nRemaining uncertain constraints:")
        for c in CG_remaining:
            print(f"  [?] {c} (P={probabilities_final.get(c, 0.5):.3f})")
    
    stats = {
        'queries': queries_used,
        'time': total_duration,
        'validated': len(C_validated),
        'rejected': len(candidate_constraints) - len(C_validated) - len(CG_remaining),
        'uncertain': len(CG_remaining),
        'unique_queries': len(query_signature_cache),
        'positive_queries': len(positive_query_examples),
        'query_history': copy.deepcopy(query_history),
        'phase2_positive_examples': copy.deepcopy(positive_query_examples),
        'query_assignments': copy.deepcopy(query_assignments),
        'negative_query_assignments': copy.deepcopy(negative_query_assignments),
        'query_signature_cache': list(query_signature_cache),
        'positive_examples_repository': copy.deepcopy(positive_examples_repository),
        'phase1_positive_examples_initial': copy.deepcopy(original_phase1_positive_examples)
    }
    
    return C_validated, stats, validation_log


def construct_instance(experiment_name):
    
    if 'graph_coloring_register' in experiment_name.lower() or experiment_name.lower() == 'register':
        result = construct_graph_coloring_register()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'graph_coloring_scheduling' in experiment_name.lower() or experiment_name.lower() == 'scheduling':
        result = construct_graph_coloring_scheduling()

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'latin_square' in experiment_name.lower() or 'latin' in experiment_name.lower():
        result = construct_latin_square(n=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'jsudoku' in experiment_name.lower():
        result = construct_jsudoku(grid_size=9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif ('sudoku' in experiment_name.lower() and 'gt' in experiment_name.lower()) or 'sudoku_greater' in experiment_name.lower():
        if '4x4' in experiment_name.lower():
            print("Constructing Sudoku GT 4x4...")
            result = construct_sudoku_greater_than(2, 2, 4)
        else:
            result = construct_sudoku_greater_than(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'sudoku' in experiment_name.lower():
        result = construct_sudoku(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v1' in experiment_name.lower() or 'examtt_variant1' in experiment_name.lower():
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5, 
                                           slots_per_day=6, days_for_exams=10)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt_v2' in experiment_name.lower() or 'examtt_variant2' in experiment_name.lower():
        result = construct_examtt_variant2(nsemesters=30, courses_per_semester=25, 
                                           slots_per_day=15, days_for_exams=40)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'examtt' in experiment_name.lower():
        result = ces_global(nsemesters=9, courses_per_semester=6, 
                           slots_per_day=9, days_for_exams=14)

        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'nurse' in experiment_name.lower():
        result = nr_global()
        if len(result) == 3:
            instance, oracle, _ = result
        else:
            instance, oracle = result
    
    elif 'uefa' in experiment_name.lower():
        from benchmarks_global.uefa import construct_uefa as construct_uefa_instance
        
        teams_data = {
            "RealMadrid": {"country": "ESP", "coefficient": 134000},
            "BayernMunich": {"country": "GER", "coefficient": 129000},
            "ManchesterCity": {"country": "ENG", "coefficient": 128000},
            "PSG": {"country": "FRA", "coefficient": 112000},
            "Liverpool": {"country": "ENG", "coefficient": 109000},
            "Barcelona": {"country": "ESP", "coefficient": 98000},
            "Juventus": {"country": "ITA", "coefficient": 95000},
            "AtleticoMadrid": {"country": "ESP", "coefficient": 94000},
            "ManchesterUnited": {"country": "ENG", "coefficient": 92000},
            "Chelsea": {"country": "ENG", "coefficient": 91000},
            "BorussiaDortmund": {"country": "GER", "coefficient": 88000},
            "Ajax": {"country": "NED", "coefficient": 82000},
            "RB Leipzig": {"country": "GER", "coefficient": 79000},
            "InterMilan": {"country": "ITA", "coefficient": 76000},
            "Sevilla": {"country": "ESP", "coefficient": 75000},
            "Napoli": {"country": "ITA", "coefficient": 74000},
            "Benfica": {"country": "POR", "coefficient": 73000},
            "Porto": {"country": "POR", "coefficient": 72000},
            "Arsenal": {"country": "ENG", "coefficient": 71000},
            "ACMilan": {"country": "ITA", "coefficient": 70000},
            "RedBullSalzburg": {"country": "AUT", "coefficient": 69000},
            "ShakhtarDonetsk": {"country": "UKR", "coefficient": 68000},
            "BayerLeverkusen": {"country": "GER", "coefficient": 67000},
            "Olympiacos": {"country": "GRE", "coefficient": 66000},
            "Celtic": {"country": "SCO", "coefficient": 65000},
            "Rangers": {"country": "SCO", "coefficient": 64000},
            "PSVEindhoven": {"country": "NED", "coefficient": 63000},
            "SportingCP": {"country": "POR", "coefficient": 62000},
            "Marseille": {"country": "FRA", "coefficient": 61000},
            "ClubBrugge": {"country": "BEL", "coefficient": 60000},
            "Galatasaray": {"country": "TUR", "coefficient": 59000},
            "Feyenoord": {"country": "NED", "coefficient": 58000}
        }
        
        instance, oracle = construct_uefa_instance(teams_data)
    
    elif 'vm_allocation' in experiment_name.lower():
        print("Constructing VM Allocation...")
        from benchmarks_global.vm_allocation import construct_vm_allocation as construct_vm_instance
        from vm_allocation_model import PM_DATA, VM_DATA
        
        instance, oracle = construct_vm_instance(PM_DATA, VM_DATA)
    
    else:
        print(f"Unknown experiment: {experiment_name}")
        sys.exit(1)
    
    return instance, oracle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HCAR AllDifferent Phase 2'
    )
    parser.add_argument('--experiment', type=str, default='sudoku',
                       help='Benchmark name (sudoku, examtt, nurse, uefa, vm_allocation)')
    parser.add_argument('--phase1_pickle', type=str, default=None,
                       help='Path to Phase 1 pickle file (optional)')
    parser.add_argument('--alpha', type=float, default=0.42,
                       help='Bayesian learning rate (default: 0.42)')
    parser.add_argument('--theta_max', type=float, default=0.9,
                       help='Acceptance threshold (default: 0.9)')
    parser.add_argument('--theta_min', type=float, default=0.1,
                       help='Rejection threshold (default: 0.1)')
    parser.add_argument('--max_queries', type=int, default=500,
                       help='Maximum total queries (default: 500)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds (default: 600)')
    parser.add_argument('--prior', type=float, default=0.5)
    parser.add_argument('--bias_weight', type=float, default=0.5,
                       help='Weight for B_fixed bias violations in COP objective (default: 100)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"HCAR AllDifferent COP Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Alpha: {args.alpha}")
    print(f"Theta_max: {args.theta_max}")
    print(f"Theta_min: {args.theta_min}")
    print(f"Max queries: {args.max_queries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Prior: {args.prior}")
    print(f"Bias weight: {args.bias_weight}")
    print(f"{'='*60}\n")

    instance, oracle = construct_instance(args.experiment)

    oracle.variables_list = cpm_array(instance.X)

    
    
    CG = extract_alldifferent_constraints(oracle)
    print(f"\nExtracted {len(CG)} AllDifferent constraints from oracle")
    
    phase1_data = None
    B_fixed = None
    if args.phase1_pickle:
        phase1_data = load_phase1_data(args.phase1_pickle)
        
        
        B_fixed = phase1_data.get('B_fixed', None)
        if B_fixed is not None:
            print(f"\nExtracted B_fixed from Phase 1: {len(B_fixed)} binary constraints")
        else:
            print(f"\nNo B_fixed found in Phase 1 data")
        
        if 'initial_probabilities' in phase1_data:
            
            old_probs = phase1_data['initial_probabilities']
            probabilities = {}
            
            
            old_prob_map = {str(c): p for c, p in old_probs.items()}
            
            
            for c in phase1_data['CG']:
                print(c)
                c_str = str(c)
                if c_str in old_prob_map:
                    probabilities[c] = old_prob_map[c_str]
                else:
                    probabilities[c] = args.prior
            
            print(f"Mapped {len(probabilities)} probabilities from Phase 1")
        else:
            probabilities = initialize_probabilities(CG, prior=args.prior)
            print(f"No initial_probabilities in pickle, using uniform prior={args.prior}")
    else:
        probabilities = initialize_probabilities(CG, prior=args.prior)
    
    if len(phase1_data['CG']) == 0:
        print(f"\n No AllDifferent constraints found")
        sys.exit(0)

    C_validated, stats, validation_log = cop_based_refinement(
        experiment_name=args.experiment,
        oracle=oracle,
        candidate_constraints=phase1_data['CG'],
        initial_probabilities=probabilities,
        variables=instance.X,
        alpha=args.alpha,
        theta_max=args.theta_max,
        theta_min=args.theta_min,
        max_queries=args.max_queries,
        timeout=args.timeout,
        B_fixed=B_fixed,
        bias_weight=args.bias_weight
    )

    print(f"\n{'='*60}")
    print(f"Comparison with Target Model")
    print(f"{'='*60}")
    
    target_alldiff = extract_alldifferent_constraints(oracle)
    target_strs = set(str(c) for c in target_alldiff)
    learned_strs = set(str(c) for c in C_validated)
    
    correct = len(learned_strs & target_strs)
    missing = len(target_strs - learned_strs)
    spurious = len(learned_strs - target_strs)
    
    print(f"Target AllDifferent constraints: {len(target_alldiff)}")
    print(f"Learned AllDifferent constraints: {len(C_validated)}")
    print(f"Correct: {correct}")
    print(f"Missing: {missing}")
    print(f"Spurious: {spurious}")
    
    if correct == len(target_alldiff) and spurious == 0:
        print(f"\n[SUCCESS] Perfect learning!")
    else:
        if missing > 0:
            print(f"\n[ERROR] Missing constraints:")
            for c in target_alldiff:
                if str(c) not in learned_strs:
                    print(f"  - {c}")
        
        if spurious > 0:
            print(f"\n[ERROR] Spurious constraints:")
            for c in C_validated:
                if str(c) not in target_strs:
                    print(f"  - {c}")

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
    elif not C_validated:
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
        print(f"Validated constraints to check: {len(C_validated)}")

        for idx, constraint in enumerate(C_validated, start=1):
            violation_expr = build_constraint_violation(constraint)

            test_model = Model()
            test_model += base_constraints
            test_model += violation_expr

            has_counterexample = test_model.solve()

            if has_counterexample:
                assignment = variables_to_assignment(instance.X)
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
              f"not_implied={not_implied_count}, checked={len(C_validated)}")

        cp_implication_results = {
            'skipped': False,
            'checked': len(C_validated),
            'implied': implied,
            'not_implied': counterexamples,
            'status': 'all_implied' if not not_implied else 'partial',
            'implied_count': implied_count,
            'not_implied_count': not_implied_count
        }

    if isinstance(stats, dict):
        cp_implication_results['target_constraint_count'] = len(target_constraint_list)
        stats['cp_implication'] = cp_implication_results
    
    print(f"\n{'='*60}")
    print(f"Final Statistics")
    print(f"{'='*60}")
    print(f"Total queries: {stats['queries']}")
    print(f"Total time: {stats['time']:.2f}s")
    print(f"Queries per second: {stats['queries']/stats['time']:.2f}")
    print(f"{'='*60}\n")
    
    
    if validation_log:
        print(f"\n{'='*60}")
        print(f"Validation Log")
        print(f"{'='*60}")
        print(f"Total validations: {len(validation_log)}\n")
        
        
        by_method = {}
        for entry in validation_log:
            method = entry['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(entry)
        
        for method, entries in sorted(by_method.items()):
            print(f"\n{method.upper().replace('_', ' ')} ({len(entries)} constraints):")
            print("-" * 60)
            for entry in entries:
                constraint_str = entry['constraint']
                prob = entry.get('probability', 0.0)
                depth = entry.get('depth', 0)
                iteration = entry.get('iteration', 0)
                
                
                if len(constraint_str) > 80:
                    constraint_display = constraint_str[:77] + "..."
                else:
                    constraint_display = constraint_str
                
                info = f"  [D={depth},I={iteration},P={prob:.3f}] {constraint_display}"
                print(info)
        
        print(f"\n{'='*60}\n")

    phase2_output = {
        'C_validated': C_validated,  
        'C_validated_strs': [str(c) for c in C_validated],  
        'probabilities': probabilities,  
        'experiment_name': args.experiment,
        'phase2_stats': stats,

        'phase1_data': phase1_data if args.phase1_pickle else None,
        'E_plus': phase1_data['E_plus'] if args.phase1_pickle and 'E_plus' in phase1_data else None,
        'B_fixed': phase1_data['B_fixed'] if args.phase1_pickle and 'B_fixed' in phase1_data else None,
        'all_variables': list(instance.X),
        'query_assignments': stats.get('query_assignments', []),
        'negative_query_assignments': stats.get('negative_query_assignments', []),
        'phase1_positive_examples_initial': stats.get('phase1_positive_examples_initial', []),
        'phase2_positive_examples': stats.get('phase2_positive_examples', []),
        'positive_examples_repository': stats.get('positive_examples_repository', []),
        'query_signature_cache': stats.get('query_signature_cache', []),
        'metadata': {
            'alpha': args.alpha,
            'theta_max': args.theta_max,
            'theta_min': args.theta_min,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_queries': stats['queries'],
            'total_time': stats['time']
        }
    }

    phase2_output_dir = "phase2_output"
    os.makedirs(phase2_output_dir, exist_ok=True)
    phase2_pickle_path = os.path.join(phase2_output_dir, f"{args.experiment}_phase2.pkl")
    
    with open(phase2_pickle_path, 'wb') as f:
        pickle.dump(phase2_output, f)
    
    positive_examples_path = os.path.join(phase2_output_dir, f"{args.experiment}_positive_examples.pkl")
    positive_examples_payload = {
        'phase1_initial': stats.get('phase1_positive_examples_initial', []),
        'phase2_positive_examples': stats.get('phase2_positive_examples', []),
        'merged_positive_examples': stats.get('positive_examples_repository', [])
    }

    with open(positive_examples_path, 'wb') as f:
        pickle.dump(positive_examples_payload, f)

    cp_implication_log_path = os.path.join(phase2_output_dir, f"{args.experiment}_cp_implication.log")
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

    query_history_path = os.path.join(phase2_output_dir, f"{args.experiment}_query_history.pkl")
    with open(query_history_path, 'wb') as f:
        pickle.dump({
            'query_history': stats.get('query_history', []),
            'query_assignments': stats.get('query_assignments', []),
            'negative_query_assignments': stats.get('negative_query_assignments', []),
            'query_signature_cache': stats.get('query_signature_cache', [])
        }, f)

    print(f"\n Phase 2 outputs saved to: {phase2_pickle_path}")
    print(f"  - Validated constraints: {len(C_validated)}")
    print(f"  - Positive examples saved to: {positive_examples_path}")
    print(f"  - Query log saved to: {query_history_path}")
    print(f"  - CP implication log saved to: {cp_implication_log_path}")
