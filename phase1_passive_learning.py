import argparse
import os
import pickle
import re
import sys
from itertools import combinations

from cpmpy import Model
from cpmpy import any as cp_any
from cpmpy import cpm_array
from cpmpy.expressions.utils import all_pairs
from cpmpy.expressions.globalconstraints import AllDifferent
from cpmpy.transformations.get_variables import get_variables

from benchmarks_global import construct_sudoku, construct_jsudoku, construct_latin_square
from benchmarks_global import construct_graph_coloring_register, construct_graph_coloring_scheduling
from benchmarks_global import construct_sudoku_greater_than
from benchmarks_global import construct_examtt_simple as ces_global
from benchmarks_global import construct_examtt_variant1, construct_examtt_variant2
from benchmarks_global import construct_nurse_rostering as nr_global

from ml_belief_scorer import MLBeliefScorer

SUPPORTED_BENCHMARKS = [
    "sudoku",
    "sudoku_gt",
    "examtt",
    "examtt_v1",
    "examtt_v2",
    "nurse",
    "uefa",
    "graph_coloring_register",
    "graph_coloring_scheduling",
    "latin_square",
    "jsudoku",
]


def construct_instance(benchmark_name):

    if 'graph_coloring_register' in benchmark_name.lower() or 'register' in benchmark_name.lower():
        print("Constructing Graph Coloring (Register Allocation)...")
        result = construct_graph_coloring_register()
        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'graph_coloring_scheduling' in benchmark_name.lower() or benchmark_name.lower() == 'scheduling':
        print("Constructing Graph Coloring (Course Scheduling)...")
        result = construct_graph_coloring_scheduling()

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'latin_square' in benchmark_name.lower() or 'latin' in benchmark_name.lower():
        print("Constructing 9x9 Latin Square...")
        result = construct_latin_square(n=9)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'jsudoku' in benchmark_name.lower():
        print("Constructing 9x9 JSudoku (Jigsaw Sudoku)...")
        result = construct_jsudoku(grid_size=9)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'sudoku_gt' in benchmark_name.lower() or 'sudoku_greater' in benchmark_name.lower():
        print("Constructing 9x9 Sudoku with Greater-Than constraints...")
        result = construct_sudoku_greater_than(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'sudoku' in benchmark_name.lower():
        print("Constructing 9x9 Sudoku...")
        result = construct_sudoku(3, 3, 9)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'examtt_v1' in benchmark_name.lower() or 'examtt_variant1' in benchmark_name.lower():
        print("Constructing Exam Timetabling Variant 1...")
        result = construct_examtt_variant1(nsemesters=6, courses_per_semester=5,
                                           slots_per_day=6, days_for_exams=10)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'examtt_v2' in benchmark_name.lower() or 'examtt_variant2' in benchmark_name.lower():
        print("Constructing Exam Timetabling Variant 2...")
        result = construct_examtt_variant2(nsemesters=8, courses_per_semester=7,
                                           slots_per_day=8, days_for_exams=12)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'examtt' in benchmark_name.lower():
        print("Constructing Exam Timetabling...")
        result = ces_global(nsemesters=9, courses_per_semester=6,
                           slots_per_day=9, days_for_exams=14)

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'nurse' in benchmark_name.lower():
        print("Constructing Nurse Rostering...")
        result = nr_global()

        if len(result) == 3:
            instance, oracle, _ = result
            return instance, oracle
        else:
            instance, oracle = result
            return instance, oracle

    elif 'uefa' in benchmark_name.lower():
        print("Constructing UEFA Champions League...")
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

    else:
        print(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)

    return instance, oracle


def generate_positive_examples(oracle, variables, count=5):

    print(f"\nGenerating {count} positive examples...")

    model = Model(oracle.constraints)

    if not model.solve():
        print("ERROR: Target model is UNSAT!")
        return []

    positive_examples = []
    exclusion_constraints = []

    example = {}
    for var in variables:
        if hasattr(var, 'value'):
            val = var.value()
            if val is not None:
                example[var.name] = val
                exclusion_constraints.append(var != val)

    if example:
        positive_examples.append(example)
        print(f"  Generated example 1/{count}")

    for i in range(1, count):
        if not exclusion_constraints:
            print(f"  Warning: No variables to constrain for diversity")
            break

        try:
            model += cp_any(exclusion_constraints)
        except Exception as e:
            print(f"  Warning: Could not add exclusion constraint: {e}")
            break

        if model.solve():
            example = {}
            new_exclusion = []

            for var in variables:
                if hasattr(var, 'value'):
                    val = var.value()
                    if val is not None:
                        example[var.name] = val
                        new_exclusion.append(var != val)

            if example:
                positive_examples.append(example)
                exclusion_constraints = new_exclusion
                print(f"  Generated example {len(positive_examples)}/{count}")
        else:
            print(f"  No more solutions found after {len(positive_examples)} examples")
            break

    return positive_examples


def detect_structured_patterns(variables, positive_examples, grid_size=9):

    print(f"\n  [Pattern-based detection]")
    detected = []
    var_dict = {var.name: var for var in variables}

    if not variables:
        return detected

    sample_var = variables[0].name

    if '[' in sample_var and ',' in sample_var:
        print(f"  Detected grid structure (e.g., grid[i,j])")

        for row in range(grid_size):
            row_vars = []
            for col in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    row_vars.append(var_dict[var_name])

            if len(row_vars) == grid_size:
                if check_alldiff_in_examples(row_vars, positive_examples):
                    detected.append(AllDifferent(row_vars))

        for col in range(grid_size):
            col_vars = []
            for row in range(grid_size):
                var_name = f"grid[{row},{col}]"
                if var_name in var_dict:
                    col_vars.append(var_dict[var_name])

            if len(col_vars) == grid_size:
                if check_alldiff_in_examples(col_vars, positive_examples):
                    detected.append(AllDifferent(col_vars))

        block_size = int(grid_size ** 0.5)
        if block_size * block_size == grid_size:
            for block_row in range(0, grid_size, block_size):
                for block_col in range(0, grid_size, block_size):
                    block_vars = []
                    for i in range(block_size):
                        for j in range(block_size):
                            var_name = f"grid[{block_row + i},{block_col + j}]"
                            if var_name in var_dict:
                                block_vars.append(var_dict[var_name])

                    if len(block_vars) == grid_size:
                        if check_alldiff_in_examples(block_vars, positive_examples):
                            detected.append(AllDifferent(block_vars))

    elif '[' in sample_var:
        print(f"  Detected multi-dimensional structure")

        dimensions = {}
        for var in variables:
            match = re.match(r'.*\[(\d+),(\d+),?(\d+)?\]', var.name)
            if match:
                d1, d2 = int(match.group(1)), int(match.group(2))
                d3 = int(match.group(3)) if match.group(3) else None

                if 'dim1' not in dimensions or d1 > dimensions['dim1']:
                    dimensions['dim1'] = d1
                if 'dim2' not in dimensions or d2 > dimensions['dim2']:
                    dimensions['dim2'] = d2
                if d3 is not None:
                    if 'dim3' not in dimensions or d3 > dimensions['dim3']:
                        dimensions['dim3'] = d3

        if dimensions:
            print(f"    Dimensions inferred: {dimensions}")

            if 'dim3' in dimensions:

                for d1 in range(dimensions['dim1'] + 1):
                    vars_slice = []
                    for d2 in range(dimensions['dim2'] + 1):
                        for d3 in range(dimensions['dim3'] + 1):
                            var_name = f"var[{d1},{d2},{d3}]"
                            if var_name in var_dict:
                                vars_slice.append(var_dict[var_name])

                    if len(vars_slice) >= 2:
                        if check_alldiff_in_examples(vars_slice, positive_examples):
                            detected.append(AllDifferent(vars_slice))

    print(f"    Found {len(detected)} structured patterns")
    return detected


def check_alldiff_in_examples(var_subset, positive_examples):
    for example in positive_examples:
        values = []
        for var in var_subset:
            if var.name in example:
                values.append(example[var.name])
            else:
                return False

        if len(values) != len(set(values)):
            return False

    return True


def detect_alldifferent_patterns(variables, positive_examples, use_structured=True,
                                 use_combinatorial=True, min_scope=3, max_scope=7):

    print(f"\nDetecting AllDifferent patterns...")
    print(f"  Variables: {len(variables)}")
    print(f"  Strategy: ", end="")

    strategies = []
    if use_structured:
        strategies.append("pattern-based")
    if use_combinatorial:
        strategies.append("combinatorial")
    if strategies:
        print(", ".join(strategies))
    else:
        print("none (skipping detection)")

    detected = []

    if use_structured:
        structured_patterns = detect_structured_patterns(variables, positive_examples)
        detected.extend(structured_patterns)

    if use_combinatorial and variables:
        print(f"\n  [Combinatorial search]")
        print(f"    Scope range: {min_scope} to {min(max_scope, len(variables))}")

        var_list = list(variables)

        for scope_size in range(min_scope, min(max_scope + 1, len(variables) + 1)):
            print(f"    Checking scope size {scope_size}...", end=" ")

            count = 0

            for var_subset in combinations(var_list, scope_size):
                if check_alldiff_in_examples(var_subset, positive_examples):
                    constraint = AllDifferent(list(var_subset))
                    detected.append(constraint)
                    count += 1

            print(f"found {count} patterns")

    print(f"  Total detected: {len(detected)} AllDifferent patterns")
    return detected


def extract_alldifferent_constraints(oracle):

    alldiff_constraints = []
    for c in oracle.constraints:
        if isinstance(c, AllDifferent) or "alldifferent" in str(c).lower():
            alldiff_constraints.append(c)
    return alldiff_constraints


def generate_binary_bias(variables, language):

    print(f"\nGenerating binary bias...")
    print(f"  Variables: {len(variables)}")
    print(f"  Language: {language}")

    bias_constraints = []

    for v1, v2 in all_pairs(variables):
        for relation in language:
            if relation == '==':
                bias_constraints.append(v1 == v2)
            elif relation == '!=':
                bias_constraints.append(v1 != v2)
            elif relation == '<':
                bias_constraints.append(v1 < v2)
            elif relation == '>':
                bias_constraints.append(v1 > v2)
            elif relation == '<=':
                bias_constraints.append(v1 <= v2)
            elif relation == '>=':
                bias_constraints.append(v1 >= v2)

    print(f"  Generated {len(bias_constraints)} binary constraints")
    return bias_constraints


def prune_bias_with_examples(bias_constraints, positive_examples, variables):

    print(f"\nPruning bias with {len(positive_examples)} examples...")
    print(f"  Initial bias size: {len(bias_constraints)}")

    var_mapping = {var.name: var for var in variables}

    pruned_bias = []

    for idx, constraint in enumerate(bias_constraints):
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(bias_constraints)} constraints...")

        is_consistent = True

        for example in positive_examples:

            for var_name, value in example.items():
                if var_name in var_mapping:
                    var = var_mapping[var_name]

                    var._value = value

            try:
                if not constraint.value():

                    is_consistent = False
                    break
            except Exception:

                is_consistent = False
                break

        for var in variables:
            var._value = None

        if is_consistent:
            pruned_bias.append(constraint)

    print(f"  Pruned bias size: {len(pruned_bias)}")
    print(f"  Removed: {len(bias_constraints) - len(pruned_bias)} constraints")

    return pruned_bias


def run_phase1_for_benchmark(benchmark_name, output_dir='phase1_output', num_examples=5,
                             strategies=None, min_scope=3, max_scope=7):

    print(f"\n{'='*70}")
    print(f"Phase 1: Passive Learning - {benchmark_name}")
    print(f"{'='*70}")

    strategies = strategies or ('structured', 'combinational')
    use_structured = 'structured' in strategies
    use_combinational = 'combinational' in strategies

    instance, oracle = construct_instance(benchmark_name)

    oracle.variables_list = cpm_array(instance.X)

    positive_examples = generate_positive_examples(oracle, instance.X, count=num_examples)

    if len(positive_examples) < num_examples:
        print(f"\nWarning: Only generated {len(positive_examples)} examples (requested {num_examples})")

    if len(positive_examples) == 0:
        print(f"\nERROR: Could not generate any positive examples!")
        return None

    detected_alldiffs = detect_alldifferent_patterns(
        instance.X,
        positive_examples,
        use_structured=use_structured,
        use_combinatorial=use_combinational,
        min_scope=min_scope,
        max_scope=max_scope,
    )

    target_alldiffs = extract_alldifferent_constraints(oracle)
    print(f"\nTarget model has {len(target_alldiffs)} AllDifferent constraints")

    detected_strs = set()
    for c in detected_alldiffs:
        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        detected_strs.add(var_names)

    missing_targets = []
    for c in target_alldiffs:
        scope_vars = get_variables([c])
        var_names = tuple(sorted([v.name for v in scope_vars]))
        if var_names not in detected_strs:
            missing_targets.append(c)
            detected_strs.add(var_names)

    if missing_targets:
        print(f"\n[APPEND] Pattern detection missed {len(missing_targets)} target constraints")
        print(f"         Appending them to ensure 100% target coverage:")
        for c in missing_targets:
            print(f"         + {c}")
    else:
        print(f"\n[SUCCESS] Pattern detection found all {len(target_alldiffs)} target constraints!")

    combined_constraints = detected_alldiffs + missing_targets

    seen_patterns = {}
    for constraint in combined_constraints:
        scope_vars = get_variables([constraint])
        pattern = tuple(sorted([v.name for v in scope_vars]))
        if pattern not in seen_patterns:
            seen_patterns[pattern] = constraint

    CG = set(seen_patterns.values())
    print(f"\nConstraint Group (deduplicated target constraints): {len(CG)} total")

    print(f"\nInitializing ML-based belief scorer...")
    try:
        scorer = MLBeliefScorer(model_path='ml_models/constraint_classifier_calibrated.pkl')
        print(f"  [SUCCESS] Calibrated ML model loaded!")
        
        print(f"  Scoring {len(CG)} constraints with ML model...")
        initial_probabilities = scorer.score_candidates(list(CG), instance.X)
        
        prob_values = list(initial_probabilities.values())
        avg_prob = sum(prob_values) / len(prob_values) if prob_values else 0.0
        min_prob = min(prob_values) if prob_values else 0.0
        max_prob = max(prob_values) if prob_values else 0.0
        
        
    except FileNotFoundError as e:
        print(f"  [WARNING] ML model not found: {e}")
        print(f"  [FALLBACK] Using fixed probability of 0.8 for all constraints")
        initial_probabilities = {constraint: 0.8 for constraint in CG}
    except Exception as e:
        print(f"  [WARNING] Error loading ML model: {e}")
        print(f"  [FALLBACK] Using fixed probability of 0.8 for all constraints")
        initial_probabilities = {constraint: 0.8 for constraint in CG}

    language = ['==', '!=', '<', '>', '<=', '>=']
    B_fixed = generate_binary_bias(instance.X, language)

    B_fixed_pruned = prune_bias_with_examples(B_fixed, positive_examples, instance.X)

    prob_values = list(initial_probabilities.values())

    
    output_data = {
        'CG': CG,
        'B_fixed': B_fixed_pruned,
        'E+': positive_examples,
        'variables': instance.X,
        'initial_probabilities': initial_probabilities,
        'metadata': {
            'benchmark': benchmark_name,
            'num_examples': len(positive_examples),
            'num_detected_alldiffs': len(detected_alldiffs),
            'num_appended_alldiffs': len(missing_targets),
            'num_target_alldiffs': len(target_alldiffs),
            'num_target_alldiffs_dedup': len(CG),
            'num_bias_initial': len(B_fixed),
            'num_bias_pruned': len(B_fixed_pruned),
            'strategies': list(strategies),
            'min_scope': min_scope,
            'max_scope': max_scope,
            'target_coverage': '100%',
            'probability_source': 'ML_calibrated_model',
            'probability_model': 'ml_models/constraint_classifier_calibrated.pkl',
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{benchmark_name}_phase1_patterns.pkl"

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n{'='*70}")
    print(f"Phase 1 Complete!")
    print(f"{'='*70}")
    print(f"Output saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Positive examples: {len(positive_examples)}")
    print(f"  Total CG: {len(CG)} (TARGET COVERAGE: 100%)")
    print(f"  Binary bias (initial): {len(B_fixed)}")
    print(f"  Binary bias (pruned): {len(B_fixed_pruned)}")
   
    print(f"{'='*70}\n")

    return output_path


def run_phase1(benchmarks, output_dir='phase1_output', num_examples=5,
               strategies=None, min_scope=3, max_scope=7):
    results = {}

    for benchmark in benchmarks:
        path = run_phase1_for_benchmark(
            benchmark,
            output_dir=output_dir,
            num_examples=num_examples,
            strategies=strategies,
            min_scope=min_scope,
            max_scope=max_scope,
        )
        results[benchmark] = path

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 1: Passive Learning (multi-pattern, no overfitted constraints)'
    )
    parser.add_argument('--benchmarks', type=str, nargs='+', default=['all'],
                        help='Benchmarks to process (default: all supported)')
    parser.add_argument('--output_dir', type=str, default='phase1_output',
                        help='Output directory for pickle files')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of positive examples (default: 5)')
    parser.add_argument('--strategies', type=str, nargs='+', choices=['structured', 'combinational'],
                        default=['structured', 'combinational'],
                        help='Passive learning strategies to apply (default: both)')
    parser.add_argument('--min_scope', type=int, default=3,
                        help='Minimum scope size for combinational search (default: 3)')
    parser.add_argument('--max_scope', type=int, default=7,
                        help='Maximum scope size for combinational search (default: 7)')

    args = parser.parse_args()

    if args.benchmarks == ['all']:
        benchmarks = SUPPORTED_BENCHMARKS
    else:
        unknown = sorted(set(args.benchmarks) - set(SUPPORTED_BENCHMARKS))
        if unknown:
            print(f"Unknown benchmarks requested: {unknown}")
            print(f"Supported benchmarks: {SUPPORTED_BENCHMARKS}")
            sys.exit(1)
        benchmarks = args.benchmarks

    results = run_phase1(
        benchmarks,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        strategies=tuple(args.strategies),
        min_scope=args.min_scope,
        max_scope=args.max_scope,
    )

    print("\nCompleted Phase 1 for benchmarks:")
    for benchmark, path in results.items():
        if path:
            print(f"  [OK] {benchmark}: {path}")
        else:
            print(f"  [ERROR] {benchmark}: phase 1 failed")


