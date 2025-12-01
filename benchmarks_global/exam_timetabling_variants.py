import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_variant1(nsemesters=6, courses_per_semester=5, slots_per_day=6, days_for_exams=10):
    
    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams,
        'variant': 1
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()


    model += cp.AllDifferent(variables)

    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)

    all_exams = variables.flatten()
    max_exams_per_day = (total_courses + days_for_exams - 1) // days_for_exams + 1
    for day in range(days_for_exams):
        exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
        model += (exams_on_day <= max_exams_per_day)

    C_T = list(model.constraints)




    overfitted_constraints = []



    if nsemesters >= 5 and courses_per_semester >= 3:
        non_adj_vars = []
        for sem in [0, 2, 4]:
            for course in range(3):
                if sem < nsemesters and course < courses_per_semester:
                    non_adj_vars.append(variables[sem, course])
        if len(non_adj_vars) >= 9:
            overfitted_c1 = cp.AllDifferent(non_adj_vars)
            overfitted_constraints.append(overfitted_c1)
            model += overfitted_c1


    if nsemesters >= 5:
        odd_sems = [variables[1, :].flatten(), variables[3, :].flatten()]
        if nsemesters >= 5:
            odd_sems.append(variables[5, :].flatten() if nsemesters > 5 else variables[4, :].flatten())
        odd_combined = []
        for sem in odd_sems:
            odd_combined.extend(list(sem))
        overfitted_c2 = cp.AllDifferent(odd_combined[:min(12, len(odd_combined))])
        overfitted_constraints.append(overfitted_c2)
        model += overfitted_c2


    if nsemesters >= 5 and courses_per_semester >= 2:
        first_two_cols = variables[:, :2].flatten()
        overfitted_c3 = cp.AllDifferent(first_two_cols)
        overfitted_constraints.append(overfitted_c3)
        model += overfitted_c3


    if nsemesters >= 5:
        even_sem_vars = []
        for sem in [0, 2, 4]:
            if sem < nsemesters:
                even_sem_vars.extend(list(variables[sem, :].flatten()))
        overfitted_c4 = cp.AllDifferent(even_sem_vars)
        overfitted_constraints.append(overfitted_c4)
        model += overfitted_c4


    if nsemesters >= 4 and courses_per_semester >= 3:
        last_three_cols = variables[:4, -3:].flatten()
        overfitted_c5 = cp.AllDifferent(last_three_cols)
        overfitted_constraints.append(overfitted_c5)
        model += overfitted_c5

    if nsemesters >= 5 and courses_per_semester >= 5:
        diagonal = [variables[i, i] for i in range(min(5, nsemesters, courses_per_semester))]
        anti_diag = [variables[i, courses_per_semester - 1 - i] 
                     for i in range(min(5, nsemesters, courses_per_semester))]
        combined = diagonal + anti_diag
        overfitted_c6 = cp.AllDifferent(combined)
        overfitted_constraints.append(overfitted_c6)
        model += overfitted_c6


    if nsemesters >= 5 and courses_per_semester >= 3:
        mid_sem = nsemesters // 2
        mid_course = courses_per_semester // 2
        middle_block = []
        for s_offset in [-1, 0, 1]:
            sem_idx = mid_sem + s_offset
            if 0 <= sem_idx < nsemesters:
                for c_offset in [-1, 0, 1]:
                    course_idx = mid_course + c_offset
                    if 0 <= course_idx < courses_per_semester:
                        middle_block.append(variables[sem_idx, course_idx])
        if len(middle_block) >= 7:
            overfitted_c7 = cp.AllDifferent(middle_block)
            overfitted_constraints.append(overfitted_c7)
            model += overfitted_c7


    if nsemesters >= 6 and courses_per_semester >= 4:
        checkerboard = []
        for sem in range(0, min(6, nsemesters), 2):  
            for course in range(0, min(4, courses_per_semester), 2):  
                checkerboard.append(variables[sem, course])
        for sem in range(1, min(6, nsemesters), 2):  
            for course in range(1, min(4, courses_per_semester), 2):  
                if sem < nsemesters and course < courses_per_semester:
                    checkerboard.append(variables[sem, course])
        if len(checkerboard) >= 8:
            overfitted_c8 = cp.AllDifferent(checkerboard)
            overfitted_constraints.append(overfitted_c8)
            model += overfitted_c8

    if nsemesters >= 10 and courses_per_semester >= 8:
        first_ten_sems = variables[:10, :].flatten()
        overfitted_c9 = cp.AllDifferent(first_ten_sems[:min(80, len(first_ten_sems))])
        overfitted_constraints.append(overfitted_c9)
        model += overfitted_c9

    if nsemesters >= 12 and courses_per_semester >= 10:
        large_diagonal = [variables[i, i] for i in range(min(12, nsemesters, courses_per_semester))]
        overfitted_c10 = cp.AllDifferent(large_diagonal)
        overfitted_constraints.append(overfitted_c10)
        model += overfitted_c10

    if nsemesters >= 12 and courses_per_semester >= 10:
        large_anti_diag = [variables[i, courses_per_semester - 1 - i] 
                          for i in range(min(12, nsemesters, courses_per_semester))]
        overfitted_c11 = cp.AllDifferent(large_anti_diag)
        overfitted_constraints.append(overfitted_c11)
        model += overfitted_c11

    if nsemesters >= 15 and courses_per_semester >= 12:
        quarter_subset = variables[:15, :12].flatten()
        overfitted_c12 = cp.AllDifferent(quarter_subset[:min(180, len(quarter_subset))])
        overfitted_constraints.append(overfitted_c12)
        model += overfitted_c12

    if nsemesters >= 20 and courses_per_semester >= 15:
        first_half_sems = variables[:nsemesters//2, :].flatten()
        overfitted_c13 = cp.AllDifferent(first_half_sems[:min(150, len(first_half_sems))])
        overfitted_constraints.append(overfitted_c13)
        model += overfitted_c13

    if nsemesters >= 15 and courses_per_semester >= 10:
        every_third_sem = variables[::3, :].flatten()
        overfitted_c14 = cp.AllDifferent(every_third_sem[:min(50, len(every_third_sem))])
        overfitted_constraints.append(overfitted_c14)
        model += overfitted_c14

    if nsemesters >= 12 and courses_per_semester >= 8:
        last_eight_cols = variables[:, -8:].flatten()
        overfitted_c15 = cp.AllDifferent(last_eight_cols[:min(96, len(last_eight_cols))])
        overfitted_constraints.append(overfitted_c15)
        model += overfitted_c15

    if nsemesters >= 18 and courses_per_semester >= 12:
        middle_block = variables[nsemesters//3:2*nsemesters//3, courses_per_semester//3:2*courses_per_semester//3].flatten()
        overfitted_c16 = cp.AllDifferent(middle_block[:min(72, len(middle_block))])
        overfitted_constraints.append(overfitted_c16)
        model += overfitted_c16

    AV = absvar(2)

    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day),
        cp.Count(AV, AV[0]) <= AV[1],
        cp.Count(AV, AV[0]) == AV[1]
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_v1_sem{nsemesters}_c{courses_per_semester}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints


def construct_examtt_variant2(nsemesters=30, courses_per_semester=25, slots_per_day=15, days_for_exams=60):
    
    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams,
        'variant': 2
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()


    model += cp.AllDifferent(variables)

    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)

    all_exams = variables.flatten()
    max_exams_per_day = (total_courses + days_for_exams - 1) // days_for_exams + 1
    for day in range(days_for_exams):
        exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
        model += (exams_on_day <= max_exams_per_day)

    C_T = list(model.constraints)


    overfitted_constraints = []

    if nsemesters >= 3:
        first_three_sems = variables[:3, :].flatten()
        first_three_days = [day_of_exam(exam, slots_per_day) for exam in first_three_sems]
        overfitted_c1 = cp.AllDifferent(first_three_days)
        overfitted_constraints.append(overfitted_c1)
        model += overfitted_c1

    if nsemesters >= 4 and courses_per_semester >= 2:
        first_four = variables[:4, :].flatten()
        time_slots = [exam % slots_per_day for exam in first_four]
        overfitted_c2 = cp.AllDifferent(time_slots[:min(slots_per_day, len(time_slots))])
        overfitted_constraints.append(overfitted_c2)
        model += overfitted_c2

    if nsemesters >= 5 and courses_per_semester >= 2:
        first_two_cols = variables[:, :2].flatten()
        overfitted_c3 = cp.AllDifferent(first_two_cols[:min(15, len(first_two_cols))])
        overfitted_constraints.append(overfitted_c3)
        model += overfitted_c3

    if nsemesters >= 5 and courses_per_semester >= 5:
        diagonal = [variables[i, i] for i in range(min(5, nsemesters, courses_per_semester))]
        overfitted_c4 = cp.AllDifferent(diagonal)
        overfitted_constraints.append(overfitted_c4)
        model += overfitted_c4

    if nsemesters >= 5 and courses_per_semester >= 5:
        anti_diag = [variables[i, courses_per_semester - 1 - i] 
                     for i in range(min(5, nsemesters, courses_per_semester))]
        overfitted_c5 = cp.AllDifferent(anti_diag)
        overfitted_constraints.append(overfitted_c5)
        model += overfitted_c5

    if nsemesters >= 6 and courses_per_semester >= 3:
        even_subset = variables[::2, :3].flatten()
        overfitted_c6 = cp.AllDifferent(even_subset[:min(12, len(even_subset))])
        overfitted_constraints.append(overfitted_c6)
        model += overfitted_c6

    if nsemesters >= 5 and courses_per_semester >= 3:
        odd_subset = variables[1::2, -3:].flatten()
        overfitted_c7 = cp.AllDifferent(odd_subset[:min(12, len(odd_subset))])
        overfitted_constraints.append(overfitted_c7)
        model += overfitted_c7

    if nsemesters >= 6 and courses_per_semester >= 1:
        first_col = [variables[sem, 0] for sem in range(min(6, nsemesters))]
        overfitted_c8 = cp.AllDifferent(first_col)
        overfitted_constraints.append(overfitted_c8)
        model += overfitted_c8

    if nsemesters >= 5 and courses_per_semester >= 2:
        last_col = [variables[sem, -1] for sem in range(min(5, nsemesters))]
        overfitted_c9 = cp.AllDifferent(last_col)
        overfitted_constraints.append(overfitted_c9)
        model += overfitted_c9

    if nsemesters >= 6 and courses_per_semester >= 3:
        mid_idx = courses_per_semester // 2
        middle_col = [variables[sem, mid_idx] for sem in range(min(6, nsemesters))]
        overfitted_c10 = cp.AllDifferent(middle_col)
        overfitted_constraints.append(overfitted_c10)
        model += overfitted_c10

    AV = absvar(2)

    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day),
        cp.Count(AV, AV[0]) <= AV[1],
        cp.Count(AV, AV[0]) == AV[1]
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_v2_sem{nsemesters}_c{courses_per_semester}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints

