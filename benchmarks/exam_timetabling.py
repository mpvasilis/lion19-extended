import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14):
    

    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()
    model += cp.AllDifferent(variables).decompose()

    for semester_index, row in enumerate(variables):

        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days).decompose()

    for sem1 in range(nsemesters):
        for course1 in range(courses_per_semester):
            for sem2 in range(sem1, nsemesters):
                start_course2 = course1 + 1 if sem2 == sem1 else 0
                for course2 in range(start_course2, courses_per_semester):
                    exam1 = variables[sem1][course1]
                    exam2 = variables[sem2][course2]

                    if abs(sem1 - sem2) == 1:
                        model += [exam1 + 2 <= exam2 - 1]  

                    if (abs(sem1 - sem2) == 1 and 
                        ((course1 == 0 and course2 == courses_per_semester - 1) or
                         (course1 == courses_per_semester - 1 and course2 == 0))):
                        model += [exam1 + slots_per_day <= exam2 - 1]  

                    if sem1 < sem2 and course1 == 0 and course2 == 0:
                        model += [exam1 < exam2]

                    if (sem1 != sem2 and course1 == courses_per_semester - 1 and 
                        course2 == courses_per_semester - 1):
                        model += [day_of_exam(exam1, slots_per_day) != day_of_exam(exam2, slots_per_day)]

    C_T = list(model.constraints)

    AV = absvar(2)  

    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day)
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_semesters{nsemesters}_courses{courses_per_semester}_"
             f"timeslots{slots_per_day}_days{days_for_exams}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle
