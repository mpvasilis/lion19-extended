import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar
import json
from datetime import datetime


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_simple(nsemesters=30, courses_per_semester=20, slots_per_day=18, days_for_exams=35):
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

    if nsemesters >= 4:
        first_four_sems = variables[:4, :].flatten()
        first_four_days = [day_of_exam(exam, slots_per_day) for exam in first_four_sems]
        overfitted_c1 = cp.AllDifferent(first_four_days)
        overfitted_constraints.append(overfitted_c1)
        model += overfitted_c1

    if nsemesters >= 5 and courses_per_semester >= 2:
        first_five = variables[:5, :].flatten()
        time_of_day = [exam % slots_per_day for exam in first_five]
        overfitted_c2 = cp.AllDifferent(time_of_day[:min(slots_per_day, len(time_of_day))])
        overfitted_constraints.append(overfitted_c2)
        model += overfitted_c2

    if nsemesters >= 6 and courses_per_semester >= 3:
        first_three_cols = variables[:, :3].flatten()
        overfitted_c3 = cp.AllDifferent(first_three_cols[:min(18, len(first_three_cols))])
        overfitted_constraints.append(overfitted_c3)
        model += overfitted_c3

    if nsemesters >= 6 and courses_per_semester >= 6:
        diagonal = [variables[i, i] for i in range(min(6, nsemesters, courses_per_semester))]
        overfitted_c4 = cp.AllDifferent(diagonal)
        overfitted_constraints.append(overfitted_c4)
        model += overfitted_c4

    if nsemesters >= 6 and courses_per_semester >= 6:
        anti_diag = [variables[i, courses_per_semester - 1 - i] 
                     for i in range(min(6, nsemesters, courses_per_semester))]
        overfitted_c5 = cp.AllDifferent(anti_diag)
        overfitted_constraints.append(overfitted_c5)
        model += overfitted_c5

    if nsemesters >= 8 and courses_per_semester >= 4:
        even_subset = variables[::2, :4].flatten()
        overfitted_c6 = cp.AllDifferent(even_subset[:min(16, len(even_subset))])
        overfitted_constraints.append(overfitted_c6)
        model += overfitted_c6

    if nsemesters >= 7 and courses_per_semester >= 4:
        odd_subset = variables[1::2, -4:].flatten()
        overfitted_c7 = cp.AllDifferent(odd_subset[:min(16, len(odd_subset))])
        overfitted_constraints.append(overfitted_c7)
        model += overfitted_c7

    if nsemesters >= 7 and courses_per_semester >= 1:
        first_col = [variables[sem, 0] for sem in range(min(7, nsemesters))]
        overfitted_c8 = cp.AllDifferent(first_col)
        overfitted_constraints.append(overfitted_c8)
        model += overfitted_c8

    if nsemesters >= 6 and courses_per_semester >= 2:
        last_col = [variables[sem, -1] for sem in range(min(6, nsemesters))]
        overfitted_c9 = cp.AllDifferent(last_col)
        overfitted_constraints.append(overfitted_c9)
        model += overfitted_c9

    if nsemesters >= 7 and courses_per_semester >= 3:
        mid_idx = courses_per_semester // 2
        middle_col = [variables[sem, mid_idx] for sem in range(min(7, nsemesters))]
        overfitted_c10 = cp.AllDifferent(middle_col)
        overfitted_constraints.append(overfitted_c10)
        model += overfitted_c10

    if nsemesters >= 6 and courses_per_semester >= 2:
        second_col = [variables[sem, 1] for sem in range(min(6, nsemesters))]
        overfitted_c11 = cp.AllDifferent(second_col)
        overfitted_constraints.append(overfitted_c11)
        model += overfitted_c11

    if nsemesters >= 8 and courses_per_semester >= 3:
        mid_idx = courses_per_semester // 2
        alternating = [variables[sem, mid_idx] for sem in range(0, min(8, nsemesters), 2)]
        overfitted_c12 = cp.AllDifferent(alternating)
        overfitted_constraints.append(overfitted_c12)
        model += overfitted_c12

    if nsemesters >= 10 and courses_per_semester >= 5:
        first_ten_sems = variables[:10, :].flatten()
        overfitted_c13 = cp.AllDifferent(first_ten_sems[:min(50, len(first_ten_sems))])
        overfitted_constraints.append(overfitted_c13)
        model += overfitted_c13

    if nsemesters >= 12 and courses_per_semester >= 8:
        large_diagonal = [variables[i, i] for i in range(min(12, nsemesters, courses_per_semester))]
        overfitted_c14 = cp.AllDifferent(large_diagonal)
        overfitted_constraints.append(overfitted_c14)
        model += overfitted_c14

    if nsemesters >= 12 and courses_per_semester >= 8:
        large_anti_diag = [variables[i, courses_per_semester - 1 - i] 
                          for i in range(min(12, nsemesters, courses_per_semester))]
        overfitted_c15 = cp.AllDifferent(large_anti_diag)
        overfitted_constraints.append(overfitted_c15)
        model += overfitted_c15

    if nsemesters >= 15 and courses_per_semester >= 10:
        quarter_subset = variables[:15, :10].flatten()
        overfitted_c16 = cp.AllDifferent(quarter_subset[:min(100, len(quarter_subset))])
        overfitted_constraints.append(overfitted_c16)
        model += overfitted_c16

    if nsemesters >= 10 and courses_per_semester >= 6:
        first_three_cols_large = variables[:, :3].flatten()
        overfitted_c17 = cp.AllDifferent(first_three_cols_large[:min(30, len(first_three_cols_large))])
        overfitted_constraints.append(overfitted_c17)
        model += overfitted_c17

    if nsemesters >= 12 and courses_per_semester >= 5:
        last_five_cols = variables[:, -5:].flatten()
        overfitted_c18 = cp.AllDifferent(last_five_cols[:min(60, len(last_five_cols))])
        overfitted_constraints.append(overfitted_c18)
        model += overfitted_c18

    if nsemesters >= 20 and courses_per_semester >= 10:
        first_half_sems = variables[:nsemesters//2, :].flatten()
        overfitted_c19 = cp.AllDifferent(first_half_sems[:min(100, len(first_half_sems))])
        overfitted_constraints.append(overfitted_c19)
        model += overfitted_c19

    if nsemesters >= 15 and courses_per_semester >= 8:
        every_third_sem = variables[::3, :].flatten()
        overfitted_c20 = cp.AllDifferent(every_third_sem[:min(40, len(every_third_sem))])
        overfitted_constraints.append(overfitted_c20)
        model += overfitted_c20

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
        name=f"exam_timetabling_semesters{nsemesters}_courses{courses_per_semester}_"
             f"timeslots{slots_per_day}_days{days_for_exams}"
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints


def generate_exam_timetabling_instance(instance_params=None):
    if instance_params is None:
        instance_params = {}
    
    nsemesters = instance_params.get('nsemesters', 9)
    courses_per_semester = instance_params.get('courses_per_semester', 6)
    slots_per_day = instance_params.get('slots_per_day', 9)
    days_for_exams = instance_params.get('days_for_exams', 14)
    
    return construct_examtt_simple(nsemesters, courses_per_semester, slots_per_day, days_for_exams)


class ExamTimetablingModel:
    def __init__(self, nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14, instance_id=1):
        self.nsemesters = nsemesters
        self.courses_per_semester = courses_per_semester
        self.slots_per_day = slots_per_day
        self.days_for_exams = days_for_exams
        self.instance_id = instance_id
        self.total_courses = nsemesters * courses_per_semester
        self.total_slots = slots_per_day * days_for_exams
        
        self.exam_slots = cp.intvar(1, self.total_slots, shape=(nsemesters, courses_per_semester), name="exam_slots")
        
        self.model = cp.Model()
        
        self.add_constraints()
        
        self.solution = None
    
    def add_constraints(self):
        self.model += cp.AllDifferent(self.exam_slots)
        
        for semester_index, row in enumerate(self.exam_slots):
            exam_days = [day_of_exam(course, self.slots_per_day) for course in row]
            self.model += cp.AllDifferent(exam_days)

        all_exams = self.exam_slots.flatten()
        total_courses = self.nsemesters * self.courses_per_semester
        max_exams_per_day = (total_courses + self.days_for_exams - 1) // self.days_for_exams + 1
        
        for day in range(self.days_for_exams):
            exams_on_day = cp.Count([day_of_exam(exam, self.slots_per_day) for exam in all_exams], day)
            self.model += (exams_on_day <= max_exams_per_day)
    
    def solve(self):
        if self.model.solve():
            self.create_solution_json()
            return True
        return False
    
    def create_solution_json(self):
        solution = {
            "instance_id": self.instance_id,
            "parameters": {
                "nsemesters": self.nsemesters,
                "courses_per_semester": self.courses_per_semester,
                "slots_per_day": self.slots_per_day,
                "days_for_exams": self.days_for_exams
            },
            "schedule": {}
        }
        
        for semester in range(self.nsemesters):
            semester_name = f"Semester_{semester+1}"
            solution["schedule"][semester_name] = []
            
            for course_idx in range(self.courses_per_semester):
                course_name = f"Course_{semester+1}_{course_idx+1}"
                slot = self.exam_slots[semester, course_idx].value()
                day = day_of_exam(slot, self.slots_per_day) + 1  
                time_slot = (slot - 1) % self.slots_per_day + 1  
                
                solution["schedule"][semester_name].append({
                    "course": course_name,
                    "day": day,
                    "time_slot": time_slot,
                    "slot_id": slot
                })
        
        self.solution = solution


def generate_multiple_solutions(num_solutions=5, nsemesters=9, courses_per_semester=6, 
                               slots_per_day=9, days_for_exams=14):
    all_solutions = []
    
    for i in range(num_solutions):
        current_nsemesters = max(2, nsemesters - 2 + i % 5)  
        current_courses = max(2, courses_per_semester - 1 + i % 3)  
        
        print(f"\nGenerating solution {i+1}/{num_solutions}...")
        print(f"Parameters: {current_nsemesters} semesters, {current_courses} courses per semester")
        
        model = ExamTimetablingModel(
            nsemesters=current_nsemesters,
            courses_per_semester=current_courses,
            slots_per_day=slots_per_day,
            days_for_exams=days_for_exams,
            instance_id=i+1
        )
        
        if model.solve():
            print("Solution found!")
            all_solutions.append(model.solution)
        else:
            print("Failed to find solution.")
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_solutions": len(all_solutions),
            "problem_type": "exam_timetabling"
        },
        "solutions": all_solutions
    }
    
    with open("exam_timetabling_solutions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(all_solutions)} solutions saved to 'exam_timetabling_solutions.json'")
    return all_solutions


if __name__ == "__main__":
    solutions = generate_multiple_solutions(5)
