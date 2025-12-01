import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_nurse_rostering(shifts_per_day=3, num_days=7, num_nurses=10, nurses_per_shift=2, max_workdays=6):

    parameters = {"shifts_per_day": shifts_per_day, "num_days": num_days, "num_nurses": num_nurses,
                  "nurses_per_shift": nurses_per_shift, "max_workdays": max_workdays}

    roster_matrix = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")

    model = cp.Model()

    for day in range(num_days):
        model += cp.AllDifferent(roster_matrix[day, ...])


    for day in range(num_days - 1):
        last_shift_nurses = roster_matrix[day, shifts_per_day - 1, :]
        first_shift_next_day = roster_matrix[day + 1, 0, :]
        combined = list(last_shift_nurses) + list(first_shift_next_day)
        model += cp.AllDifferent(combined)


    for nurse in range(1, num_nurses + 1):
        model += cp.Count(roster_matrix, nurse) <= max_workdays

    C_T = list(model.constraints)




    overfitted_constraints = []


    if num_days >= 3:
        overfitted_positions = list(roster_matrix[0, 0, :]) + list(roster_matrix[2, 0, :])
        overfitted_c1 = cp.AllDifferent(overfitted_positions)
        overfitted_constraints.append(overfitted_c1)



    if num_days >= 2 and shifts_per_day >= 2:
        mid_shift = shifts_per_day // 2
        overfitted_positions = list(roster_matrix[0, mid_shift, :]) + list(roster_matrix[num_days-1, mid_shift, :])
        overfitted_c2 = cp.AllDifferent(overfitted_positions)
        overfitted_constraints.append(overfitted_c2)



    if num_days >= 4 and shifts_per_day >= 2:
        overfitted_positions = list(roster_matrix[1, 0, :]) + list(roster_matrix[3, 1, :])
        overfitted_c3 = cp.AllDifferent(overfitted_positions)
        overfitted_constraints.append(overfitted_c3)


    AV = absvar(2)
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1]
    ]

    instance = ProblemInstance(variables=roster_matrix, params=parameters, language=lang,
                               name=f"nurse_rostering_nurses{num_nurses}_shifts{shifts_per_day}_days{num_days}_"
                                    f"nurses_per_shift{nurses_per_shift}_max_workdays{max_workdays}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle, overfitted_constraints

