import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_nurse_rostering(shifts_per_day=3, num_days=7, num_nurses=8, nurses_per_shift=2, max_workdays=5):
    

    parameters = {"shifts_per_day": shifts_per_day, "num_days": num_days, "num_nurses": num_nurses,
                  "nurses_per_shift": nurses_per_shift, "max_workdays": max_workdays}

    roster_matrix = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="var")

    model = cp.Model()

    for day in range(num_days):
        model += cp.AllDifferent(roster_matrix[day, ...]).decompose()

    for day in range(num_days - 1):
        model += cp.AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()


    for nurse in range(1, num_nurses + 1):
        model += cp.sum([cp.sum(roster_matrix[d, ...] == nurse) for d in range(num_days)]) <= max_workdays

    for day1 in range(num_days):
        for shift1 in range(shifts_per_day):
            for pos1 in range(nurses_per_shift):
                for day2 in range(day1, num_days):
                    start_shift2 = shift1 + 1 if day2 == day1 else 0
                    for shift2 in range(start_shift2, shifts_per_day):
                        start_pos2 = pos1 + 1 if (day2 == day1 and shift2 == shift1) else 0
                        for pos2 in range(start_pos2, nurses_per_shift):
                            nurse1 = roster_matrix[day1, shift1, pos1]
                            nurse2 = roster_matrix[day2, shift2, pos2]

                            if (shift1 == shifts_per_day - 1 and day2 == day1 + 1 and shift2 == 0):
                                model += [nurse1 != nurse2]

                            if (day1 == day2 and abs(shift1 - shift2) == 1):
                                model += [nurse1 != nurse2]

                            if (day2 == day1 + 1 and shift1 == shifts_per_day - 1 and shift2 == 0):
                                model += [nurse1 != nurse2]

                            if (day1 >= 5 and day2 >= 5 and day1 != day2):  
                                model += [nurse1 != nurse2]

                            if (pos1 == 0 and pos2 == nurses_per_shift - 1 and 
                                day1 == day2 and shift1 == shift2):
                                model += [nurse1 < nurse2]  




    C_T = list(model.constraints)

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

    return instance, oracle




