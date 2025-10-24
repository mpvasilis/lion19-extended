import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_latin_square(n=9):
    
    
    parameters = {"n": n}

    grid = cp.intvar(1, n, shape=(n, n), name="grid")
    
    model = cp.Model()

    for row in grid:
        model += cp.AllDifferent(row)

    for col in grid.T:
        model += cp.AllDifferent(col)
    
    C_T = list(set(toplevel_list(model.constraints)))

    mock_constraints = []


















    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="latin_square")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_latin_square_4x4():
    
    return construct_latin_square(n=4)


def construct_latin_square_6x6():
    
    return construct_latin_square(n=6)


def construct_latin_square_9x9():
    
    return construct_latin_square(n=9)

