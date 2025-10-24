import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_latin_square_binary(n=9):
    
    
    parameters = {"n": n}

    grid = cp.intvar(1, n, shape=(n, n), name="grid")
    
    model = cp.Model()

    for row in grid:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                model += (row[i] != row[j])

    for col in grid.T:
        for i in range(len(col)):
            for j in range(i + 1, len(col)):
                model += (col[i] != col[j])
    
    C_T = list(set(toplevel_list(model.constraints)))

    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="latin_square")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle


def construct_latin_square_binary_4x4():
    
    return construct_latin_square_binary(n=4)


def construct_latin_square_binary_6x6():
    
    return construct_latin_square_binary(n=6)


def construct_latin_square_binary_9x9():
    
    return construct_latin_square_binary(n=9)

