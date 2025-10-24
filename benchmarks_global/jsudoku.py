import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar


def construct_jsudoku(grid_size=9, regions=None):
    
    
    parameters = {"grid_size": grid_size}

    if regions is None and grid_size == 9:
        regions = [

            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)],

            [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)],

            [(0,6), (0,7), (0,8), (1,6), (1,7), (1,8), (2,6), (2,7), (2,8)],

            [(3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1), (5,2)],

            [(3,3), (3,4), (3,5), (4,3), (4,4), (4,5), (5,3), (5,4), (5,5)],

            [(3,6), (3,7), (3,8), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8)],

            [(6,0), (6,1), (6,2), (7,0), (7,1), (7,2), (8,0), (8,1), (8,2)],

            [(6,3), (6,4), (6,5), (7,3), (7,4), (7,5), (8,3), (8,4), (8,5)],

            [(6,6), (6,7), (6,8), (7,6), (7,7), (7,8), (8,6), (8,7), (8,8)],
        ]
    elif regions is None:
        raise ValueError(f"No default regions for grid_size={grid_size}. Please provide custom regions.")
    
    parameters["num_regions"] = len(regions)

    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")
    
    model = cp.Model()

    for row in grid:
        model += cp.AllDifferent(row)

    for col in grid.T:
        model += cp.AllDifferent(col)

    for region_idx, region in enumerate(regions):
        region_cells = [grid[r, c] for r, c in region]
        model += cp.AllDifferent(region_cells)
    
    C_T = list(set(toplevel_list(model.constraints)))


    mock_constraints = []


    if grid_size >= 9:
        non_adj_rows = []
        for row in [0, 3, 6]:
            for col in range(3):
                non_adj_rows.append(grid[row, col])
        mock_c1 = cp.AllDifferent(non_adj_rows)  
        mock_constraints.append(mock_c1)
        model += mock_c1

    if grid_size >= 9:
        non_adj_cols = []
        for col in [1, 4, 7]:
            for row in range(3):
                non_adj_cols.append(grid[row, col])
        mock_c2 = cp.AllDifferent(non_adj_cols)  
        mock_constraints.append(mock_c2)
        model += mock_c2

    if grid_size >= 9:
        diagonal = [grid[i, i] for i in range(grid_size)]
        mock_c3 = cp.AllDifferent(diagonal)
        mock_constraints.append(mock_c3)
        model += mock_c3

    if grid_size >= 9:
        anti_diag = [grid[i, grid_size - 1 - i] for i in range(grid_size)]
        mock_c4 = cp.AllDifferent(anti_diag)
        mock_constraints.append(mock_c4)
        model += mock_c4

    if grid_size >= 9 and len(regions) >= 9:
        corner_cells = []

        for region_idx in [0, 2, 6, 8]:
            if region_idx < len(regions):
                region_cells = [grid[r, c] for r, c in regions[region_idx][:3]]
                corner_cells.extend(region_cells)
        if len(corner_cells) >= 12:
            mock_c5 = cp.AllDifferent(corner_cells)
            mock_constraints.append(mock_c5)
            model += mock_c5

    if grid_size >= 9:
        border_cells = []

        border_cells.extend([grid[0, col] for col in range(6)])

        border_cells.extend([grid[8, col] for col in range(6)])
        mock_c6 = cp.AllDifferent(border_cells)  
        mock_constraints.append(mock_c6)
        model += mock_c6

    if grid_size >= 9:
        middle_area = []

        for row in range(2, 7):
            middle_area.append(grid[row, 4])

        for col in range(2, 7):
            if col != 4:  
                middle_area.append(grid[4, col])
        mock_c7 = cp.AllDifferent(middle_area)  
        mock_constraints.append(mock_c7)
        model += mock_c7

    if grid_size >= 9:
        l_shape = []

        for row in range(3):
            l_shape.append(grid[row, 0])

        for col in range(1, 6):
            l_shape.append(grid[0, col])
        mock_c8 = cp.AllDifferent(l_shape)  
        mock_constraints.append(mock_c8)
        model += mock_c8

    if grid_size >= 9:
        star_pattern = [
            grid[0, 0], grid[0, grid_size-1],  
            grid[grid_size-1, 0], grid[grid_size-1, grid_size-1],  
            grid[grid_size//2, grid_size//2],  
            grid[0, grid_size//2], grid[grid_size-1, grid_size//2],  
            grid[grid_size//2, 0], grid[grid_size//2, grid_size-1]  
        ]
        mock_c9 = cp.AllDifferent(star_pattern)
        mock_constraints.append(mock_c9)
        model += mock_c9

    if grid_size >= 9:
        z_pattern = []

        for col in range(1, 4):
            z_pattern.append(grid[0, col])

        for i in range(3, 6):
            z_pattern.append(grid[i, i])

        for col in range(5, 8):
            z_pattern.append(grid[8, col])
        mock_c10 = cp.AllDifferent(z_pattern)  
        mock_constraints.append(mock_c10)
        model += mock_c10

    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]
    
    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="jsudoku")
    oracle = ConstraintOracle(C_T)
    
    return instance, oracle, mock_constraints


def construct_jsudoku_4x4():
    
    regions = [

        [(0,0), (0,1), (1,0), (1,1)],

        [(0,2), (0,3), (1,2), (1,3)],

        [(2,0), (2,1), (3,0), (3,1)],

        [(2,2), (2,3), (3,2), (3,3)],
    ]
    return construct_jsudoku(grid_size=4, regions=regions)


def construct_jsudoku_6x6():
    
    regions = [

        [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)],

        [(0,2), (0,3), (1,2), (1,3), (2,2), (2,3)],

        [(0,4), (0,5), (1,4), (1,5), (2,4), (2,5)],

        [(3,0), (3,1), (4,0), (4,1), (5,0), (5,1)],

        [(3,2), (3,3), (4,2), (4,3), (5,2), (5,3)],

        [(3,4), (3,5), (4,4), (4,5), (5,4), (5,5)],
    ]
    return construct_jsudoku(grid_size=6, regions=regions)

