import cpmpy as cp
import numpy as np
from cpmpy import cpm_array
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_uefa(teams_data, n_groups=8, teams_per_group=4):
    

    parameters = {
        "n_groups": n_groups,
        "teams_per_group": teams_per_group,
        "n_teams": len(teams_data)
    }

    group_assignments = {}
    for team in teams_data:
        group_assignments[team] = cp.intvar(0, n_groups-1, name=f"group{team}")
 

    model = cp.Model()

    all_constraints = []

    global_constraints = []

    for g in range(n_groups):
        group_size = sum(group_assignments[team] == g for team in teams_data)
        constraint = (group_size == teams_per_group)
        all_constraints.append(constraint)
        global_constraints.append(constraint)  

    countries = {}
    for team, data in teams_data.items():
        country = data['country']
        if country not in countries:
            countries[country] = []
        countries[country].append(team)

    for country, country_teams in countries.items():
        if 1 < len(country_teams) <= 4:  
            country_vars = [group_assignments[team] for team in country_teams]
            constraint = cp.AllDifferent(country_vars).decompose()
            all_constraints.append(constraint)
            global_constraints.append(constraint)  

    teams_by_coefficient = sorted(teams_data.items(), key=lambda x: x[1]['coefficient'], reverse=True)
    for pot_idx in range(0, len(teams_data), n_groups):
        pot_end = min(pot_idx + n_groups, len(teams_data))
        pot_teams = teams_by_coefficient[pot_idx:pot_end]
        if len(pot_teams) > 1:
            pot_vars = [group_assignments[team[0]] for team in pot_teams]
            constraint = cp.AllDifferent(pot_vars).decompose()
            all_constraints.append(constraint)
            global_constraints.append(constraint)  


    team_names = list(teams_data.keys())
    for i in range(len(team_names)):
        for j in range(i+1, len(team_names)):
            team1, team2 = team_names[i], team_names[j]

            if (teams_data[team1]['country'] in ['Spain', 'Portugal'] and 
                teams_data[team2]['country'] in ['Spain', 'Portugal']):
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)

            if (teams_data[team1]['coefficient'] > 90 and 
                teams_data[team2]['coefficient'] > 90):
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)

            coeff_diff = abs(teams_data[team1]['coefficient'] - teams_data[team2]['coefficient'])
            if coeff_diff < 5:  
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)
































































    for constraint in all_constraints:
        model += constraint

    AV = absvar(2)  

    lang = [
        sum(AV) == AV[0],     
        sum(AV) <= AV[0],     
        sum(AV) >= AV[0],     
    ]

    variables = []

    for team in sorted(teams_data.keys()):
        variables.append(group_assignments[team])










    instance = ProblemInstance(variables=cpm_array(variables), params=parameters, language=lang, name="uefa")
    oracle = ConstraintOracle(global_constraints)  

    return instance, oracle 