import argparse
import glob
import re
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from pathlib import Path
from properties import timed_props

SUTS = ["Interfuser", "TCP", "LAV"]
ROUTES = list(range(16, 26, 1))
PROPERTIES = [prop.name for prop in timed_props]

class Violation:
    def __init__(self, property, route, start, end):
        self.property = property
        self.route = route
        self.start = start
        self.end = end

    def duration(self) -> int:
        return self.end - self.start


def latex_format(latex: str) -> str:
    latex = latex.replace('\\_duration', '')
    latex = latex.replace(''.join(SUTS), 'All')
    latex = latex.replace('.0', '')
    latex = re.sub(r'Phi(\d+)', r'$\\psi_\1$', latex)
    return latex


def main():
    parser = argparse.ArgumentParser(prog='Property checker')
    parser.add_argument('-bf', '--base_folder', type=Path, default='.')
    parser.add_argument('-of', '--output_folder', type=Path, default='./violation_tables/')
    args = parser.parse_args()
    
    # Sum of property violations accross all routes
    route_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    count_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    duration_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    route_with_most_violations = (16, 0)
    route_with_longest_violations = (16, 0)

    args.output_folder.mkdir(parents=True, exist_ok=True)
    violations: Dict[str, List[Violation]] = defaultdict(list)
    for route in ROUTES:
        n_violations = 0
        duration_violations = 0
        # Property violations per route
        count_results = pd.DataFrame(columns=["SUT", *PROPERTIES])
        duration_results = pd.DataFrame(columns=["SUT", *PROPERTIES])
        route_results = pd.DataFrame(columns=["SUT", *PROPERTIES])
        for sut in SUTS:
            count_row = {"SUT": sut}
            duration_row = {"SUT": sut}
            route_row = {"SUT": sut}
            # Initialize all properties with no violations
            for prop in PROPERTIES:
                count_row[prop] = 0
                duration_row[prop] = 0
                route_row[prop] = 0
            properties_path = args.base_folder / f"{sut}_results/done_RouteScenario_{route}/"
            if properties_path.exists():
                for violation_file in glob.glob(f'{properties_path}/violations_*.csv'):
                    prop_name = violation_file.split(".csv")[0][violation_file.find('violations_')+len('violations_'):]
                    if prop_name not in PROPERTIES:
                        continue
                    df = pd.read_csv(violation_file)
                    for index, row in df.iterrows():
                        violation = Violation(prop_name, route, row['start'], row['end'])
                        violations[prop_name].append(violation)
                        route_row[prop_name] = 1
                        count_row[prop_name] += 1
                        duration_row[prop_name] += violation.duration()
            count_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[count_row])
            n_violations += count_r.iloc[:, 1:].sum(axis=1)[0]
            count_results = pd.concat([count_results, count_r])
            duration_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[duration_row])
            duration_violations += duration_r.iloc[:, 1:].sum(axis=1)[0]
            duration_results = pd.concat([duration_results, duration_r])
            route_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[route_row])
            route_results = pd.concat([route_results, route_r])

        count_results["Total"] = count_results.iloc[:, 1:].sum(axis=1)
        # count_results.loc['Total', :] = count_results.sum(axis=0)
        count_results.to_csv(args.output_folder / f"RouteScenario_{route}_counts.csv", index=False)
        route_results["Total"] = route_results.iloc[:, 1:].sum(axis=1)
        # route_results.loc['Total', :] = route_results.sum(axis=0)
        route_results.to_csv(args.output_folder / f"RouteScenario_{route}_num_routes.csv", index=False)
        duration_results["Total"] = duration_results.iloc[:, 1:].sum(axis=1)
        # duration_results.loc['Total', :] = duration_results.sum(axis=0)
        duration_results.to_csv(args.output_folder / f"RouteScenario_{route}_durations.csv", index=False)
        if n_violations > route_with_most_violations[1]:
            route_with_most_violations = (route, n_violations)
        if duration_violations > route_with_longest_violations[1]:
            route_with_longest_violations = (route, duration_violations)

        if len(count_summary) == 0:
            count_summary = count_results
        else:
            count_summary.iloc[:, 1:] += count_results.iloc[:, 1:]

        if len(duration_summary) == 0:
            duration_summary = duration_results
        else:
            duration_summary.iloc[:, 1:] += duration_results.iloc[:, 1:]

        if len(route_summary) == 0:
            route_summary = route_results
        else:
            route_summary.iloc[:, 1:] += route_results.iloc[:, 1:]
    count_summary.loc['Total'] = count_summary.sum(axis=0)
    duration_summary.loc['Total'] = duration_summary.sum(axis=0)
    route_summary.loc['Total'] = route_summary.sum(axis=0)
    count_summary.to_csv(args.output_folder / "count_summary.csv", index=False)
    duration_summary.to_csv(args.output_folder / "duration_summary.csv", index=False)
    route_summary.to_csv(args.output_folder / "route_summary.csv", index=False)
    print(f"Route results")
    print(latex_format(route_summary.to_latex(index=False)))
    print(f"Route {route_with_most_violations[0]} has the most violations: {route_with_most_violations[1]}")
    print(latex_format(count_summary.to_latex(index=False)))
    print()
    print(f"Route {route_with_longest_violations[0]} has the longest total violations: {route_with_longest_violations[1]}")
    print(latex_format(duration_summary.to_latex(index=False)))


if __name__ == "__main__":
    main()
