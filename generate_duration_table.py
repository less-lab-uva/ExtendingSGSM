import argparse
import glob
import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path
from properties import timed_props

SUTS = ["Interfuser", "TCP", "LAV"]
ROUTES = list(range(16, 26, 1))
PROPERTIES = [prop.name for prop in timed_props]
# PROPERTIES = [prop.name for prop in timed_props if 'S_1' not in prop.name]
# PROPERTIES = [prop.name for prop in timed_props if 'Phi4' not in prop.name]


class Violation:
    def __init__(self, property, route, start, end):
        self.property = property
        self.route = route
        self.start = start
        self.end = end

    def duration(self) -> int:
        return self.end - self.start


def latex_format(latex: str, full=False) -> str:
    # centering_fill = '|k{0.7cm}' + ('k{0.85cm}' * 2) + 'k{0.7cm}' + ('k{0.85cm}' * 3) + (
    #             'k{0.85cm}' * 6) + 'k{0.7cm}' + '|k{1cm}'
    # latex = re.sub(r'{tabular}{(.*)}', r'{tabular}{c' + centering_fill + r'}', latex)

    # all 3 of the Phi4 configs are 0 in all cases, so condense them into 1
    # latex = latex.replace('Phi4\\_S\\_5\\_duration', 'Phi4\\_S\\_5\\_duration, Phi4\\_S\\_10\\_duration, Phi4\\_S\\_15\\_duration')
    # latex = latex.replace('Phi4\\_S\\_5\\_duration', 'Phi4$^{S\\in\\{5, 10, 15\\}}$')
    latex = latex.replace('\\_duration', '')
    latex = latex.replace(''.join(SUTS), 'All')
    # latex = latex.replace('.0', '')
    latex = re.sub(r'Phi(\d+)', r'$\\psi_\1$', latex)
    # handle for complex reset
    latex = re.sub(r'\$\\_complex\\_reset\\_(\d+)', r'^{\$[\1]}$', latex)
    # handle for parameterized
    if not full:
        latex = re.sub(r'\$\\_(\w+)\\_(\d+)', r'\\atop{\1=\2}$', latex)
    else:
        latex = re.sub(r'\$\\_(\w+)\\_(\d+)', r'^{\1=\2}$', latex)
    latex = latex.replace('$$', '')
    if not full:
        latex = latex.replace('All', '\\hline\nAll')
    else:
        # Interfuser~\cite{shao2023safety}
        # TCP~\cite{wu2022trajectory}
        # LAV~\cite{chen2022lav}
        SUT_BLOCK = '\\\\cite{shao2023safety} & \\\\cite{wu2022trajectory} & \\\\cite{chen2022lav}'
        new_header = ' & \\\\multicolumn{4}{c||}{\\\\# Routes} & \\\\multicolumn{4}{c||}{\\\\# Violations} & \\\\multicolumn{4}{c||}{Total Duration (\\\\# Frames)} & \\\\multicolumn{4}{c|}{Max Duration (\\\\# Frames)} \\\\\\\\ \n'
        # $\\\\Sigma$
        new_header += 'SUT & ' + (SUT_BLOCK + ' & Sum & ') * 3 + SUT_BLOCK + ' & Max \\\\\\\\ \n'

        latex = re.sub(r'index.*\n', new_header, latex)
        # \begin{tabular}{lllllllllllllllllllll}
        # latex = latex.replace(r'\\begin{tabular}{lllllllllllllllllllll}',
        #                       r'\\begin{tabular}{|l|lll|l||lll|l||lll|l||lll|l||lll|l|}')
        latex = re.sub(r'{tabular}{(.*)}', r'{tabular}{|c|rrr|r||rrr|r||rrr|r||rrr|r|}', latex)
        latex = re.sub(r'\\\\', r'\\\\[2pt]', latex)
        latex = latex.replace('                  Sum', '\\midrule\nSum')
    latex = latex.replace('\\midrule', '\\hline\\hline')
    latex = latex.replace('\\toprule', '\\hline')
    latex = latex.replace('\\bottomrule', '\\hline')
    return latex


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    parser = argparse.ArgumentParser(prog='Property checker')
    parser.add_argument('-bf', '--base_folder', type=Path, default='.')
    parser.add_argument('-of', '--output_folder', type=Path, default='./violation_tables/')
    args = parser.parse_args()

    # Sum of property violations accross all routes
    route_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    count_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    duration_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Total"])
    max_duration_summary = pd.DataFrame(columns=["SUT", *PROPERTIES, "Max"])
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
        max_duration_results = pd.DataFrame(columns=["SUT", *PROPERTIES])
        route_results = pd.DataFrame(columns=["SUT", *PROPERTIES])
        for sut in SUTS:
            count_row = {"SUT": sut}
            duration_row = {"SUT": sut}
            max_duration_row = {"SUT": sut}
            route_row = {"SUT": sut}
            # Initialize all properties with no violations
            for prop in PROPERTIES:
                count_row[prop] = 0
                duration_row[prop] = 0
                max_duration_row[prop] = 0
                route_row[prop] = 0
            properties_path = args.base_folder / f"{sut}_results/done_RouteScenario_{route}/"
            if properties_path.exists():
                for violation_file in glob.glob(f'{properties_path}/violations_*.csv'):
                    prop_name = violation_file.split(".csv")[0][
                                violation_file.find('violations_') + len('violations_'):]
                    if prop_name not in PROPERTIES:
                        continue
                    df = pd.read_csv(violation_file)
                    for index, row in df.iterrows():
                        violation = Violation(prop_name, route, row['start'], row['end'])
                        violations[prop_name].append(violation)
                        route_row[prop_name] = 1
                        count_row[prop_name] += 1
                        duration_row[prop_name] += violation.duration()
                        max_duration_row[prop_name] = max(violation.duration(), max_duration_row[prop_name])
            count_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[count_row])
            n_violations += count_r.iloc[:, 1:].sum(axis=1)[0]
            count_results = pd.concat([count_results, count_r])
            duration_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[duration_row])
            duration_violations += duration_r.iloc[:, 1:].sum(axis=1)[0]
            duration_results = pd.concat([duration_results, duration_r])
            # print(list(max_duration_row.values())[1:])
            max_duration_row["Max"] = max(list(max_duration_row.values())[1:])
            max_duration_r = pd.DataFrame(columns=["SUT", *PROPERTIES, "Max"], data=[max_duration_row])
            max_duration_results = pd.concat([max_duration_results, max_duration_r])
            route_r = pd.DataFrame(columns=["SUT", *PROPERTIES], data=[route_row])
            route_results = pd.concat([route_results, route_r])

        count_results["Total"] = count_results.iloc[:, 1:].sum(axis=1)
        # count_results.loc['Total', :] = count_results.sum(axis=0)
        count_results.to_csv(args.output_folder / f"RouteScenario_{route}_counts.csv", index=False)
        route_results["Total"] = route_results.iloc[:, 1:].sum(axis=1)
        # route_results.loc['Total', :] = route_results.sum(axis=0)
        route_results.to_csv(args.output_folder / f"RouteScenario_{route}_num_routes.csv", index=False)
        # print('duration_results')
        # print(duration_results.iloc[:, 1:])
        # print('max')
        # print(duration_results.iloc[:, 1:].max(axis=1))
        max_duration_results.to_csv(args.output_folder / f"RouteScenario_{route}_max_durations.csv", index=False)
        duration_results["Total"] = duration_results.iloc[:, 1:].sum(axis=1)
        # duration_results.loc['Total', :] = duration_results.sum(axis=0)
        duration_results.to_csv(args.output_folder / f"RouteScenario_{route}_durations.csv", index=False)
        # print(duration_results.iloc[:, 1:])
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

        if len(max_duration_summary) == 0:
            max_duration_summary = max_duration_results
        else:
            max_duration_summary.iloc[:, 1:] = max_duration_summary.iloc[:, 1:].where(
                max_duration_summary.iloc[:, 1:] > max_duration_results.iloc[:, 1:], max_duration_results.iloc[:, 1:])

        if len(route_summary) == 0:
            route_summary = route_results
        else:
            route_summary.iloc[:, 1:] += route_results.iloc[:, 1:]
    count_summary.loc['Total'] = count_summary.sum(axis=0)
    duration_summary.loc['Total'] = duration_summary.sum(axis=0)
    max_duration_summary.loc['Max'] = max_duration_summary.max(axis=0)
    max_duration_summary.loc['Max', 'SUT'] = 'All'
    route_summary.loc['Total'] = route_summary.sum(axis=0)
    average_duration_summary = duration_summary.copy(deep=True)
    average_duration_summary.iloc[:, 1:] = duration_summary.iloc[:, 1:].div(
        count_summary.iloc[:, 1:].replace(0, np.nan)).replace(np.nan, 0)
    count_summary.to_csv(args.output_folder / "count_summary.csv", index=False)
    duration_summary.to_csv(args.output_folder / "duration_summary.csv", index=False)
    route_summary.to_csv(args.output_folder / "route_summary.csv", index=False)
    float_format = '%.0f'
    average_duration_summary.to_csv(args.output_folder / "average_duration_summary.csv", index=False)
    # print(f"Route results")
    print_table(route_summary, 'Number of routes with $\\geq 1$ violation', 'tab:route_violations', float_format)
    # print(f"Route {route_with_most_violations[0]} has the most violations: {route_with_most_violations[1]}")
    print_table(count_summary, 'Total number of violations', 'tab:total_violations', float_format)
    # print(latex_format(count_summary.to_latex(index=False, float_format=float_format)))
    # print()
    # print(f"Route {route_with_longest_violations[0]} has the longest total violations: {route_with_longest_violations[1]}")
    print_table(duration_summary, 'Total duration of violations (\\# frames)', 'tab:duration_violations', float_format)
    # print(latex_format(duration_summary.to_latex(index=False, float_format=float_format)))
    # print('Average duration')
    float_format = '%.1f'
    # print(latex_format(average_duration_summary.to_latex(index=False, float_format=float_format)))
    print_table(average_duration_summary, 'Average duration of violations (\\# frames)', 'tab:avg_duration_violations',
                float_format)
    # print('Max duration')
    float_format = '%.0f'
    # print(latex_format(max_duration_summary.to_latex(index=False, float_format=float_format)))
    print_table(max_duration_summary, 'Maximum duration of violations (\\# frames)', 'tab:max_duration_violations',
                float_format)
    all_dfs = [
        route_summary.transpose(),
        count_summary.transpose(),
        duration_summary.transpose(),
        # average_duration_summary.transpose(),
        max_duration_summary.transpose()
    ]
    combined = pd.concat(all_dfs, axis=1)
    new_header = [a if a != 'InterfuserTCPLAV' else 'All' for a in combined.iloc[0].tolist()]
    combined.columns = new_header  # replace generated headers with intended headers
    combined = combined[1:]  # drop fake head
    combined = combined[:-2]  # drop max and total
    combined.loc['Sum'] = combined.sum()
    combined.loc['Max'] = combined[:-1].max()
    combined.reset_index(inplace=True)  # make the prop labels actually in the table
    # print(combined)
    combined.to_csv('test.csv')
    print()
    print_table(combined, 'Full Study Results', 'tab:full_study', lambda x: '%.1f' % x, full=True)


def print_table(route_summary, caption, label, float_format, full=False):
    print('\\begin{table}[h]')
    print('\\begin{center}')
    print('\\footnotesize')
    print('\\caption{' + caption + '}')
    print('\\label{' + label + '}')
    print(latex_format(route_summary.to_latex(index=False, float_format=float_format, na_rep=''), full))
    print('\\end{center}')
    print('\\end{table}')


if __name__ == "__main__":
    main()
