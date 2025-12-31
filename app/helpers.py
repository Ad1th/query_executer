import re
from typing import List
import itertools
from app.models import QueryParameter, ReadyQuery


def extract_variables(query: str) -> List[QueryParameter]:
    """
    Extract variables from query which are defined inside double curly braces {{ ... }}
    """
    pattern = r"\{\{(\w+):(\w+)\}\}"
    matches = re.findall(pattern, query)
    return [QueryParameter(name, data_type) for name, data_type in matches]


def build_single_query(template: str, variables: list) -> str:
    pattern = r"\{\{(\w+):(INT|FLOAT)\}\}"

    # Convert list of dicts to a name-value-type lookup
    value_map = {var['name']: var['value'] for var in variables}
    type_map = {var['name']: var['type'] for var in variables}

    def replace_placeholder(match):
        var_name = match.group(1)
        var_type = match.group(2)

        if var_name not in value_map:
            raise ValueError(f"Missing value for variable '{var_name}'")

        val = value_map[var_name]

        if var_type == "INT":
            return str(int(val))
        elif var_type == "FLOAT":
            return str(float(val))
        else:
            raise ValueError(f"Unsupported type '{var_type}'")

    return re.sub(pattern, replace_placeholder, template)


def build_all_queries(template: str, variable_ranges: list) -> list[ReadyQuery]:
    variable_names = [var['name'] for var in variable_ranges]
    variable_types = {var['name']: var['type'] for var in variable_ranges}

    # Generate ranges
    value_lists = [range(start, end + 1, step) for var in variable_ranges
                   for (start, end, step) in [var['range']]]

    # Cartesian product of all variable values
    combinations = itertools.product(*value_lists)

    all_queries: List[ReadyQuery] = []
    for combo in combinations:
        # Prepare variable dicts with name, type, value
        variable_values = [
            {'name': name, 'type': variable_types[name], 'value': value}
            for name, value in zip(variable_names, combo)
        ]
        query = build_single_query(template, variable_values)
        ready_query = ReadyQuery(query, variable_values)
        all_queries.append(ready_query)

    return all_queries
