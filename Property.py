from typing import Dict, List, Union

from LTLfDFA import LTLfDFA, ltlf_to_python
from functools import partial

from PIL import Image
from io import BytesIO
import networkx as nx


class Property:
    def __init__(self, property_name, property_string, predicates, reset_string=None,
                 reset_init_trace: Dict[str, List[bool]] = None):
        self.resetable = False
        if reset_string is None:
            # if no reset string is specified, then the DFA can't leave the trap state
            reset_string = "False"
        else:
            self.resetable = True
        self.name = property_name
        self.ltldfa = LTLfDFA(property_string)
        self.data = {}
        self.predicates = {}
        for a, b in predicates:
            self.data[a] = []
            self.predicates[a] = b
        self.reset_string = ltlf_to_python(reset_string)
        if reset_init_trace is not None:
            self.reset_init_trace = {key: [(i, val) for i, val in enumerate(bool_list)]
                                     for key, bool_list in reset_init_trace.items()}
            # validate that the reset_init_trace can be run from the init state
            try:
                self.ltldfa.from_init(self.reset_init_trace)
            except NameError:
                raise AttributeError("reset_init_trace not valid over the DFA produced by the provided property_string")
        else:
            self.reset_init_trace = None
        self.violations = []
        self.in_violation = False
        self.time = 0

    def update_data(self, sg, save_usage_information=False):
        sg.graph[f'save_usage_information_{self.name}'] = save_usage_information
        if save_usage_information:
            sg.graph[f'usage_information_{self.name}'] = []
        for atomic_predicate, sg_predicate in self.predicates.items():
            self.data[atomic_predicate].append((
                len(self.data[atomic_predicate]),
                self.__evaluate_predicate(sg_predicate, sg)
            ))

    def save_relevant_subgraph(self, sg, file_name):
        svg = file_name is not None and file_name.endswith('svg')
        if not sg.graph[f'save_usage_information_{self.name}']:
            raise ValueError(
                "Cannot save relevant subgraph without save_usage_information set and calling update_data before this.")
        all_nodes = set()
        for data_dict in sg.graph[f'usage_information_{self.name}']:
            func_name = data_dict['func']  # TODO: filter only by those used in comparison expressions?
            data = data_dict['data']
            # print(func_name, data)
            all_nodes.update(data)
        all_nodes.update([node for node in sg.nodes if node.name == 'ego'])
        # print(all_nodes)
        graph_copy = nx.induced_subgraph(sg, all_nodes)
        # graph_copy = copy.deepcopy(sg)
        # graph_copy = nx.induced_subgraph(graph_copy, all_nodes)
        # graph_copy.graph['size'] = (100, 100)
        if svg:
            img = nx.nx_pydot.to_pydot(graph_copy).create_svg()
            with open(file_name, 'wb') as f:
                f.write(img)
        else:
            img = nx.nx_pydot.to_pydot(graph_copy).create_png()
            img = Image.open(BytesIO(img))
            if file_name is None:
                return img
            else:
                img.save(file_name)

    def check_from_init(self):
        """
        Checks if the data given leads to at least one violation. Does not handle multiple violations
        """
        return self.ltldfa.from_init(self.data, return_state=False)

    def check_step(self, return_state=False):
        """
        Updates the DFA based on the data. Handles multiple violations according to the reset criteria provided
        """
        self.time += 1
        acc, state = self.ltldfa.step(self.get_last_predicates(), return_state=True)
        if self.ltldfa.is_trap_state(state):
            if not self.in_violation:
                self.in_violation = True
                self.violations.append([self.time, -1])
            if eval(self.reset_string, self.get_last_predicates()):
                self.in_violation = False
                self.violations[-1][-1] = self.time
                dfa_trace = self.ltldfa.from_init(self.reset_init_trace, return_state=True)
                # set the DFA state to the state it ended the reset trace in
                state = dfa_trace[-1][-1]
                self.ltldfa.set_state(state)
        if return_state:
            return acc, state
        return acc

    def get_last_predicates(self):
        result = {}
        for key, val in self.data.items():
            result[key] = val[-1][1]
        return result

    def __evaluate_predicate(self, predicate, sg, func_chain=None):
        if func_chain is None:
            func_chain = ''
        func_chain += '.' + predicate.func.__name__
        param_list = []
        for arg in predicate.args:
            if isinstance(arg, partial):
                param_list.append(self.__evaluate_predicate(arg, sg, func_chain=func_chain))
            else:
                param_list.append(arg)
        if sg.graph[f'save_usage_information_{self.name}']:
            data = set()
            for param in param_list:
                if type(param) == set:
                    data.update(param)
            sg.graph[f'usage_information_{self.name}'].append({
                'func': func_chain,
                'data': data
            })
        if predicate.func.__name__ in ['filterByAttr', 'relSet']:
            return predicate.func(*param_list, sg, **predicate.keywords)
        else:
            return predicate.func(*param_list, **predicate.keywords)
