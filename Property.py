import copy
import itertools
from typing import Dict, List, Union, Tuple, Any, Optional

import sympy

from LTLfDFA import LTLfDFA, get_pydot_image
from functools import partial

from PIL import Image
from io import BytesIO
import networkx as nx


class Property:
    def __init__(self,
                 property_name: str,
                 property_string: str,
                 predicates: List[Tuple[str, Union[partial, Any]]],
                 reset_prop: Union[str, 'Subproperty'] = None,
                 reset_init_trace: Union[Dict[str, List[bool]], 'str'] = None):
        self.resettable = False
        self.name = property_name
        self.ltldfa = LTLfDFA(property_string)
        self.data = {}
        self.predicates = {}
        for a, b in predicates:
            self.data[a] = []
            self.predicates[a] = b
        needed_symbols = set(self.ltldfa.symbols)
        have_symbols = set(self.predicates.keys())
        missing_symbols = needed_symbols - have_symbols
        if len(missing_symbols) > 0:
            raise AttributeError("The provided list of predicates is insufficient to evaluate the formula."
                                 f"Found symbols: {have_symbols}, Need symbols: {needed_symbols}, "
                                 f"Missing: {missing_symbols}")
        if type(reset_prop) == Subproperty:
            self.reset_prop = reset_prop
            self.resettable = True
        elif type(reset_prop) == str:
            self.resettable = True
            prop_string = f"~({reset_prop}) U ({reset_prop})"
            self.reset_prop = Subproperty(self,
                                          property_name=f"{self.name}_reset_prop",
                                          property_string=prop_string)
        elif reset_prop is None:
            if type(self) == NeverAccepting:
                # break the recursion
                self.reset_prop = self
            else:
                self.reset_prop = NeverAccepting()
        else:
            raise AttributeError("reset_prop must be Subproperty or str")
        self.reset_state = None
        if reset_init_trace is not None:
            if type(reset_init_trace) == str:
                self.reset_init_trace = Subproperty(self,
                                                    property_name=f'{self.name}_reset_state_formula',
                                                    property_string=reset_init_trace)
                # validate that the reset_init_trace Subproperty points to exactly one state
                reset_states = self.__compute_reset_state_from_product(self.reset_init_trace)
                if len(reset_states) != 1:
                    if len(reset_states) == 0:
                        raise AttributeError("The provided reset formula over-constrains the input. "
                                             "No valid reset state found.")
                    else:
                        raise AttributeError("The provided reset formula under-constrains the input. "
                                             f"Found {len(reset_states)} possible reset states.")
                else:
                    self.reset_state = reset_states.pop()  # pop the only element
            elif type(reset_init_trace) == dict:
                # convert the trace to the form used by the DFA function
                self.reset_init_trace = {key: [(i, val) for i, val in enumerate(bool_list)]
                                         for key, bool_list in reset_init_trace.items()}
                # validate that the reset_init_trace can be run from the init state
                try:
                    ret_val = self.ltldfa.from_init(self.reset_init_trace, return_state=True)
                    self.reset_state = ret_val[-1][-1]
                except NameError:
                    raise AttributeError(
                        "reset_init_trace not valid over the DFA produced by the provided property_string")
            else:
                raise AttributeError("reset_init_trace must be type Dict[str, List[bool]] or Subproperty")
        else:
            self.reset_init_trace = None
        self.violations = []
        self.in_violation = False
        self.time = 0

    def __compute_reset_state_from_product(self, reset_prop: 'Subproperty', save_product=False):
        """Computes the product of the DFA for this property with the DFA for the Subproperty."""
        assert reset_prop.is_subproperty_of(self), "The reset formula provided is not a Subproperty of this property"
        reset_dfa = copy.copy(reset_prop.ltldfa._dfa)
        orig_dfa = copy.copy(self.ltldfa._dfa)
        for dfa in [reset_dfa, orig_dfa]:
            # the init state is special and isn't handled by the cross product.
            # it will be added back into the cross product DFA later.
            dfa.remove_node('init')
        predicates = set(reset_prop.predicates.keys())
        predicates.update(self.predicates.keys())
        symbols = {p: sympy.symbols(p) for p in predicates}
        symbols['sympy'] = sympy
        nodes = []
        edges = []
        for u, v in itertools.product(reset_dfa.nodes, orig_dfa.nodes):
            nodes.append((u, v))
            first_edges = list(reset_dfa.out_edges(u, data=True))
            second_edges = list(orig_dfa.out_edges(v, data=True))
            for edge1, edge2 in itertools.product(first_edges, second_edges):
                data = dict(symbols)
                data['label1'] = edge1[2]['orig_label'].replace('true', 'True')
                data['label2'] = edge2[2]['orig_label'].replace('true', 'True')
                reduced_label = eval('sympy.simplify(sympy.simplify(label1) & sympy.simplify(label2))', data)
                # check that the reduced label isn't False
                if reduced_label:
                    edges.append(((u, v), (edge1[1], edge2[1]), {'label': str(reduced_label)}))
        calc_prod = nx.DiGraph()
        calc_prod.update(nodes=nodes)
        calc_prod.update(edges=edges)
        # remove nodes that have no way to get in
        done = False
        while not done:
            remove_nodes = []
            for node in calc_prod.nodes:
                if node == ('1', '1'):
                    continue
                # find all nodes that are not this node that can get you into this node
                src_nodes = {u for (u,v) in calc_prod.in_edges(node) if u != node}
                if len(src_nodes) == 0:
                    remove_nodes.append(node)
            for node in remove_nodes:
                calc_prod.remove_node(node)
            done = len(remove_nodes) == 0
        # set accepting
        reset_nodes = []
        for node in calc_prod.nodes():
            u, v = node
            calc_prod.nodes[node]['accepting'] = reset_dfa.nodes[u]['accepting']
            if calc_prod.nodes[node]['accepting']:
                reset_nodes.append(v)
        reset_nodes = set(reset_nodes)
        if save_product:
            sg_img = get_pydot_image(calc_prod, color=False, svg=True)
            with open(f'{self.name}_prod.svg', 'wb') as f:
                f.write(sg_img)
        return reset_nodes

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
            all_nodes.update(data)
        all_nodes.update([node for node in sg.nodes if node.name == 'ego'])
        graph_copy = nx.induced_subgraph(sg, all_nodes)
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
            if self.reset_prop.ltldfa.step(self.get_last_predicates()):
                self.in_violation = False
                self.violations[-1][-1] = self.time
                self.ltldfa.set_state(self.reset_state)
                self.reset_prop.ltldfa.reset()
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


class Subproperty(Property):
    """
    A subproperty is another property that depends on the same predicate values as another property
    Subproperties cannot be reset.
    """

    def __init__(self, parent: Optional[Property], property_name, property_string):
        if type(parent) == Subproperty:
            raise AttributeError(f"Got {property_name} as a subproperty of {parent.name}. "
                                 "Subproperty objects cannot be subproperties of other Subproperty objects.")
        # This isn't the prettiest, but convert it back into the format the init func expects
        preds = [(name, value) for name, value in parent.predicates.items()] if parent is not None else []
        super().__init__(property_name=property_name,
                         property_string=property_string,
                         predicates=preds)
        self.parent = parent

    def is_subproperty_of(self, prop: Property) -> bool:
        return self.parent == prop


class NeverAccepting(Subproperty):
    def __init__(self):
        super().__init__(None, "NeverAccepting", "False")

    def is_subproperty_of(self, prop: Property) -> bool:
        return True
