from os import listdir

from aalpy.SULs import MealySUL
from aalpy.learning_algs import run_Lstar, run_RPNI
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file, compare_automata


def get_paths(t, paths=None, current_path=None):
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []

    if len(t.children) == 0:
        paths.append(current_path)
    else:
        for inp, child in t.children.items():
            current_path.append(inp)
            get_paths(child, paths, list(current_path))
    return paths


def data_from_cache(l_star_eq_oracle):
    cache = l_star_eq_oracle.sul.cache

    cache_data = []
    for inputs in get_paths(cache.root_node):
        outputs = eq_oracle.sul.query(inputs)
        cache_data.append(list(zip(inputs, outputs)))

    return cache_data


def data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set=True):
    data = []
    prefixes = [state.prefix for state in hypothesis.states]

    if include_extended_s_set:
        extended_prefixes = []
        for p in prefixes:
            for a in hypothesis.get_input_alphabet():
                extended_prefixes.append(p + tuple([a]))

        prefixes.extend(extended_prefixes)

    for prefix in prefixes:
        for suffix in e_set:
            inputs = prefix + suffix
            output = hypothesis.compute_output_seq(hypothesis.initial_state, inputs)
            data.append(list(zip(inputs, output)))

    return data


def data_from_computed_e_set(hypothesis, include_extended_s_set=True):
    e_set = hypothesis.compute_characterization_set()
    return data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set)


bluetooth_models = []

for dot_file in listdir('./automata'):
    model_name = dot_file[:-4]
    model = load_automaton_from_file(f'./automata/{dot_file}', automaton_type='mealy')
    bluetooth_models.append((model_name, model))

for model_name, model in bluetooth_models:
    sul = MealySUL(model)
    alphabet = model.get_input_alphabet()
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, return_data=True)

    e_set = data['characterization set']
    learning_queries = data['queries_learning']
    eq_oracle_queries = data['queries_eq_oracle']

    # data = data_from_cache(eq_oracle) # TODO I think get_paths does not work
    data = data_from_l_star_E_set(l_star_model, e_set, include_extended_s_set=True)
    data = data_from_computed_e_set(l_star_model, include_extended_s_set=True)

    rpni_model = run_RPNI(data, automaton_type='mealy', input_completeness='sink_state', print_info=False)

    print(f'Experiment: {model_name}')

    if set(rpni_model.get_input_alphabet()) != set(l_star_model.get_input_alphabet()):
        print('Learned models do not have the same input alphabets => RPNI model is not input complete.')
        continue

    cex = compare_automata(rpni_model, l_star_model)
    if cex:
        cex = cex[0]
        print('Counterexample found between models learned by RPNI and L*.')
        print('Inputs :', cex)
        print('L*     :', l_star_model.compute_output_seq(l_star_model.initial_state, cex))
        print('RPNI   :',  rpni_model.compute_output_seq(rpni_model.initial_state, cex))

    else:
        print('RPNI and L* learned same models.')
        if rpni_model.size != l_star_model.size:
            print(f'    Models do have different size.\n    RPNI {rpni_model.size} vs. L* {l_star_model.size}')
    print('-----------------------------------------------')