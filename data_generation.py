import random

from aalpy.SULs import MealySUL
from aalpy.base.SUL import CacheSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle


def data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set=True):
    observation_table_data = []
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
            observation_table_data.append(list(zip(inputs, output)))

    return observation_table_data


def data_from_computed_e_set(hypothesis, include_extended_s_set=True):
    return data_from_l_star_E_set(hypothesis, hypothesis.compute_characterization_set(), include_extended_s_set)


def minimized_char_set_data(hypothesis, include_extended_s_set=True):
    from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import extract_unique_sequences, createPTA
    data = data_from_computed_e_set(hypothesis, include_extended_s_set)
    pruned_data = []
    for seq in extract_unique_sequences(createPTA(data, automaton_type='mealy')):
        pruned_data.append([io[0] for io in seq])
    return pruned_data


def generate_random_data(model, num_sequences, min_sequence_len, max_sequence_len):
    data = []
    input_alphabet = model.get_input_alphabet()
    for _ in range(num_sequences):
        inputs = random.choices(input_alphabet, k=random.randint(min_sequence_len, max_sequence_len))
        output = model.compute_output_seq(model.initial_state, inputs)
        data.append(list(zip(inputs, output)))
    return data


def l_star_with_populated_cache(model, cache_data):
    sul = CacheSUL(MealySUL(model))

    for sample in cache_data:
        sul.cache.reset()
        for i, o in sample:
            sul.cache.step_in_cache(i, o)

    alphabet = model.get_input_alphabet()
    eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=8)

    l_star_model, data = run_Lstar(alphabet, sul, eq_oracle, 'mealy', print_level=0, cache_and_non_det_check=False,
                                   return_data=True)

    cache_hits = sul.num_cached_queries
    queries_posed = data['queries_learning'] + data['queries_eq_oracle'] - cache_hits

    return queries_posed, cache_hits