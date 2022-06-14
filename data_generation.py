import random

from math import ceil

from aalpy.SULs import MealySUL
from aalpy.base.SUL import CacheSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils.HelperFunctions import all_prefixes


def data_from_l_star_E_set(hypothesis, e_set, include_extended_s_set=True, prefix_closed=True):
    observation_table_data = []
    prefixes = [state.prefix for state in hypothesis.states]

    if include_extended_s_set:
        extended_prefixes = []
        for p in prefixes:
            for a in hypothesis.get_input_alphabet():
                extended_prefixes.append(p + tuple([a]))

        prefixes.extend(extended_prefixes)

    data_set = set()
    for prefix in prefixes:
        for suffix in e_set:
            cell = prefix + suffix
            if prefix_closed:
                data_set.update(all_prefixes(cell))
            else:
                data_set.add(cell)

    for seq in list(data_set):
        output = hypothesis.compute_output_seq(hypothesis.initial_state, seq)[-1]
        observation_table_data.append((seq, output))

    return observation_table_data


def data_from_computed_e_set(hypothesis, include_extended_s_set=True, prefix_closed=True):
    return data_from_l_star_E_set(hypothesis, hypothesis.compute_characterization_set(), include_extended_s_set, prefix_closed)


def minimized_char_set_data(hypothesis, include_extended_s_set=True, prefix_closed=True):
    from aalpy.learning_algs.deterministic_passive.rpni_helper_functions import extract_unique_sequences, createPTA
    data = data_from_computed_e_set(hypothesis, include_extended_s_set, prefix_closed)
    input_sequences = []
    for seq in extract_unique_sequences(createPTA(data, automaton_type='mealy')):
        input_sequences.append([io[0][0] for io in seq])

    if prefix_closed:
        prefix_closed_seq = set()
        for seq in input_sequences:
            prefix_closed_seq.update(all_prefixes(seq))
        input_sequences = list(prefix_closed_seq)

    pruned_data = []
    for seq in input_sequences:
        output = hypothesis.compute_output_seq(hypothesis.initial_state, seq)[-1]
        pruned_data.append((seq, output))

    return pruned_data


def generate_random_data(model, num_sequences, min_sequence_len, max_sequence_len, prefix_closed=True):
    data = []
    input_alphabet = model.get_input_alphabet()
    random_sequences = [random.choices(input_alphabet, k=random.randint(min_sequence_len, max_sequence_len))
                        for _ in range(num_sequences)]

    sequence_step_sum = sum([len(i) for i in random_sequences])

    average_length = sequence_step_sum / num_sequences

    if prefix_closed:
        prefix_closed_seq = set()
        for seq in random_sequences:
            prefix_closed_seq.update(all_prefixes(seq))
        random_sequences = list(prefix_closed_seq)

    for seq in random_sequences:
        output = model.compute_output_seq(model.initial_state, seq)[-1]
        data.append((seq, output))

    return data


def l_star_with_populated_cache(model, cache_data):
    # Note: This circumvents the fact if the data is not prefix-closed
    cache_io_seq = []
    for seq in cache_data:
        i = seq[0]
        o = model.compute_output_seq(model.initial_state, i)
        cache_io_seq.append(zip(i, o))

    sul = CacheSUL(MealySUL(model))

    for sample in cache_io_seq:
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