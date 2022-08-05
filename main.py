import sys

from csv_export import *
from data_classes import *
from data_generation import *
from learning_setups import *
from model_comparison import *



if __name__ == "__main__":

    args = sys.argv[1:]

    # load all automata from benchmark
    benchmark = args[0] if len(args) == 1 else "BLE"  # 'MQTT' or 'BLE'
    benchmark_models = load_dot_files(benchmark)

    # generate test suite for conformance testing after learning
    num_tests = 10000
    test_cases_coverage = create_test_cases(benchmark_models, num_tests, 'coverage')
    test_cases_random = create_test_cases(benchmark_models, num_tests, 'random')

    # number of repetition of each learning algorithm
    repeats_per_experiment = 5  # 5

    # levels on which output is printed
    # 0: no output is printed
    # 1: data summary is printed
    # 2: data summary + debug output
    verbose_level = 1

    # export csv files that contain the results of the performed evaluation
    csv = True
    l_star_data_export = DataExporter(LStarExportEntry.pretty_printed_attr())
    rpni_data_export = RPNIDataExporter(RPNIExportEntry.pretty_printed_attr())
    cached_l_star_data_export = DataExporter(CachedLStarExportEntry.pretty_printed_attr())

    # rpni_model_l_star_str = "l* data"
    rpni_model_random_l_star_length_str = "random |l* data|"
    rpni_model_random_large_set_str = "random 2*|l* data|"
    rpni_model_random_long_traces_str = "random long traces"
    rpni_model_minimized_char_set_str = "l* data (minimized)"
    # rpni_model_random_good_enough_str = "random corr"

    rpni_data = {
        # rpni_model_l_star_str: data_l_star,
        rpni_model_random_l_star_length_str: None,
        rpni_model_random_large_set_str: None,
        rpni_model_random_long_traces_str: None,
        # rpni_model_minimized_char_set_str: data_minimized_char_set,
        # rpni_model_random_good_enough_str: data_random_good_enough
    }

    rpni_data_names = rpni_data.keys()

    for model_name, model in benchmark_models:

        l_star_experiment_data = defaultdict(list)
        rpni_experiment_data = defaultdict(defaultdict)
        cached_l_star_experiment_data = defaultdict(list)

        # parameter for equivalence oracle
        walks_per_state = 25
        walk_len = 30

        sul = MealySUL(model)
        alphabet = model.get_input_alphabet()

        rpni_data_export.add_model(model_name)

        for _ in range(repeats_per_experiment):
            eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=walks_per_state, walk_len=walk_len)
            l_star_res = l_star_experiment(model, test_cases_coverage[model_name], test_cases_random[model_name],
                                           alphabet, eq_oracle)
            l_star_experiment_data[model_name].append(l_star_res)

        avg_query_steps = data_stats("average_trace_len", l_star_experiment_data[model_name])[0]
        max_sequence_length = round((avg_query_steps - 0.5) * 2)

        learning_queries = round(data_stats("sum_queries", l_star_experiment_data[model_name])[0])

        data_l_star = data_from_computed_e_set(model, include_extended_s_set=True, verbose=verbose_level == 2)

        data_minimized_char_set = minimized_char_set_data(model, include_extended_s_set=True,
                                                          verbose=verbose_level == 2)

        model_size = model.size

        rpni_experiment_data[model_name] = defaultdict(list)

        for _ in range(repeats_per_experiment):
            # data generation:

            # random |l* data|
            data_random_l_star_length = generate_random_data(model, num_sequences=learning_queries, min_sequence_len=1,
                                                             max_sequence_len=max_sequence_length,
                                                             verbose=verbose_level == 2)

            # random 2*|l* data|
            data_random_large_set = generate_random_data(model, num_sequences=(learning_queries * 2),
                                                         min_sequence_len=1, max_sequence_len=max_sequence_length,
                                                         verbose=verbose_level == 2)

            # random long traces
            data_random_long_traces = generate_random_data(model, num_sequences=learning_queries,
                                                           min_sequence_len=model_size,
                                                           max_sequence_len=(model_size * 2),
                                                           verbose=verbose_level == 2)

            # good enough data (randomly generated data that learns correctly) data_random_good_enough =
            # generate_random_data(model, num_sequences= learning_queries * 1, min_sequence_len=model_size,
            # max_sequence_len=max(10,model_size)  + model_size, verbose=verbose)

            rpni_data = {
                # rpni_model_l_star_str: data_l_star,
                rpni_model_random_l_star_length_str: data_random_l_star_length,
                rpni_model_random_large_set_str: data_random_large_set,
                rpni_model_random_long_traces_str: data_random_long_traces,
                # rpni_model_minimized_char_set_str: data_minimized_char_set,
                # rpni_model_random_good_enough_str: data_random_good_enough
            }

            for data_name, data in rpni_data.items():
                rpni_experiment_data[model_name][data_name].append(
                    rpni_experiment(data, model, test_cases_coverage[model_name], test_cases_random[model_name]))

            eq_oracle = StatePrefixEqOracle(alphabet, sul, walks_per_state=walks_per_state, walk_len=walk_len)
            cached_l_star_experiment = l_star_with_initial_cache(data_random_l_star_length, model, eq_oracle,
                                                                 test_cases_coverage[model_name])
            # L* cached with random samples
            cached_l_star_experiment_data[model_name].append(cached_l_star_experiment)

        rpni_experiment_data[model_name][rpni_model_minimized_char_set_str].append(
            rpni_experiment(data_minimized_char_set, model, test_cases_coverage[model_name],
                            test_cases_random[model_name]))

        print(f'\n\n------------------{model_name}------------------')
        l_star_data_export.add_entry(model_name,
                                     l_star_summary(l_star_experiment_data[model_name], verbose=verbose_level >= 1))

        rpni_data_export.add_entry(model_name, rpni_summary(rpni_experiment_data[model_name], rpni_data_names,
                                                            rpni_model_minimized_char_set_str,
                                                            verbose=verbose_level >= 1))

        cached_l_star_data_export.add_entry(model_name, cached_l_star_summary(cached_l_star_experiment_data[model_name],
                                                                              verbose=verbose_level >= 1))

    if csv:
        l_star_data_export.export_csv(f'{benchmark}_l_star_data')
        rpni_experiments = [*rpni_data] + [rpni_model_minimized_char_set_str]
        rpni_data_export.export_csv(f'{benchmark}_rpni_data', rpni_experiments)
        cached_l_star_data_export.export_csv(f'{benchmark}_cached_l_star_data')
