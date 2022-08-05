import csv

from collections import defaultdict

class Entry:
    def __init__(self):
        pass

class LStarExportEntry(Entry):
        def __init__(self, model_size, output_queries, steps_output_queries, eq_oracle_queries, steps_eq_oracle, conformance_coverage, learning_rounds, sum_queries, sum_steps, average_trace_len) -> None:
            self.model_size = model_size
            self.output_queries = output_queries
            self.steps_output_queries = steps_output_queries
            self.eq_oracle_queries = eq_oracle_queries
            self.steps_eq_oracle = steps_eq_oracle
            self.conformance_coverage = conformance_coverage
            self.learning_rounds = learning_rounds
            self.sum_queries = sum_queries
            self.sum_steps = sum_steps
            self.average_trace_len = average_trace_len
        
        @staticmethod
        def pretty_printed_attr():
            return {
                "States" : "model_size",
                "Output queries" : "output_queries",
                "Steps output queries" : "steps_output_queries",
                "Equivalence queries" : "eq_oracle_queries",
                "Steps equivalence queries" : "steps_eq_oracle",
                "Conformance (coverage) %" : "conformance_coverage",
                "Learning rounds": "learning_rounds",
                "Sum queries" : "sum_queries",
                "Sum steps" : "sum_steps",
                "Average trace length": "average_trace_len"
            }

class RPNIExportEntry(Entry):

        def __init__(self, model_size, conformance_coverage, conformance_random, data_size, average_len, correctly_learned_model) -> None:
            self.model_size = model_size
            self.conformance_coverage = conformance_coverage
            self.conformance_random = conformance_random
            self.data_size = data_size
            self.average_len = average_len
            self.correctly_learned_model = correctly_learned_model
        
        @staticmethod
        def pretty_printed_attr():
            return {
                "States" : "model_size",
                "Conformance (coverage) %" : "conformance_coverage",
                "Conformance (random) %" : "conformance_random",
                "Data size" : "data_size",
                "Average trace length": "average_len",
                "Correctly learned model" : "correctly_learned_model"
            }

class CachedLStarExportEntry(Entry):

        def __init__(self, conformance_coverage, random_sample_size, performed_queries, cache_hits, learning_rounds) -> None:
            self.conformance_coverage = conformance_coverage
            self.random_sample_size = random_sample_size
            self.performed_queries = performed_queries
            self.cache_hits = cache_hits
            self.learning_rounds = learning_rounds
        
        @staticmethod
        def pretty_printed_attr():
            return {
                "Conformance (coverage) %" : "conformance_coverage",
                "Random sample" : "random_sample_size",
                "Active Queries" : "performed_queries",
                "Cache hits": "cache_hits",
                "Learning rounds" : "learning_rounds"
            }

class DataExporter:

    def __init__(self, attributes) -> None:
        self.export_data = defaultdict(Entry)
        self.attributes = attributes

    def add_entry(self, model_name, export_entry):
        self.export_data[model_name] =  export_entry
        

    def export_csv(self, filename):
        with open(f'{filename}.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            header = [""] + list(self.export_data.keys())

            # write the header
            writer.writerow(header)

            for k in self.attributes:
                row = [k]
                for model in self.export_data.keys():
                    elem = getattr(self.export_data[model], self.attributes[k])
                    row.append(f'{round(elem[0],2):.2f} ({round(elem[1],2):.2f})')
                writer.writerow(row)

class RPNIDataExporter(DataExporter):

    def __init__(self, attributes) -> None:
        super().__init__(attributes)
        self.export_data = defaultdict(defaultdict)

    def add_model(self, model_name):
        self.export_data[model_name] = defaultdict(Entry)

    def export_csv(self, filename, experiments):
        with open(f'{filename}.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            header = [""]*2 + list(self.export_data.keys())

            # write the header
            writer.writerow(header)
            for e in experiments:
                for k in self.attributes:
                    row = [e, k] 
                    for model in self.export_data.keys():
                        elem = getattr(self.export_data[model][e], self.attributes[k])
                        row.append(f'{round(elem[0],2):.2f} ({round(elem[1],2):.2f})')
                    writer.writerow(row)
