"""
Search for the largest batch size in a range that can run a model successfully using a given GPU configuration.

python search_batch_size.py -m bart -g 3 -s 150 -o /workspace/search_results/batch_size.json -lo 165 -hi 175 
"""

import os
import subprocess
import argparse
import json
import time

from run_model import ModelRunner


class BatchSizeModelRunner(ModelRunner):
    def __init__(self, model:str, n_gcd:int, max_steps:int,
                    project_dir:str='/workspace'):
        super().__init__(project_dir=project_dir)
        self.model = model
        self.n_gcd = n_gcd
        self.max_steps = max_steps
    
    def run(self, batch_size):
        self.batch_size = batch_size
        super().run(model=self.model, n_gcd=self.n_gcd, batch_size=self.batch_size, max_steps=self.max_steps)

    def search_max_batch_size(self, range_min:int=1, range_max:int=500) -> int:
        """ Find the largest batch size in a range that results in a successful training run.
        Let the largest batch size be `bs_opt`.
        Assume that the maximum batch size exists within the range: range_min <= bs_opt < range_max .
        Assume that model runs are successful for all batch sizes bs <= bs_opt .

        This function also creates `self.search_records`.
        """
        self.search_records = []

        # Try highest value in user-specified range
        record = {}
        record['batch_size'] = range_max
        self.run(range_max)
        record['success'] = self._run_success
        record['train_metrics'] = self._run_train_metrics
        self.search_records.append(record)
        if self._run_success:
            return self.search_records
        
        # Initialize search range
        low, high = range_min, range_max
        mid = int((low + high) // 2)

        while low < mid:
            # Initialize single run record
            record = {}
            record['batch_size'] = mid

            # Run with current batch size
            self.run(mid)

            # Log run outcome
            record['success'] = self._run_success
            record['train_metrics'] = self._run_train_metrics
            print(record)
            self.search_records.append(record)

            # Update search range
            if self._run_success == True:
                low = mid
            else:
                high = mid

            # Update batch size
            mid = int((low + high) // 2)

        return self.search_records


# def is_success(batch_size, opt:int=1) -> bool:
#     if batch_size <= opt:
#         return True
#     return False
#
#
# def find_max_batch_size(bs_range_min:int=1, bs_range_max:int=100):
#     if is_success(bs_range_max) == True:
#         return bs_range_max
    
#     low, high = bs_range_min, bs_range_max
#     mid = int((low + high) // 2)

#     while low < mid:
#         if is_success(mid) == True:
#             low = mid
#         else:
#             high = mid
#         mid = int((low + high) // 2)

#     return low


def search_model_max_batch_size(model:str, n_gcd:int, max_steps:int,
                                batch_size_min, batch_size_max,
                                project_dir:str=None) -> dict:
    record = {}
    record['model'] = model
    record['n_gcd'] = n_gcd
    record['max_steps'] = max_steps
    record['batch_size_min'] = batch_size_min
    record['batch_size_max'] = batch_size_max

    model_runner = BatchSizeModelRunner(model=model, n_gcd=n_gcd, max_steps=max_steps)
    search_records = model_runner.search_max_batch_size(batch_size_min, batch_size_max)
    record['search_records'] = search_records

    return record


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-odir', '--output_dir', type=str, default='/workspace/search_results', help="directory= where JSON output should be written")
    parser.add_argument('-lo', '--batch_size_min', type=int, help="lowest batch size in search range")
    parser.add_argument('-hi', '--batch_size_max', type=int, help="highest batch size in search range")
    parser.add_argument('-m', '--model', type=str, help="the name of the model")
    parser.add_argument('-g', '--n_gcd', type=int, help="number of GCDs to use when running model")
    parser.add_argument('-s', '--max_steps', type=int, help="maximum number of training steps")
    args=parser.parse_args()
    return args


def main():
    args = parse_args()
    record = search_model_max_batch_size(model=args.model, n_gcd=args.n_gcd, max_steps=args.max_steps, batch_size_min=args.batch_size_min, batch_size_max=args.batch_size_max)

    timestr = time.strftime("%Y%m%dT%H%M%S")
    output_file = f'{args.output_dir}/search_batch_size-{timestr}.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(json.dumps(record, indent=4))


if __name__ == '__main__':
    main()