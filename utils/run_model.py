"""
Run a single hugging-bench model with a single set of parameters (n_gcd, batch_size, max_steps)

python run_model.py -m bart -g 8 -bs 16 -s 150
"""

import subprocess
import argparse
from typing import List

# import metrics
from metrics import extract_metrics_from_stdout


def generate_model_run_command(model:str, n_gcd:int, batch_size:int, max_steps:int,
    project_dir:str='/workspace') -> List[str]:
    # Set default param values if not given
    n_gcd = 1 if n_gcd is None else n_gcd
    batch_size = 1 if batch_size is None else batch_size
    max_steps = 150 if max_steps is None else max_steps

    command = [ f'{project_dir}/scripts/run-{model}.sh' ]
    command += ['--n_gcd', n_gcd]
    command += ['--batch_size', batch_size]
    command += ['--max_steps', max_steps]

    command = [str(item) for item in command]
    return command


def run_model(model:str, n_gcd:int, batch_size:int, max_steps:int,
                    project_dir:str='/workspace') -> str:
    """
    Run a hugging-bench model and return the command line stdout as decoded text.
    """
    model_run_command = generate_model_run_command(model=model, n_gcd=n_gcd, batch_size=batch_size, max_steps=max_steps)
    sproc = subprocess.run(model_run_command, capture_output=True)
    stdout = sproc.stdout.decode("utf-8")
    return stdout


def extract_run_train_metrics(stdout:str) -> dict:
    """
    Extract the train metrics from the stdout from a completed model run.
    """
    train_metrics = extract_metrics_from_stdout(stdout)
    return train_metrics


def is_run_oom(stdout:str) -> bool:
    """
    Determine if a model run had a GPU "out of memory" error from its stdout.
    """
    if "out of memory" in stdout:
        return True
    return False


def is_run_successful(stdout:str) -> bool:
    """
    Determine if a model run was successful from its stdout.  A run is considered successful if training metrics were generated successfully.
    """
    train_metrics = extract_metrics_from_stdout(stdout)
    if (len(train_metrics) > 0):
        return True
    return False


class ModelRunner:
    def __init__(self, project_dir:str='/workspace'):
        self._project_dir = project_dir
    
    def run(self, model:str, n_gcd:int, batch_size:int, max_steps:int):
        self._model = model
        self._n_gcd = n_gcd
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._run_stdout = run_model(model=self._model, n_gcd=self._n_gcd, batch_size=self._batch_size, max_steps=self._max_steps, project_dir=self._project_dir)
        self._run_train_metrics = extract_run_train_metrics(self._run_stdout)
        self._run_success = is_run_successful(self._run_stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('-pdir', '--project_dir', type=str, help="directory where project lives")
    parser.add_argument('-m', '--model', type=str, help="the name of the model")
    parser.add_argument('-g', '--n_gcd', type=int, help="number of GCDs to use when running model")
    parser.add_argument('-bs', '--batch_size', type=int, help="training batch size")
    parser.add_argument('-s', '--max_steps', type=int, help="maximum number of training steps")
    args=parser.parse_args()
    return args


def main():
    args = parse_args()
    model_runner = ModelRunner()
    model_runner.run(model=args.model, n_gcd=args.n_gcd, batch_size=args.batch_size, max_steps=args.max_steps)
    print(model_runner._run_train_metrics)
    print(f"run successful? {model_runner._run_success}")


if __name__=='__main__':
    main()