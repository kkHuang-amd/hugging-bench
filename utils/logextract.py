import re
import json
import argparse

train_metrics = [
    'epoch',
    'train_loss',
    'train_runtime',
    'train_samples',
    'train_samples_per_second',
    'train_steps_per_second'
]
metrics_line = r'***** train metrics *****'


def extract(log_file:str) -> dict:
    out = {}

    if not log_file:
        return out

    with open(log_file, 'r') as file:
        lines = file.read().splitlines()
        
    i_metrics = None
    for i, line in enumerate(lines):
        if line.startswith(metrics_line):
            i_metrics = i
            break
    
    if not i_metrics:
        print('Training metrics not found!')
        return out

    def generate_metric_re_string(metric:str) -> str:
        return f'\s*{metric}\s*=\s*(.*)\s*'

    re_strings = [ generate_metric_re_string(metric) for metric in train_metrics ]
    
    for line in lines[i_metrics:]:
        for metric, re_string in zip(train_metrics, re_strings):
            g = re.match(re_string, line)
            if g:
                out[metric] = g.groups()[0]
                break

    print(out)

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--log_file', type=str, help='the name of the log file to extract data from')
    args=parser.parse_args()
    return args


def main():
    args = parse_args()
    train_metrics = extract(args.log_file)


if __name__ == '__main__':
    main() 