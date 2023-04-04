from datetime import timedelta
import itertools
from pathlib import Path
import time

import click
import yaml

from .ml_trainer import SetupTrainer


def create_comb_dicts(original_dict):
    dict_combinations = []
    # Generate all possible combinations of the values in
    # each nested dictionary
    combinations = []
    for dict_key, dict_value in original_dict.items():
        combinations.append(itertools.product(*dict_value.values()))

    for combo in itertools.product(*combinations):
        new_dict = {}
        for i, (dict_key, dict_value) in enumerate(original_dict.items()):
            nested_dict = {}
            for j, (key, value) in enumerate(dict_value.items()):
                nested_dict[key] = combo[i][j]
            new_dict[dict_key] = nested_dict
        dict_combinations.append(new_dict)
    return dict_combinations


@click.command(help="Script to start the training")
@click.option("--params_path",
              type=click.Path(exists=True,
                              dir_okay=False,
                              resolve_path=True,
                              path_type=Path),
              help="Path to params file (.yaml)")
def main(params_path):
    params = yaml.safe_load(open(params_path))
    params_list = create_comb_dicts(params)
    print("Total number of experiments:", len(params_list))
    for params in params_list:
        trainer = SetupTrainer.with_params(params)
        tik = time.time()
        print("Started training.....")
        trainer.start_train()
        tok = time.time() - tik
        train_time = str(timedelta(seconds=tok)).split('.')[0]
        print(f"Total training time: {train_time}")


if __name__ == "__main__":
    main()
