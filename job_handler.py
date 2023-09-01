import click
import itertools
from pathlib import Path
import subprocess as sp

import yaml

params_path = Path(__file__).parents[0] / "params" / "unet_params.yaml"
bash_path = Path(__file__).parents[0] / "slurm_template" / "train_hpc.sh"

main_params = yaml.safe_load(open(params_path))


def get_exp_name(params):
    path1 = "unet_tune_{{conv_block}}_d{{depth}}_f{{filters}}_di{{dilation}}" \
            "_dr{{dropout}}_BN_{{batch_norm}}_{{up_mode}}_at{{attention}}_" \
            "relu{{relu}}_{{weight_init}}_aug{{augmentation}}_" \
            "b{{batch_size}}_al{{alpha}}_gm{{gamma}}_lr{{learn_rate}}"
    path2 = "unet_{{weight_init}}_aug{{augmentation}}_b{{batch_size}}_" \
            "al{{alpha}}_gm{{gamma}}_lr{{learn_rate}}"

    req_params = {**params.get("model"), **params.get("dataset"),
                  **params.get("criterion"), **params.get("optimizer")}

    if "depth" in req_params.keys():
        for key, val in req_params.items():
            val = "F" if str(val) == "False" else "T" if str(
                val) == "True" else val
            path1 = path1.replace("{{" + key + "}}", str(val))
        return Path(path1)
    else:
        for key, val in req_params.items():
            path2 = path2.replace("{{" + key + "}}", str(val))
        return Path(path2)


def create_params_combinations(original_dict):
    """ Take a dictionary of dictionaries as input and then generates
    all possible combinations of the values in each nested dictionary.
    Parameters
    ----------
    original_dict
        Nested dictionary with params
    Returns
    -------
    A list of dictionaries
    """
    dict_combinations = []
    # Generate all possible combinations of the values in
    # each nested dictionary
    combinations = []
    for o_key, o_value in original_dict.items():
        if o_key != "hpc_params":
            params = [v if isinstance(v, list) else [v] for v in
                      o_value.values()]
            combinations.append(itertools.product(*params))

    for combo in itertools.product(*combinations):
        new_dict = {}
        for i, (dict_key, dict_value) in enumerate(original_dict.items()):
            if dict_key != "hpc_params":
                nested_dict = {}
                for j, (key, value) in enumerate(dict_value.items()):
                    nested_dict[key] = combo[i][j]
                new_dict[dict_key] = nested_dict
        dict_combinations.append(new_dict)
    return dict_combinations


@click.command(help="Script for handling jobs")
@click.option("--slurm", is_flag=True, help="Run slurm jobs")
@click.option("--local", is_flag=True, help="Run jobs locally")
def main(slurm=False, local=False):
    experiment_dicts = create_params_combinations(main_params)
    print("="*80)
    print(f"Total experiments: {len(experiment_dicts)}")
    for n, exp_dict in enumerate(experiment_dicts):
        # Create experiment name using params dict
        exp_name = get_exp_name(exp_dict)
        # Get the pathout from the params dict
        path_out = exp_dict["others"]["path_out"]
        # Create experiment directory
        exp_path = Path(path_out) / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        # Replace pathout with experiment path
        exp_dict["others"]["path_out"] = str(exp_path)
        # Save experiment yaml file
        params_path = exp_path / "params.yaml"
        with open(params_path, 'w') as file:
            yaml.dump(exp_dict, file, sort_keys=False)
        if slurm:
            # Read the slurm job template
            slurm_file = bash_path.read_text()
            # Create directory to save hpc output files
            job_dir = Path(path_out) / exp_name / "job_files"
            job_dir.mkdir(parents=True, exist_ok=True)
            # Create experiment params and slurm job files
            slurm_path = exp_path / "job.sh"
            hpc_params = main_params["hpc_params"]
            hpc_job_dict = {
                "EXP_NAME": exp_name,
                "PATH_OUT": path_out,
                "JOB_NAME": exp_name,
                "MAIL_ID": hpc_params["mail_id"],
                "MAX_MEM": f"{hpc_params['max_mem_GB']:.0f}G",
                "MAX_TIME": f"{int(hpc_params['max_time_hours']):02d}:00:00",
                "PARAMS_PATH": params_path
            }
            # Update and save slurm job file
            for k, v in hpc_job_dict.items():
                slurm_file = slurm_file.replace("{{" + k + "}}", str(v))
            slurm_path.write_text(slurm_file)
            # Run slurm jobs
            sp.run(["cd", str(Path(path_out) / exp_name)], shell=True)
            slout = sp.check_output(f'sbatch {slurm_path}', shell=True)
            for line in slout.decode().split("\n"):
                line = line.strip().lower()
                if line.startswith("submitted batch job"):
                    print(f"{n}) {line}")
        elif local:
            sp.run(["python", "-m", "semanticsegmenter", "--params_path",
                    str(Path(params_path))], shell=True)
        else:
            print("Please specify job type (--slurm or --local)!")


if __name__ == "__main__":
    main()
