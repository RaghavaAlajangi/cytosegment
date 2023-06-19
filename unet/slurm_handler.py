import itertools
from pathlib import Path
import subprocess as sp

import yaml

params_path = Path(__file__).parents[1] / "params" / "unet_params.yaml"
bash_path = Path(__file__).parents[1] / "slurm_template" / "train_hpc.sh"


def get_exp_name(params):
    path1 = "unet_tune_d{{depth}}_f{{filters}}_dr{{dropout}}_aug" \
            "{{augmentation}}_b{{batch_size}}_alpha{{alpha}}_gamma{{gamma}}_" \
            "lr{{learn_rate}}"
    path2 = "unet_aug{{augmentation}}_b{{batch_size}}_alpha{{alpha}}" \
            "_gamma{{gamma}}_lr{{learn_rate}}"

    req_params = {**params.get("model"), **params.get("dataset"),
                  **params.get("criterion"), **params.get("optimizer")}

    if "depth" in req_params.keys():
        for key, val in req_params.items():
            path1 = path1.replace("{{" + key + "}}", str(val))
        return Path(path1)
    else:
        for key, val in req_params.items():
            path2 = path2.replace("{{" + key + "}}", str(val))
        return Path(path2)


def create_params_combinations(original_dict):
    """
    The function takes a dictionary of dictionaries as input and then generates
    all possible combinations of the values in each nested dictionary.
    For example, if we have the following input:
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


params = yaml.safe_load(open(params_path))
slurm_file = bash_path.read_text()
experiment_dicts = create_params_combinations(params)

for n, exp_dict in enumerate(experiment_dicts):
    path_out = params["others"]["path_out"][0]
    exp_name = get_exp_name(exp_dict)
    exp_path = Path(path_out) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    slurm_path = exp_path / "job.sh"
    params_path = exp_path / "params.yaml"
    hpc_params = params["hpc_params"]
    hpc_job_dict = {
        "EXP_NAME": exp_name,
        "PATH_OUT": path_out,
        "JOB_NAME": exp_name,
        "MAIL_ID": hpc_params["mail_id"],
        "MAX_MEM": f"{hpc_params['max_mem_GB']:.0f}G",
        "MAX_TIME": f"{int(hpc_params['max_time_hours']):02d}:00:00",
        "PARAMS_PATH": params_path
    }
    for k, v in hpc_job_dict.items():
        slurm_file = slurm_file.replace("{{" + k + "}}", str(v))
    slurm_path.write_text(slurm_file)

    with open(params_path, 'w') as file:
        # Update path_out in params.yaml with experiment name
        exp_dict["others"]["path_out"] = str(exp_path)
        yaml.dump(exp_dict, file, sort_keys=False)

    slout = sp.check_output(f'sbatch {slurm_path}', shell=True)
    for line in slout.decode().split("\n"):
        if line.startswith("submitted batch job"):
            print(f"{n}) {line}")
