import yaml

from .helper_methods import retrieve_train_data_path


def test_cli(script_runner):
    """Test the cytosegment command line interface."""
    path_in = retrieve_train_data_path("test_dataset.zip") / "dataset"
    path_out = path_in / "output"
    print(path_out)
    result = script_runner.run(
        [
            "cytosegment",
            f"data.path={str(path_in)}",
            f"path_out={str(path_out)}",
            # Test only a small model
            "model.depth=1",
            "model.filters=2",
            "model.conv_block=single",
            # Test only a small number of epochs
            "max_epochs=2",
        ]
    )

    assert result.success

    log_files = list(path_out.glob("**/*.yaml"))

    for file in log_files:
        if file.name == "train_logs.yaml":
            train_logs = yaml.safe_load(open(file))
            print(train_logs)
            assert train_logs["params"]["epochs"] == 2
            assert train_logs["params"]["train_samples"] == 8
            assert train_logs["params"]["valid_samples"] == 2
            assert train_logs["params"]["test_samples"] == 3
        elif file.name == "run_params.yaml":
            run_params = yaml.safe_load(open(file))
            assert run_params["model"]["depth"] == 1
            assert run_params["model"]["filters"] == 2
