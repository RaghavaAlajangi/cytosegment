from torch.optim import lr_scheduler


def get_scheduler_with_params(params, optimizer):
    assert {"scheduler"}.issubset(params)
    scheduler_params = params.get("scheduler")
    assert {"type", "lr_decay_rate"}.issubset(scheduler_params)
    scheduler_type = scheduler_params.get("type")
    lr_decay_rate = scheduler_params.get("lr_decay_rate")

    if scheduler_type.lower() == "steplr":
        assert {"lr_step_size"}.issubset(scheduler_params)
        lr_step_size = scheduler_params.get("lr_step_size")
        return lr_scheduler.StepLR(optimizer=optimizer,
                                   step_size=lr_step_size,
                                   gamma=lr_decay_rate)

    if scheduler_type.lower() == "reducelronplateau":
        assert {"mode"}.issubset(scheduler_params)
        mode = scheduler_params.get("mode")
        return lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                              mode=mode,
                                              factor=lr_decay_rate)
