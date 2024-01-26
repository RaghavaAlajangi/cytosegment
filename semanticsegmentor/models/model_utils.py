import torch.nn.init as init


def init_weights(net, init_type="HeNormal", gain=0.02):
    """
    Initializes the weights of a network.
    Parameters
    ----------
    net
        Pass the network to be initialized
    init_type
        Specify the type of initialization to be used
    gain
        Scale the weights of the network
    Returns
    -------
    The initialized network
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "HeNormal":
                init.kaiming_normal_(m.weight.data, mode="fan_in",
                                     nonlinearity="relu")
            elif init_type == "HeUniform":
                init.kaiming_uniform_(m.weight.data, mode="fan_in",
                                      nonlinearity="relu")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "init method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"Initialize network with {init_type}")
    return net.apply(init_func)
