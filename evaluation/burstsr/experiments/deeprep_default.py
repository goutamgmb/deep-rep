from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    network_list.append(NetworkParam(network_path='deeprep_sr_burstsr_default.pth', unique_name='DeepRep'))

    return network_list

