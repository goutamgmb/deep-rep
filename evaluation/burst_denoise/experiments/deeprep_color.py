from evaluation.common_utils.network_param import NetworkParam


def main():
    network_list = []

    # Check the documentation of NetworkParam for other example use cases
    # network_list.append(NetworkParam('deeprep', 'denoise_color_pwcnet', epoch=None, burst_sz=8))
    # network_list.append(NetworkParam(network_path='deeprep_denoise_color_pwcnet.pth', unique_name='DeepRep-PWCNet'))
    network_list.append(NetworkParam(network_path='deeprep_denoise_color_pwcnet.pth', unique_name='DeepRep-Custom'))

    return network_list

