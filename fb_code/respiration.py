import biosppy
import numpy as np

def get_resp_features(resp_data):
    """
    Function:

    :param:

    :return
    
    """
    resp_rate, filtered, zeros, resp_rate_ts, resp_rate = biosppy.signals.resp.resp(resp_data, sampling_rate=700, show=False)
    extremas, values = biosppy.signals.tools.find_extrema(signal=filtered, mode='both')
    inhal_durations = []
    exhal_durations = []
    last_index = 0
    for i in range(len(extremas)):
        if values[i] * values[last_index] < 0:
            if values[last_index] < 0:
                inhal_durations.append((extremas[i] - extremas[last_index]) / 700)
            else:
                exhal_durations.append((extremas[i] - extremas[last_index]) / 700)
            last_index = i
    return np.mean(resp_rate), np.mean(inhal_durations), np.std(inhal_durations), np.mean(exhal_durations), np.std(exhal_durations), np.sum(inhal_durations) / np.sum(exhal_durations)