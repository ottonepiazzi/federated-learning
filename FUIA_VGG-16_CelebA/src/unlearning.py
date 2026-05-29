import random

from model import VGG16
from fl_training import run_fl_rounds


#Randomly select 1 sample per client to forget (paper Sec VI.B.1)
def select_forgotten_samples(client_data, private_data):
    forgotten = {}
    for client_id, sample_indices in client_data.items():
        forgotten[client_id] = random.choice(sample_indices)
    return forgotten


#Retrain from pretrained model, removing ALL forgotten samples at once
def retrain_without_samples(pretrained_sd, private_data, client_data,
                            forgotten_samples, round_selections, target_client):
    model = VGG16()
    model.load_state_dict(pretrained_sd)
    modified_data = {}
    for client_id, sample_indices in client_data.items():
        if client_id in forgotten_samples:
            modified_data[client_id] = [i for i in sample_indices
                                        if i != forgotten_samples[client_id]]
        else:
            modified_data[client_id] = list(sample_indices)
    return run_fl_rounds(model, modified_data, private_data, round_selections,
                         target_client=target_client)
