import numpy as np
import torch.nn as nn
import torch
from sklearn.cluster import KMeans




def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    


def extract_features(model, data_loader):
    # extract features from dataloader to numpy array
    features = []
    targets = []
    model.eval()
    print('extracting train features')
    with  torch.no_grad():
        for (image, label,) in data_loader:
            image= image.cuda()
            latent_z, _, _ = model.encoder(image)
            features.append(latent_z)
            targets.append(label)
    features = torch.cat(features, 0).detach().cpu().numpy()
    targets = torch.cat(targets, 0).detach().numpy()
    return features, targets



def reset_prototypes(model, data_loader, n_sub_prototypes=3):
    # reset prototypes by clustering
    print('Resetting Prototypes')
    features, targets = extract_features(model, data_loader)
    labelnum = len(np.unique(targets))
    mu_list = []
    for i in range(labelnum):
        features_l = features[targets==[i]]
        kmeans = KMeans(n_clusters=n_sub_prototypes, random_state=0).fit(features_l)
        mu = kmeans.cluster_centers_
        mu_list.append(mu)
    mu_list = np.vstack(mu_list)
    new_prototypes = torch.from_numpy(np.vstack(mu_list)).cuda().float()
    new_prototypes = nn.Parameter(new_prototypes, requires_grad=True)
    return new_prototypes

