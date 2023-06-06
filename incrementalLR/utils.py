import os
import logging
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy.linalg as LA
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix


def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.

    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array

    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels


def init_pipeline(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'checkpoints'))
        os.makedirs(os.path.join(model_dir, 'figures'))
        os.makedirs(os.path.join(model_dir, 'plabels'))
    if headers is None:
        headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e",
                   "discrimn_loss_t", "compress_loss_t"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))


def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path


def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)


def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory
    to new directoy. """
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)


def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict


def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n' + ','.join(map(str, entries)))


def save_ckpt(model_dir, net, epoch, name):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints'+name,
                                              'model-epoch{}.pt'.format(epoch)))


def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)


def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


def clustering_accuracy(labels_true, labels_pred):
    """Compute clustering accuracy."""
    from sklearn.metrics.cluster import supervised
    from scipy.optimize import linear_sum_assignment
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)


def init_pipeline_AE(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'checkpoints'))
        os.makedirs(os.path.join(model_dir, 'figures'))
        os.makedirs(os.path.join(model_dir, 'plabels'))
    if headers is None:
        headers = ["epoch", "step", "mcr_loss", "discrimn_loss_e", "compress_loss_e",
                   "discrimn_loss_t", "compress_loss_t", "recon_loss", "loss"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))



def get_features_AE(encoder, decoder, trainloader):
    '''Extract all features out into one single batch.

    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar
    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    X_all = []
    X_bar_all = []
    Z_all = []
    Z_bar_all = []
    labels_all = []

    train_bar = tqdm(trainloader, desc="extracting all features from dataset")

    with torch.no_grad():
        for step, (X, labels) in enumerate(train_bar):

            Z = encoder(X.cuda())
            # the dim of z is [bs, dim], should resize it to [bs, dim, 1, 1] first.
            # TODO use it as a standard and change the output of the encoder
            X_bar = decoder(Z.reshape(len(Z), -1, 1, 1))

            Z_bar = encoder(X_bar.detach())

            X_all.append(X.cpu().detach())
            Z_all.append(Z.view(-1, Z.shape[1]).cpu().detach())
            X_bar_all.append(X_bar.cpu().detach())
            Z_bar_all.append(Z_bar.view(-1, Z_bar.shape[1]).cpu().detach())

            labels_all.append(labels)

    return torch.cat(X_all), torch.cat(Z_all), torch.cat(X_bar_all), torch.cat(Z_bar_all), torch.cat(labels_all)


def nearsub(train_features, train_labels, test_features, test_labels, n_comp=15, test=1):
    """Perform nearest subspace classification.

    Options:
        n_comp (int): number of components for PCA or SVD

    """
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1  # should be correct most of the time
      
    if test == 3:
        features_sort, _ = sort_dataset(test_features.numpy(), test_labels.numpy(),
                                          num_classes=num_classes, stack=False)
    else:
        features_sort, _ = sort_dataset(train_features.numpy(), train_labels.numpy(),
                                          num_classes=num_classes, stack=False)
    fd = features_sort[0].shape[1]
    for j in range(num_classes):
        pca = PCA(n_components=n_comp).fit(features_sort[j])
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        if test == 1:
            pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                    @ (test_features.numpy() - mean).T
        else:
            pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                    @ (train_features.numpy() - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        if test == 1:
            svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                    @ (test_features.numpy()).T
        else:
            svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                    @ (train_features.numpy()).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)

        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    if test == 1:
        acc_pca = compute_accuracy(test_predict_pca, test_labels.numpy())
        acc_svd = compute_accuracy(test_predict_svd, test_labels.numpy())
        print(confusion_matrix(test_predict_pca, test_labels.numpy()))    
        
    else:
        acc_pca = compute_accuracy(test_predict_pca, train_labels.numpy())
        acc_svd = compute_accuracy(test_predict_svd, train_labels.numpy())
        print(confusion_matrix(test_predict_pca, train_labels.numpy()))
    
    acc_pca = round(acc_pca, 4)
    acc_svd = round(acc_svd, 4)

    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_pca, acc_svd


def sort_feature(features, labels):

    uniq_label = np.unique(labels)
    sorted_features = []
    for l in uniq_label:
        sorted_features.append(features[labels == l])

    return sorted_features


def find_support(features, labels, num_class=10, n_component=10, num_per_direction=30, num_explore = 40):
    """Find corresponding images to the nearests component per class. """
    features_sort = sort_feature(features.numpy(), labels.numpy())

    Z = []
    Z_label = []

    print("feature_sort dim: ", len(features_sort))

    for class_ in range(num_class):

        """

        this_class = features_sort[class_]
        len_this_class = len(this_class)
        random_choice = np.array(random.sample(range(len_this_class), n_component*num_per_direction))

        Z.append(torch.tensor(this_class[random_choice]))
        Z_label.append(torch.ones(n_component*num_per_direction) * class_)

        """
        # try:
        #     pca = TruncatedSVD(n_components=n_component, random_state=10).fit(features_sort[class_])
        # except:
        #     pca = TruncatedSVD(n_components=n_component, random_state=10).fit(features_sort)
        pca = TruncatedSVD(n_components=n_component, random_state=10).fit(features_sort[class_])

        for j in range(n_component):
            proj = features_sort[class_] @ pca.components_.T[:, j]
            img_idx = np.argsort(np.abs(proj), axis=0)[::-1][:num_explore]

            comp = np.array(features_sort[class_])[img_idx]
            
            comp_mean = np.mean(comp, axis = 0)
            comp_cov = np.cov(comp.T)
            
            sampled = np.array([np.random.multivariate_normal(comp_mean, comp_cov, size=None, check_valid='warn', tol=1e-8) for i in range(num_per_direction)])
            
            Z.append(torch.tensor(sampled).float())
            Z_label.append(torch.ones(num_per_direction)*class_)
            
    # print(Z)
    Z = torch.cat(Z, 0)
    Z_label = torch.cat(Z_label, 0)
    return Z, Z_label


def subspace_dist(features, log_dir):
    print(len(features))

    u_set = []
    for feature in tqdm(features):
        u, s, vh = LA.svd(np.vstack(feature).T, full_matrices=False)
        u_set.append(u)
    print("finishing svd")
    log = f"{log_dir}/subspace_dis"
    os.makedirs(log, exist_ok=True)
    for n_compoment in [5, 10, 15]:
        u_set = [u[:, :n_compoment] for u in u_set]
        log_ = f"{log}/uset_{n_compoment}"
        os.makedirs(log_, exist_ok=True)
        subspace_dis = np.zeros((len(u_set), len(u_set)))
        for i, ui in enumerate(tqdm(u_set)):
            for j, uj in enumerate(u_set):
                subspace_dis[i, j] = LA.norm(ui.T @ uj)

        plt.figure(), plt.imshow(subspace_dis)
        plt.colorbar()
        plt.savefig(f"{log_}/confusion_subspace.jpg"), plt.close()
