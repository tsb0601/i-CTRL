import os
import torch
import argparse
import torch.optim as optim
import random
import numpy as np
from incrementalLR.default import _C as config
from incrementalLR.default import update_config
from incrementalLR.datasets import get_dataloader
from incrementalLR.models import get_model
from incrementalLR.utils import get_features_AE, nearsub, find_support
from incrementalLR.trainer import train
from incrementalLR.loss import MCRGANloss, MCRGANlossGEN, MCRGANlossDISC

seed = config.SEED
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)

def _to_yaml(obj, filename=None, default_flow_style=False,
             encoding="utf-8", errors="strict",
             **yaml_kwargs):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding=encoding, errors=errors) as f:
        obj.dump(stream=f, default_flow_style=default_flow_style, **yaml_kwargs)


def evaluate(netG, netD, train_loader, test_loader, n_comp=15, test=1):
    train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_loader)
    test_X, test_Z, test_X_bar, test_Z_bar, test_labels = get_features_AE(netD, netG, test_loader)

    acc_pca, acc_svd = nearsub(train_Z, train_labels, test_Z, test_labels, n_comp, test)
    return acc_pca, acc_svd

def create_dirs(log_dir):
    _to_yaml(config, os.path.join(log_dir, 'config.yaml'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpointsDISC'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpointsGEN'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'figures'), exist_ok=True)
    log = open(f"{log_dir}/log.txt", mode='w+')
    log.close()
    result = open(f"{log_dir}/result.txt", mode='w+')
    result.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='should add the .yaml file',
                        required=True,
                        type=str,
                        )
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)


def main():

    # step 1: pre-setting
    parse_args()

    create_dirs(config.LOG_DIR)
    
    assert config.TRAIN.START_CLASS % config.TRAIN.CLASS_PER_STEP == 0
    dataset_id = int(config.TRAIN.START_CLASS / config.TRAIN.CLASS_PER_STEP)
    start = config.TRAIN.START_CLASS
    end = config.TRAIN.NUM_CLASSES
    class_per_step = config.TRAIN.CLASS_PER_STEP
    review = config.TRAIN.REVIEW
    review_round = config.TRAIN.REVIEW_ROUND
    
    (train_loader, test_loader), train_inc_loader, (train_accinc_loader, test_accinc_loader) = get_dataloader(
        data_name=config.TRAIN.DATASET,
        root="./data",
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.CUDNN.NUM_WORKERS,
        class_per_step=class_per_step
    )

    # load model and ckpt
    netG, netD = get_model()
    
    ACC_PCA = []
    ACC_SVD = []
    ACC_PCA_IL = []
    ACC_SVD_IL = []
    
    if config.TRAIN.NETD_CKPT and config.TRAIN.NETG_CKPT:
        netD.load_state_dict(torch.load(config.TRAIN.NETD_CKPT))
        netG.load_state_dict(torch.load(config.TRAIN.NETG_CKPT))
    else:
        raise ValueError()
    
    train_accinc_loader.dataset.reset_task(0)
    test_accinc_loader.dataset.reset_task(0)

    acc_pca_IL, acc_svd_IL = evaluate(netG, netD, train_accinc_loader, test_accinc_loader, n_comp=config.MODEL.PCACOMP)
    
    result = f"Trained {start} classes, 2 testing:" + \
        f"\nPCA: {acc_pca_IL}\tSVD: {acc_svd_IL}\n"
    
    ACC_PCA_IL.append(acc_pca_IL)
    ACC_SVD_IL.append(acc_svd_IL)
        
    with open(f"{config.LOG_DIR}/result.txt", 'a+') as f:
        f.write(result)
    
    train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_loader)

    Z, Z_label = find_support(
        train_Z,
        train_labels,
        num_class=config.TRAIN.START_CLASS,
        n_component=config.MODEL.SAMPLE_N,
        num_per_direction=config.MODEL.SAMPLE_K)

    Z = Z.cuda()
    Z_label = Z_label.int()

    # initial loss
    criterionMCR = MCRGANloss(gam1=1, gam2=1, eps=config.TRAIN.EPS, numclasses=start,
                              mode=config.TRAIN.MODE, lambd=config.TRAIN.LAMBD, class_step=class_per_step)

    criterionGEN = MCRGANlossGEN(gam1=1, gam2=1, eps=config.TRAIN.EPS, numclasses=start,
                                 mode=config.TRAIN.MODE)

    criterionDISC = MCRGANlossDISC(gam1=1, gam2=1, eps=config.TRAIN.EPS, numclasses=start,
                                   mode=config.TRAIN.MODE, class_step=class_per_step)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.TRAIN.LRD, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=config.TRAIN.LRG, betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    

    for k, IncrementalClass in enumerate(range(start, end, class_per_step)):

        # update data-loader
        train_inc_loader = torch.utils.data.DataLoader(
            train_inc_loader.dataset,
            batch_size=config.MODEL.SAMPLE_N * config.MODEL.SAMPLE_K * (IncrementalClass * class_per_step),
            shuffle=True,
            num_workers=config.CUDNN.NUM_WORKERS)
        train_inc_loader.dataset.reset_task(dataset_id + k)

        # update data-loader for calculate accuracy
        train_accinc_loader.dataset.reset_task(dataset_id + k)
        test_accinc_loader.dataset.reset_task(dataset_id + k)

        # update num of class in loss function
        criterionMCR.num_class = IncrementalClass + class_per_step
        criterionGEN.num_class, criterionDISC.num_class = IncrementalClass + class_per_step, IncrementalClass + class_per_step

        X_memory = netG(torch.reshape(Z, (len(Z), config.MODEL.NZ, 1, 1))).detach()
        Z_bar_memory = netD(X_memory).detach()

        for epoch in range(1, config.TRAIN.EPOCHS + 1):
            train(
                netG,
                netD,
                IncrementalClass + class_per_step,
                train_inc_loader,
                optimizerG,
                optimizerD,
                criterionMCR,
                epoch,
                Z,
                Z_label,
                criterionGEN,
                criterionDISC,
                X_memory,
                Z_bar_memory,
                config,
                False
            )

        train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_inc_loader)

        for class_add in range(class_per_step):
            new_Z, new_Z_label = find_support(
                train_Z[train_labels == IncrementalClass + class_add],
                train_labels[train_labels == IncrementalClass + class_add],
                num_class=1,
                n_component=config.MODEL.SAMPLE_N,
                num_per_direction=config.MODEL.SAMPLE_K
            )

            Z = torch.cat((Z, new_Z.cuda()))
            # Batch size maybe need to match number of support
            Z_label = torch.cat((Z_label, (torch.ones(new_Z.shape[0]) * (IncrementalClass + class_add)).int()))

        # n_comp = 14/15 maybe better
        print(f"All training samples, All testing:")
        acc_pca, acc_svd = evaluate(netG, netD, train_loader, test_loader, n_comp=config.MODEL.PCACOMP)
        ACC_PCA.append(acc_pca)
        ACC_SVD.append(acc_svd)

        print(f"Trained {IncrementalClass + class_per_step} classes, {IncrementalClass + class_per_step} testing:")
        acc_pca_IL, acc_svd_IL = evaluate(netG, netD, train_accinc_loader, test_accinc_loader, n_comp=config.MODEL.PCACOMP)
        ACC_PCA_IL.append(acc_pca_IL)
        ACC_SVD_IL.append(acc_svd_IL)
        
        
        result = f"All training samples, All testing acc:" + \
            f"\nPCA: {acc_pca}\tSVD: {acc_svd}" + \
            f"\nTrained {IncrementalClass + class_per_step} classes, {IncrementalClass + class_per_step} testing:" + \
            f"\nPCA: {acc_pca_IL}\tSVD: {acc_svd_IL}\n"
            
        with open(f"{config.LOG_DIR}/result.txt", 'a+') as f:
            f.write(result)
    
    final_result = f"All training samples, All testing acc for every step:" + \
        f"\nPCA: {ACC_PCA}\tAvg :{sum(ACC_PCA)/len(ACC_PCA)}" + \
        f"\nSVD: {ACC_SVD}\tAvg :{sum(ACC_SVD)/len(ACC_SVD)}" + \
        f"\nIncremental training samples, Incremental testing acc for every step {class_per_step}:" + \
        f"\nPCA: {ACC_PCA_IL}\tAvg :{sum(ACC_PCA_IL)/len(ACC_PCA_IL)}" +\
        f"\nSVD: {ACC_SVD_IL}\tAvg :{sum(ACC_SVD_IL) /len(ACC_SVD_IL)}\n"

    with open(f"{config.LOG_DIR}/result.txt", 'a+') as f:
        f.write(final_result)
        
    if review:
        ACCUS_REVIEW = []
        
        for RRound in range(review_round):
            ACCUS_PER_ROUND = []
            for k, IncrementalClass in enumerate(range(0, end, class_per_step)):
                 # update data-loader
                train_inc_loader = torch.utils.data.DataLoader(
                    train_inc_loader.dataset,
                    batch_size=int(10 * 192),
                    shuffle=True,
                    num_workers=config.CUDNN.NUM_WORKERS)
                train_inc_loader.dataset.reset_task(k)

                # update data-loader for calculate accuracy
                train_accinc_loader.dataset.reset_task(k)
                test_accinc_loader.dataset.reset_task(k)

                # update num of class in loss function
                criterionMCR.num_class = IncrementalClass + class_per_step
                criterionGEN.num_class, criterionDISC.num_class = IncrementalClass + class_per_step, IncrementalClass + class_per_step
        
                print(f"For Review {IncrementalClass} Training:")
    
                train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_loader)
                test_X, test_Z, test_X_bar, test_Z_bar, test_labels = get_features_AE(netD, netG, test_loader)
    
                print(f"For Reviewed All {config.TRAIN.NUM_CLASSES} class")
                print("Train:")
                nearsub(train_Z, train_labels, test_Z, test_labels, test=0)
                print("Test:")
                test_accu = nearsub(train_Z, train_labels, test_Z, test_labels, config.MODEL.PCACOMP, test=1)
                ACCUS_PER_ROUND.append(test_accu)
    
                train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_accinc_loader)
                test_X, test_Z, test_X_bar, test_Z_bar, test_labels = get_features_AE(netD, netG, test_accinc_loader)
    
                print(f"For Reviewed {IncrementalClass} class")
                print("Train:")
                nearsub(train_Z, train_labels, test_Z, test_labels, config.MODEL.PCACOMP, test=0)
                print("Test:")
                nearsub(train_Z, train_labels, test_Z, test_labels, config.MODEL.PCACOMP, test=1)
    
                if IncrementalClass == end - 1:
                    break
    
                X_memory = netG(torch.reshape(Z, (len(Z), config.MODEL.NZ, 1, 1))).detach()
                Z_bar_memory = netD(X_memory).detach()
    
                for epoch in range(1, config.TRAIN.EPOCHS + 1):
                    train(
                        netG,
                        netD,
                        IncrementalClass + class_per_step,
                        train_inc_loader,
                        optimizerG,
                        optimizerD,
                        criterionMCR,
                        epoch,
                        Z,
                        Z_label,
                        criterionGEN,
                        criterionDISC,
                        X_memory,
                        Z_bar_memory,
                        config,
                        True
                    )
    
                train_X, train_Z, train_X_bar, train_Z_bar, train_labels = get_features_AE(netD, netG, train_inc_loader)
                
                new_Z, new_Z_label = find_support(train_Z[train_labels == IncrementalClass], train_labels[train_labels == IncrementalClass], num_class=1)
                
                Z[Z_label == IncrementalClass] = new_Z.cuda()
            ACCUS_REVIEW.append(ACCUS_PER_ROUND)
        
        review_result = ""
        for review_round, accus_per_round in enumerate(ACCUS_REVIEW):
            review_result += f"Review Round {review_round}:\n" + \
                f"\tAccuracy: {accus_per_round}\n" + \
                f"\tAvg: {sum(accus_per_round)/ len(accus_per_round)}\n"
        
        with open(f"{config.LOG_DIR}/result.txt", 'a+') as f:
            f.write(review_result)

if __name__ == '__main__':
    main()
