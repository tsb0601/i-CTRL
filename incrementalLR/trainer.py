import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import save_ckpt
import torchvision.utils as vutils
import torchvision.transforms.functional as FF
import time


def train(gen, disc, classIL, dataloader, optimizerG, optimizerD, criterion, epoch, Z_memory, Z_memory_label,
          criterionGEN, criterionDISC, X_memory, Z_bar_memory, config=None, review=False):

    # gen.train()
    # disc.train()
    
    steps = 0
    t = time.time()
    for i, (data, label) in enumerate(dataloader):

        label = torch.cat((label.cpu(), Z_memory_label.cpu()))

        for j in range(config.TRAIN.N_ITER_DIS):
            disc.zero_grad()
            # Forward pass real batch through D(X->Z)
            Z = disc(data.cuda())
            Z = torch.cat((Z, Z_memory))

            errD, _, _, _ = criterionDISC(Z, Z, label, train_mode=2)

            errD.backward()
            optimizerD.step()

            # Update Discriminator

        disc.zero_grad()
        # Forward pass real batch through D(X->Z)
        Z = disc(data.cuda())

        X_bar = gen(torch.reshape(Z, (len(Z), config.MODEL.NZ, 1, 1)))

        X_bar = torch.cat((X_bar, X_memory))

        Z_bar = disc(X_bar.detach())

        Z = torch.cat((Z, Z_memory))

        errDI, firstD, secondD, _ = criterion(Z, Z_bar, label)

        errDII, _, _, thirdD, _ = criterionGEN(Z, Z_bar, label)

        # errD = errDI - errDII - MSEloss(Z_bar[num_batch:], Z_bar_memory)
        errD = errDI - errDII

        errD.backward()
        optimizerD.step()

        for j in range(config.TRAIN.N_ITER_GEN):
            # Update Generator
            gen.zero_grad()

            X = data.cuda()

            Z = disc(data.cuda())

            Z_IL = Z

            Z_IL = torch.cat((Z_IL, Z_memory))
            X_bar = gen(torch.reshape(Z, (len(Z), config.MODEL.NZ, 1, 1)))

            X_bar = torch.cat((X_bar, X_memory))

            X_IL_bar = gen(torch.reshape(Z_IL, (len(Z_IL), config.MODEL.NZ, 1, 1)))

            Z_IL = disc(X_IL_bar)

            Z_bar = disc(X_bar)

            Z = torch.cat((Z, Z_memory))


            errG_I, firstG, secondG, thirdG, all_third = criterionGEN(Z_IL, Z_bar, label)

            errG_II, firstG, secondG, thirdG = criterion(Z, Z_bar, label)

            errG = (-1) * (errG_II + errG_I)

            errG.backward()
            optimizerG.step()

    out = f"Training Epoch:{epoch}, at {config.LOG_DIR}: " + \
        f"\n|ErrD is {errD:.5f} \n|1-term: {firstD:.5f} \n|2-term: {secondD:.5f} \n|3-term: {thirdD:.5f}" + \
        f"\n|ErrG is {errG:.5f} \n|1-term: {firstG:.5f} \n|2-term: {secondG:.5f} \n|3-term: {thirdG:.5f}" + \
        f"\n|epoch time: {(time.time() - t):.4f}\n"
    print(out)

    with open(f"{config.LOG_DIR}/log.txt", 'a+') as f:
        f.write(out)

    if epoch % 5 == 0:
        save_ckpt(config.LOG_DIR, gen, epoch, "GEN")
        save_ckpt(config.LOG_DIR, disc, epoch, "DISC")
        with torch.no_grad():
            real = gen(torch.reshape(Z[:32], (32, config.MODEL.NZ, 1, 1))).detach().cpu()
            memory = gen(torch.reshape(Z_memory[:32], (32, config.MODEL.NZ, 1, 1))).detach().cpu()
            
            if review:
                save_fig(vutils.make_grid(real, padding=2, normalize=True), epoch, f"Review_class_{str(classIL)}_Generated", config.LOG_DIR)
                save_fig(vutils.make_grid(data[:32], padding=2, normalize=True), epoch, f"Review_class_{str(classIL)}_{config.TRAIN.DATASET}", config.LOG_DIR)
                save_fig(vutils.make_grid(memory, padding=2, normalize=True), epoch, f"Review_class_{str(classIL)}_Memory", config.LOG_DIR)
            else:
                save_fig(vutils.make_grid(real, padding=2, normalize=True), epoch, f"class_{str(classIL)}_Generated", config.LOG_DIR)
                save_fig(vutils.make_grid(data[:32], padding=2, normalize=True), epoch, f"class_{str(classIL)}_{config.TRAIN.DATASET}", config.LOG_DIR)
                save_fig(vutils.make_grid(memory, padding=2, normalize=True), epoch, f"class_{str(classIL)}_Memory", config.LOG_DIR)


def save_fig(imgs, epoch, prefix, dir):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig(f"{dir}/figures/{prefix}_epoch_{epoch}.png")
    plt.close()


