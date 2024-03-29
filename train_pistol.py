from cgi import test
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from pistol_yolo import Yolov1
from dataset_yolo import PistolDataset
import matplotlib.pyplot as plt
from utils_pistol import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from augmentations import aug_transforms
from loss_pistols import YoloLoss
import time

# seed = 123
# torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 6
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
VAL_MODEL_FILE = "validated.pth.tar"
IMG_DIR = "../../Datasets/Guns_In_CCTV/VOC/"
LABEL_DIR = "../../Datasets/Guns_In_CCTV/VOC/modified/"
VALIDATE=True

class Compose(object):
    def __init__(self,transforms, img_size):
        self.transforms = transforms
        self.img_size = img_size

    def __call__(self, img, bboxes):
        img, anchor = img
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return (img, anchor), bboxes

transform = Compose([transforms.Resize((224, 224)),transforms.ToTensor()], 224)

# def show_losses(losses, filename="losses.png"):
#     plt.figure()
#     box = []
#     obj = []
#     noobj = []
#     for l in losses[-200:]:
#         b, c = l
#         box.append(b.item())
#         obj.append(o.item())
#         noobj.append(n.item())
#         # classes.append(c.item())
#     plt.plot(box, label="Box Loss")
#     plt.plot(classes, label="Class Loss")
#     plt.legend(loc="best")
#     plt.savefig(filename)
#     plt.close()



def train_fn(train_loader, model, optimizer, loss_fn, validation_set=None):
    print("Train:")
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    batch_losses = []
    displ = []
    val_loss = []
    lambda_coord = 0

    

    for batch_idx, ((x,x_pos), y) in enumerate(loop):
        x, y = (x.to(DEVICE), x_pos.to(DEVICE)), y.to(DEVICE)
        out = model(x)
        loss, losses = loss_fn(out, y, lambda_coord=lambda_coord)
        if lambda_coord < 1:
            lambda_coord += 2e-2
        batch_losses.append(loss.item())
        mean_loss.append(loss.item())
        displ.append(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b, o, n = losses
        # if batch_idx % 20 == 1:
        #     show_losses(displ)

        b, o, n = round(b.item(), 4), round(o.item(), 4), round(n.item(), 4)

        loop.set_postfix(loss=round(loss.item(), 4), box=b, obj_loss=o, noobj_loss=n)

    if validation_set is not None:
        print("Validation:")
        val_loop = tqdm(validation_set, leave=True)
        
        for batch_idx, ((x,x_pos), y) in enumerate(val_loop):
            x, y = (x.to(DEVICE), x_pos.to(DEVICE)), y.to(DEVICE)
            with torch.no_grad():
                out = model(x)
                loss, losses = loss_fn(out, y)
            val_loss.append(loss.item())
            displ.append(losses)
            
            b, o, n = losses
            b, o, n = round(b.item(), 4), round(o.item(), 4), round(n.item(), 4)

            val_loop.set_postfix(loss=round(loss.item(), 4), box=b, obj_loss=o, noobj_loss=n)
        
        print(f"Validation loss was {sum(val_loss) / len(val_loss)}")
        print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
        return batch_losses, sum(mean_loss) / len(mean_loss), sum(val_loss) / len(val_loss)

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
    return batch_losses, sum(mean_loss) / len(mean_loss)

def main():
    model = Yolov1(split_size=1, num_boxes=1, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = PistolDataset(
        "CCTV/train_copy.txt",
        transform=transform,
        img_dir=IMG_DIR + "train/",
        label_dir=LABEL_DIR,
        augmentations=aug_transforms,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    
    if VALIDATE:
        test_dataset = PistolDataset(
            "CCTV/test_copy.txt",
            transform=transform,
            img_dir=IMG_DIR + "test/",
            label_dir=LABEL_DIR
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=True
        )
    else:
        test_loader=None

    batch_losses, mean_losses, val_losses= [], [], []
    mAPs = []
    val_mAPs = []

    for epoch in range(EPOCHS):
        print("----------------------------------------------")
        print("----------------------------------------------")
        print(f"Epoch {epoch + 1}:")
        if VALIDATE:
            b_loss, m_loss, v_loss = train_fn(train_loader, model, optimizer, loss_fn, validation_set=test_loader)
            batch_losses.extend(b_loss)
            mean_losses.append(m_loss)
            val_losses.append(v_loss)
        else:
            b_loss, m_loss = train_fn(train_loader, model, optimizer, loss_fn )
            batch_losses.extend(b_loss)
            mean_losses.append(m_loss)
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="corners")
        if VALIDATE:
            
            val_pred_boxes, val_target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4, validation=True
            )
            mean_avg_prec_val = mean_average_precision(val_pred_boxes, val_target_boxes, iou_threshold=0.5, box_format="corners")
        
        print(f"Train mAP: {mean_avg_prec}")
        if VALIDATE:
            print(f"Validation mAP: {mean_avg_prec_val}")
            val_mAPs.append(mean_avg_prec_val)
        mAPs.append(mean_avg_prec)

        if mean_avg_prec > 0.5:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(3)
        if VALIDATE:
            if mean_avg_prec_val > 0.5:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_checkpoint(checkpoint, filename=VAL_MODEL_FILE)
                time.sleep(3)


        if (epoch + 1) % 1 == 0:
            plt.figure()
            plt.plot(batch_losses)
            plt.xlabel("Batches")
            plt.ylabel("Loss")
            plt.title("Batch Losses")
            plt.savefig("batch_losses.png")
            plt.close()

            plt.figure()
            plt.plot(mean_losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Mean Losses")
            plt.savefig("mean_losses.png")
            plt.close()

            plt.figure()
            plt.plot(mAPs)
            plt.xlabel("Epochs")
            plt.ylabel("mAP")
            plt.title("Mean Average Precision")
            plt.savefig("mAP.png")
            plt.close()

            if VALIDATE:
                plt.figure()
                plt.plot(val_mAPs)
                plt.xlabel("Epochs")
                plt.ylabel("mAP")
                plt.title("Mean Average Precision")
                plt.savefig("val_mAP.png")
                plt.close()
                

                plt.figure()
                plt.plot(val_losses)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title("Validation Loss")
                plt.savefig("val_losses.png")
                plt.close()


if __name__ == "__main__":
    main()