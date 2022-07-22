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
from loss_pistols import YoloLoss
import time

# seed = 123
# torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 6
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "../../Datasets/Guns_In_CCTV/VOC/"
LABEL_DIR = "../../Datasets/Guns_In_CCTV/VOC/modified/"

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        img, anchor = img
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return (img, anchor), bboxes

transform = Compose([transforms.Resize((448, 448)),transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    hands_coords = torch.Tensor([0.5,0.5]).to(DEVICE)

    batch_losses = []

    for batch_idx, ((x,x_pos), y) in enumerate(loop):
        x, y = (x.to(DEVICE), x_pos.to(DEVICE)), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y, hands_coords)
        batch_losses.append(loss)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

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
        label_dir=LABEL_DIR
    )
    
    test_dataset = PistolDataset(
        "CCTV/test_copy.txt",
        transform=transform,
        img_dir=IMG_DIR + "test/",
        label_dir=LABEL_DIR
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    batch_losses, mean_losses= [], []

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(10)
            

        print(f"Train mAP: {mean_avg_prec}")

        b_loss, m_loss = train_fn(train_loader, model, optimizer, loss_fn)
        batch_losses.extend(b_loss)
        mean_losses.append(m_loss)
    plt.figure()
    plt.plot(batch_losses)
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Batch Losses")
    plt.savefig("batch_losses.png")

    plt.figure()
    plt.plot(mean_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean Losses")
    plt.savefig("mean_losses.png")


if __name__ == "__main__":
    main()