from torchvision.transforms import RandomHorizontalFlip, ToTensor
import torch

hflip = RandomHorizontalFlip(1)

def horizontal_flip_im(image):
    """
    Horizontally flip an image.

    :param image: Image to be flipped.
    :return: Image with the flipped image.
    """
    image, hands = image
    new_hands = [0, 0, 0, 0]
    if hands[:2] != [0, 0]:
        new_hands[:2] = [1 - hands[0],hands[1]]

    if hands[2:] != [0, 0]:
        new_hands[2:] = [1 - hands[2],hands[3]]
    return torch.flip(image, (-2,)), torch.tensor(new_hands)

def horizontal_flip_la(label):
    """
    Horizontally flip a label.

    :param label: Label to be flipped.
    :return: Label with the flipped label.
    """
    # print(type(label))
    # quit()
    new_label = [[label[0,0], 0, 0], [label[1,0], 0, 0]]
    if label[0, 1:] != [0, 0]:
        new_label[0][1:] = [1 - label[0, 1],label[0, 2]]
    
    if label[1, 1:] != [0, 0]:
        new_label[1][1:] = [1 - label[1, 1],label[1, 2]]
    return torch.tensor(new_label)
def vertical_flip_im(image):
    """
    vertically flip an image.

    :param image: Image to be flipped.
    :return: Image with the flipped image.
    """
    image, hands = image
    new_hands = [0, 0, 0, 0]
    if hands[:2] != [0, 0]:
        new_hands[:2] = [hands[0], 1 - hands[1]]

    if hands[2:] != [0, 0]:
        new_hands[2:] = [hands[2], 1 - hands[3]]
    return torch.flip(image, (-1,)), torch.tensor(new_hands)

def vertical_flip_la(label):
    """
    vertically flip a label.

    :param label: Label to be flipped.
    :return: Label with the flipped label.
    """
    # print(type(label))
    # quit()
    new_label = [[label[0,0], 0, 0], [label[1,0], 0, 0]]
    if label[0, 1:] != [0, 0]:
        new_label[0][1:] = [label[0, 1], 1 - label[0, 2]]
    
    if label[1, 1:] != [0, 0]:
        new_label[1][1:] = [label[1, 1], 1 - label[1, 2]]
    return torch.tensor(new_label)

aug_transforms = [
    (horizontal_flip_im, horizontal_flip_la),
    (vertical_flip_im, vertical_flip_la)
]