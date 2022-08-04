
import cv2
# from xml.dom import minidom
import xml.etree.ElementTree as ET

IMG_DIR = "../../Datasets/Guns_In_CCTV/VOC/"
LABEL_DIR = "../../Datasets/Guns_In_CCTV/VOC/"
OUTPUT_LABEL_DIR = LABEL_DIR + "modified/"
ds_file = "CCTV/valid_copy.txt"

with open(ds_file + "_stats.txt", 'w') as f:
    stats = f.read().strip()
    print(stats)

quit()
def write_to_xml(xml_filename, filename, img_size, data):
    width, height, _ = img_size
    xml = ET.Element("annotation")
    # xml = root.createElement("annotation")
    # root.appendChild(xml)

    filenameAttr = ET.SubElement(xml, "filename")
    filenameAttr.text = filename

    sizeAttr = ET.SubElement(xml,"size")
    widthAttr = ET.SubElement(sizeAttr,"width")
    widthAttr.text = str(width)

    heightAttr = ET.SubElement(sizeAttr,"height")
    heightAttr.text = str(height)

    objAttr = ET.SubElement(xml, "object")
    lhand, rhand = data["left"], data["right"]
    # lhandcoord, rhandcoord = None, None
    # lhandbbox, rhandbbox = None, None

    if len(lhand) == 2:
        lhandAttr = ET.SubElement(objAttr, "lhand")
        lhandxAttr = ET.SubElement(lhandAttr, "x")
        lhandxAttr.text = str(lhand[0])
        lhandyAttr = ET.SubElement(lhandAttr, "y")
        lhandyAttr.text = str(lhand[1])

    elif len(lhand) == 4:
        lhandAttr = ET.SubElement(objAttr, "lhand")
        lhandxAttr = ET.SubElement(lhandAttr, "x")
        lhandxAttr.text = str((lhand[0] + lhand[2]) // 2 )
        lhandyAttr = ET.SubElement(lhandAttr, "y")
        lhandyAttr.text = str((lhand[1] + lhand[3]) // 2)

        bboxAttr = ET.SubElement(objAttr, "lbbox")
        xmin_bboxAttr = ET.SubElement(bboxAttr, "xmin")
        xmin_bboxAttr.text = str(lhand[0])
        
        xmax_bboxAttr = ET.SubElement(bboxAttr, "xmax")
        xmax_bboxAttr.text = str(lhand[2])

        ymin_bboxAttr = ET.SubElement(bboxAttr, "ymin")
        ymin_bboxAttr.text = str(lhand[1])
        
        ymax_bboxAttr = ET.SubElement(bboxAttr, "ymax")
        ymax_bboxAttr.text = str(lhand[3])


    if len(rhand) == 2:
        rhandAttr = ET.SubElement(objAttr, "rhand")
        rhandxAttr = ET.SubElement(rhandAttr, "x")
        rhandxAttr.text = str(rhand[0])
        rhandyAttr = ET.SubElement(rhandAttr, "y")
        rhandyAttr.text = str(rhand[1])

    elif len(rhand) == 4:
        rhandAttr = ET.SubElement(objAttr, "rhand")
        rhandxAttr = ET.SubElement(rhandAttr, "x")
        rhandxAttr.text = str((rhand[0] + rhand[2]) // 2 )
        rhandyAttr = ET.SubElement(rhandAttr, "y")
        rhandyAttr.text = str((rhand[1] + rhand[3]) // 2)

        bboxAttr = ET.SubElement(objAttr, "rbbox")
        xmin_bboxAttr = ET.SubElement(bboxAttr, "xmin")
        xmin_bboxAttr.text = str(rhand[0])
        
        xmax_bboxAttr = ET.SubElement(bboxAttr, "xmax")
        xmax_bboxAttr.text = str(rhand[2])

        ymin_bboxAttr = ET.SubElement(bboxAttr, "ymin")
        ymin_bboxAttr.text = str(rhand[1])
        
        ymax_bboxAttr = ET.SubElement(bboxAttr, "ymax")
        ymax_bboxAttr.text = str(rhand[3])




    tree = ET.ElementTree(xml)
    with open(xml_filename, "w"):
        tree.write(xml_filename)
    # xml_str = ET.tostring(tree)
    # print(xml_str)

    


found_gun = False

lbtn_down = False

rbtn_down = False


def mouse_handler(event, x, y, flags, data):
    global lbtn_down, rbtn_down, found_gun
    data, img = data
    left_hand_coords = data["left"]
    right_hand_coords = data["right"]
    if event == cv2.EVENT_LBUTTONDOWN:
        # You have just clicked on a button
        lbtn_down = True
        if len(left_hand_coords) == 4:
            left_hand_coords.clear()
        left_hand_coords.append(x)
        left_hand_coords.append(y)

        if len(left_hand_coords) == 4:
            
            left_hand_coords[0], left_hand_coords[2] = min(left_hand_coords[0], left_hand_coords[2]), max(left_hand_coords[0], left_hand_coords[2])
            left_hand_coords[1], left_hand_coords[3] = min(left_hand_coords[1], left_hand_coords[3]), max(left_hand_coords[1], left_hand_coords[3])
            mod_img = cv2.rectangle(img, (left_hand_coords[0], left_hand_coords[1]), (left_hand_coords[2], left_hand_coords[3]), (255, 0, 0), 1)
            cv2.imshow("Modified Image", mod_img)

    # elif event == cv2.EVENT_MOUSEMOVE and lbtn_down:
    #     found_gun = True
    #     # m_img = cv2.rectangle(img, (left_hand_coords[0], left_hand_coords[1]), (x, y), (255, 0, 0), 10)
    # elif event == cv2.EVENT_LBUTTONUP and found_gun:
    #     found_gun = False
    #     lbtn_down = False
    #     left_hand_coords.append(x)
    #     left_hand_coords.append(y)
    #     mod_img = cv2.rectangle(img, (left_hand_coords[0], left_hand_coords[1]), (left_hand_coords[2], left_hand_coords[3]), (255, 0, 0), 1)
    #     cv2.imshow("Modified Image", mod_img)
    elif event == cv2.EVENT_MBUTTONDOWN:
        # You have just clicked on a button
        if len(right_hand_coords) == 4:
            right_hand_coords.clear()
        right_hand_coords.append(x)
        right_hand_coords.append(y)

        if len(right_hand_coords) == 4:
            right_hand_coords[0], right_hand_coords[2] = min(right_hand_coords[0], right_hand_coords[2]), max(right_hand_coords[0], right_hand_coords[2])
            right_hand_coords[1], right_hand_coords[3] = min(right_hand_coords[1], right_hand_coords[3]), max(right_hand_coords[1], right_hand_coords[3])        
            mod_img = cv2.rectangle(img, (right_hand_coords[0], right_hand_coords[1]), (right_hand_coords[2], right_hand_coords[3]), (0, 255, 0), 1)
            cv2.imshow("Modified Image", mod_img)
    elif event == cv2.EVENT_MOUSEMOVE and rbtn_down:
        found_gun = True
    elif event == cv2.EVENT_MBUTTONUP and found_gun:
        found_gun = False
        rbtn_down = False
        right_hand_coords.append(x)
        right_hand_coords.append(y)
        
        mod_img = cv2.rectangle(img, (right_hand_coords[0], right_hand_coords[1]), (right_hand_coords[2], right_hand_coords[3]), (0, 255, 0), 1)
        cv2.imshow("Modified Image", mod_img)


image_file_locations = []
label_file_locations = []
files = None
left_hands = 0
left_boxes = 0

right_hands = 0
right_boxes = 0
with open(ds_file, 'r') as f:
    files = f.readlines()
    for ff in files:
        image_file_locations.append(IMG_DIR + ff.strip() + ".jpg")
        label_file_locations.append(OUTPUT_LABEL_DIR + ff.strip() + ".xml")

for im, la in zip(image_file_locations, label_file_locations):
    # 1. Open image
    # 2. Open labels
    # 3. Left mouse = left hand, right mouse = right hand
    # 4. No click = not visible
    # 5. Click + drag = draw bbox around gun - hand in center of bbox
    # 6. Click = hand visible but no gun
    hand_coords = {
        "left": [],
        "right": []
    }
    img = cv2.imread(im)
    # img = cv2.resize(img, (448, 448))
    cv2.imshow("Modified Image", img)
    cv2.setMouseCallback("Modified Image", mouse_handler, (hand_coords, img))
    cv2.waitKey(0)
    print(hand_coords)
    curr_entry = files.pop(0)
    if len(hand_coords['left']) > 0:
        left_hands += 1
        left_boxes += 1 if len(hand_coords['left']) > 2 else 0
    if len(hand_coords['right']) > 0:
        right_hands += 1
        right_boxes += 1 if len(hand_coords['right']) > 2 else 0


    if len(hand_coords["left"]) > 0 or len(hand_coords["right"]) > 0:
        write_to_xml(la, im, img.shape, hand_coords)
        with open(ds_file + "_modified.txt" , 'a') as f:
            f.write(curr_entry)
    with open(ds_file, 'w') as f:
        f.write("".join(files))
    with open(ds_file + "_stats.txt", 'w') as f:
        out_str = f"left_hands: {left_hands}\nright_hands: {right_hands}\n\n"
        out_str += f"left_boxes: {left_boxes}\nright_boxes: {right_boxes}"
        f.write(out_str)

print("Done!")
cv2.destroyAllWindows()

