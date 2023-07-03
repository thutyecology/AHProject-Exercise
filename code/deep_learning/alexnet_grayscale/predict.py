import os
import sys
import torch
import json
from torchvision import transforms, datasets
from tqdm import tqdm
import cv2
import numpy as np

from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 32

    data_transform = transforms.Compose(
        [transforms.Resize((input_size, input_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         #transforms.Grayscale(num_output_channels=1)
        ]
    )

    # Load model
    model = AlexNet(num_classes=5).to(device)
    weights_path = './AlexNet'+str(input_size)+'.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Load data path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    image_path = os.path.join(data_root, "data", "img2_gray_size" + str(input_size))
    all_path = os.path.join(image_path, "all")

    # Load validation data
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=2)

    # Compute validation accuracy
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num

    acc_dict = {'accuracy': val_accurate}
    json_str = json.dumps(acc_dict)
    with open('accuracy.json', 'w') as json_file:
        json_file.write(json_str)

    print()
    print("Model: AlexNet")
    print("Validation accuracy: %.2f" % (val_accurate))


    # Load image
    org_path = os.path.join(data_root, "data", "Fig2_2019.jpeg")
    org_path1 = os.path.join(data_root, "data", "Fig1_1939.jpeg")
    img2 = cv2.imread(org_path)
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2gray3 = cv2.cvtColor(img2gray, cv2.COLOR_GRAY2BGR)
    img1 = cv2.imread(org_path1)
    m, n, d = img2.shape

    # Expand image
    img2expand = np.zeros([m + input_size, n + input_size, d], dtype=np.uint8)
    img1expand = np.zeros([m + input_size, n + input_size, d], dtype=np.uint8)
    for b in range(0, d):
        start = int(input_size / 2)
        img2expand[start:start + m, start:start + n, b] = img2gray3[:, :, b]
        img1expand[start:start + m, start:start + n, b] = img1[:, :, b]

    # Predict land cover types
    print()
    print("#############Execute prediction#############")
    outimg = np.zeros([m, n], dtype=np.int8)
    for i in range(0, m, 1):
        for j in range(0, n, 1):

            img = img2expand[i:i+input_size, j:j+input_size,:]
            img = img1expand[i:i + input_size, j:j + input_size, :]

            # [N, C, H, W]
            img = transforms.ToPILImage()(img)
            img = data_transform(img)

            # Expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # Predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            outimg[i][j] = predict_cla

        print('Row ' + str(i) + ' finished!')

    # Save predictions
    outname = "predict_img1.tif"
    outfile = os.path.join(image_path, outname)
    cv2.imwrite(outfile, outimg)
    print()
    print(outname + ' saved!')
    print("#############End of prediction#############")

if __name__ == '__main__':
    main()
