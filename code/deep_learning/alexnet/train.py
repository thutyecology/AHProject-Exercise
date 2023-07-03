import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

def main():
    #Use GPU device to speed up the process
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    #Set image input size as 32
    input_size = 32

    #Transform training and validation data
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(input_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((input_size, input_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    #Get data root
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))

    #Get image path
    image_path = os.path.join(data_root, "data", "img2_rgb_size"+str(input_size))

    #Get training path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    #Get classes {1:"Cropland" ,2:"Built-up (Low)", 3:"Built-up (High)", 4:"Vegetation",5:"Bareland"}
    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())

    #Write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    #Set training batch size
    batch_size = 32

    #Set number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 2])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    #Load training data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    #Load validation data
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    #Create AlexNet
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device) #to GPU

    #Set loss function
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())

    #Set optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # Set epochs
    epochs = 100

    save_path = './AlexNet'+str(input_size)+'.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        print ()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()