from binary import Net,imshow,get_path
import torch
import torchvision

PATH = get_path()
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load(PATH,map_location=device))

data_path = './test/'
train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=0,
        shuffle=True
    )
# get some random training images
dataiter = iter(train_loader)
images= dataiter.next()
print(images[0].shape)
x = images[0][0].unsqueeze_(0)

x = torch.max(x, dim=1,keepdim = True)[0].to(device)

predic = net(x).detach()
imshow(torchvision.utils.make_grid(predic.cpu()))
torchvision.utils.save_image(predic,"./result.jpg")
