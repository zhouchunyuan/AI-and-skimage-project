import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F


PATH = './binary_net.pth'
def get_path():
    return PATH
#w, h, batch_size = 40, 40, 1
# Create dummy input and target tensors (data)
    
#x = torch.randn(batch_size,1,w,h).clamp(min=0,max=1)
#y = torch.randn(batch_size,1,w,h).clamp(min=0,max=1).int().float()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 9,padding =4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 1, 9,padding = 4)
        self.conv3 = nn.Conv2d(1,1,9,padding = 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=4)
        return x

    

def main():

    data_path = './imgs'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False
    )

    gnd_path = './gnd'
    gnd_dataset = torchvision.datasets.ImageFolder(
        root=gnd_path,
        transform=torchvision.transforms.ToTensor()
    )
    gnd_loader = torch.utils.data.DataLoader(
        gnd_dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False
    )
    # get some random training images
    dataiter = iter(train_loader)
    images= dataiter.next()

    gnditer = iter(gnd_loader)
    gnds= gnditer.next()

    x = images[0]#[0].unsqueeze_(0)
    y = gnds[0]#[0].unsqueeze_(0)

    x = torch.max(x, dim=1,keepdim = True)[0]
    y = torch.max(y, dim=1,keepdim = True)[0]
    print(x.shape)
    print(y.shape)

    imshow(torchvision.utils.make_grid(x))
    imshow(torchvision.utils.make_grid(y))

    model = Net()
    import os
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
    else:
        print("load file wrong!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    #Construct the loss function
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # Gradient Descent

    for epoch in range(2100):
        # Forward pass: Compute predicted y by passing x to the model

        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if epoch % 10 ==0 :
            print('epoch: ', epoch,' loss: ', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()

    torch.save(model.state_dict(), PATH)    
    y_pred = model(x)
    y_pred = y_pred.detach()

    imshow(torchvision.utils.make_grid(y_pred.cpu()))
    torchvision.utils.save_image(y_pred,"./result1.jpg")

if __name__ == '__main__':
    main()
