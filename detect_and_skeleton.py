from binary import Net,imshow,get_path
from skeleton import showLines
import torch
import torchvision
from skimage import io,img_as_ubyte
import numpy as np

PATH = get_path()
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
weights = torch.load(PATH,map_location=device)
net.load_state_dict(weights)

net = net.double()
tiffimg = io.imread("MAX_na1.2 ais cortex-2.tif")
rr,cc,_ =tiffimg.shape
red_img = tiffimg[:,:,2]/4095.0

#swap R & B
rgbtiff = tiffimg.copy()
rgbtiff[:,:,0] = tiffimg[:,:,2]
rgbtiff[:,:,2] = tiffimg[:,:,0]
rgbtiff = rgbtiff/4095.0
io.imshow(rgbtiff)
io.show()

red_img = torch.from_numpy(red_img)
red_img = red_img.view([1,1,rr,cc])
red_img = red_img.to(device)

msk_img = net(red_img)

msk_img = msk_img.detach()
msk_img = msk_img.cpu()
imshow(torchvision.utils.make_grid(msk_img))

msk_img = msk_img.view([rr,cc])
msk_img = msk_img.numpy()

#sav_img = np.zeros([1024,1024,3])
#sav_img[:,:,0]=msk_img
#io.imsave("./test.jpg",msk_img)
overlay = showLines(msk_img)
rgbtiff[:,:,0] = overlay+rgbtiff[:,:,0]
rgbtiff = rgbtiff.clip(0.0,1.0)
io.imshow(rgbtiff)
io.imsave('overlay.jpg',overlay)