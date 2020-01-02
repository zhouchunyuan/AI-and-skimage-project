import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io
import numpy as np
from skimage.morphology import skeletonize,area_closing,binary_closing,binary_opening,binary_dilation
from skimage.measure import label,regionprops
from skimage.filters import threshold_otsu, rank
from skimage.feature import peak_local_max , corner_harris, match_template
from scipy.signal import convolve2d as conv2
from scipy.ndimage.morphology import binary_hit_or_miss
from skimage.draw import line

def conv(im1):
    kernel = np.ones([7,7])*-1
    #kernel[3-1:3+2,3-1:3+2]=20
    kernel[3,3]=48
    print(kernel)
    im = conv2(im1,kernel, 'same')
    return im

def save(name,img):
    io.imsave('./desktop/debug/'+name, img)
    io.imshow(img)
    io.show()

class Node:
    
    visit_table = []
    longest_chain = []
    
    #layer_level = 0 # root level is 0
    longest_end = None
    max_length = 0
    
    def __init__(self,row,col,image,lastNode):
        if lastNode == None: # init of the root node
            Node.visit_table = []
            Node.longest_chain = []
            Node.longest_end = None
            Node.max_length = 0
        self.coord = (row,col)
        Node.visit_table.extend([(row,col)])
        self.branches = []
        self.last = lastNode
        if(lastNode == None):
            self.count = 1
        else:
            self.count = lastNode.count+1
        #Node.layer_level +=1
        self.find_branches(image)
        #Node.layer_level -=1

    def find_branches(self,image):
        for j in [-1,0,1]:
            for i in [-1,0,1]:
                if not(i==0 and j == 0):
                    r = self.coord[0]+j
                    c = self.coord[1]+i
                    if r >= 0 and r < image.shape[0] and c >=0 and c < image.shape[1]:
                        if image[r,c] >0 :
                            if not ((r,c)  in Node.visit_table):
                                new_node = Node(r,c,image,self)
                                self.branches.extend([new_node])
    def find_branch_ends(self):
        if len(self.branches) == 0:
            #print(self.count,self.coord)
            if Node.max_length < self.count:
                Node.max_length = self.count
                Node.longest_end = self
                
                # below code to copy longest chain 
                Node.longest_chain = [None]*self.count
                temp_node = self
                Node.longest_chain[temp_node.count-1]=temp_node.coord
                while temp_node.last != None:
                    temp_node = temp_node.last
                    Node.longest_chain[temp_node.count-1]=temp_node.coord
        else:
            for b in self.branches:
                #Node.layer_level+=1 # layer index
                b.find_branch_ends()
                #Node.layer_level-=1 # layer index

            
    # debug tool, to show longest chain
    def print_longest(self):
        for c in Node.longest_chain:
            print(c)
            
    # debug tool. To show all ends
    def print_table(self):
        if len(self.branches) == 0:
            print(self.count)
        else:
            for b in self.branches:
                b.print_table()

    

    

        
img = io.imread('C:/Users/f3412/Desktop/pixelwise/sample.jpg')
channel_img = img[:,:,0]
img = channel_img
#save('chanel.jpg',channel_img)

####################################
'''
test = np.zeros((100,100))
rr, cc = line(1, 1, 99, 99)
test[rr, cc] = 1
rr, cc = line(10, 10, 30, 10)
test[rr, cc] = 1
io.imshow(test)
io.show()
ch = Node(1,1,test,None)
ch.find_branch_ends()
print(ch.longest_end.count)
ch.print_longest()
'''
#####################################

img = conv(img).astype(int)
#save('convolved.jpg',img)


thresh = threshold_otsu(img)
binary_img = binary_closing(img > thresh)

binary_img = area_closing(binary_img*255)
binary_img = binary_img > 200


#save('binary.jpg',binary_img*255)

#peak_img = peak_local_max(gaussian(binary_img*255,3),indices = False)
#save('peak.jpg',peak_img*255)

skele_img = skeletonize(binary_img)
#save('skele_img.jpg',skele_img*255)

label_img = label(skele_img)
props = regionprops(label_img)

msk_img = np.zeros(skele_img.shape)

#ist_dot_img = np.zeros(skele_img.shape)
for p in props:
    if p.area > 50:
        #io.imshow(p.image)
        #io.show()
        #rmin,cmin,rmax,cmax = p.bbox
        #msk_img[rmin:rmax,cmin:cmax] = p.image
        
        #print(p.coords.shape)
        the_1st_node_row = p.coords[0][0]
        the_1st_node_col = p.coords[0][1]
        
        #ist_dot_img[the_1st_node_row,the_1st_node_col]=255
        if not (the_1st_node_row == 0 or the_1st_node_col == 0):
            chain = Node(the_1st_node_row,the_1st_node_col,skele_img,None)
            chain.find_branch_ends()
            #print("longest_chain:",len(chain.longest_chain))
            for dot in chain.longest_chain:
                #print(dot)
                msk_img[dot[0],dot[1]]=255


'''
save('1st_dot_img.jpg',ist_dot_img)

kernel = np.zeros([3,3])
kernel[1,1]=1
kernel[2,1]=1
#print(kernel)
corners = binary_hit_or_miss(msk_img,kernel)
save("corners.jpg",binary_dilation(corners)*255)
'''      
save('msk_img.jpg',msk_img)
rgb_overlay = np.zeros([skele_img.shape[0],skele_img.shape[1],3])
#rgb_overlay[:,:,0] = msk_img*255
rgb_overlay[:,:,1] = channel_img
save('rgb_overlay.jpg',rgb_overlay)