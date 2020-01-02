from skimage.morphology import binary_erosion,skeletonize,binary_dilation,binary_closing,remove_small_holes
from skimage.filters import threshold_otsu
from skimage import io
from skimage.color import rgb2gray,label2rgb
from skimage.measure import label, regionprops,approximate_polygon
from skimage.draw import line
from scipy import signal
import numpy as np
from skimage.filters import meijering
#################################################################
def findJoints(skeleimage):
    tshape1 = np.array([[0,1,0],
                        [1,1,0],
                        [0,1,0]])
    tshape2 = np.array([[1,0,1],
                        [0,1,0],
                        [1,0,0]])
    yshape1 = np.array([[1,0,1],
                        [0,1,0],
                        [0,1,0]])
    yshape2 = np.array([[0,0,1],
                        [1,1,0],
                        [0,1,0]])
    pshape1 = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]])
    pshape2 = np.array([[1,0,1],
                        [0,1,0],
                        [1,0,1]])

    joints = [tshape1,tshape1.T,np.fliplr(tshape1),np.flipud(tshape1.T)]
    joints.extend([tshape2,np.fliplr(tshape2),np.flipud(tshape2),np.flipud(np.fliplr(tshape2))])
    joints.extend([yshape1,yshape1.T,np.flipud(yshape1),np.fliplr(yshape1.T)])
    joints.extend([yshape2,yshape2.T,np.flipud(yshape2),np.fliplr(yshape2)])
    joints.extend([pshape1,pshape2])
    
    im = np.zeros(skeleimage.shape)
    for j in joints:
        #print(j)
        im += (signal.convolve2d(skeleimage,j,mode='same')/j.sum()).astype(int)
    return im>0;
################################################################

debug_mode = True
debug_folder = './debug/'

def showLines(img):
    thresh = threshold_otsu(img)

    binary = img >thresh
    binary = remove_small_holes(binary)
    skeleton = skeletonize(binary)

    # label image regions
    label_image = label(skeleton)
    
    if debug_mode:
        image_label_overlay = label2rgb(label_image, image=img)
        io.imsave(debug_folder+"image_label_overlay.jpg",image_label_overlay)
    

    prop = regionprops(label_image)
    #print(len(prop))

    mask = np.ones([3,3])
    kernel = mask<0

    
    overlay_polylineImage = np.zeros(img.shape)
    for obj in prop:
        #print(label.convex_image)
        if obj.area > 10:
            b = findJoints(obj.image)
            joints = label(b)
            dots = regionprops(joints)
            img4edit = np.copy(obj.image)
            
            #io.imshow(closing(obj.image))
            #io.show()
            min_row = obj.bbox[0]
            min_col = obj.bbox[1]
            for d in dots:
                y,x = np.array(d.centroid).astype(int)
                mask = np.copy(img4edit[y-1:y+2,x-1:x+2])

                img4edit[y-1:y+2,x-1:x+2] = 1

                branchLabel = label(img4edit)
                branches = regionprops(branchLabel)
                branchsize= [branch.area for branch in branches]
                branchcoord = [branch.coords for branch in branches]
                if len(branchsize) > 1:
                    minIndex = np.argmin(np.array(branchsize))
                    for i,j in branchcoord[minIndex]:
                        img4edit[i,j] = 0
                    img4edit[y-1:y+2,x-1:x+2]=mask
                    #img4edit[y,x]=1

            ##### to fit into polyline
            img4edit = skeletonize(binary_dilation(binary_dilation(img4edit)))

            label4coord = label(img4edit)
            coords = regionprops(label4coord)[0]
            polyline = approximate_polygon(coords.coords, tolerance=0.02)
            
            xlast=-1
            ylast=-1
            row = []
            col = []

            for i,j in polyline:
                if xlast != -1:
                    rr,cc = line(min_row+ylast,min_col+xlast,min_row+i,min_col+j)
                    row.extend(rr)
                    col.extend(cc)
                    #polylineImage[rr,cc] = 1
                xlast = j
                ylast = i    
                #polylineImage[i,j] = 1

            overlay_polylineImage[row, col] = 1  




            #io.imshow(polylineImage)
            #io.show()
    return overlay_polylineImage



if __name__ == '__main__':
    msk_img = io.imread("./test.jpg")
    msk_img = rgb2gray(msk_img)
    overlay = showLines(msk_img)
    io.imsave('overlay.jpg',overlay)
    io.imshow(overlay)