from xml.dom import minidom
import matplotlib.image as mpig
import matplotlib.pyplot as plt
import numpy as np
import cv2
'''
doc=minidom.parse('E:/mask_dataset/label_nomask/0000.xml')
xmin=int(doc.getElementsByTagName('xmin')[0].firstChild.data)
plt.figure()
img=mpig.imread("E:/mask_dataset/image_nomask/0000.jpg")
img=img[99:228,282:390,:]
#plt.savefig('d.jpg')
plt.imshow(img)
'''


pic=[]
temp = [str(i) for i in range(0, 650)]
for i in range(len(temp)):
    while (len(temp[i]) < 4):
        temp[i] = '0' + temp[i]
for i in range(650):
    doc=minidom.parse("D:/MaskFace_Project/pytorch-CycleGAN-and-pix2pix/datasets/mask_dataset/label_nomask/" +temp[i]+ '.xml')
    xmin=int(doc.getElementsByTagName('xmin')[0].firstChild.data)
    xmax=int(doc.getElementsByTagName('xmax')[0].firstChild.data)
    ymin=int(doc.getElementsByTagName('ymin')[0].firstChild.data)
    ymax=int(doc.getElementsByTagName('ymax')[0].firstChild.data)
    img=mpig.imread("D:/MaskFace_Project/pytorch-CycleGAN-and-pix2pix/datasets/mask_dataset/image_nomask/"+temp[i]+".jpg")
    img=img[ymin:ymax,xmin:xmax,:]
    pic.append(img)
    
pic = np.array(pic)
for i in range(650):
    # plt.imshow(pic[i])
    img = cv2.resize(pic[i], (256, 256))
    plt.imsave('no_mask_face' + str(i) + '.jpg', img)
    # cv2.imwrite(, img)
    # plt.savefig('E:/nomask1/'+temp[i]+'.jpg',bbox_inches='tight',pad_inches=-0.4)
    if (i+1)%50==0:
        print("已完成50张")
