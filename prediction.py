
import cv2
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def predict_video():
    
    # Read the video from specified path
    videoLink=input("input video location: ")
    cam = cv2.VideoCapture(videoLink)
    
    try:
        
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            # print(currentframe)
            if(currentframe==6000):
                break

        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    frameSize = (1280, 720)

    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    video = cv2.VideoWriter('/content/predicted_video.mp4', fourcc, float(20), frameSize)
    # video = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'MP42'), 30, frameSize)

    for i in range(5000):
        imgL = cv2.imread('/content/data/frame'+str(i)+'.jpg',0)
        imgR = cv2.imread('/content/data/frame'+str(i+5)+'.jpg',0)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL,imgR)
        # plt.imshow(disparity,'gray')
        # plt.show()
        # print(type(disparity))
        if(i%500==0):
            print(i)
        disparity=disparity.astype(np.uint8)
        video.write(disparity+imgL)

    video.release()
    
  

if __name__ == '__main__':
  predict_video()
  
