from hand_tracker import HandTracker
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import warnings
warnings.filterwarnings("ignore")

palm_model_path = "./models/palm_detection.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"
# Opens the Video file
cap= cv2.VideoCapture('./epic.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    #cv2.imwrite('ego_'+str(i)+'.jpg',frame)
    img = frame[:,:,::-1]
    # box_shift determines
    detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                           box_shift=0.2, box_enlarge=1.3)

    hands = detector(img)

    fig, ax = plt.subplots(1,1,figsize=(20, 20))
    ax.imshow(img)
    for h in hands:
        kp = h['joints']
        box = h['bbox']
        ax.scatter(kp[:,0], kp[:,1])
        ax.add_patch(Polygon(box, color="#00ff00", fill=False))

    fig.savefig('out_'+str(i)+'.jpg')


    #if i == 501:
        #break
    i+=1


cap.release()
cv2.destroyAllWindows()



os.system("ffmpeg -f image2  -i ./out_%d.jpg -vcodec libx264 -s 640x480 -pix_fmt yuv420p epic_pose.mp4")

os.system("rm *.jpg")
