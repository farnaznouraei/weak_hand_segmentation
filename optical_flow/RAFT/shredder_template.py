import cv2
import os
import numpy

os.system("ffmpeg -i P11_06.MP4 -vcodec libx264 -crf 20 epic.mp4")

# Opens the Video file
cap= cv2.VideoCapture('./epic.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('epic_'+str(i)+'.jpg',frame)
    #if i == 501:
        #break
    i+=1


cap.release()
cv2.destroyAllWindows()
for ind in range(i):
    os.system("MKL_THREADING_LAYER=GNU python run.py --model default --first ./epic_"+str(ind)+".jpg --second ./epic_"+str(ind+1)+".jpg --out ./out"+str(ind)+".flo")

os.system("python -m flowiz ./*.flo")
os.system("ffmpeg -f image2  -i ./out%d.flo.png -vcodec libx264 -s 640x480 -pix_fmt yuv420p epic_flow.mp4")
os.system("ffmpeg -f image2  -i ./epic_%d.jpg -vcodec libx264 -s 640x480 -pix_fmt yuv420p epic_input.mp4")
#os.system("convert -delay 2 'out%d.flo.png'[0-99] out.gif")
os.system("rm *.png")
os.system("rm *.jpg")
os.system("rm *.flo")
