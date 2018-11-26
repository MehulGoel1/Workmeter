from imutils import face_utils
import dlib
import cv2
import numpy as np
import math

facePath = 'haarcascade_frontalface_default.xml'
smilePath = 'haarcascade_smile.xml'
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

print('[INFO] loading facial landmark predictor...')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,340)

count=0
countout=0

image_points = np.array([
                            (0, 0),     # Nose tip 34
                            (0, 0),     # Chin 9
                            (0, 0),     # Left eye left corner 37
                            (0, 0),     # Right eye right corne 46
                            (0, 0),     # Left Mouth corner 49
                            (0, 0)      # Right mouth corner 55
                        ], dtype='double')

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])
image_oints = np.array([
                            (0, 0),     # Nose tip 34
                            (0, 0),     # Chin 9
                            (0, 0),     # Left eye left corner 37
                            (0, 0),     # Right eye right corne 46
                            (0, 0),     # Left Mouth corner 49
                            (0, 0)      # Right mouth corner 55
                        ], dtype='double')


lists=[[0,0,0]]
list=[0,0,0]
count=0
countout=0
smiles=0
dist=[[0,0,0,]]
oldx=0
oldy=0
diffx=0
diffy=0
tupx=[]
tupy=[]
x=0
y=0
percent=0

flist=[]
work=1
slist=[]
ss=0

file= open("stillness.txt","w+")
file2= open("smile.txt","w+")

nosmile=0


t=6
tcount=0

nosmile=0
smileflag=0
sm=0

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    
    
    
    rects = detector(gray, 0)
    
    faceflag=0

    for rect in rects:
        faceflag=1
        f=0
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
        
        
        nosmile=nosmile+1
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        roi_gray = gray[bY:bY + bH,bX:bX + bW]
        roi_color = frame[bY:bY + bH,bX:bX + bW]
        
        
        leftb=shape[19]
        rightb=shape[24]
        
        image_oints[0]=np.array([shape[19][0],shape[19][1]])
        image_oints[1]=np.array([shape[24][0],shape[24][1]])
        
        p1x =int(image_oints[0][0])
        p1y =int(image_oints[0][1])
        p2x =int(image_oints[1][0])
        p2y =int(image_oints[1][1])
        
        distance=abs(shape[27][1]-shape[8][1])
        
        slope=(p2y-p1y)/(p2x-p1x)
        #slope=(p1y-p2y)/(p1x-p2x)
        angle=math.degrees(math.atan(slope))
        num=(22/7)*(angle/180)
        divby=math.cos(num)
        ans=distance/divby
        
        if ans < 96 :
            ans=ans-65
            t=int(ans/4)
            t=t+3
        elif ans > 136 and ans < 170:
            t=17
        elif ans>170 and ans <183:
            t=18
        elif ans > 183 and ans < 196 :
            t=19
        elif ans >197:
            t=20
        else:
            ans=ans-95
            ans=int(ans/5)
            t=ans+1+8
            
        r=t*6
        
        #cv2.putText(frame, 'This is t: {} %'.format(t), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.2,
            minNeighbors=r,
            minSize=(t,t),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        
        image_points[0]=np.array([shape[33][0],shape[33][1]])
                 
       
        for (x, y, w, h) in smile:
            f=1
            #print ("Found", count, "smiles!")
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
            count=count+1

             
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
                
        #print(p1)
        tup=(p1[0],p1[1],f)
        
        x=p1[0]
        y=p1[1]
        
        dist.append(tup)
        
        if len(dist) is 2:
            oldx=x
            oldy=y            
        
        elif len(dist)< 9 and len(dist)!=2 :
            tupx.append(oldx-x)
            tupy.append(oldy-y)
            print(p1)
            print(oldx-x)
            diffx=diffx+abs(oldx-x)
            diffy=diffy+abs(oldy-y)
            oldx=x
            oldy=y
        else:
            #diffx=0
            #diffy=0
            g=dist.pop(0)
            
            fx=abs(tupx.pop(0))
            fy=abs(tupy.pop(0))
            
            tupx.append(abs(oldx-x))
            tupy.append(abs(oldy-y))
            
            diffx=diffx-fx+abs(oldx-x)
            diffy=diffy-fy+abs(oldy-y)
            oldx=x
            oldy=y

        sum=diffx+diffy
        if sum < 150:
            mar=(150-(diffx+diffy))/150
        else:
            mar=0
        
        mar=mar*100
        
        print(diffx,diffy)
        
        lists.append(tup)
        if len(lists)>16:
            h=lists.pop(0)
            if h[2] is 1:
                smiles=smiles-1
        
        if f is 1:
            smiles=smiles+1
            
            
#TODO work
        if mar>90:
            slist.append(1)
        else :
            slist.append(0)
        if mar>90:
            ss=ss+1
        if len(slist)>40:
            it=slist.pop(0)
            if it is 1:
                ss=ss-1
                if ss<20:
                    work=1
        if ss>20:
            work=2
        
        if work is 2:
            print('WATCHING SOMETHING IMMERSIVELY')
            cv2.putText(frame, 'WATCHING SOMETHING IMMERSIVELY', (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        
            
        flist.append(f)
        if f is 1:
            sm=sm+1
        
        if sm >=17:
            work=0
            
        if len(flist)>80:
            item=flist.pop(0)
            if item is 1:
                sm=sm-1
            if sm < 40:
                work=1
        if work is 1:
            print('WORKING',sm)
            cv2.putText(frame, 'WORKING', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        elif work is 0:
            print('PLAYING',sm)
            cv2.putText(frame, 'PLAYING', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        
        per=smiles-5
        if per>5:
            percent=percent+40+smiles*6
        else:
            per=per+5
            percent=per*8
            
        if percent>100:
            percent=100
        print('total smiles yet',smiles)
        

    if faceflag is 0:
        mar=0
        percent=0
        
    #cv2.putText(frame, 'Stillness %: {} %'.format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    #cv2.putText(frame, 'Smile Meter: {} %'.format(percent), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 2)
    
    #file.write(str(countout))
    countout=countout+1
 
    cv2.imshow('Smile Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        file.close()
        file2.close()
        break

cap.release()
cv2.destroyAllWindows()
