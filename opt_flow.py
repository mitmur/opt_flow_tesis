
import numpy as np
import cv2
#import imutils
#import video
#import glob
#from common import anorm2, draw_str
#from time import clock
import image
#from collections import deque
#import matplotlib

import sys


lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

feature_params = dict( maxCorners = 500, 
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

image_folder = '/Test/Test001_gt'


segundos=0
track_len = 10
detect_interval = 5
tracks = []
frame_idx = 0
velocidadT=0
cntAnomalos=0

cam=image.captura(image_folder)
    
def todo():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
        
    cam=image.captura(image_folder)

    #cam = cv2.VideoCapture(fn)
    ret, prev = cam.read()
    fps = cam.get(cv2.CAP_PROP_FPS)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    kernel = np.ones((5,5),np.uint8)
    outputTxt = open('outputTxt.txt','w')
    #x = collections.deque(10*[0], 10)

    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
        vis = img.copy()
        text="Normal"
        gray4 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray4)
        blur = cv2.medianBlur(fgmask,5)
        #blur = cv2.GaussianBlur(blur,(5,5),0)
        _ ,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        frame_gray=morph.copy()
        num_Rects=0

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            
            velocidadT=d.sum()
            
            #outputTxt.write(text+" - "+str(velocidadT) + "\n")
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            #cv2.polylines(vis,prueba,False, (0, 255, 0)) 
            #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
            draw_str(vis, (20, 455),"Velocidad: "+str(velocidadT))
        

        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])            


        #Dilatamos para armar contornos            
        morph2 = cv2.dilate(morph, None, iterations=6)

        #Definición de contornos y visualización de recuadros de tracking
        im2, cnts, hierarchy = cv2.findContours(morph2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5] # get largest five contour area
        rects = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h >= 80:
                rect = (x, y, w, h)
                rects.append(rect)
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                num_Rects=num_Rects+1
        

        if num_Rects==0:
            pass
        else:
            velocidadT=velocidadT/num_Rects
            if velocidadT>60:
                text="Anomalo"
                cv2.imwrite('Anomalo/'+str(round(segundos,2))+'.png',vis)
                cntAnomalos=cntAnomalos+1
            else:
                cntAnomalos=0            

        if cntAnomalos==10:
            text="Alta"
            cv2.imwrite('Alta/'+str(round(segundos,2))+'.png',vis)
            pass


        #x.append(text)        
        #x.popleft()            

        frame_idx += 1
        prev_gray = frame_gray

        if fps==0:
            fps=30
        segundos=segundos+(1/fps)
        
        print(text+" - "+str(round(velocidadT))+" - "+str(round(segundos,2)))

        draw_str(vis, (550, 20),text)
        draw_str(vis, (20, 470),"Segundos: "+str(round(segundos,2)))
        outputTxt.write(text+" - "+str(velocidadT) +" - "+ str(round(segundos,2)) + "\n")
        #cv2.imshow('original',img2)
        #cv2.imshow('fgmask',fgmask)
        #cv2.imshow('thresh',thresh)
        cv2.imshow('morph',morph2)
        cv2.imshow('vis',vis)
        #cv2.imshow('pr',frame_gray)
        #cv2.imshow('key',im_with_keypoints)
        #cv2.imshow('thresh',thresh)
        

        ch = cv2.waitKey(5)
        if ch == 27:
            break
        if ch == ord('s'):
            cv2.imwrite('shots/vis_FINAL2.png',vis)
            cv2.imwrite('shots/morph_FINAL2.png',morph)
    cv2.destroyAllWindows()    
    


if __name__ == '__main__':
    main()
    todo()
    
