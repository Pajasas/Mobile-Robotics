
# coding: utf-8

# In[139]:

#imports
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
import math
from collections import defaultdict


if 'oldsysstdout' not in locals():
    import sys
    oldsysstdout = sys.stdout
    class flushfile():
        def __init__(self, f):
            self.f = f
        def __getattr__(self,name): 
            return object.__getattribute__(self.f, name)
        def write(self, x):
            self.f.write(x)
            self.f.flush()
        def flush(self):
            self.f.flush()
    sys.stdout = flushfile(sys.stdout)


# In[ ]:

visualize = False
def detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    coords = np.transpose(np.nonzero(dst>0.01*dst.max()))
    # Threshold for an optimal value, it may vary depending on the image.
    #if visualize:
        #img[dst>0.01*dst.max()]=[0,0,255]
    #print coords
    #cv2.imshow('dst',img)
    return coords

def dist2(a, b):
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])

def subimage(image, center, ts, tc, width, height):
    
    v_x = (tc, ts)
    v_y = (-ts,tc)
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])


    return cv2.warpAffine(
        image,
        mapping,
        (int(width), int(height)),
        flags=cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)

def subimage2(image, c1, c2, width):
    w = c1[0]-c2[0]
    h = c1[1]-c2[1]
    center = topoint((c1+c2)/2)
    c = math.sqrt(dist2(c1, c2))
    ts = -w/c
    tc = h/c
    #print (topoint(c1), topoint(c2), center, ts, tc, w, h, c)
    return subimage(image, center, ts, tc, width, c)

def mergeCoords(coords,maxDist2=100):
    clusters = []
    for c in coords:
        matched = 0
        for cluster in clusters:
            ok = 1
            for n in cluster:
                if dist2(c, n)>maxDist2:
                    ok =0
                    break
            if ok == 1:
                cluster += [c]
                matched = 1
                break
        if matched == 0:
            clusters += [[c]]
    return clusters

def circleCoords(coords, mask_zero, radius = 2, maxSize = 22):
    mask = mask_zero[:,:,0].copy()
    for c in coords:
        cv2.circle(mask, (c[1],c[0]), radius,255,-1)
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if visualize:#for visualization only
        max_ = np.amax(markers)
        markers *= 255/max_
        markers = markers.astype(np.uint8)
        markers_col = cv2.applyColorMap(markers, cv2.COLORMAP_JET)
    candidates = []
    for (i,c) in enumerate(centroids):
        if  (stats[i,cv2.CC_STAT_WIDTH] < maxSize and stats[i,cv2.CC_STAT_HEIGHT] < maxSize and
            (stats[i,cv2.CC_STAT_WIDTH] > 4*radius or stats[i,cv2.CC_STAT_HEIGHT] >4*radius)):
            if visualize:
                cv2.circle(markers_col, (int(c[0]),int(c[1])), maxSize/2,(0,0,255),1)
            candidates += [c]
    if visualize:#for visualization only
        cv2.imshow('markers_col', markers_col)
    return candidates
    

def topoint(x):
    return (int(x[0]),int(x[1]))
    

def findGraph(candidates, img):
    img_gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gs = clahe.apply(img_gs)
    mask = np.zeros_like(img_gs)
    graph = defaultdict(lambda:[])
    if visualize:
        cv2.imshow('clahe', img_gs)
    
    for (i,c) in enumerate(candidates):
        for j in range(i):
            sub = subimage2(img_gs, c, candidates[j],10)
            min_ = np.amin(sub, axis = 1)
            
            
            if min_.mean() < 40:
                if visualize:
                    cv2.line(mask, topoint(c), topoint(candidates[j]),255,1)
                graph[i] += [j]
                graph[j] += [i]
    return graph, mask

def findHouse(graph, mask_, img, primary=5, secondary=4, name='_X', color=(0,0,255)):
    if visualize:
        mask = mask_.copy()
    for i in graph:
        if len(graph[i]) != 2:
            continue
        [a,b] = graph[i]
        if len(graph[a]) != primary or len(graph[a]) != primary:
            continue
        ok = 1
        for c in graph[a]:
            if c != b and c!= i and len(graph[c]) != secondary:
                ok = 0
                break
        if ok == 0:
            continue
        print [a]+graph[a]
        for x in [a]+graph[a]:
            for y in graph[x]:
                if x<y:
                    if visualize:
                        cv2.line(mask, topoint(candidates[x]), topoint(candidates[y]),255,3)
                    cv2.line(img, topoint(candidates[x]), topoint(candidates[y]),color,1)
        
    if visualize:          
        cv2.imshow('mask'+name, mask)
    return

mypath = 'frames'
if 'cap' in locals():
    cap.release()
#cap = cv2.VideoCapture(0)
#while(True):
for f in listdir(mypath):
    if not isfile(join(mypath, f)):
        continue
    
    #ret, frame = cap.read()
    frame = cv2.imread(join(mypath, f))
    start = time.time()
    coords_ = detect(frame)
    candidates = circleCoords(coords_, np.zeros_like(frame))
    graph, mask = findGraph(candidates, frame)
    findHouse(graph, mask, frame, 5, 4, '_X')
    findHouse(graph, mask, frame,3, 2, '_N', color=(0,255,0))
    end = time.time()
    cv2.imshow('dst',frame)
    print end - start
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    #break
cap.release()
#coords_


# In[18]:

len(clusters)


# In[23]:



if 'cap' in locals():
    cap.release()
cap = cv2.VideoCapture(0)
i = 0
while(True):
    ret, frame = cap.read()
    #coords_ = detect(frame)
    #clusters = mergeCoords(coords_)
    cv2.imwrite('frame_%d.jpg'%i,frame)
    cv2.imshow('dst',frame)
    i+=1
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    #break
cap.release()

