
# coding: utf-8

# In[1]:

#imports
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
import os
import time
import math
from collections import defaultdict
from numpy.linalg import inv

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


# In[2]:

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def nothing(x):
    pass


# In[ ]:

cv2.destroyAllWindows()


# In[3]:

visualize = True
def detect(img):
    img_gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gs = clahe.apply(img_gs)
    img_gs_f = np.float32(img_gs)
    dst = cv2.cornerHarris(img_gs_f,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    coords = np.transpose(np.nonzero(dst>0.01*dst.max()))
    # Threshold for an optimal value, it may vary depending on the image.
    #if visualize:
        #img[dst>0.01*dst.max()]=[0,0,255]
    #print coords
    return (coords,img_gs_f, img_gs)

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

#get image of a line between two points with given width
def subimage2(image, c1, c2, width):
    w = c1[0]-c2[0]
    h = c1[1]-c2[1]
    center = topoint((c1+c2)/2)
    c = math.sqrt(dist2(c1, c2))
    ts = -w/c
    tc = h/c
    #print (topoint(c1), topoint(c2), center, ts, tc, w, h, c)
    return subimage(image, center, ts, tc, width, c)

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
    

def findGraph(candidates, img_gs, max_mean=40):
    mask = np.zeros_like(img_gs)
    graph = defaultdict(lambda:[])
    if visualize:
        cv2.imshow('clahe', img_gs)
    
    for (i,c) in enumerate(candidates):
        for j in range(i):
            sub = subimage2(img_gs, c, candidates[j],10)
            min_ = np.amin(sub, axis = 1)
            
            
            if min_.mean() < max_mean:
                if visualize:
                    cv2.line(mask, topoint(c), topoint(candidates[j]),255,1)
                graph[i] += [j]
                graph[j] += [i]
    if visualize:
        cv2.imshow('graph', mask)
    return graph, mask

#http://stackoverflow.com/a/246063
def CrossProductZ(a,b):
    return a[0] * b[1] - a[1] * b[0]

def Orientation(a, b, c):
    return CrossProductZ(a, b) + CrossProductZ(b, c) + CrossProductZ(c, a)

def OrientationC(g, a, b, c):
    return Orientation(g[a],g[b],g[c])

def add1(a):
    return np.append(a,[1])

def findHouse(graph, mask_, img, candidates, primary=5, secondary=4, name='_X', color=(0,0,255),img_gs=None):
    if visualize:
        mask = mask_.copy()
    for i in graph: #subgraph localization
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
        if OrientationC(candidates,i,a,b)<0:
            (a,b) = (b,a)
        for x in [a]+graph[a]:
            for y in graph[x]:
                if x<y:
                    if visualize:
                        cv2.line(mask, topoint(candidates[x]), topoint(candidates[y]),255,3)
                    cv2.line(img, topoint(candidates[x]), topoint(candidates[y]),color,1)
        used = {i: 1, a:1, b:1}
        
        x = 0
        s = 0
        if name == '_X':#find middle point
            for j in filter(lambda x: x not in used, graph[a]):
                s_ = 0
                for k in graph[j]:
                    s_ += dist2(candidates[j],candidates[k])
                if x == 0 or s_ < s:
                    x = j
                    s = s_
            used[x] = 1
        cd = filter(lambda x: x not in used, graph[a])
        if len(cd) != 2:
            #print "cd has len %d" %len(cd)
            return (None, None)
        [c,d] = cd
        if OrientationC(candidates,x,c,d)<0:
            (c,d) = (d,c)
    
        points_imdg = [candidates[__] for __ in [a,b,c,d]]#,x,i
        if name == '_X':
            points_img = [candidates[__] for __ in [a,b,c,d,x]]#,x,i
            new_3d = np.float32([[100,0,0],[0,0,0],[100,100,0],[0,100,0],[50,50,0]])
        else:
            points_img = [candidates[__] for __ in [a,b,c,d]]#,x,i
            new_3d = np.float32([[100,0,0],[0,0,0],[100,100,0],[0,100,0]])
        p_img = np.array(points_img, np.float32)
        cv2.cornerSubPix(img_gs,p_img,(11,11),(-1,-1),criteria)
        #cv2.circle(img, topoint(p_img[5]), 5, (255,0,0),-1)
        cv2.circle(img, topoint(p_img[0]), 5, (0,0,255),-1)#tr
        cv2.circle(img, topoint(p_img[1]), 5, (0,255,0),-1)#tl
        #cv2.circle(img, topoint(p_img[4]), 5, (255,255,255),-1)
        cv2.circle(img, topoint(p_img[2]), 5, (0,255,255),-1)#br
        cv2.circle(img, topoint(p_img[3]), 5, (255,0,255),-1)#bl
        return draw3d(img,p_img,new_3d)
    #   
    return (None,None)

goal_3d = np.float32([[50,50,-1]])
def draw3d(img, pts, new_3d):
    fx = 0.5 + cv2.getTrackbarPos('focal', 'dst') / 50.0
    h, w = img.shape[:2]
    K = np.float64([[fx*w, 0, 0.5*(w-1)],
                    [0, fx*w, 0.5*(h-1)],
                    [0.0,0.0,      1.0]])
    dist_coef = np.zeros(4)
    ret, rvec, tvec = cv2.solvePnP(new_3d, pts, K, dist_coef)
    goal_3d[0,2] = -1 * cv2.getTrackbarPos('height', 'dst')
    verts = cv2.projectPoints(goal_3d, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
    cv2.circle(img, topoint(verts[0]), 5, (255,255,255),-1)
    for p in pts:
        cv2.line(img, topoint(p), topoint(verts[0]), (255,255,255),2, -1)

def detectHouse(frame, house_type='_X'):
    (coords_,img_gs_f, img_gs) = detect(frame)
    candidates = circleCoords(coords_, np.zeros_like(frame))
    graph, mask = findGraph(candidates, img_gs,max_mean=cv2.getTrackbarPos('max_mean', 'dst'))
    findHouse(graph, mask, frame, candidates, 5, 4, '_X', img_gs=img_gs)
    #findHouse(graph, mask, frame, candidates, 3, 2, '_N', img_gs=img_gs)
    cv2.imshow('dst',frame)


# In[ ]:

#camera example
cv2.namedWindow('dst')
cv2.createTrackbar('focal', 'dst', 25, 50, nothing)
cv2.createTrackbar('height', 'dst', 50, 200, nothing)
cv2.createTrackbar('max_mean', 'dst', 40, 200, nothing)

if 'cap' in locals():
    cap.release()
cap = cv2.VideoCapture(1)#select correct camera\

while(True):
    ret, frame = cap.read()
    detectHouse(frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
if 'cap' in locals():
    cap.release()


# In[4]:

#scanned pictures example
mypath = 'frames'
cv2.namedWindow('dst')
cv2.createTrackbar('focal', 'dst', 25, 50, nothing)
cv2.createTrackbar('height', 'dst', 50, 200, nothing)
cv2.createTrackbar('max_mean', 'dst', 40, 200, nothing)

for f in glob.glob(os.path.join(mypath,'*.jpg')):
    frame_orig = cv2.imread(f)
    while True:
        frame = frame_orig.copy()
        detectHouse(frame)
        key = cv2.waitKey(100)
        if key & 0xFF == ord('q') or key & 0xFF == ord(' '):
            break
    if key & 0xFF == ord('q'):
        break
#for f in listdir(mypath)


# In[5]:

cv2.__version__


# In[ ]:

#store frames

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

