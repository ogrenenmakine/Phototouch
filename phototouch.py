from __future__ import print_function
import os
import sys
import random
import string
import math
import numpy as np
import skimage.io
import queue
import threading
from PIL import ImageOps
from PIL import ImageTk
from PIL import Image, ImageDraw
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import matplotlib
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from collections import*
import speech_recognition as sr
ROOT_DIR = os.path.abspath("MaskRCNN/")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
file_names = next(os.walk(IMAGE_DIR))[2]

####### Draw on image##########
###############################

def drawOnImage(canvas):
    wDraw=Toplevel(canvas.data.mainWindow)
    wDraw.title="Draw"
    drawing=Frame(wDraw)
    rB=Button(drawing, bg="red", width=2,command=lambda: chooseColor(wDraw,canvas, "red"))
    rB.grid(row=0,column=0)
    bB=Button(drawing, bg="blue", width=2,command=lambda: chooseColor(wDraw,canvas, "blue"))
    bB.grid(row=0,column=1)
    gB=Button(drawing, bg="green",width=2,command=lambda: chooseColor(wDraw,canvas, "green"))
    gB.grid(row=0,column=2)
    mB=Button(drawing, bg="magenta", width=2,command=lambda: chooseColor(wDraw,canvas, "magenta"))
    mB.grid(row=1,column=0)
    cB=Button(drawing, bg="cyan", width=2,command=lambda: chooseColor(wDraw,canvas, "cyan"))
    cB.grid(row=1,column=1)
    yB=Button(drawing, bg="yellow",width=2,command=lambda: chooseColor(wDraw,canvas, "yellow"))
    yB.grid(row=1,column=2)
    oB=Button(drawing, bg="orange", width=2,command=lambda: chooseColor(wDraw,canvas, "orange"))
    oB.grid(row=2,column=0)
    pB=Button(drawing, bg="purple",width=2,command=lambda: chooseColor(wDraw,canvas, "purple"))
    pB.grid(row=2,column=1)
    bB=Button(drawing, bg="brown",width=2,command=lambda: chooseColor(wDraw,canvas, "brown"))
    bB.grid(row=2,column=2)
    blB=Button(drawing, bg="black",width=2,command=lambda: chooseColor(wDraw,canvas, "black"))
    blB.grid(row=3,column=0)
    wB=Button(drawing, bg="white",width=2,command=lambda: chooseColor(wDraw,canvas, "white"))
    wB.grid(row=3,column=1)
    grB=Button(drawing, bg="gray",width=2,command=lambda: chooseColor(wDraw,canvas, "gray"))
    grB.grid(row=3,column=2)
    drawing.pack(side=BOTTOM)

def chooseColor(wDraw, canvas, color):
    canvas.data.drawingColor=color
    canvas.data.mainWindow.bind("<B1-Motion>", lambda event: drawDraw(event, canvas))
    wDraw.destroy()

def drawDraw(event, canvas):
    x=int(round((event.x-canvas.data.imageTopX)*canvas.data.imageScale))
    y=int(round((event.y-canvas.data.imageTopY)*canvas.data.imageScale))
    draw = ImageDraw.Draw(canvas.data.image)
    draw.ellipse((x-5, y-5, x+5, y+5), fill=canvas.data.drawingColor,outline=None)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

####### Histogram ##########
############################

def histogram(canvas):
    wHist=Toplevel(canvas.data.mainWindow)
    wHist.title("Histogram")
    canvas.data.ww=350
    canvas.data.hh=250
    histCanvas = Canvas(wHist, width=canvas.data.ww, height=canvas.data.hh)
    histCanvas.pack()
    OkHistFrame=Frame(wHist)
    OkHistButton=Button(OkHistFrame, text="Exit", command=lambda: closeHist(wHist))
    OkHistButton.grid(row=0,column=0)
    OkHistFrame.pack(side=BOTTOM)
    displayHistogram(canvas,wHist,histCanvas)

def closeHist(wHist):
    wHist.destroy()

def displayHistogram(canvas,histogramWindow, histCanvas):
    ww=canvas.data.ww
    hh=canvas.data.hh
    margin=50
    histCanvas.delete(ALL)
    im=canvas.data.image
    histCanvas.create_line(margin-1, hh-margin+1,margin-1+ 258, hh-margin+1)
    xmarkerStart=margin-1
    for i in range(0,257,64):
        xmarker="%d" % (i)
        histCanvas.create_text(xmarkerStart+i,hh-margin+7, text=xmarker)
    histCanvas.create_line(margin-1,hh-margin+1, margin-1, margin)
    ymarkerStart= hh-margin+1
    for i in range(0, hh-2*margin+1, 50):
        ymarker="%d" % (i)
        histCanvas.create_text(margin-1-10,ymarkerStart-i, text=ymarker)
    R, G, B=im.histogram()[:256], im.histogram()[256:512],im.histogram()[512:768]
    for i in range(len(R)):
        pixelNo=R[i]
        histCanvas.create_oval(i+margin,hh-pixelNo/100.0-1-margin,i+2+margin,hh-pixelNo/100.0+1-margin,fill="red",outline="red")
    for i in range(len(G)):
        pixelNo=G[i]
        histCanvas.create_oval(i+margin,hh-pixelNo/100.0-1-margin,i+2+margin,hh-pixelNo/100.0+1-margin,fill="green", outline="green")
    for i in range(len(B)):
        pixelNo=B[i]
        histCanvas.create_oval(i+margin,hh-pixelNo/100.0-1-margin,i+2+margin,hh-pixelNo/100.0+1-margin,fill="blue", outline="blue")
    
def cgray(canvas,mag):
    data = []
    for col in range(canvas.data.image.size[1]):
        for row in range(canvas.data.image.size[0]):
            r, g, b= canvas.data.image.getpixel((row, col))
            avg= int(round((r + g + b)/3.0))
            if mag > 0.7:
                R, G, B= round(avg*5/4), round(avg*5/4), round(avg*5/4)
            else:
                R, G, B= round(avg*3/4), round(avg*3/4), round(avg*3/4)
            data.append((R, G, B))
    canvas.data.image.putdata(data)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

def invert(canvas):
    canvas.data.image=ImageOps.invert(canvas.data.image)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

def sepia(canvas,mag):
    sepiaData=[]
    for col in range(canvas.data.image.size[1]):
        for row in range(canvas.data.image.size[0]):
            r, g, b= canvas.data.image.getpixel((row, col))
            avg= int(round((r + g + b)/3.0))
            if mag > 0.7:
                R, G, B= avg+150, avg+75, avg
            else:
                R, G, B= avg+75, avg+25, avg
            sepiaData.append((R, G, B))
    canvas.data.image.putdata(sepiaData)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

def posterize(canvas,mag):
    posterData=[]
    if mag > 0.7:
        for col in range(canvas.data.imageSize[1]):
            for row in range(canvas.data.imageSize[0]):
                r, g, b= canvas.data.image.getpixel((row, col))
                if r in range(96):
                    R=0
                elif r in range(96, 224):
                    R=128
                elif r in range(224,256):
                    R=255
                if g in range(96):
                    G=0
                elif g in range(96, 224):
                    G=128
                elif g in range(224,256):
                    G=255
                if b in range(96):
                    B=0
                elif b in range(96, 224):
                    B=128
                elif b in range(224,256):
                    B=255
                posterData.append((R, G, B))
    else:
        for col in range(canvas.data.imageSize[1]):
            for row in range(canvas.data.imageSize[0]):
                r, g, b= canvas.data.image.getpixel((row, col))
                if r in range(32):
                    R=0
                elif r in range(32, 96):
                    R=64
                elif r in range(96, 160):
                    R=128
                elif r in range(160, 224):
                    R=192
                elif r in range(224,256):
                    R=255
                if g in range(32):
                    G=0
                elif g in range(32, 96):
                    G=64
                elif g in range(96, 160):
                    G=128
                elif g in range(160, 224):
                    G=192
                elif g in range(224,256):
                    G=255
                if b in range(32):
                    B=0
                elif b in range(32, 96):
                    B=64
                elif b in range(96, 160):
                    B=128
                elif b in range(160, 224):
                    B=192
                elif b in range(224,256):
                    B=255
                posterData.append((R, G, B))
    canvas.data.image.putdata(posterData)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

def popupFocusOut(event):
    popup.unpost()

def doPopup(event):
    popup.post(event.x_root, event.y_root)

def helper(canvas,mask,counter,r):
    return lambda event:maskcrop(canvas,mask,counter,r)

def analyze(canvas):
    image = skimage.io.imread(filename)
    results = model.detect([image], verbose=1)
    r = results[0]
    counter = 0
    cobj = []
    for l in r['rois']:
        cobj.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)))
        counter = counter + 1
    counter = 0
    for obj in cobj:
        obj = canvas.create_rectangle(r['rois'][counter][3], r['rois'][counter][2], r['rois'][counter][1], r['rois'][counter][0], outline='lightyellow', fill='gray', stipple='@docs/transparent.xbm', width=2)
        A = np.multiply(image[:,:,0],r['masks'][:,:,counter]);
        B = np.multiply(image[:,:,1],r['masks'][:,:,counter]);
        C = np.multiply(image[:,:,2],r['masks'][:,:,counter]);
        mask =(np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
        canvas.tag_bind(obj,"<ButtonPress-3>", doPopup)
        canvas.tag_bind(obj,"<ButtonPress-1>", helper(canvas,mask,counter,r))
        counter = counter + 1
    canvas.bind("<ButtonPress-1>",popupFocusOut)

def maskcrop(canvas,mask,counter,r):
    image = skimage.io.imread(filename)
    w, h, d = image.shape
    ind = counter
    if cropAll.get() == True and addBackground.get() == True:
        counter = 0
        mmask = np.sum(r['masks'],2);
        background = askbackground()
        background = background[0:w,0:h,:]
        A1 = background[:,:,0] - np.multiply(background[:,:,0],mmask);
        B1 = background[:,:,1] - np.multiply(background[:,:,1],mmask);
        C1 = background[:,:,2] - np.multiply(background[:,:,2],mmask);
        A = A1 + np.multiply(image[:,:,0],mmask);
        B = B1 + np.multiply(image[:,:,1],mmask);
        C = C1 + np.multiply(image[:,:,2],mmask);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
    if cropAll.get() == True and addBackground.get() == False:
        counter = 0
        mmask = np.sum(r['masks'],2);
        A = np.multiply(image[:,:,0],mmask);
        B = np.multiply(image[:,:,1],mmask);
        C = np.multiply(image[:,:,2],mmask);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
    if addBackground.get() == True and cropAll.get() == False:
        background = askbackground()
        background = background[0:w,0:h,:]
        A1 = np.multiply(mask[:,:,0],r['masks'][:,:,ind]);
        B1 = np.multiply(mask[:,:,1],r['masks'][:,:,ind]);
        C1 = np.multiply(mask[:,:,2],r['masks'][:,:,ind]);
        A = A1 + np.multiply(background[:,:,0],~r['masks'][:,:,ind]);
        B = B1 + np.multiply(background[:,:,1],~r['masks'][:,:,ind]);
        C = C1 + np.multiply(background[:,:,2],~r['masks'][:,:,ind]);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))    
    im = Image.fromarray(np.uint8(mask))
    canvas.data.image=im
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)

def crop(canvas):
    canvas.data.colourPopToHappen=False
    canvas.data.drawOn=False
    canvas.data.cropPopToHappen=True
    canvas.data.mainWindow.bind("<ButtonPress-1>", lambda event: startCrop(event, canvas))
    canvas.data.mainWindow.bind("<B1-Motion>", lambda event: drawCrop(event, canvas))
    canvas.data.mainWindow.bind("<ButtonRelease-1>", lambda event: endCrop(event, canvas))
    canvas.data.mainWindow.bind("<Return>", lambda event: performCrop(event, canvas))

def startCrop(event, canvas):
    if canvas.data.endCrop==False and canvas.data.cropPopToHappen==True:
        canvas.data.startCropX=event.x
        canvas.data.startCropY=event.y

def drawCrop(event,canvas):
    if canvas.data.endCrop==False and canvas.data.cropPopToHappen==True:
        canvas.data.tempCropX=event.x
        canvas.data.tempCropY=event.y
        canvas.create_rectangle(canvas.data.startCropX, canvas.data.startCropY, canvas.data.tempCropX, canvas.data.tempCropY, fill="gray", stipple="gray12", width=0)
        
def endCrop(event, canvas):
    if canvas.data.cropPopToHappen==True:
        canvas.data.endCrop=True
        canvas.data.endCropX=event.x
        canvas.data.endCropY=event.y
        canvas.create_rectangle(canvas.data.startCropX, canvas.data.startCropY, canvas.data.endCropX, canvas.data.endCropY, fill="gray", stipple="gray12", width=0 )

def performCrop(event,canvas):
    imageWidth=canvas.data.image.size[0] 
    imageHeight=canvas.data.image.size[1]
    im = canvas.data.originalImage.crop(\
    (int(round((canvas.data.startCropX-canvas.data.imageTopX)*canvas.data.imageScale)),
    int(round((canvas.data.startCropY-canvas.data.imageTopY)*canvas.data.imageScale)),
    int(round((canvas.data.endCropX-canvas.data.imageTopX)*canvas.data.imageScale)),
    int(round((canvas.data.endCropY-canvas.data.imageTopY)*canvas.data.imageScale))))
    canvas.data.image = im
    data = []
    for col in range(canvas.data.image.size[1]):
        for row in range(canvas.data.image.size[0]):
            r, g, b= canvas.data.image.getpixel((row, col))
            avg= int(round((r + g + b)/3.0))
            R, G, B= avg, avg, avg
            data.append((R, G, B))
    canvas.data.image.putdata(data)
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)
    
def makeImage(canvas):
    im = canvas.data.image
    canvas.data.originalImage=im.copy()
    canvas.data.imageSize=im.size
    imageWidth=canvas.data.image.size[0] 
    imageHeight=canvas.data.image.size[1]
    if imageWidth>imageHeight:
        resizedImage=im.resize((canvas.data.width,int(round(float(imageHeight)*canvas.data.width/imageWidth))))
        canvas.data.imageScale=float(imageWidth)/canvas.data.width
    else:
        resizedImage=im.resize((int(round(float(imageWidth)*canvas.data.height/imageHeight)),canvas.data.height))
        canvas.data.imageScale=float(imageHeight)/canvas.data.height
    canvas.data.imageForTk = ImageTk.PhotoImage(resizedImage)
    canvas.data.resizedIm = resizedImage
    return ImageTk.PhotoImage(resizedImage)

def drawImage(canvas):
    canvas.create_image(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0, canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0, anchor=NW, image=canvas.data.imageForTk)
    canvas.data.imageTopX=int(round(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0))
    canvas.data.imageTopY=int(round(canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0))

def saveAs(canvas):
    filename=filedialog.asksaveasfilename(defaultextension=".jpg")
    im=canvas.data.image
    im.save(filename)

def save(canvas):
    im=canvas.data.image
    im.save(canvas.data.imageLocation)

def openfile(canvas):
    global filename
    filename = filedialog.askopenfilename(initialdir = "MaskRCNN/images/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    im = Image.open(filename)
    canvas.data.image=im
    canvas.data.imageForTk=makeImage(canvas)
    drawImage(canvas)
    analyze(canvas)
    histogram(canvas)

def askbackground():
    r = sr.Recognizer()
    r.energy_threshold = 300
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.record(source, duration = 5)
        print("Done listening!")
    try:
        print("You said: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    global text
    text = r.recognize_google(audio)
    client = language.LanguageServiceClient()
    document = types.Document(content=text,type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    print('Text: {}'.format(text))
    if sentiment.score > 0.15:
        background = skimage.io.imread("MaskRCNN/images/bg1.jpg")
    elif sentiment.score < -0.15:
        background = skimage.io.imread("MaskRCNN/images/bg3.jpg")
    else:
        background = skimage.io.imread("MaskRCNN/images/bg2.jpg")
    return background
    
def autoprocess(canvas):
    client = language.LanguageServiceClient()
    document = types.Document(content=text,type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    print('Text: {}'.format(text))
    print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))
    if sentiment.score > 0.15:
        posterize(canvas,sentiment.magnitude)
    elif sentiment.score < -0.15:
        sepia(canvas,sentiment.magnitude)
    else:
        cgray(canvas,sentiment.magnitude)

def autoadd(canvas):
    r = sr.Recognizer()
    r.energy_threshold = 300
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.record(source, duration = 5)
        print("Done listening!")
    try:
        print("You said: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    text = r.recognize_google(audio)
    #text = "Please add a puppy."
    client = language.LanguageServiceClient()
    document = types.Document(content=text,type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    print('Text: {}'.format(text))
    if "cat" in text:
        background=np.array(canvas.data.image)
        image = skimage.io.imread("MaskRCNN/images/cat.jpg")
        mmask = np.sum(image != 0,axis=2)
        mmask = (mmask != 0)
        A1 = background[:,:,0] - np.multiply(background[:,:,0],mmask);
        B1 = background[:,:,1] - np.multiply(background[:,:,1],mmask);
        C1 = background[:,:,2] - np.multiply(background[:,:,2],mmask);
        A = A1 + np.multiply(image[:,:,0],mmask);
        B = B1 + np.multiply(image[:,:,1],mmask);
        C = C1 + np.multiply(image[:,:,2],mmask);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
        im = Image.fromarray(np.uint8(mask))
        canvas.data.image=im
        canvas.data.imageForTk=makeImage(canvas)
        drawImage(canvas)
    elif "puppy" in text:
        background=np.array(canvas.data.image)
        image = skimage.io.imread("MaskRCNN/images/puppy.jpg")
        mmask = np.sum(image != 0,axis=2)
        mmask = (mmask != 0)
        A1 = background[:,:,0] - np.multiply(background[:,:,0],mmask);
        B1 = background[:,:,1] - np.multiply(background[:,:,1],mmask);
        C1 = background[:,:,2] - np.multiply(background[:,:,2],mmask);
        A = A1 + np.multiply(image[:,:,0],mmask);
        B = B1 + np.multiply(image[:,:,1],mmask);
        C = C1 + np.multiply(image[:,:,2],mmask);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
        im = Image.fromarray(np.uint8(mask))
        canvas.data.image=im
        canvas.data.imageForTk=makeImage(canvas)
        drawImage(canvas)
    elif "bird" in text:
        background=np.array(canvas.data.image)
        image = skimage.io.imread("MaskRCNN/images/bird.jpg")
        mmask = np.sum(image != 0,axis=2)
        mmask = (mmask != 0)
        A1 = background[:,:,0] - np.multiply(background[:,:,0],mmask);
        B1 = background[:,:,1] - np.multiply(background[:,:,1],mmask);
        C1 = background[:,:,2] - np.multiply(background[:,:,2],mmask);
        A = A1 + np.multiply(image[:,:,0],mmask);
        B = B1 + np.multiply(image[:,:,1],mmask);
        C = C1 + np.multiply(image[:,:,2],mmask);
        mask = (np.concatenate((A[...,np.newaxis],B[...,np.newaxis],C[...,np.newaxis]),axis=2))
        im = Image.fromarray(np.uint8(mask))
        canvas.data.image=im
        canvas.data.imageForTk=makeImage(canvas)
        drawImage(canvas)

def donothing():
    filewin = Toplevel(root)
    button = Button(filewin, text="Do nothing button")
    button.pack()

def menuInit(root,canvas):
    menubar=Menu(root)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=lambda:openfile(canvas))
    filemenu.add_command(label="Save", command=lambda:save(canvas))
    filemenu.add_command(label="Save as...", command=lambda:saveAs(canvas))
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)
    
    editmenu = Menu(menubar, tearoff=0)
    editmenu.add_command(label="Undo", command=donothing)
    editmenu.add_separator()
    editmenu.add_command(label="Cut", command=donothing)
    editmenu.add_command(label="Copy", command=donothing)
    editmenu.add_command(label="Paste", command=donothing)
    editmenu.add_command(label="Delete", command=donothing)
    editmenu.add_command(label="Select All", command=donothing)
    editmenu.add_separator()
    editmenu.add_command(label="Crop", command=lambda:crop(canvas))
    menubar.add_cascade(label="Edit", menu=editmenu)
    
    processmenu = Menu(menubar, tearoff=0)
    processmenu.add_command(label="Draw", command=lambda:drawOnImage(canvas))
    processmenu.add_separator()
    processmenu.add_command(label="Gray", command=lambda:cgray(canvas))
    processmenu.add_command(label="Invert", command=lambda:invert(canvas))
    processmenu.add_command(label="Sepia", command=lambda:sepia(canvas))
    processmenu.add_command(label="Posterize", command=lambda:posterize(canvas))
    processmenu.add_separator()
    processmenu.add_command(label="Histogram", command=lambda:histogram(canvas))
    processmenu.add_command(label="Analyze", command=lambda:analyze(canvas))
    menubar.add_cascade(label="Process", menu=processmenu)
    
    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help Index", command=donothing)
    helpmenu.add_command(label="About...", command=donothing)
    menubar.add_cascade(label="Help", menu=helpmenu)
    menubar.add_command(label="StrictCommand", command=lambda:autoadd(canvas))
    menubar.add_command(label="Process", command=lambda:autoprocess(canvas))
    
    global popup
    global cropAll
    global addBackground
    cropAll = BooleanVar()
    addBackground = BooleanVar()
    popup = Menu(root, tearoff=0)
    popup.add_checkbutton(label="cropAll?",variable=cropAll)
    popup.add_checkbutton(label="addBackground?",variable=addBackground)
    
    root.config(menu=menubar)

def init(root, canvas):
    canvas.data.image=None
    canvas.data.angleSelected=None
    canvas.data.rotateWindowClose=False
    canvas.data.histWindowClose=False
    canvas.data.colourPopToHappen=False
    canvas.data.cropPopToHappen=False
    canvas.data.endCrop=False
    canvas.data.drawOn=True
    canvas.pack()
    
def main():
    root = Tk()
    canvasWidth = 640
    canvasHeight = 480
    canvas = Canvas(root, width=canvasWidth, height=canvasHeight)
    class Struct: pass
    canvas.data = Struct()
    canvas.data.width=canvasWidth
    canvas.data.height=canvasHeight
    canvas.data.mainWindow=root
    init(root, canvas)
    menuInit(root, canvas)
    root.bind("<Key>", lambda event:keyPressed(canvas, event))
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main())
