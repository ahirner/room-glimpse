
# coding: utf-8


from __future__ import division

from config import *
from creds.credentials import *
from vision import *

from device.D2CMsgSender import D2CMsgSender
import json
import io, os
import socket
import time, datetime

import numpy as np
import PIL.Image
import picamera
import picamera.array

from queue import Queue
from collections import namedtuple  #Forgo typing to maintain vanilla python 3.4 compatibility on RPi

#Schema
Motion = namedtuple('Motion', 'timestamp, triggered, vectors_x, vectors_y, sad, magnitude')
Snapshot = namedtuple('Snapshot','timestamp, img_rgb, motion')
PictureEvent = namedtuple('PictureEvent', 'timestamp, type, on, data')
SceneCapture = namedtuple('SceneCapture', 'pic_on, pic_off')



def to_jpg(rgb):
    f = io.BytesIO()
    PIL.Image.fromarray(rgb).save(f, 'jpeg')
    return f.getvalue()
    
def to_ISO(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).isoformat()

def to_ID(timestamp, on):
    return str(to_ISO(timestamp).replace(':', '_')) + ('_on' if on else '_off')

def get_convert_jpg(pic: PictureEvent, modify=True):
    jpg = pic.data
    if pic.type == 'rgb':
        jpg = to_jpg(pic.data)
        if modify: 
            pic.data = jpg
            pic.type = 'jpg'
    #Todo: Exception handling
    return jpg

def save_jpg(jpg, _id):
    if DATA_FOLDER is not None:
        with open(os.path.join(DATA_FOLDER, _id+'.jpg'), "wb") as f:   
            f.write(jpg)     



#Derive more constants
BLOCKSIZE = 16
MOTION_W = RESOLUTION[0] // BLOCKSIZE + 1
MOTION_H = RESOLUTION[1] // BLOCKSIZE + 1
BLOCKS = (MOTION_W)*(MOTION_H)
MD_BLOCKS = int(MD_BLOCK_FRACTION * BLOCKS)
MD_MAGNITUDE = int(MD_SPEED / FPS * RESOLUTION[0])
print("Motion detected if >%i out of %i blocks show >%i pixel movement in a %i wide frame" % (MD_BLOCKS, BLOCKS, MD_MAGNITUDE, RESOLUTION[0]))



class MyRGBAnalysis(picamera.array.PiRGBAnalysis):
    def __init__(self, camera):
        super().__init__(camera)
        self.rgb = None
        
    def analyse(self, a): 
        self.rgb = a 

class MyMotionDetector(picamera.array.PiMotionAnalysis):
    
    def __init__(self, camera, rgb_detect: MyRGBAnalysis, scene_queue, motion_queue, picture_queue):
        super().__init__(camera)
        
        self.last_md_time_false = time.time()
        self.last_md_time_true = None
        self.rgb_detect = rgb_detect
        self.md = False
        
        self.scene_queue = scene_queue
        self.motion_queue = motion_queue
        self.picture_queue = picture_queue
        
    def analyse(self, a):
        m = np.sqrt(
            np.square(a['x'].astype(np.float)) +
            np.square(a['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
    
        #If there're more than MD_BLOCKS vectors with a magnitude greater
        #than MD_MAGNITUDE, then say we've detected motion
        md = ((m > MD_MAGNITUDE).sum() > MD_BLOCKS)
        
        now = time.time()
        motion = Motion(now, md, a['x'], a['y'], a['sad'], m)
        
        #Todo: does motion or RGB analysis come first? In the former case, current_state['rgb'] lags one frame
        # --> use separate picamera capture        
        snap = Snapshot(now, rgb_detect.rgb, motion)
        
        self.md_update(snap)
        
    def md_update(self, snap: Snapshot):
        now = snap.timestamp
        before = self.last_md_time_true
        is_motion = snap.motion.triggered

        #Test if motion detection flipped over
        if is_motion:
            self.last_md_time_true = now
            if not self.md:
                self.md_rising(snap)
                self.md = True
        else:
            self.last_md_time_false = now
            if self.md is True and before is not None and (now - before) > MD_FALLOFF:
                self.md_falling(snap)
                self.md = False 

        #Queue motion data
        if self.md:
            if (self.motion_queue): self.motion_queue.put(snap.motion)


    #Attention: runs in blocking sync with motion detection
    def md_rising(self, snap: Snapshot):
        now = snap.timestamp    
        motion = snap.motion

        #Calculate Summary statistics only for debugging purposes
        avg_x = motion.vectors_x.sum() / RESOLUTION[0]
        avg_y = motion.vectors_y.sum() / RESOLUTION[1]
        avg_m = motion.magnitude.sum() / (RESOLUTION[0] * RESOLUTION[1])

        print('Motion detected, avg_x: %i, avg_y: %i, mag: %i' % (avg_x, avg_y, avg_m) )

        pic = PictureEvent(now, 'rgb', True, snap.img_rgb)
        self.last_pic_on = pic
        if (self.picture_queue): self.picture_queue.put(pic)

    def md_falling(self, snap: Snapshot):
        now = snap.timestamp
        print("Motion vanished after %f secs" % (now - self.last_md_time_true))

        pic = PictureEvent(now, 'jpg', False, to_jpg(snap.img_rgb))
        
        if (self.picture_queue): self.picture_queue.put(pic)
        if (self.scene_queue): self.scene_queue.put(SceneCapture(self.last_pic_on, pic))



#Custom encoder for objects containing numpy 
class MsgEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MsgEncoder, self).default(obj)

#Normalized versions with summary stats to be sent to the cloud
MotionEvent = namedtuple('MotionEvent', 'timestamp, triggered, blocks_x, blocks_y, vectors_x, vectors_y, avg_x, avg_y, mag, sad')
SceneEvent = namedtuple('SceneEvent', 'timestamp_on, timestamp_off, caption, caption_conf, tags')        

def dispatch_scene(scene_queue, azure_msg):
    while True:
        scene = scene_queue.get()
        
        jpg_off = get_convert_jpg(scene.pic_off, False)
        
        result = analyze_img(jpg_off)
        caption = result['description']['captions'][0]['text']
        caption_confidence = result['description']['captions'][0]['confidence']
        tags = result['description']['tags']
        on = to_ISO(scene.pic_on.timestamp)
        off = to_ISO(scene.pic_off.timestamp)

        event = SceneEvent(on, off, caption, caption_confidence, tags)
        print(event)
        azure_msg.sendD2CMsg(AZURE_DEV_ID, json.dumps(event._asdict(), cls=MsgEncoder))
        
        scene_queue.task_done()
                
def dispatch_motiondata(motion_queue, azure_msg):
    
    #Todo: we could reduce overhead by enabling batch transfer (with caveats interleaving low- and high latency messages).
    #This is only feasible with the official Python SDK. On *nix you still need to manually compile it.
    #Therefore, we go with HTTP and REST for now.
    #Discussion here: https://github.com/Azure/azure-iot-sdk-python/issues/15
    nr_motion = 0
    while True:
        m = motion_queue.get()

        #MotionEvent = namedtuple('MotionEvent', 'timestamp, triggered, blocks_x, blocks_y, vectors_x, vectors_y, avg_x, avg_y, min_x, min_y, mag')
        avg_x = m.vectors_x.sum() / RESOLUTION[0]
        avg_y = m.vectors_y.sum() / RESOLUTION[1]
        avg_m = m.magnitude.sum() / (RESOLUTION[0] * RESOLUTION[1])
        
        me = MotionEvent(to_ISO(m.timestamp), to_ISO(m.triggered), MOTION_W, MOTION_H, list(m.vectors_x.flatten()), list(m.vectors_y.flatten()),                          avg_x, avg_y, avg_m, list(m.sad.flatten()))

        azure_msg.sendD2CMsg(AZURE_DEV_ID, json.dumps(me._asdict(), cls=MsgEncoder))
        
        nr_motion += 1
        print(nr_motion, to_ISO(m.timestamp), to_ISO(time.time()), avg_m)
        
        motion_queue.task_done()

def publish_pictures(picture_queue):
    while True:
        p = picture_queue.get()

        jpg = get_convert_jpg(p, False)
        _id = to_ID(p.timestamp, p.on)
        save_jpg(jpg, _id)
        
        print("File saved: " +_id)
        picture_queue.task_done()  



import _thread

if __name__ == "__main__":

    #Create queues
    #Todo: make cloud telemetrics and saving pictures optional    
    scene_queue = Queue(3)          #Queue three full scenes
    motion_queue = Queue(FPS * 10)  #Queue a maxmimum of 10 seconds motion-data
    picture_queue = Queue(4)        #Queue a maximum pair of two on and off snapshots
    
    with picamera.PiCamera() as camera:
        camera.resolution = RESOLUTION
        camera.framerate = FPS
        camera.rotation = ROTATION
        
        rgb_detect = MyRGBAnalysis(camera)
        motion_detect = MyMotionDetector(camera, rgb_detect, scene_queue, motion_queue, picture_queue)

        print("Starting camera and motion detection...")    
        #Set up motion and video stream analyzer
        camera.start_recording(
            '/dev/null',
            format='h264',
            motion_output=motion_detect
            )
        #Set up RGB capture on a splitter port
        camera.start_recording(
            rgb_detect,
            format='rgb',
            splitter_port=2
        )
        camera.wait_recording(0.5)

        #Todo: make cloud telemetrics and saving pictures optional
        print("Starting threads for data dispatch...")
        azure_msg = D2CMsgSender(AZURE_DEV_CONNECTION_STRING)    
        _thread.start_new_thread(dispatch_scene, (scene_queue, azure_msg))
        _thread.start_new_thread(dispatch_motiondata, (motion_queue, azure_msg))
        _thread.start_new_thread(publish_pictures, (picture_queue,))

        print("--- Exit with Ctrl-C ---")
        while True:       
            try:
                time.sleep(1)       
            except KeyboardInterrupt:
                break

        camera.stop_recording(splitter_port=2)
        camera.stop_recording()

        print("Camera stopped, waiting for queued data to dispatch")

        scene_queue.join()
        motion_queue.join()
        picture_queue.join()

