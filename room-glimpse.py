
# coding: utf-8


from __future__ import division

#Config Camera
RESOLUTION = (640, 480)
FPS = 30
ROTATION = 180
MD_BLOCK_FRACTION = 0.008 #Fraction of blocks that must show movement
MD_SPEED = 2.0            #How many screens those blocks must move per second
MD_FALLOFF = 0.5          #How many seconds no motion must be present to trigger completion of a scene

#Config Azure
AZURE_COG_HOST = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/analyze'
AZURE_COG_RETRIES = 3
from creds.credentials import *
from device.D2CMsgSender import D2CMsgSender

import numpy as np
import PIL.Image
import picamera
import picamera.array

from collections import namedtuple  #Forgo typing to maintain vinalla python 3.4 on RPi
import json
import io
import socket
import time, datetime
from queue import Queue

#Schema
Snapshot = namedtuple('Snapshot','timestamp, img_rgb, motion_raw, motion_magnitude_raw')
Motion = namedtuple('Motion', 'timestamp, triggered, vectors_x, vectors_y, sad, magnitude')

#Normalized versions with summary stats that can be sent to the cloud
#Todo: Identifiy MotionEvents with SnapshotEvents
MotionEvent = namedtuple('MotionEvent', 'timestamp, triggered, blocks_x, blocks_y, vectors_x, vectors_y, avg_x, avg_y, mag, sad')
SnapshotEvent = namedtuple('SnapshotEvent', 'timestamp_on, timestamp_off, caption, caption_conf, tags')



#Derive more constants
BLOCKSIZE = 16
MOTION_W = RESOLUTION[0] // BLOCKSIZE + 1
MOTION_H = RESOLUTION[1] // BLOCKSIZE + 1
BLOCKS = (MOTION_W)*(MOTION_H)
MD_BLOCKS = int(MD_BLOCK_FRACTION * BLOCKS)
MD_MAGNITUDE = int(MD_SPEED / FPS * RESOLUTION[0])
print("MD if >%i out of %i blocks show >%i pixel movement in a %i wide frame" % (MD_BLOCKS, BLOCKS, MD_MAGNITUDE, RESOLUTION[0]))



snapshot_queue = Queue(3)
motion_queue = Queue(FPS * 10) #Queue a maxmimum of 10 seconds motion-data

#Shared state for image and video analyzers
#Todo: Encapsulate state better by passing into constructor or multiple inheritance from both detectors 
current_state = {
    'motion_vectors_raw' : None,
    'motion_magnitude_raw': None,
    'last_md_time_true' : None,
    'last_md_time_false' : time.time(),
    'md': False,
    'rgb' : None,
    'last_jpg_motion' : None,
    'last_jpg_still' : None
}

class MyMotionDetector(picamera.array.PiMotionAnalysis):
    def analyse(self, a):
        m = np.sqrt(
            np.square(a['x'].astype(np.float)) +
            np.square(a['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
        # If there're more than 10 vectors with a magnitude greater
        # than 60, then say we've detected motion
        current_state['motion_vectors_raw'] = a
        current_state['motion_magnitude_raw'] = m
        
        #Todo: does motion- or RGB analysis come first? In the former case current_state['rgb'] lags one frame
        snap = Snapshot(time.time(), current_state['rgb'], a, m)
        
        if (m > MD_MAGNITUDE).sum() > MD_BLOCKS:
            md_update(True, snap)    
        else: md_update(False, snap)
        
class MyRGBAnalysis(picamera.array.PiRGBAnalysis):
    def analyse(self, a): 
        current_state['rgb'] = a 

def md_update(is_motion, snap: Snapshot):
    now = snap.timestamp
    before = current_state['last_md_time_true']
    md = current_state['md']
    
    #Queue motion_data
    if md:
        mv = snap.motion_raw
        mm = snap.motion_magnitude_raw
        motion_queue.put(Motion(now, is_motion, mv['x'], mv['y'], mv['sad'], mm))

    #Test if 
    if is_motion:
        current_state['last_md_time_true'] = now
        if not md:
            md_rising(snap)
            current_state['md'] = True
            return
    else:
        current_state['last_md_time_false'] = now
        if md is True and before is not None and (now - before) > MD_FALLOFF:
            md_falling(snap)
            current_state['md'] = False
            
#Attention: runs synchronous to motion detection
def md_rising(snap):
    a, m = snap.motion_raw, snap.motion_magnitude_raw
    avg_x, avg_y = a['x'].sum() / MOTION_W, a['y'].sum() / MOTION_H
    avg_m = m.sum()
    print('Motion detected, avg_x: %i, avg_y: %i, mag: %i' % (avg_x, avg_y, avg_m) )

def md_falling(snap):
    now = snap.timestamp
    print("Motion vanished after %f secs" % (now - current_state['last_md_time_true']))

    jpg = to_jpg(snap.img_rgb)
    result = analyze_pic(jpg)
    
    caption = result['description']['captions'][0]['text']
    caption_confidence = result['description']['captions'][0]['confidence']
    tags = result['description']['tags']
    on = to_ISO(current_state['last_md_time_true'])
    off = to_ISO(now)
    
    event = SnapshotEvent(on, off, caption, caption_confidence, tags)
    snapshot_queue.put(event)



import requests
import operator


def processRequest( json, data, headers, params ):
    """
    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """
    retries = 0
    result = None

    while True:
        response = requests.request( 'post', AZURE_COG_HOST, json = json, data = data, headers = headers, params = params )
        if response.status_code == 429: 
            print( "Message: %s" % ( response.json()['error']['message'] ) )
            if retries <= AZURE_COG_RETRIES: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:
            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )

        break
        
    return result

def analyze_pic(jpg, features='Color,Categories,Tags,Description'):
    # Computer Vision parameters
    params = { 'visualFeatures' : features} 

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = AZURE_COG_KEY
    headers['Content-Type'] = 'application/octet-stream'

    result = processRequest(None, jpg, headers, params )
    return result



def to_jpg(rgb):
    f = io.BytesIO()
    PIL.Image.fromarray(rgb).save(f, 'jpeg')
    return f.getvalue()
    
def to_ISO(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).isoformat()    


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

def dispatch_msgs(azure_msg):

    #SnapshotEvents
    #SetOption Batching to False to come closer to realtime HTTP calls.
    #Caveat: If MotionEvent queue is not emptied yet, it will block this messages. Todo: run in separate thread.
    #Discussion here: https://github.com/Azure/azure-iot-sdk-python/issues/15
    
    while snapshot_queue.empty() == False:
        se = snapshot_queue.get()
        print(se)
        azure_msg.sendD2CMsg(AZURE_DEV_ID, json.dumps(se._asdict()))
    
    #MotionEvents
    #SetOption Batching to True to save HTTP calls    
    nr_motion = 0
    while motion_queue.empty() == False:
        m = motion_queue.get()

        #MotionEvent = namedtuple('MotionEvent', 'timestamp, triggered, blocks_x, blocks_y, vectors_x, vectors_y, avg_x, avg_y, min_x, min_y, mag')
        #Todo: Normalize Fully
        avg_x, avg_y = m.vectors_x.sum() / MOTION_W, m.vectors_y.sum() / MOTION_H
        avg_m = m.magnitude.sum()
        me = MotionEvent(m.timestamp, m.triggered, MOTION_W, MOTION_H, list(m.vectors_x.flatten()), list(m.vectors_y.flatten()),                          avg_x, avg_y, avg_m, list(m.sad.flatten()))

        print(nr_motion)
        azure_msg.sendD2CMsg(AZURE_DEV_ID, json.dumps(me._asdict(), cls=MsgEncoder))
        nr_motion += 1



with picamera.PiCamera() as camera:      
    camera.resolution = RESOLUTION
    camera.framerate = FPS
    camera.rotation = ROTATION
    
    #Motion and video
    camera.start_recording(
        '/dev/null',
        format='h264',
        motion_output=MyMotionDetector(camera)
        )
    #RGB
    camera.start_recording(
        MyRGBAnalysis(camera),
        format='rgb',
        splitter_port=2
    )
    camera.wait_recording(0.5)

    azure_msg = D2CMsgSender(AZURE_DEV_CONNECTION_STRING)    
    while True:       
        try:
            dispatch_msgs(azure_msg)
            time.sleep(0.1)
                
        except KeyboardInterrupt:
            break
            
    camera.stop_recording(splitter_port=2)
    camera.stop_recording()





