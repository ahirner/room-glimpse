**A fun IoT app using a Raspberry Pi + camera.**
The app detects [motion from the h264 encoder](http://picamera.readthedocs.io/en/release-1.12/recipes2.html#recording-motion-vector-data) with little CPU drain. Once a motion threshold is reached, a first snapshot is taken. The second snapshot is taken after the scene turns static again and is then getting analyzed. Thus, this _Thing of the Internet_ is a (wonky) surveillance camera and a selfie-machine at the same time --however you want to view it. The purpose was to demo Azure IoT and cognitive services on top of building an image acquisition framework on the RPi.

## Automatic Image Captioning
The importance of (data) privacy grows from day to day, but having a NN talk about its observations might just be ok... Thus, the actual pictures get only persisted on the local file system. However, we use Microsoft's [computer vision API](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api) to request tags, categories and a caption from the second snapshot and publish them.
## IoT Telemetry:
I wanted to try Azure's IoT Hub for data ingestion. All the results that describe a scene well, are forwarded via device-to-cloud-messages. This includes motion vectors for each frame during a scene. Learning gestures from this dataset would be even more fun!

#Example
**I'm entering the living room from the left**
![alt-text](https://raw.githubusercontent.com/ahirner/room-glimpse/master/example_snapshots/2017-02-21T10_52_21.227244_on.jpg)

The first snapshot is is triggered and saved ono the RPi. Meanhwile, motion vector data from each video frame is being sent to the cloud synchronously. 

**I pause to to complete the scene**
![alt-text](https://raw.githubusercontent.com/ahirner/room-glimpse/master/example_snapshots/2017-02-21T10_52_22.553099_off.jpg)

```javascript
    caption: 'a man that is standing in the living room'
    confidence: 0.1240666986256891
    tags: 'floor', 'indoor', 'person', 'window', 'table', 'room', 'man', 'living', 'holding', 'young', 'black', 'standing', 'woman', 'dog', 'kitchen', 'remote', 'playing', 'white'
```
This is how the second snapshot is described by Azure's cognitive API. Fair enough... Unfortunately, the caption doesn't mention my awesome guitar performance. All this info along with meta-information like timestamps gets sent to the cloud.

**I leave the room after much of applause** (snapshot omitted).
After no motion was detected for an predefined amount of time (0.75 secs in that case), another scene is analyzed. 
**Now it's just the bare room**
![alt-text](https://raw.githubusercontent.com/ahirner/room-glimpse/master/example_snapshots/2017-02-21T10_52_24.915270_off.jpg)
```javascript
    caption: 'a living room with hard wood floor'
    confidence: 0.9247661343688557
    tags: 'floor', 'indoor', 'room', 'living', 'table', 'building', 'window', 'wood', 'hard', 'wooden', 'sitting', 'television', 'black', 'furniture', 'kitchen', 'small', 'large', 'open', 'area', 'computer', 'view', 'home', 'white', 'modern', 'door', 'screen', 'desk', 'laptop', 'dog', 'refrigerator', 'bedroom'
```
Now this is pretty accurate (and confident).

#Installation
[Setup](https://azure.microsoft.com/en-us/resources/samples/iot-hub-c-raspberrypi-getstartedkit/) an Azure IoT Hub and add the RPi as a device.
Create `credentials.py` in `./creds` with the Azure Cognitive API key, the IoT device ID and a device connection string.
```python
AZURE_COG_KEY= 'xxx'
AZURE_DEV_ID= 'yyy'
AZURE_DEV_CONNECTION_STRING='HostName=zzz.azure-devices.net;SharedAccessKeyName=zzz;SharedAccessKey=zzz='
```
Start with `python3 room-glimpse.py`

`requirements.txt` tbd... To keep installation simple, the HTTP API is used instead of the dedicated azure-iot (not available via pip3 yet).
Parameters to configure video stream and motion thresholds are at the top of the file (proper configuration tbd).

#More ideas
1) Of course, nothing prevents you from running/training your own version of a [talking NN](https://github.com/tensorflow/models/tree/master/im2txt). In fact, this project is a vantage point to try pushing computing on the edge. Sam maintains a pip wheel to install TensorFlow on the little RPi. [Pete Warden](https://petewarden.com/2016/12/30/rewriting-tensorflow-graphs-with-the-gtt/) has done amazing work to trim down NNs in a principled way recently (e.g. quantization for fixed point math).

2) In general, make use of spare cores. Most of the time, the CPU idles at 15% (remember the h264 motion detection). So there is plenty of room left for [beefier tasks](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/) on the edge. 

3) Overlay the motion vectors (there is a 2D vector for each 16x16 pixel block) in a live web view.
