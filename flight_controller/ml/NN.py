import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import collections
import threading
import queue
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from typing import Optional

Gst.init(None)
cam_index_1 = 0
#cam_index_2 = 1

def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! appsink"
    )

class CameraReader:
    def __init__(self, cam_index):
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={cam_index} ! "
            f"video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.q = queue.Queue(maxsize=2)
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            sample = self.appsink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w = caps.get_value("width")
            h = caps.get_value("height")
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            buf.unmap(mapinfo)
            if not self.q.empty():
                self.q.get_nowait()
            self.q.put(frame)

cam_forward = CameraReader(cam_index_1) 
#cam_under = CameraReader(cam_index_2) 
model = YOLO("models/best.pt")
SOURCE = None 
CAP_WIDTH = 1280
CAP_HEIGHT = 720 
latest_detection = None

buffer_size = 7 
detection_buffer = collections.deque(maxlen=buffer_size)

def normalised_coordinates(frame):
    results = model.track(frame, tracker="custom_bytetrack.yaml", persist=True, conf=0.6, device=0)
    annotated = results[0].plot()   

    boxes = results[0].boxes.xywhn
    ids = results[0].boxes.id
    if ids is None:
        return None, annotated     
    for box, track_id in zip(boxes, ids):
        x, y, w, h = box
    return (x, y, w, h, track_id), annotated

def get_normalized_coordinates(camera_object):
    global latest_detection
    local_buffer = collections.deque(maxlen=buffer_size)

    import os
    from datetime import datetime
    os.makedirs("recordings", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"recordings/flight_forward_{ts}.mkv"
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    writer = None  
    print(f"[NN] enregistrement video -> {out_path}")

    while True:
        frame = camera_object.q.get()
        result, annotated = normalised_coordinates(frame)

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
            if not writer.isOpened():
                print(f"[NN] ATTENTION: VideoWriter n'a pas pu s'ouvrir ({out_path}) - essai MJPG")
                writer = cv2.VideoWriter(out_path.replace(".mkv", ".avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (w, h))

        if writer is not None and writer.isOpened():
            writer.write(annotated)

        if result is not None:
            x, y, w, h, track_id = result
            local_buffer.appendleft(track_id)
            if local_buffer.count(track_id) >= int(buffer_size * 0.6):
                latest_detection = (x, y, w, h, track_id)
                print(f"Detection robuste pour l'ID : {track_id}")
            else:
                latest_detection = None
        else:
            local_buffer.appendleft(None)
            latest_detection = None
    
if __name__ == "__main__":
    threading.Thread(target=get_normalized_coordinates, args=(cam_forward,), daemon=True).start()
    
    #threading.Thread(target=get_normalized_coordinates, args=(cam_under,), daemon=True).start()
    
    import time
    while True: time.sleep(1)
