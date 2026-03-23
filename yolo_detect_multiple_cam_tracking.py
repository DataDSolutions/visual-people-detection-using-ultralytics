import cv2
import argparse
import numpy as np
import os
import time
import uuid
from datetime import datetime
from ultralytics import YOLO

import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

# ============================================================
# ADDITIONAL MODULES (UID TRACKING + REID)
# ============================================================

class FeatureExtractor:

    def __init__(self, device="cpu"):

        self.device = device

        model = resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()

        self.model = model.to(device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256,128)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],
                        [0.229,0.224,0.225])
        ])

    def extract(self,img):

        with torch.no_grad():

            x=self.transform(img).unsqueeze(0).to(self.device)

            feat=self.model(x)

            feat=feat.cpu().numpy().flatten()

        feat=feat/np.linalg.norm(feat)

        return feat


class UIDDatabase:

    def __init__(self):

        self.people={}

        self.match_threshold=0.35

        self.disappear_time=15*60
        self.retention_time=2*60*60

    def new_uid(self):

        return str(uuid.uuid4())

    def find_match(self,embedding):

        best_uid=None
        best_dist=999

        for uid,data in self.people.items():

            dist=cosine(embedding,data["embedding"])

            if dist<best_dist:

                best_dist=dist
                best_uid=uid

        if best_dist<self.match_threshold:
            return best_uid

        return None


    def register(self,embedding,cam_id):

        now=time.time()

        uid=self.find_match(embedding)

        if uid is None:

            uid=self.new_uid()

            self.people[uid]={
                "embedding":embedding,
                "last_seen":now,
                "cams":{},
                "dwell":0
            }

        person=self.people[uid]

        person["last_seen"]=now

        if cam_id not in person["cams"]:

            person["cams"][cam_id]={
                "entry":now,
                "exit":None
            }

        return uid


    def update_exit(self,uid,cam_id):

        now=time.time()

        person=self.people.get(uid)

        if person is None:
            return

        if cam_id in person["cams"]:

            if person["cams"][cam_id]["exit"] is None:

                person["cams"][cam_id]["exit"]=now

                entry=person["cams"][cam_id]["entry"]

                person["dwell"]+=now-entry


    def cleanup(self):

        now=time.time()

        remove=[]

        for uid,data in self.people.items():

            if now-data["last_seen"]>self.retention_time:

                remove.append(uid)

        for uid in remove:

            del self.people[uid]


class MultiCameraTracker:

    def __init__(self,extractor,db):

        self.extractor=extractor
        self.db=db

        self.active_tracks={}

    def process_detection(self,frame,bbox,cam_id):

        x1,y1,x2,y2=bbox

        crop=frame[y1:y2,x1:x2]

        if crop.size==0:
            return None

        emb=self.extractor.extract(crop)

        uid=self.db.register(emb,cam_id)

        if cam_id not in self.active_tracks:

            self.active_tracks[cam_id]=set()

        self.active_tracks[cam_id].add(uid)

        return uid


    def end_frame(self,cam_id,current):

        prev=self.active_tracks.get(cam_id,set())

        disappeared=prev-current

        for uid in disappeared:

            self.db.update_exit(uid,cam_id)

        self.active_tracks[cam_id]=current


# ============================================================
# INITIALIZE TRACKING
# ============================================================

feature_extractor=FeatureExtractor()
uid_database=UIDDatabase()
tracker=MultiCameraTracker(feature_extractor,uid_database)

# ============================================================
# SCRIPT
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True,
                    help="Comma separated RTSP streams OR a text file with camera URLs")
parser.add_argument('--thresh', default=0.5)

args = parser.parse_args()

model_path = args.model
min_thresh = float(args.thresh)

# -----------------------------
# Source Parsing
# -----------------------------
if os.path.isfile(args.source) and args.source.lower().endswith(".txt"):
    sources = []
    with open(args.source, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line)
else:
    sources = args.source.split(',')

WIDTH = 640
HEIGHT = 480

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(model_path, task="detect")
labels = model.names

# -----------------------------
# Category Map
# -----------------------------
category_map = {

"Adult - Female":"Female",
"Adult - Female sitting":"Female",
"Adult - Female standing":"Female",

"Adult - Male":"Male",
"Adult - Male sitting":"Male",
"Adult - Male standing":"Male",

"Kid - Boy":"Child Boy",
"Kid - Boy sitting":"Child Boy",
"Kid - Boy standing":"Child Boy",

"Kid - Girl":"Child Girl",
"Kid - Girl sitting":"Child Girl",
"Kid - Girl standing":"Child Girl"

}

# -----------------------------
# Daily Log Folder
# -----------------------------
today = datetime.now().strftime("%Y-%m-%d")
log_dir = os.path.join("logs", today)
os.makedirs(log_dir, exist_ok=True)

# -----------------------------
# Create CSV files
# -----------------------------
log_files = []

for i in range(len(sources)):

    path = os.path.join(log_dir,f"cam_{i}.csv")

    f = open(path,"a")

    if os.stat(path).st_size == 0:
        f.write("Timestamp,Female,Male,Child Boy,Child Girl,Total\n")

    log_files.append(f)

# -----------------------------
# Open Cameras
# -----------------------------
caps=[]

for src in sources:

    cap=cv2.VideoCapture(src,cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,2)

    caps.append(cap)

# -----------------------------
# FPS
# -----------------------------
fps_buffer=[]
fps_len=200
avg_fps=0

# -----------------------------
# Main Loop
# -----------------------------
while True:

    t_start=time.perf_counter()

    frames=[]

    for i,cap in enumerate(caps):

        ret,frame=cap.read()

        if not ret:
            frames.append(np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8))
            continue

        frame=cv2.resize(frame,(WIDTH,HEIGHT))

        results=model(frame,verbose=False)
        detections=results[0].boxes

        female=male=boy=girl=0
        people_detected=0

        current_uids=set()

        for det in detections:

            conf=det.conf.item()

            if conf<min_thresh:
                continue

            xyxy=det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin,ymin,xmax,ymax=xyxy

            cls=int(det.cls.item())
            classname=labels[cls]

            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)

            label=f"{classname}:{int(conf*100)}%"
            cv2.putText(frame,label,(xmin,ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            people_detected+=1

            # UID Tracking
            uid=tracker.process_detection(frame,(xmin,ymin,xmax,ymax),i)

            if uid:

                current_uids.add(uid)

                cv2.putText(frame,f"UID:{uid[:8]}",
                            (xmin,ymax+15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,0),1)

            if classname in category_map:

                g=category_map[classname]

                if g=="Female":
                    female+=1
                elif g=="Male":
                    male+=1
                elif g=="Child Boy":
                    boy+=1
                elif g=="Child Girl":
                    girl+=1

        tracker.end_frame(i,current_uids)

        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_files[i].write(
            f"{timestamp},{female},{male},{boy},{girl},{people_detected}\n"
        )
        log_files[i].flush()

        # DISPLAY TEXT
        cv2.putText(frame,f"Camera {i}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.putText(frame,f"People Detected: {people_detected}",
                    (10,60),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,255),2)

        cv2.putText(frame,f"Female: {female}",(10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Male: {male}",(10,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Child Boy: {boy}",(10,150),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Child Girl: {girl}",(10,180),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        frames.append(frame)

    rows = int(np.ceil(np.sqrt(len(frames))))
    cols = int(np.ceil(len(frames)/rows))

    grid=[]

    for r in range(rows):

        row_frames=[]

        for c in range(cols):

            idx=r*cols+c

            if idx<len(frames):
                row_frames.append(frames[idx])
            else:
                row_frames.append(np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8))

        grid.append(np.hstack(row_frames))

    display=np.vstack(grid)

    cv2.imshow("Multi Camera Detection",display)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

    t_stop=time.perf_counter()

    fps=1/(t_stop-t_start)

    fps_buffer.append(fps)

    if len(fps_buffer)>fps_len:
        fps_buffer.pop(0)

    avg_fps=np.mean(fps_buffer)

    uid_database.cleanup()

# -----------------------------
# Cleanup
# -----------------------------
for cap in caps:
    cap.release()

for f in log_files:
    f.close()

cv2.destroyAllWindows()