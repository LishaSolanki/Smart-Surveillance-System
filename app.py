"""
SENTINEL v2.1 — Smart Public Safety Surveillance System
Fixes: better detection thresholds, zone trigger, test mode, weapon simulation
"""
 
import cv2
import numpy as np
import asyncio
import base64
import time
import json
import os
import csv
import tempfile
import random
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
 
# ─── YOLOv8 Import ────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 available")
except ImportError:
    YOLO_AVAILABLE = False
    print(" YOLOv8 not found — using CV fallback. Run: pip install ultralytics")
 
BASE_DIR   = Path(__file__).parent
SNAP_DIR   = BASE_DIR / "snapshots"
EXPORT_DIR = BASE_DIR / "exports"
SNAP_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
 
app = FastAPI(title="SENTINEL")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
 
# ─── State ────────────────────────────────────────────────────────────────────
class SurveillanceState:
    def __init__(self):
        self.active_alerts   = []
        self.alert_history   = []
        self.stats = {
            "total_alerts": 0, "people_count": 0,
            "zone_violations": 0, "loitering_incidents": 0,
            "anomalies": 0, "weapon_detections": 0, "fight_detections": 0,
        }
        self.settings = {
            "crowd_threshold": 3,
            "loiter_seconds":  12,
            "fight_speed":     25,
            "enabled_features": {
                "zone_intrusion": True, "crowd_density": True,
                "loitering": True, "anomaly": True, "weapon": True,
            }
        }
        self.connected_clients = []
        self.camera_active     = False
        self.start_time        = time.time()
        self.yolo_status       = "not_loaded"
 
state = SurveillanceState()
 
# ─── Model Manager ────────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.person_model = None
        self.pose_model   = None
        self.loaded       = False
 
    def load(self):
        if not YOLO_AVAILABLE:
            return False
        try:
            print(" Loading YOLOv8n...")
            self.person_model = YOLO("yolov8n.pt")
            self.person_model.to("cpu")
            print(" Loading YOLOv8n-pose...")
            self.pose_model = YOLO("yolov8n-pose.pt")
            self.pose_model.to("cpu")
            self.loaded = True
            print("Models ready!")
            return True
        except Exception as e:
            print(f" Model error: {e}")
            return False
 
    def detect_persons(self, frame):
        if not self.loaded:
            return []
        try:
            res = self.person_model(frame, classes=[0], conf=0.35, verbose=False, device="cpu")[0]
            out = []
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                out.append({"bbox":[x1,y1,x2,y2],"conf":round(conf,2),
                             "center":((x1+x2)//2,(y1+y2)//2)})
            return out
        except Exception:
            return []
 
    def detect_objects(self, frame, classes, conf=0.30):
        if not self.loaded:
            return []
        try:
            res = self.person_model(frame, classes=classes, conf=conf, verbose=False, device="cpu")[0]
            out = []
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                label = res.names[int(box.cls[0])]
                out.append({"bbox":[x1,y1,x2,y2],"label":label,"conf":round(float(box.conf[0]),2)})
            return out
        except Exception:
            return []
 
    def get_pose_keypoints(self, frame):
        if not self.loaded or not self.pose_model:
            return []
        try:
            res = self.pose_model(frame, conf=0.35, verbose=False, device="cpu")[0]
            if res.keypoints is not None:
                return [kps.tolist() for kps in res.keypoints.xy]
            return []
        except Exception:
            return []
 
models = ModelManager()
 
# ─── Person Tracker ───────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self):
        self.tracks  = {}
        self.next_id = 1
 
    def update(self, detections):
        active = set()
        for det in detections:
            cx,cy     = det["center"]
            best_id   = None
            best_dist = 130
            for tid,td in self.tracks.items():
                px,py = td["pos"]
                d = np.sqrt((cx-px)**2+(cy-py)**2)
                if d < best_dist:
                    best_dist = d; best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "first_seen": time.time(), "pos":(cx,cy),
                    "positions":[(cx,cy)], "bbox":det["bbox"],
                    "conf":det.get("conf",1.0),
                    "in_zone":False, "loiter_alerted":False,
                }
            else:
                t = self.tracks[best_id]
                t["pos"] = (cx,cy); t["bbox"] = det["bbox"]
                t["positions"].append((cx,cy))
                if len(t["positions"]) > 60:
                    t["positions"].pop(0)
            det["id"] = best_id
            active.add(best_id)
        stale = [t for t,td in self.tracks.items()
                 if t not in active and time.time()-td["first_seen"] > 3]
        for t in stale:
            del self.tracks[t]
        return detections
 
    def reset(self):
        self.tracks = {}; self.next_id = 1
 
tracker = PersonTracker()
 
# ─── Detection Engine ─────────────────────────────────────────────────────────
class DetectionEngine:
    def __init__(self):
        self.frame_count  = 0
        self.bg_sub       = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
        self.cooldowns    = {}
        self.pose_history = {}
 
    def detect(self, frame):
        self.frame_count += 1
        h,w = frame.shape[:2]
        result = {"persons":[],"weapons":[],"alerts":[]}
 
        # Person detection
        raw = models.detect_persons(frame) if models.loaded else self._cv_persons(frame)
        persons = tracker.update(raw)
        result["persons"] = persons
 
        # Zone (top-right 38% width, 50% height)
        zx1,zy1 = int(w*0.62), int(h*0.04)
        zx2,zy2 = int(w*0.98), int(h*0.54)
        feat = state.settings["enabled_features"]
 
        # Zone intrusion
        if feat["zone_intrusion"]:
            for p in persons:
                cx,cy = p["center"]
                if zx1 < cx < zx2 and zy1 < cy < zy2:
                    p["in_zone"] = True
                    if self._cd(f"ZONE_{p['id']}", 5):
                        a = self._alert("ZONE_INTRUSION",
                            f"Person #{p['id']} entered Restricted Zone",
                            "critical", p["bbox"])
                        result["alerts"].append(a)
                        state.stats["zone_violations"] += 1
                        self._snap(frame, a)
 
        # Crowd density
        if feat["crowd_density"]:
            n = len(persons)
            if n >= state.settings["crowd_threshold"] and self._cd("CROWD", 8):
                a = self._alert("CROWD_DENSITY",
                    f"Crowd: {n} people detected (limit {state.settings['crowd_threshold']})",
                    "warning", None)
                result["alerts"].append(a)
                state.stats["anomalies"] += 1
 
        # Loitering — fires ONCE per person
        if feat["loitering"]:
            lim = state.settings["loiter_seconds"]
            for p in persons:
                pid   = p["id"]
                tdata = tracker.tracks.get(pid)
                if not tdata or tdata.get("loiter_alerted"):
                    continue
                dur = time.time() - tdata["first_seen"]
                if dur > lim:
                    pts = tdata["positions"][-20:]
                    if len(pts) > 5:
                        spread = np.std([pt[0] for pt in pts]) + np.std([pt[1] for pt in pts])
                        if spread < 50:
                            tdata["loiter_alerted"] = True
                            a = self._alert("LOITERING",
                                f"Person #{pid} loitering for {int(dur)}s",
                                "warning", p["bbox"])
                            result["alerts"].append(a)
                            state.stats["loitering_incidents"] += 1
 
        # Fight / anomaly
        if feat["anomaly"]:
            fa = self._fight(frame, persons)
            if fa:
                result["alerts"].append(fa)
                self._snap(frame, fa)
 
        # Weapon detection
        # Detects: phone(67), remote(65), knife(43), scissors(76)
        # These are real COCO objects visible in webcam scenes
        if feat["weapon"] and models.loaded and self.frame_count % 4 == 0:
            objs = models.detect_objects(frame, classes=[67,65,43,76], conf=0.28)
            LABELS = {"knife":"Knife","scissors":"Sharp Object",
                      "cell phone":"Suspicious Device","remote":"Device/Remote"}
            for obj in objs:
                label   = LABELS.get(obj["label"], obj["label"])
                result["weapons"].append({**obj,"label":label})
                if self._cd(f"WEP_{obj['label']}", 8):
                    a = self._alert("WEAPON",
                        f"⚠ {label} detected ({int(obj['conf']*100)}% confidence)",
                        "critical", obj["bbox"])
                    result["alerts"].append(a)
                    state.stats["weapon_detections"] += 1
                    self._snap(frame, a)
 
        return result, (zx1, zy1, zx2, zy2)
 
    def _fight(self, frame, persons):
        thresh = state.settings["fight_speed"]
        for p in persons:
            pid   = p["id"]
            tdata = tracker.tracks.get(pid)
            if not tdata:
                continue
            pts = tdata["positions"]
            if len(pts) >= 5:
                recent = pts[-5:]
                dists  = [np.sqrt((recent[i+1][0]-recent[i][0])**2 +
                                  (recent[i+1][1]-recent[i][1])**2)
                          for i in range(len(recent)-1)]
                if np.mean(dists) > thresh and self._cd("FIGHT", 6):
                    state.stats["fight_detections"] += 1
                    state.stats["anomalies"]        += 1
                    return self._alert("FIGHT",
                        f" Rapid movement near Person #{pid} — possible fight",
                        "critical", p["bbox"])
 
        # Pose check
        if models.loaded and self.frame_count % 6 == 0:
            try:
                for i,kps in enumerate(models.get_pose_keypoints(frame)):
                    if len(kps) < 13:
                        continue
                    lw,rw,lh,rh = kps[9],kps[10],kps[11],kps[12]
                    if all(v[0] > 0 for v in [lw,rw,lh,rh]):
                        ratio = abs(rw[0]-lw[0]) / (abs(rh[0]-lh[0])+1)
                        pid2  = 900+i
                        self.pose_history.setdefault(pid2,[]).append(ratio)
                        if len(self.pose_history[pid2]) > 8:
                            self.pose_history[pid2].pop(0)
                        if np.mean(self.pose_history[pid2]) > 3.2 and self._cd("POSE_FIGHT",8):
                            state.stats["fight_detections"] += 1
                            return self._alert("FIGHT",
                                f" Aggressive posture detected",
                                "critical", None)
            except Exception:
                pass
        return None
 
    def _cv_persons(self, frame):
        mask = self.bg_sub.apply(frame)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, k)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            if cv2.contourArea(c) < 1800:
                continue
            x,y,cw,ch = cv2.boundingRect(c)
            out.append({"bbox":[x,y,x+cw,y+ch],"center":(x+cw//2,y+ch//2),"conf":1.0})
        return out
 
    def _cd(self, key, secs):
        now = time.time()
        if now - self.cooldowns.get(key,0) > secs:
            self.cooldowns[key] = now
            return True
        return False
 
    def _alert(self, atype, msg, sev, bbox):
        return {"id":f"{atype}_{int(time.time()*1000)}","type":atype,
                "message":msg,"severity":sev,
                "timestamp":datetime.now().isoformat(),"bbox":bbox}
 
    def _snap(self, frame, alert):
        try:
            fname = f"{alert['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(SNAP_DIR/fname), frame)
            alert["snapshot"] = fname
        except Exception:
            pass
 
engine = DetectionEngine()
 
# ─── Frame Renderer ───────────────────────────────────────────────────────────
def draw_frame(frame, detections, zone_coords):
    h,w = frame.shape[:2]
    zx1,zy1,zx2,zy2 = zone_coords
    ov = frame.copy()
    cv2.rectangle(ov,(zx1,zy1),(zx2,zy2),(0,0,200),-1)
    cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
    cv2.rectangle(frame,(zx1,zy1),(zx2,zy2),(0,0,255),2)
    cv2.putText(frame,"RESTRICTED ZONE",(zx1+5,zy1+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,60,255),1)
 
    alert_types = {a["type"] for a in detections["alerts"]}
 
    for p in detections["persons"]:
        x1,y1,x2,y2 = p["bbox"]
        pid   = p.get("id",0)
        color = (0,0,255) if p.get("in_zone") else (0,255,100)
        tdata = tracker.tracks.get(pid)
        if tdata and time.time()-tdata["first_seen"] > state.settings["loiter_seconds"]:
            color = (0,165,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        lbl = f"P#{pid} {int(p.get('conf',1)*100)}%" if models.loaded else f"P#{pid}"
        cv2.putText(frame,lbl,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        if tdata:
            pts = tdata["positions"]
            for i in range(1,len(pts)):
                a = i/len(pts)
                cv2.line(frame,pts[i-1],pts[i],(int(80*a),int(180*a),int(255*a)),1)
 
    for wep in detections.get("weapons",[]):
        x1,y1,x2,y2 = wep["bbox"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(180,0,255),3)
        cv2.putText(frame,f"!{wep['label']}",(x1,y1-7),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,0,255),2)
 
    BADGE = {"ZONE_INTRUSION":(0,0,220),"CROWD_DENSITY":(0,100,255),
             "LOITERING":(0,140,255),"FIGHT":(0,0,180),"WEAPON":(140,0,255)}
    y = 28
    for atype in alert_types:
        col = BADGE.get(atype,(0,200,200))
        cv2.rectangle(frame,(8,y-15),(245,y+6),col,-1)
        cv2.putText(frame,f"! {atype}",(12,y),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
        y += 24
 
    cnt  = len(detections["persons"])
    mode = "YOLO" if models.loaded else "CV"
    cv2.rectangle(frame,(w-200,h-48),(w,h),(12,12,12),-1)
    cv2.putText(frame,f"People:{cnt} [{mode}]",(w-193,h-28),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
    cv2.putText(frame,datetime.now().strftime("%H:%M:%S"),
                (w-193,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)
    return frame, cnt
 
# ─── Broadcast ────────────────────────────────────────────────────────────────
async def broadcast(data):
    msg  = json.dumps(data)
    dead = []
    for ws in state.connected_clients:
        try:    await ws.send_text(msg)
        except: dead.append(ws)
    for ws in dead:
        if ws in state.connected_clients:
            state.connected_clients.remove(ws)
 
# ─── Video Loop ───────────────────────────────────────────────────────────────
async def process_stream(cap, name="Camera"):
    idx      = 0
    local_cd = {}
    while state.camera_active and cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret,frame = cap.read()
            if not ret: break
        idx += 1
        if idx % 2 != 0:
            await asyncio.sleep(0.01)
            continue
        frame = cv2.resize(frame,(854,480))
        dets,zone = engine.detect(frame)
        ann,count = draw_frame(frame,dets,zone)
        state.stats["people_count"] = count
        _,buf = cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,78])
        b64   = base64.b64encode(buf).decode()
        new_alerts = []
        for a in dets["alerts"]:
            now = time.time()
            if now - local_cd.get(a["type"],0) > 5:
                local_cd[a["type"]] = now
                new_alerts.append(a)
                state.active_alerts.insert(0,a)
                state.alert_history.insert(0,a)
                state.stats["total_alerts"] += 1
                if len(state.active_alerts)  > 30:  state.active_alerts.pop()
                if len(state.alert_history)  > 500: state.alert_history.pop()
        await broadcast({"type":"frame","frame":b64,"stats":state.stats.copy(),
                         "alerts":new_alerts,"active_alerts":state.active_alerts[:15],
                         "people_count":count,"source":name,"yolo":models.loaded})
        await asyncio.sleep(0.033)
    cap.release()
    state.camera_active = False
    await broadcast({"type":"stream_ended"})
 
# ─── Export ───────────────────────────────────────────────────────────────────
def do_export_csv():
    p = EXPORT_DIR/f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(p,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["id","type","severity","message","timestamp","snapshot"])
        w.writeheader()
        for a in state.alert_history:
            w.writerow({k:a.get(k,"") for k in ["id","type","severity","message","timestamp","snapshot"]})
    return p
 
def do_export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as C
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        p      = EXPORT_DIR/f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc    = SimpleDocTemplate(str(p),pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []
        story.append(Paragraph("SENTINEL — Surveillance Report",
            ParagraphStyle("t",fontSize=18,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=4)))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ParagraphStyle("s",fontSize=10,fontName="Helvetica",alignment=TA_CENTER,textColor=C.grey,spaceAfter=18)))
        uptime = int((time.time()-state.start_time)/60)
        summary = [["Metric","Value"],
            ["Total Alerts",str(state.stats["total_alerts"])],
            ["Zone Violations",str(state.stats["zone_violations"])],
            ["Loitering Incidents",str(state.stats["loitering_incidents"])],
            ["Fight Detections",str(state.stats["fight_detections"])],
            ["Weapon Detections",str(state.stats["weapon_detections"])],
            ["Uptime (min)",str(uptime)],
            ["Engine","YOLOv8" if models.loaded else "OpenCV"]]
        st = Table(summary,colWidths=[8*cm,8*cm])
        st.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C.HexColor("#0d1426")),("TEXTCOLOR",(0,0),(-1,0),C.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#f0f4ff"),C.white]),
            ("GRID",(0,0),(-1,-1),0.5,C.grey),("PADDING",(0,0),(-1,-1),7)]))
        story += [st,Spacer(1,.5*cm),Paragraph("Alert Log",styles["Heading2"]),Spacer(1,.2*cm)]
        SEV = {"critical":"#ff2d55","warning":"#ffaa00","info":"#00aaff"}
        if state.alert_history:
            rows = [["#","Type","Severity","Message","Timestamp"]]
            for i,a in enumerate(state.alert_history[:100],1):
                rows.append([str(i),a.get("type",""),a.get("severity","").upper(),
                              a.get("message","")[:55],a.get("timestamp","")[:19].replace("T"," ")])
            at = Table(rows,colWidths=[.8*cm,3*cm,2.5*cm,7.5*cm,3.5*cm])
            astyle = [("BACKGROUND",(0,0),(-1,0),C.HexColor("#0d1426")),("TEXTCOLOR",(0,0),(-1,0),C.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#fff5f5"),C.white]),
                ("GRID",(0,0),(-1,-1),0.3,C.lightgrey),("PADDING",(0,0),(-1,-1),5)]
            for i,a in enumerate(state.alert_history[:100],1):
                col = C.HexColor(SEV.get(a.get("severity","info"),"#aaaaaa"))
                astyle += [("TEXTCOLOR",(2,i),(2,i),col),("FONTNAME",(2,i),(2,i),"Helvetica-Bold")]
            at.setStyle(TableStyle(astyle))
            story.append(at)
        else:
            story.append(Paragraph("No alerts recorded.",styles["Normal"]))
        doc.build(story)
        return p
    except ImportError:
        return None
 
# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    if YOLO_AVAILABLE:
        state.yolo_status = "loading"
        ok = await asyncio.get_event_loop().run_in_executor(None, models.load)
        state.yolo_status = "ready" if ok else "error"
    else:
        state.yolo_status = "unavailable"
 
# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)),"index.html"))
 
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.append(websocket)
    await websocket.send_text(json.dumps({"type":"yolo_status","status":state.yolo_status}))
    try:
        while True:
            data   = json.loads(await websocket.receive_text())
            action = data.get("action")
 
            if action == "start_webcam":
                state.camera_active = False
                tracker.reset()
                engine.cooldowns = {}
                await asyncio.sleep(0.5)
                state.camera_active = True
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    state.camera_active = False
                    await websocket.send_text(json.dumps({"type":"error","message":"Camera not found"}))
                else:
                    asyncio.create_task(process_stream(cap,"Webcam"))
 
            elif action == "stop_stream":
                state.camera_active = False
 
            elif action == "update_settings":
                if "crowd_threshold" in data: state.settings["crowd_threshold"] = int(data["crowd_threshold"])
                if "loiter_seconds"  in data: state.settings["loiter_seconds"]  = int(data["loiter_seconds"])
                if "fight_speed"     in data: state.settings["fight_speed"]     = int(data["fight_speed"])
                if "features"        in data: state.settings["enabled_features"].update(data["features"])
 
            # TEST MODE — fire any alert manually from dashboard
            elif action == "test_alert":
                atype = data.get("alert_type","ZONE_INTRUSION")
                TESTS = {
                    "ZONE_INTRUSION": ("Person #1 entered Restricted Zone [TEST]",  "critical"),
                    "WEAPON":         ("⚠ Weapon detected near entry [TEST]",       "critical"),
                    "FIGHT":          (" Fight/disturbance detected [TEST]",       "critical"),
                    "CROWD_DENSITY":  ("High crowd density: 8 people [TEST]",       "warning"),
                    "LOITERING":      ("Person #2 loitering for 20s [TEST]",        "warning"),
                }
                msg,sev = TESTS.get(atype,("Test alert","warning"))
                a = {"id":f"{atype}_TEST_{int(time.time()*1000)}","type":atype,
                     "message":msg,"severity":sev,"timestamp":datetime.now().isoformat(),"bbox":None}
                state.active_alerts.insert(0,a)
                state.alert_history.insert(0,a)
                state.stats["total_alerts"] += 1
                await broadcast({"type":"frame_alert","alerts":[a],
                                 "stats":state.stats.copy(),"active_alerts":state.active_alerts[:15]})
 
    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
        if not state.connected_clients:
            state.camera_active = False
 
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4",".avi",".mov",".mkv",".webm")):
        raise HTTPException(400,"Unsupported format")
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
    tmp.write(await file.read()); tmp.close()
    state.camera_active = False; tracker.reset()
    await asyncio.sleep(0.5)
    state.camera_active = True
    asyncio.create_task(process_stream(cv2.VideoCapture(tmp.name),f"File:{file.filename}"))
    return {"status":"processing","filename":file.filename}
 
@app.get("/export/csv")
async def export_csv_route():
    p = do_export_csv()
    return FileResponse(str(p),media_type="text/csv",filename=p.name)
 
@app.get("/export/pdf")
async def export_pdf_route():
    p = do_export_pdf()
    if not p: raise HTTPException(500,"reportlab not installed. Run: pip install reportlab")
    return FileResponse(str(p),media_type="application/pdf",filename=p.name)
 
@app.get("/stats")
async def get_stats():
    return {"stats":state.stats,"settings":state.settings,
            "yolo":models.loaded,"uptime":int(time.time()-state.start_time)}
 
@app.get("/alerts")
async def get_alerts():
    return {"alerts":state.alert_history[:100]}
 
@app.post("/clear-alerts")
async def clear_alerts():
    state.active_alerts.clear(); return {"status":"cleared"}
 
@app.post("/reset-stats")
async def reset_stats():
    for k in state.stats: state.stats[k] = 0
    return {"status":"reset"}
 
@app.get("/snapshots/{filename}")
async def get_snapshot(filename: str):
    p = SNAP_DIR/filename
    if not p.exists(): raise HTTPException(404,"Not found")
    return FileResponse(str(p))
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")