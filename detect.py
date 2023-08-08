# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

'''
------------------- GPIO ------------------------------
'''
import RPi.GPIO as GPIO
import time

from pynput import keyboard
GPIO.setmode(GPIO.BCM)


pin_a=22 #A-1A :15
pin_b=23 #A-1B :16
pin_c=24 #B-1A :18
pin_d=25 #B-2A :22

'''
------------------------------------------------------
'''


'''
------------------- GUI ------------------------------
'''
import tkinter as tk
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import threading
import datetime

'''
----------------------setting------------------------------
'''
data_file=[]#0:path,1:time long,2:line token
with open('setting_file.txt', 'r') as f:
    for line in f:
        line=line.strip('\n')
        data_file.append(line)
'''
------------------------------------------------------
'''

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#---------------------- setup wheel -------------------------------
GPIO.setup(pin_a, GPIO.OUT)
GPIO.setup(pin_b, GPIO.OUT)
GPIO.setup(pin_c, GPIO.OUT)
GPIO.setup(pin_d, GPIO.OUT)

GPIO.output(pin_a, False)
GPIO.output(pin_b, False)
GPIO.output(pin_c, False)
GPIO.output(pin_d, False)
#-------------------------------------------------------------------
@smart_inference_mode()

#------------------------- car move --------------------------------
def check_location(x1,y1,x2,y2):
    width=x2-x1
    height=y2-y1
    
    #front
    if width!=0 and width < height:
        #backward
        if x1<213 and x2>426:
            tmp = width/213
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        elif (x1 + x2)/2 > 335 :#mouse's location is too right
            print("mouse is too right")
            mid = (x1 + x2)/2
            tmp=(mid-335)/106
            if tmp<1:
                tmp=1
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, True)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.05*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to right
            '''
        elif (x1 + x2)/2 < 305 :#mouse's location is too left
            mid = (x1 + x2)/2
            tmp=(305 - mid)/106
            if tmp<1:
                tmp=1
            print("mouse is too left")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, True)
            time.sleep(0.05*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to left
            '''
        elif x1<10 and (x1 + x2)/2<320:
            tmp = width/213
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        elif x2>630 and (x1 + x2)/2>320:
            tmp = width/213
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        if x1>213 and x2<426:
            tmp = 213/width
            if tmp<0.5:
                tmp=0.5
            print("mouse is too small")
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, True)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, True)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to forward
            '''
        else :
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
    
    #side
    elif width!=0 and width > height:
        #backward
        if x1<160 and x2>480:
            tmp = width/320
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        elif (x1 + x2)/2 > 340 :#mouse's location is too right
            print("mouse is too right")
            mid = (x1 + x2)/2
            tmp=(mid - 340)/160
            if tmp<1:
                tmp=1
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, True)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.05*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to right
            '''
        elif (x1 + x2)/2 < 300 :#mouse's location is too left
            mid = (x1 + x2)/2
            tmp=(300 - mid)/160
            if tmp<1:
                tmp=1
            print("mouse is too left")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, True)
            time.sleep(0.05*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to left
            '''
        elif x1<10 and (x1 + x2)/2<320:
            tmp = width/320
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        elif x2>630 and (x1 + x2)/2>320:
            tmp = width/320
            if tmp<0.5:
                tmp=0.5
            print("mouse is too big")
            GPIO.output(pin_a, True)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, True)
            GPIO.output(pin_d, False)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to backward
            '''
        if x1>160 and x2<480:
            tmp = 320/width
            if tmp<0.5:
                tmp=0.5
            print("mouse is too small")
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, True)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, True)
            time.sleep(0.1*tmp)
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
            '''
                car to forward
            '''
        else :
            GPIO.output(pin_a, False)
            GPIO.output(pin_b, False)
            GPIO.output(pin_c, False)
            GPIO.output(pin_d, False)
    

'''
-----------------global variable----------------------
'''
global image
image=cv2.imread('start.png')
image=cv2.resize(image, (640, 450))

global automatic_move
automatic_move=True

global notice_message
notice_message=False

global save_path
save_path=data_file[0]

global timer_st,timer_ed
timer_st=time.time()
timer_ed=time.time()

'''
------------------------------------------------------
'''
'''
==================== line ===========================
'''
import requests

def line_message():
    # LINE Notify 權杖
    token = str(data_file[2])

    # 要發送的訊息
    message = '你家老鼠很久沒出現囉!'

    # HTTP 標頭參數與資料
    headers = { "Authorization": "Bearer " + token }
    mdata = { 'message': message }

    # 以 requests 發送 POST 請求
    requests.post("https://notify-api.line.me/api/notify",headers = headers, data = mdata)



'''
=====================================================
'''




def run(
        weights='hamster-int8_edgetpu.tflite',  # model path or triton URL
        source='0',  # file/dir/URL/glob/screen/0(webcam)
        data='./data.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            global image
            image=im0
            if view_img:
                #if platform.system() == 'Linux' and p not in windows:
                #windows.append(p)
                '''
                p="monitor display"
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                '''
                #show_location
                x1,y1,x2,y2=annotator.get_location()
                #check_location
                if automatic_move:
                    check_location(x1,y1,x2,y2)
                    print("左上點的座標為：(" + str(x1) + "," + str(y1) + ")，右下點的座標為(" + str(x2) + "," + str(y2) + ")")
                if notice_message:
                    global timer_ed,timer_st
                    if (x1==0)and(x2==0)and(y1==0)and(y2==0):
                        timer_ed=time.time()

                        if (timer_ed-timer_st)>float(data_file[1])*60*60:
                            line_message()
                            timer_st=time.time()
                            #print(format(timer_ed-timer_st))
                    else:
                        timer_st=time.time()
                        
                    
                cv2.waitKey(1)  # 1 millisecond
                
                #============================= GUI ================================

                     

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def UI_window():
    window = tk.Tk()
    window.title('Pet Monitor')
    window.geometry('1280x900')
    window.resizable(True, True)
    window.configure(bg='#66676C')
    #window.iconbitmap("./hamster.ico")
    
    global image
    def video_stream():
        cv2image=cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
        img=Image.fromarray(cv2image)
        imgtk=ImageTk.PhotoImage(image=img)
        video.imgtk=imgtk
        video.configure(image=imgtk)
        video.after(1,video_stream)
    videoFrame=tk.Frame(window,bg="#66676C").pack()
    video=tk.Label(videoFrame)
    video.pack()
    video_stream()
    #============================= button ====================================
    
    #------------------------------------------------------------------
    # car moves automatically.(car will detect hamster ,and then moving)
    
    imgBtn0 = tk.PhotoImage(file="./move_button_off.png")
    imgBtn1 = tk.PhotoImage(file="./move_button_on.png")
    
    global move_button
    
    def move_button_event():
        global automatic_move
        
        if automatic_move:
            automatic_move=False
            move_button.configure(image=imgBtn0)
        else:
            automatic_move=True
            move_button.configure(image=imgBtn1)
        window.update()
    #button image depend on automatic_move=true or false
    move_button = tk.Button(window,image=imgBtn1,fg='#66676C',bg='#66676C',activebackground='#66676C',command=move_button_event)
    if automatic_move:
        move_button = tk.Button(window,image=imgBtn1,fg='#66676C',bg='#66676C',activebackground='#66676C',command=move_button_event)  
    else:
        move_button = tk.Button(window,image=imgBtn0,fg='#66676C',bg='#66676C',activebackground='#66676C',command=move_button_event)

    
    move_button.pack(side='right')
    #------------------------------------------------------------------
    # notice
    
    notiBtn0 = tk.PhotoImage(file="./notice_button_off.png")
    notiBtn1 = tk.PhotoImage(file="./notice_button_on.png")
    
    global notice_button
    
    def notice_button_event():
        global notice_message
        
        if notice_message:
            notice_message=False
            notice_button.configure(image=notiBtn0)
        else:
            notice_message=True
            notice_button.configure(image=notiBtn1)
            global timer_st
            timer_st=time.time()
        window.update()
    #button image depend on automatic_move=true or false
    notice_button = tk.Button(window,image=notiBtn1,fg='#66676C',bg='#66676C',activebackground='#66676C',command=notice_button_event)
    if notice_message:
        notice_button = tk.Button(window,image=notiBtn1,fg='#66676C',bg='#66676C',activebackground='#66676C',command=notice_button_event)  
    else:
        notice_button = tk.Button(window,image=notiBtn0,fg='#66676C',bg='#66676C',activebackground='#66676C',command=notice_button_event)

    
    notice_button.pack(side='right')
    #------------------------------------------------------------------
    #camera
    
    camBtn = tk.PhotoImage(file="./camera_button.png")
     
    def camera_button_event():
        #take picture
        picture_name=str(save_path)+"/"+str(datetime.datetime.now()).strip()+".png"
        cv2.imwrite(picture_name,image)
        ltoken = data_file[2]
        #----------------------------------line--------------------------
        # 要發送的訊息
        message = '圖片'
        # HTTP 標頭參數與資料
        headers = { "Authorization": "Bearer " + ltoken }
        pdata = { 'message': message }

        # 要傳送的圖片檔案
        send_image = open(picture_name, 'rb')
        files = { 'imageFile': send_image }

        # 以 requests 發送 POST 請求
        requests.post("https://notify-api.line.me/api/notify",headers = headers, data = pdata, files = files)
    #button image depend on automatic_move=true or false
    camera_button = tk.Button(window,image=camBtn,fg='#66676C',bg='#66676C',activebackground='#66676C',command=camera_button_event)

    camera_button.pack(side='right')
    #------------------------------------------------------------------
    #setting
    setBtn = tk.PhotoImage(file="./setting_button.png")
     
    def setting_button_event():
        settingWindow = tk.Toplevel(window)
        settingWindow.geometry('450x125')    
        #------------------------------save picture----------------------------------------
        def selectPath():
            path_=askdirectory()
            path.set(path_)
        
        path=tk.StringVar()
        path.set(data_file[0])
        tk.Label(settingWindow,text="圖片儲存位置:").grid(row=2,column=0)
        entry_store=tk.Entry(settingWindow,textvariable=path)
        entry_store.grid(row=2,column=1)
        tk.Button(settingWindow,text="選擇路徑",command=selectPath).grid(row=2,column=2)
        #------------------------------- set time -----------------------------------
        time_str=tk.StringVar()
        time_str.set(data_file[1])
        tk.Label(settingWindow,text="通知時間間隔:").grid(row=4,column=0)
        noti_time=tk.Entry(settingWindow,textvariable=time_str)
        noti_time.grid(row=4,column=1)
        tk.Label(settingWindow,text="hr").grid(row=4,column=2)
        #-------------------------------------------------------------------------
        line_token=tk.StringVar()
        line_token.set(data_file[2])
        tk.Label(settingWindow,text="Line notify token:").grid(row=5,column=0)
        line_notice=tk.Entry(settingWindow,textvariable=line_token)
        line_notice.grid(row=5,column=1)
        #-------------------------------------------------------------------------
        def save_setting():
            global save_path
            save_path=entry_store.get()
            data_file[1]=noti_time.get()
            data_file[2]=line_notice.get()
            with open('setting_file.txt', 'w') as f:
                f.write(str(entry_store.get())+'\n')
                f.write(str(noti_time.get())+'\n')
                f.write(str(line_notice.get())+'\n')
                
        tk.Button(settingWindow,text="儲存設定",command=save_setting).grid(row=6,column=5)
        

    #button image depend on automatic_move=true or false
    setting_button = tk.Button(window,image=setBtn,fg='#66676C',bg='#66676C',activebackground='#66676C',command=setting_button_event)

    setting_button.pack(side='left')
    
    
    #===============================================================================
    window.mainloop()
    os._exit(0)
    
    
    
    

def keyboard_control():
    while True:
        global automatic_move
        
        if automatic_move==False:
            def on_press(key):
                try:
                    #print('Alphanumeric key pressed: {0} '.format(key.char))
                    if key==keyboard.Key.up and automatic_move==False:
                        GPIO.output(pin_a, False)
                        GPIO.output(pin_b, True)
                        GPIO.output(pin_c, False)
                        GPIO.output(pin_d, True)
                    elif key==keyboard.Key.down and automatic_move==False:
                        GPIO.output(pin_a, True)
                        GPIO.output(pin_b, False)
                        GPIO.output(pin_c, True)
                        GPIO.output(pin_d, False)
                    elif key==keyboard.Key.left and automatic_move==False:
                        GPIO.output(pin_a, True)
                        GPIO.output(pin_b, False)
                        GPIO.output(pin_c, False)
                        GPIO.output(pin_d, True)
                    elif key==keyboard.Key.right and automatic_move==False:
                        GPIO.output(pin_a, False)
                        GPIO.output(pin_b, True)
                        GPIO.output(pin_c, True)
                        GPIO.output(pin_d, False)                  
                except AttributeError:
                    print('special key pressed: {0}'.format(key))

            def on_release(key):
                if (key==keyboard.Key.up or key==keyboard.Key.down or key==keyboard.Key.left or key==keyboard.Key.right)and automatic_move==False:
                    GPIO.output(pin_a, False)
                    GPIO.output(pin_b, False)
                    GPIO.output(pin_c, False)
                    GPIO.output(pin_d, False)
                #print('Key released: {0}'.format(key))
                if key == keyboard.Key.esc:
                    # Stop listener
                    return False

                # Collect events until released
            with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
                listener.join()
        else:
            
            time.sleep(5)
 
            
    
    
    
    
a = threading.Thread(target=run)  # 建立新的執行緒
b = threading.Thread(target=UI_window)  # 建立新的執行緒
c = threading.Thread(target=keyboard_control) # 建立新的執行緒

a.start()  # 啟用執行緒
b.start()  # 啟用執行緒
c.start()


