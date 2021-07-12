# !pip install mediapipe opencv-python pandas scikit-learn
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle, traceback
import threading
import ctypes
# Ref: https://google.github.io/mediapipe/solutions/pose.html
# Ref: https://codepen.io/mediapipe/pen/jOMbvxw web solution

vids = [{"cID": "qZEElv92rLM", "title": "Onda Onda", "cstart": "35137", "cend": "42571"},
{"cID": "qZEElv92rLM", "title": "Controller Crew", "cstart": "110635", "cend": "112780"},
{"cID": "qZEElv92rLM", "title": "Hit It", "cstart": "120177", "cend": "132160"},
{"cID": "qZEElv92rLM", "title": "Billy Bounce", "cstart": "154580", "cend": "160941"},
{"cID": "qZEElv92rLM", "title": "Dont' Start Now", "cstart": "175213", "cend": "188833"},
{"cID": "qZEElv92rLM", "title": "Savage", "cstart": "197961", "cend": "211121"},
{"cID": "qZEElv92rLM", "title": "The Flow", "cstart": "221555", "cend": "231141"},
{"cID": "qZEElv92rLM", "title": "The Flow", "cstart": "233278", "cend": "238461"}]

import pafy, cv2, json, math
# url = 'https://yself.outu.be/DK797d9ozN0?t=440'

class Vid_Stream():
    def __init__(self,vids, model_name, options = None, trail_frames = [5,15]):
        self.yt_url = 'https://www.yself.outube.com/watch?v='
        self.trail_frames = trail_frames
        self.vids = vids
        self.model_name = model_name
        self.out = cv2.VideoWriter(model_name + '.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,480))

        self.mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        self.mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        jumping_jack = cv2.imread('jumping_jack.jpg')
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            self.results = holistic.process(jumping_jack)
        num_coords = len(self.results.pose_landmarks.landmark)

        landmarks = ['class']
        
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        for hist_frame in self.trail_frames:
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val) + '_tf{}'.format(hist_frame), 'y{}'.format(val) + '_tf{}'.format(hist_frame), 'z{}'.format(val) + '_tf{}'.format(hist_frame), 'v{}'.format(val) + '_tf{}'.format(hist_frame)]        

        with open(self.model_name + '_coords.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks_header = landmarks
            del landmarks_header[5:42]
            csv_writer.writerow(landmarks_header)

        if not options is None:
            self.options = options
        else:
            self.options = {'side':'both', 'save_vid':True}         

    def extract(self):
        for vid in self.vids:
            vPafy = pafy.new(self.yt_url + vid['cID'])
            play = vPafy.getbest(preftype="mp4")
            cap = cv2.VideoCapture(play.url)
            cap.set(cv2.CAP_PROP_POS_MSEC, int(vid['cstart']))
            # Initiate holistic model
            frame_num = 0
            frame_list = []
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

                while cap.isOpened():
                    frame_num = frame_num + 1
                    ret, frame = cap.read()
                    

                    # Recolor Feed
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                           
                    height, width, channels = image.shape
                    if self.options['side'] == 'right':
                        image = image[:, math.floor(width/2):width,:]
                        cv2.imwrite('image.jpg',image)
#                     image.flags.writeable = False 
                    # Make Detections
                    results = holistic.process(image)
                    # print(results.face_landmarks)

                    # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True   
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #         # 1. Draw face landmarks
            #         self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS, 
            #                                  self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            #                                  self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            #                                  )

                    # 2. Right hand
                    self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                             self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                             self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                             )

                    # 3. Left Hand
                    self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                             self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                             self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                             )

                    # 4. Pose Detections
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, 
                                             self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                             self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                             )
                    # Export coordinates
                    try:
                            # Extract Pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        # remove face landmarks
                        del pose_row[3:43]

            #                 # Extract Face landmarks
            #                 face = results.face_landmarks.landmark
            #                 face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                        # Concate rows
                        row = pose_row   #+face_row
                        frame_list.insert(0,row)
                        print('len(frame_list)',len(frame_list))

                        if frame_num >= max(self.trail_frames):
                            for frame_no in self.trail_frames:
                                try:
                                    row = row + frame_list[frame_no]
                                except Exception as e:
                                    traceback.print_tb(e.__traceback__)
                                    print('frame_no',frame_no,e)

                            # Append class name 
                            row.insert(0, vid['title'])
                            print('len(row)',len(row))

                            # Export to CSV
                            if len(row)>200:
                                with open(self.model_name + '_coords.csv', mode='a', newline='') as f:
                                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    csv_writer.writerow(row)
                        try:
                            frame_list.pop(max(self.trail_frames))
                        except:
                            print ('frame_num',frame_num)

                    except Exception as e:
                        traceback.print_tb(e.__traceback__)
                        print(e)
                    cv2.putText(image, vid['title'], (5,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow( vid['title'] + ' Trick Maneuver Training video' , image)
                    # resize image
                    dim = (640, 480)
                    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                    self.out.write(image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                    cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if cur_time > int(vid['cend']):
                        break

            cap.release()
            cv2.destroyAllWindows()
        self.out.release()

    def train_model(self):
        df = pd.read_csv(self.model_name + '_coords.csv')
        X = df.drop('class', axis=1) # features
        X.dropna(axis=1, how='any', inplace=True)
        # Remove three columns as index base
        # X.drop(X.columns[4:44], axis = 1, inplace = True)
        y = df['class'] # target value
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
        }
        fit_models = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algo] = model

        fit_models['rc'].predict(X_test)

        for algo, model in fit_models.items():
            yhat = model.predict(X_test)
            # print(algo, accuracy_score(y_test, yhat))
        fit_models['rf'].predict(X_test)
        with open(self.model_name + '_body_language.pkl', 'wb') as f:
            pickle.dump(fit_models['rf'], f)

class Detect():
    def __init__(self):
        self.stop_threads = False

    def detect(self,model_name,source = 0,trail_frames=[5,15]):
        if source != 0:
            self.yt_url = 'https://www.youtube.com/watch?v='
            vPafy = pafy.new(self.yt_url + source)
            play = vPafy.getbest(preftype="mp4")
            source = play.url
        cap = cv2.VideoCapture(source)
        # Initiate holistic model
        with open(model_name + '_body_language.pkl', 'rb') as f:
            model = pickle.load(f)
        self.mp_drawing = mp.solutions.drawing_utils # Drawing helpers
        self.mp_holistic = mp.solutions.holistic # Mediapipe Solutions
        frame_num = 0
        frame_list = []
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                if self.stop_threads:
                    break
                # print(self.stop_threads)
                frame_num = frame_num + 1
                ret, frame = cap.read()
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                

                # 4. Pose Detections
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, 
                                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    # remove face landmarks
                    del pose_row[3:43]    
                    
                    # Concate rows
                    row = pose_row  #+face_row
                    frame_list.insert(0,row)
                    # print('len(frame_list)',len(frame_list))      

                    if frame_num >= max(trail_frames):
                        for frame_no in trail_frames:
                            try:
                                row = row + frame_list[frame_no]
                            except Exception as e:
                                traceback.print_tb(e.__traceback__)
                                # print('frame_no',frame_no,e)                               

                        # Make Detections
                        X = pd.DataFrame([row])
                        # print('model.n_features_',model.n_features_,'len(row)',len(row))
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        # print('model.predict_proba(X)', model.predict_proba(X))
                        # print('model.predict(X)',model.predict(X))
                        
                        # Grab ear coords
                        coords = tuple(np.multiply(
                                        np.array(
                                            (results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                            results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].y))
                                    , [640,480]).astype(int))
                        
                        cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (245, 117, 16), -1)
                        cv2.putText(image, body_language_class, coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Get status box
                        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                        
                        # Display Class
                        cv2.putText(image, 'CLASS'
                                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[0]
                                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        # print('body_language_class.split(' ')[0]',body_language_class.split(' ')[0])
                        # Display Probability
                        cv2.putText(image, 'PROB'
                                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    print('exception and ',e)
                    # pass
                self.image = image                
                # cv2.imshow(self.model_name + ' Raw Webcam Feed', image)
                # cv2.imwrite('live.jpeg',image)

                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break

        cap.release()
        cv2.destroyAllWindows()

    def start(self,YT_watchID):
        self.newStartupThread = threading.Thread(target = self.detect, args = ['Fortnite_Emotes',YT_watchID])
        self.newStartupThread.start()

    def raise_exception(self):
        thread_id = self.get_id()
        print('exception raising ')
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

# options = {'side':'right', 'save_vid':True}
# vid_processor = Vid_Stream(vids,'Fortnite_Emotes', options=options)
# vid_processor.train_model()

# Detect('Fortnite_Emotes',source = 'qZEElv92rLM')

# with open('Fortnite_Emotes' + '_body_language.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('model',model.classes_)
# print(model.get_support(indices=True))
