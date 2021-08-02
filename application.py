from flask import Flask, flash, redirect, request, send_file, abort, render_template, Response, stream_with_context
from flask_restful import Resource, Api, url_for
from werkzeug.utils import secure_filename
import json, os
import threading

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4'}

from get_video import *

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>Gesture Recognition</title> </head>\n<body>'''
instructions = '''
    <p><em>Gesture Recognition</em>:</p>'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'


jumping_jack = cv2.imread('jumping_jack.jpg')
img_str_bytes = cv2.imencode('.jpg', jumping_jack)[1].tobytes()
# EB looks for an 'application' callable by default.
application = app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
detector = Detect()


def gen_frames():
    while True:
        image_path = os.path.join(os.getcwd(),'live.jpeg')
        # image = cv2.imread(image_path)
        image = detector.image
        try:
            try:
                img_str = cv2.imencode('.jpg', image)[1].tobytes()
                yield (  b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img_str + b'\r\n')
                last_img_str = img_str
            except:
                yield (  b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + last_img_str + b'\r\n')            
        except:
            yield (  b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_str_bytes + b'\r\n')

### https://stackoverflow.com/questions/55736527/how-can-i-yield-a-template-over-another-in-flask/55755716
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print('request data',request.data)
        if  len(request.form['YouTube URL']) > 10 :
            
            YT_URL =request.form['YouTube URL']
            YT_watchID = YT_URL.split('v=')[1]
            print('starting detection on: ' + YT_watchID) 
            detector.start(YT_watchID)
        # if  len(request.form['Stop Video']) > 10 :
        #     YT_URL =request.form['YouTube URL']
        #     YT_watchID = YT_URL.split('v=')[1]
        #     detector.start(YT_watchID)
    return render_template('controls_and_stream.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    
    application.run(host='0.0.0.0')