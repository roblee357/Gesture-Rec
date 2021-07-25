from flask import Flask, flash, redirect, request, send_file, abort, render_template, Response, stream_with_context
from flask_restful import Resource, Api, url_for
from werkzeug.utils import secure_filename
import json, os
import threading

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','mp4'}

from get_video import *

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

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
upload_form =     b'''
    <!doctype html>
    <title>Gesture recognition</title>
    <h1>Upload YouTube URL for Fortnite dance gesture recognition</h1>
    <form method="POST">
        <input name="YouTube URL">
        <input type="submit">
    </form>
    '''
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>

jumping_jack = cv2.imread('jumping_jack.jpg')
img_str_bytes = cv2.imencode('.jpg', jumping_jack)[1].tobytes()
# EB looks for an 'application' callable by default.
application = app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
detector = Detect()
# # add a rule for the index page.
# application.add_url_rule('/', 'index', (lambda: header_text +
#     say_hello() + instructions + footer_text))

@app.route('/newclip', methods=['GET'])
def newclip():
   cID  = request.args.get('cID', None)
   title  = request.args.get('title', None)
   cstart  = request.args.get('cstart', None)
   cend  = request.args.get('cend', None)
   print('cID',cID,'title',title,'cstart',cstart,'cend',cend)
   mstring =  {'cID':  cID , 'title' : title , 'cstart' : cstart, 'cend': cend}
   with open('output.txt','a+') as fout:
       fout.write(json.dumps(mstring))
       fout.write('\n')
#    video = get_video.vid(cID,title,cstart,cend)
#    video.detect()
   return json.dumps({'success':True, 'input':mstring}), 200, {'ContentType':'application/json'}

def gen():
    while True:
        image_path = os.path.join(os.getcwd(),'live.jpeg')
        # image = cv2.imread(image_path)
        image = detector.image
        try:
            try:
                img_str = cv2.imencode('.jpg', image)[1].tobytes()
                yield (  upload_form + b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img_str + b'\r\n')
                last_img_str = img_str
            except:
                yield (  upload_form + b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + last_img_str + b'\r\n')            
        except:
            yield (  upload_form + b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_str_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(gen()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['GET', 'POST'])
def stop_detector():
    if request.method == 'POST':
        # detector.stop_threads = True
        detector.newStartupThread.raise_exception()
        print('raising exception')
    return redirect(url_for('upload_file'))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if  len(request.form['YouTube URL']) > 10 :
            YT_URL =request.form['YouTube URL']
            YT_watchID = YT_URL.split('v=')[1]
            detector.start(YT_watchID)

            # Detect('Fortnite_Emotes',source = YT_watchID)
            return redirect(url_for('video_feed'))

    return  upload_form


@app.route('/files')
def dir_listing():
    files = os.listdir(os.getcwd())
    return render_template('files.html', files=files)

@app.route('/download')
def download_file():
    files = os.listdir(os.path.join(os.getcwd(),'static/uploads'))
    return render_template('files.html', files=files)    


@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
def dir_listing2(req_path):
    BASE_DIR = os.getcwd()
    abs_path = os.path.join(BASE_DIR, req_path)
    if os.path.isfile(abs_path):
        return send_file(abs_path)
    else:
        return abs_path


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    
    application.run(host='0.0.0.0')