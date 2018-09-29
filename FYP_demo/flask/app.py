#!/usr/bin/env python3
import os
import sys
# sys.path.append('./end2end')
# print(sys.path)

from flask import Flask, request, jsonify, send_from_directory
from end2end import final_query_cpu
from PIL import Image
import io
import time

user_upload_dest = '/users/sunjingxuan/desktop/FYP_demo/flask/end2end/apptest_demo'
UPLOAD_FOLDER = '/users/sunjingxuan/desktop/flask'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/api/test0121', methods=['POST'])
def test0121():
    if request.method == 'POST':
        # return jsonify({'upload':True, 'name' : 'hardcoded'})
        # print(request.files)
        # print(request.data)
        
        if 'imagefile' not in request.files:
            return jsonify({'upload':True, 'name' : 'No file part'})
        # print('imagefile in request.files')
        file = request.files['imagefile']
        if file:
            filename = str(time.time()).replace(".", "") + ".jpg"
            # filename = os.path.join(user_upload_dest, file.filename)
            filename = os.path.join(user_upload_dest, filename)
            print(filename)
            file.save(filename)

        # data = request.data
        # if data:
        #     filename = str(time.time()).replace(".", "") + ".jpg"
        #     filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     print(filename)
        #     image = Image.open(io.BytesIO(data))
        #     image.save(filename)

            jsonfolder, jsonpath = final_query_cpu.final_query_cpu(filename)
            # jsonfolder = '/users/sunjingxuan/py/flask/end2end/jsons0318'
            # jsonpath = 'jsondump_1521685180880786.jpg.json'
            # return jsonify({'upload':True, 'jsonpathlist' : jsonpathlist})    app.config['UPLOAD_FOLDER']
            return send_from_directory(jsonfolder, jsonpath, mimetype='application/json')
    return 'whatever'



@app.route('/api/get_messages', methods = ['POST'])
def get_messages():
    json = request.get_json()
    if json['user'] == "larry":
        return jsonify({'messages':['test1', 'test2']})
    return jsonify({'error':'no user found'})

if __name__ == '__main__':
    # home
    # app.run(host='192.168.0.104', debug=True)
    # FSC8
    # app.run(host='158.182.188.216', debug=True)
    app.run(host='0.0.0.0', debug=True)



