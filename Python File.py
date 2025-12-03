from io import BytesIO
import cv2, numpy
from flask import Flask, render_template, request, send_file, redirect
import DetectorBox, DetectprDepthmap

#
image_site = None
detectors = []
properties = {"Box correct coefficient" : 1, "Depthmap correct coefficient": 1}
#

FLASK_APP = Flask(__name__)
@FLASK_APP.route('/')
def index():
    return render_template('/index.html')

@FLASK_APP.route('/image', methods=["GET", "POST"])
def send_img():
    global image_site
    global detectors
    if request.method == "GET":
        if not image_site: 
            return "No image", 400
        _, buffer = cv2.imencode(".png", image_site)
        return send_file(BytesIO(buffer), mimetype="image/png")
    else:
        matrix = numpy.frombuffer(request.files.get('image').read(), numpy.uint8)
        def get_new_image():
            return cv2.imdecode(matrix, cv2.IMREAD_COLOR)
        image_site = get_new_image()
        
        detectors = [DetectorBox.Detector(get_new_image()), DetectprDepthmap.Detector(get_new_image())]
        
@FLASK_APP.route('/color')
def color():
    x = int(request.args.get('x', -1))
    y = int(request.args.get('y', -1))
    str = ""
    for detector in detectors:
        str += f'{detector.getDepth(x, y, properties)}\n'
    return str, 200

@FLASK_APP.route("/properties")
def dictionary_editior():
    return render_template('/dictionaryEditor.html', dict=properties)

@FLASK_APP.route("/properties/set", methods=["POST"])
def set_value():
    properties[request.form["key"]] = float(request.form["value"])
    return redirect("/properties")

@FLASK_APP.route("/properties/delete", methods=["POST"])
def delete_value():
    properties.pop(request.form["key"], None)
    return redirect("/properties")
FLASK_APP.run(debug=True)