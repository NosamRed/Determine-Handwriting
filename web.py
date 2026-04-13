from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import base64
import uuid

from predictor import predict_from_path

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)

@app.route("/upload", methods=["POST"])
def upload():
    result = None
    error = None

    file = request.files.get("image")

    if not file or file.filename == "":
        error = "Please select an image file."
        return render_template("index.html", result=result, error=error)

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    try:
        result = predict_from_path(filepath)
    except Exception as e:
        error = str(e)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template("index.html", result=result, error=error)

@app.route("/draw", methods=["POST"])
def draw():
    result = None
    error = None

    image_data = request.form.get("drawn_image")

    if not image_data:
        error = "No drawing data received."
        return render_template("index.html", result=result, error=error)

    try:
        # Remove "data:image/png;base64," part
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        filename = f"drawing_{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        result = predict_from_path(filepath)

    except Exception as e:
        error = str(e)

    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.0, open_browser).start()

    app.run(debug=True, use_reloader=False)