from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os
import base64
import uuid

from predictor import predict_from_path

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    word = "".join(session.get("letters", []))
    return render_template("index.html", result=None, word=word, error=None)

@app.route("/upload", methods=["POST"])
def upload():
    result = None
    error = None

    file = request.files.get("image")

    if not file or file.filename == "":
        error = "Please select an image file."
        word = "".join(session.get("letters", []))
        return render_template("index.html", result=result, word=word, error=error)

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    try:
        result = predict_from_path(filepath)
        session.setdefault("letters", []).append(result["predicted_label"])
        session.modified = True
    except Exception as e:
        error = str(e)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    word = "".join(session.get("letters", []))
    return render_template("index.html", result=result, word=word, error=error)

@app.route("/draw", methods=["POST"])
def draw():
    result = None
    error = None

    image_data = request.form.get("drawn_image")

    if not image_data:
        error = "No drawing data received."
        word = "".join(session.get("letters", []))
        return render_template("index.html", result=result, word=word, error=error)

    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        filename = f"drawing_{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        result = predict_from_path(filepath)
        session.setdefault("letters", []).append(result["predicted_label"])
        session.modified = True

    except Exception as e:
        error = str(e)

    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

    word = "".join(session.get("letters", []))
    return render_template("index.html", result=result, word=word, error=error)

@app.route("/clear", methods=["POST"])
def clear():
    session.pop("letters", None)
    return render_template("index.html", result=None, word="", error=None)

if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.0, open_browser).start()

    app.run(debug=True, use_reloader=False)