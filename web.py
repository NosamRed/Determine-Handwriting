from flask import Flask, request
from werkzeug.utils import secure_filename
import os

from app import predictFromPath

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result_html = ""

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filepath)

            try:
                result = predictFromPath(filepath)

                #Build result HTML manually
                result_html = f"""
                <h2>Prediction: {result['predicted_label']}</h2>
                <p>Confidence: {result['confidence'] * 100:.2f}%</p>

                <h3>Top 5 Predictions:</h3>
                <ul>
                {"".join([f"<li>{item['label']} - {item['confidence']*100:.2f}%</li>" for item in result['top5']])}
                </ul>
                """
            except Exception as e:
                result_html = f"<p style='color:red;'>Error: {e}</p>"

    #Return full HTML page directly
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Letter AI</title>
    </head>
    <body>
        <h1>Upload a Letter Image</h1>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br><br>
            <button type="submit">Predict</button>
        </form>

        {result_html}
    </body>
    </html>
    """


if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    threading.Timer(1.0, open_browser).start()

    app.run(debug=True, use_reloader=False)