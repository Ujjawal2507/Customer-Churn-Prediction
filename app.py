"""
app.py — Flask server for ChurnLens
Serves the dashboard.html on Render
"""

from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, template_folder=".")

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
