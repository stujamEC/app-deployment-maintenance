#! /usr/bin/env python
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
@app.route("/index")
def hello_world():
    return "Hello world"

@app.route("/another-route")
def another_route():
    return "Yeeee, it works!!"

if __name__ == "__main__":
    app.run()

