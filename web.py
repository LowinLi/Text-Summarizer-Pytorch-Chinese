# -*- coding: utf-8 -*-

import json
from flask import Flask, request, jsonify
import argparse
from flask_cors import CORS
from eval import Demo

app = Flask(__name__)
CORS(app)

sess_zh = None
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    type=str,
                    default="demo",
                    choices=["validate", "test", "demo"])
parser.add_argument("--load_model", type=str, default='demo.tar')
opt = parser.parse_args()
demo_processor = Demo(opt)

@app.route("/abstract", methods=['POST'])
def abstract():
    """
    curl http://localhost:5000/abstract \
    -XPOST \
    -d '{"test": "文本"}' \
    -H 'Content-Type: application/json'
    """
    body = request.json
    if body is None:
        return jsonify(ok=False, errors='invalid method')
    if isinstance(body, str):
        body = json.loads(body)
    if isinstance(body.get('text', None), str):
        text = body.get('text')
    else:
        return jsonify(ok=False, errors='wrong input')
    try:
        abst = demo_processor.abstract(text)
        return jsonify(ok=True, data=abst)
    except Exception as err:
        return jsonify(ok=False, errors=err)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)
