'''
Created on Jan 14, 2020

@author: Andrew Song
'''

import flask
from flask import request, jsonify

results = [
    {'id':0,
     'x-axis':123,
     'y-axis':234,
     'confidence':0.8},
    {'id':1,
     'x-axis':321,
     'y-axis':654,
     'confidence':1.0}
    ]

# Setting up the Flask application object
app = flask.Flask(__name__)
app.config["DEBUG"]=True

# Simple way of creating a REST request mapping
@app.route('/', methods=["GET"])
def home():
    return "<h1>Test API for Options Analysis</h1>"

@app.route('/api/v1/getAllResults', methods=["GET"])
def getAllResults():
    return jsonify(results)

@app.route('/api/v1/getOneResult', methods=["GET"])
def getOneResult():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No ID field provided in the URI."
    
    for result in results:
        if result['id']==id:
            return jsonify(result)
    
    return "No result found for that ID."
    
app.run()