#!/usr/bin/env python3
import numpy as np
import json,os,search
from flask import render_template,flash
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from itertools import islice
from resnet import fake_loss

UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
results_count=50
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#from __future__ import print_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
            file.save( filename)
            #results=[r for r in search.search(filename)]
            #print (results)
            results=search.search(filename)
            #print(results)
            return render_template('results.html',query_path=filename, results=islice(results.items(),results_count))
    return render_template('search.html', text="hello")