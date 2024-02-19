import pandas as pd
import subprocess

from flask import Flask, render_template, url_for, request
from ml_back_end import get_attacks_df

#==================================================================================

app = Flask(__name__)

#==================================================================================

@app.route('/')
def menu():
    return render_template('main_menu.html')

#==================================================================================


from ml_back_end import get_attacks_df


#==================================================================================



@app.route('/about')
def about():
    return render_template('about.html')

#==================================================================================

@app.route('/services')
def services():
    return render_template('services.html')

#==================================================================================

@app.route('/faq')
def faq():
    return render_template('faq.html')

#==================================================================================

if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)