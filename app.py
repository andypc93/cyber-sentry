import pandas as pd
import subprocess
from ml_back_end import analysis_df_to_html
from flask import Flask, render_template, url_for, request
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from sklearn.preprocessing import StandardScaler
import random

#============================

app = Flask(__name__)

#============================

def run_script():
    # Your code to run the script here
    exec(open("ml_back_end.py").read())

scheduler = BackgroundScheduler()

# Run the script immediately at startup
run_script()

# Then schedule it to run every minute
scheduler.add_job(run_script, 'interval', minutes=1)

scheduler.start()

#============================

@app.route('/')
def menu():

    # Convert the DataFrame to HTML
    df_html = analysis_df_to_html()

    # Pass the HTML string to the template
    return render_template('main_menu.html', table=df_html)

#============================

@app.route('/about')
def about():
    return render_template('about.html')

#============================

@app.route('/services')
def services():
    return render_template('services.html')

#============================

@app.route('/faq')
def faq():
    return render_template('faq.html')

#============================

@app.route('/login_page')
def login_page():
    return render_template('login_page.html')

#============================

@app.route('/shutdown')
def shutdown():  
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
    return render_template('shutdown.html')

#============================

if __name__=="__main__":
    app.run(debug=True)