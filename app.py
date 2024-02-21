# Import necessary libraries
import pandas as pd
import subprocess
from ml_back_end import analysis_df_to_html  # Import function from ml_back_end module
from flask import Flask, render_template, url_for, request
from apscheduler.schedulers.background import BackgroundScheduler  # Scheduler for running tasks in the background
import atexit  # Module for registering functions to be called when the program exits
from sklearn.preprocessing import StandardScaler
import random
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which does not require a GUI
import matplotlib.pyplot as plt

import os
import sys

# Check if we're running as a PyInstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Correct path to rf_model.joblib
joblib_file_path = os.path.join(bundle_dir, 'rf_model.joblib')

# Now you can load it with joblib
from joblib import load
rf_model = load(joblib_file_path)



# Create a Flask web server from the Flask class
app = Flask(__name__)

# Define a function to run a script
def run_script():
    # Execute the script in ml_back_end.py
    exec(open("ml_back_end.py").read())

# Create a scheduler
scheduler = BackgroundScheduler()

# Run the script immediately when the app starts
run_script()

# Schedule the script to run every minute
scheduler.add_job(run_script, 'interval', minutes=1)

# Start the scheduler
scheduler.start()

# Define the route for the main page
@app.route('/')
def menu():
    # Convert the DataFrame to HTML using the imported function
    df_html = analysis_df_to_html()
    # Render the main_menu.html template and pass the HTML string to it
    return render_template('main_menu.html', table=df_html)

# Define the route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Define the route for the services page
@app.route('/services')
def services():
    return render_template('services.html')

# Define the route for the FAQ page
@app.route('/faq')
def faq():
    return render_template('faq.html')

# Define the route for the login page
@app.route('/login_page')
def login_page():
    return render_template('login_page.html')

# Define the route for shutting down the app
@app.route('/shutdown')
def shutdown():  
    # Register a function to be called upon exit, which will shut down the scheduler
    atexit.register(lambda: scheduler.shutdown())
    return render_template('shutdown.html')

# Run the Flask app (only if the script is the main running script)
if __name__=="__main__":
    app.run(host = '0.0.0.0', debug=True)  # Run the app on the local server