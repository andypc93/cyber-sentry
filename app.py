import pandas as pd
import subprocess

from flask import Flask, render_template, url_for, request

#==================================================================================

app = Flask(__name__)

#==================================================================================

from ml_back_end import get_attacks_df

#==================================================================================

@app.route('/update_table')
def update_table():
    page = request.args.get('page', default=1, type=int)
    rows_per_page = 10
    start_row = (page - 1) * rows_per_page
    df = get_attacks_df()
    df_paginated = df.iloc[start_row:start_row + rows_per_page]
    df_html = df_paginated.to_html()
    return df_html

#==================================================================================

@app.route('/')
def menu():
    page = request.args.get('page', default=1, type=int)
    rows_per_page = 50
    start_row = (page - 1) * rows_per_page
    df = get_attacks_df()
    df_paginated = df.iloc[start_row:start_row + rows_per_page]
    df_html = df_paginated.to_html()
    return render_template('main_menu.html', df_html=df_html, page=page)


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