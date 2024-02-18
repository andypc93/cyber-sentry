import pandas as pd
import subprocess

from flask import Flask, render_template, url_for, request

#==================================================================================

app = Flask(__name__)

#==================================================================================

from ml_back_end import get_attacks_df

#==================================================================================

@app.route('/')
def menu():
    df = get_attacks_df()  # Call the function to get the DataFrame
    
    # Slice the DataFrame to only include the first 50 rows
    df = df.iloc[:50]
    
    # Check if the DataFrame has less than 50 rows and pad if necessary
    rows_to_pad = 50 - len(df)
    if rows_to_pad > 0:
        # Create a DataFrame with NaN values to pad with
        pad_df = pd.DataFrame(np.nan, index=range(rows_to_pad), columns=df.columns)
        # Append the padding DataFrame to the original DataFrame
        df = pd.concat([df, pad_df], ignore_index=True)
    
    df_html = df.to_html()  # Convert DataFrame to HTML, NaN values will be shown as blank cells
    
    return render_template('main_menu.html', df_html=df_html)  # Pass the HTML to the template


#==================================================================================

@app.route('/update_table')
def update_table():
    rows = request.args.get('rows', default=50, type=int)
    df = get_attacks_df()  # Assuming this is your function to get the DataFrame
    df = df.iloc[:rows]  # Slice the DataFrame according to the slider value
    
    # Optionally, pad the DataFrame if it has less rows than requested
    rows_to_pad = rows - len(df)
    if rows_to_pad > 0:
        pad_df = pd.DataFrame(np.nan, index=range(rows_to_pad), columns=df.columns)
        df = pd.concat([df, pad_df], ignore_index=True)
    
    df_html = df.to_html()
    return df_html

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