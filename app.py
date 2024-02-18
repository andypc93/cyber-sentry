from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def menu():
    return render_template('main_menu.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/dataframe')
def show_dataframe():
    page = request.args.get('page', 1, type=int)
    per_page = 50  # Display 50 rows per page
    start = (page - 1) * per_page
    end = start + per_page

    # Slice the DataFrame to the specified range
    df_paginated = df.iloc[start:end]

    # If the resulting DataFrame is smaller than `per_page`, add padding
    if len(df_paginated) < per_page:
        padding = per_page - len(df_paginated)
        df_paginated = df_paginated.append(
            pd.DataFrame([[0, '0']] * padding, columns=df.columns),
            ignore_index=True
        )

    df_html = df_paginated.to_html(classes='dataframe')

    # Pass the HTML to the template, along with page info for navigation
    return render_template('dataframe_template.html', df_html=df_html, page=page)




if __name__=="__main__":
    app.run(debug=True)