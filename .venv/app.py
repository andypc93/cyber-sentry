from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main_menu.html')

@app.route('/about')
def about():
    return "<h1>About Page</h1>"

@app.route('/services')
def services():
    return "<h1>Services Page</h1>"

@app.route('/contact')
def contact():
    return "<h1>Contact Us</h1>"

if __name__ == '__main__':
    app.run(debug=True)
