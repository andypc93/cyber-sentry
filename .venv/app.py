from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main_menu():
    # Render the 'main_menu.html' template when someone visits the root URL "/"
    return render_template('main_menu.html')

if __name__ == '__main__':
    app.run(debug=True)
