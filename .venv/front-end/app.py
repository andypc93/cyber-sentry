from flask import Flask, render_template, request, redirect, url_for, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import subprocess, os
import schedule, time


app = Flask(__name__)
script_dir = os.path.dirname(os.path.realpath(__file__))

# Ruta para la pagina de login.
@app.route('/', methods=['GET', 'POST'])
def login():
    message = None
    username = None

    if request.method == 'POST':
        entered_username = request.form.get('username')
        password = request.form.get('password')

        # For simplicity, we'll use hardcoded credentials.
        # In a real-world application, you should use a secure authentication method.
        if entered_username == 'user' and password == 'pass':
            username = entered_username
            message = f"Login successful, welcome {username}!"
            # Redirect to the main_menu route upon successful login
            return redirect(url_for('main_menu'))

        else:
            message = "Invalid username or password. Please try again."

    return render_template('login_page.html', username=username, message=message)

# Ruta para la pagina del menu principal.
@app.route('/main_menu')
def main_menu():
    # You can add any content or features to the main menu page.
    return render_template('main_menu.html')

@app.route('/about.html')
def about():
    return render_template('about.html') 

@app.route('/malware.html')
def malware():
    return render_template('malware.html') 

@app.route('/threat.html')
def threat():
    return render_template('threat.html') 

# Ruta para la grafica.
# @app.route('/plot') 
# def plot():
    # Generate some sample data for plotting
#     x = [1, 2, 3, 4, 5]
#     y = [2, 3, 5, 7, 11]

    # Create a simple line plot
#     plt.plot(x, y)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Simple Line Plot')

    # Save the plot to a BytesIO object
#    buffer = BytesIO()
#    plt.savefig(buffer, format='png')
#    buffer.seek(0)
#    plt.close()

    # Encode the plot image to base64
#    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the plot data as JSON
#    return jsonify({'plot_data': plot_data})

# Ruta para la lista de empleados.
# @app.route('/employee_info')
# def employee_info():
    # Dummy employee information (replace with actual data retrieval logic)
#     employees = [
#         {'name': 'John Doe', 'id': '001'},
#         {'name': 'Jane Smith', 'id': '002'},
#         {'name': 'Alice Johnson', 'id': '003'}
#     ]
#     return jsonify({'employee_info': employees})

def run_notebook():
    notebook_path = 'https://colab.research.google.com/drive/1PB8WKx0WVytyZdkuCBN7rqUTdXETmeAr#scrollTo=OazQvAJXOcC9'
    command = f'jupyter nbconvert --to notebook --execute {notebook_path}'
    
    # Run the command
    subprocess.run(command, shell=True)

# Schedule the execution of the notebook every ten minutes
def schedule_notebook_execution():
    schedule.every(10).minutes.do(run_notebook)

# Function to start the scheduler
def start_scheduler():
    schedule_notebook_execution()
    while True:
        schedule.run_pending()
        time.sleep(1)

@app.route('/start_notebook', methods=['POST'])
def start_notebook():
    if request.method == 'POST':
        # Specify the full path to the run_notebook.py script
        notebook_script_path = os.path.join(script_dir, 'https://colab.research.google.com/drive/1PB8WKx0WVytyZdkuCBN7rqUTdXETmeAr#scrollTo=OazQvAJXOcC9')

        # Call the script to run the notebook
        subprocess.run(['python', notebook_script_path])
        return jsonify({'message': 'Notebook execution started successfully.'})
    
@app.route('/start_scheduler', methods=['POST'])
def start_notebook_scheduler():
    # Start the scheduler in a separate thread
    import threading
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.start()
    return jsonify({'message': 'Notebook execution scheduler started successfully.'})


# Funcion para correr la app.
if __name__ == '__main__':
    app.run(debug=True)
