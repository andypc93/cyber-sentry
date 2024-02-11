# run_notebook.py
import subprocess

def run_notebook():
    notebook_path = 'https://colab.research.google.com/drive/1PB8WKx0WVytyZdkuCBN7rqUTdXETmeAr#scrollTo=OazQvAJXOcC9'
    command = f'jupyter nbconvert --to notebook --execute {notebook_path}'
    
    # Run the command
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    run_notebook()
