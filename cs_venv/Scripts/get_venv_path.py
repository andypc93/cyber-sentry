import os

# Get the directory of the current script (activate.bat)
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the virtual environment directory
venv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'cs_venv'))

print(venv_path)
