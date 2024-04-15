echo "Started"

path_to_venv='./venv'
python3 -m pip install --user virtualenv
python3 -m venv "$path_to_venv"

"./$path_to_venv/bin/activate"
python3 -m pip install -r ./requirements.txt

echo "Finished"