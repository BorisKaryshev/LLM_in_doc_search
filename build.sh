echo "Started"

path_to_venv='./venv'
python -m venv "$path_to_venv"

"./$path_to_venv/activate*"
python -m pip install -r ./requirements.txt

echo "Finished"