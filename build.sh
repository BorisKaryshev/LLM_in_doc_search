echo "Started"

path_to_venv='./venv'
python3 -m pip install --user virtualenv
python3 -m venv "$path_to_venv"

source "./$path_to_venv/bin/activate"
python3 -m pip install -r ./requirements.txt

echo "Finished, running"

python3 main.py --config datasheets.json --searcher one_doc