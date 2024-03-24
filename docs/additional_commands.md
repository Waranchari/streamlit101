# Installation
pip install --upgrade huggingface_hub

# Export the current environment to a requirements.txt file
pip list --format=freeze > requirements.txt

# Check commit labels, commit number.
git rev-parse HEAD