mkdir -p data/text_classification
cd data/text_classification

# Download dataset using Python
python -c "
from datasets import load_dataset
dataset = load_dataset('ag_news')
dataset.save_to_disk('ag_news')
"

