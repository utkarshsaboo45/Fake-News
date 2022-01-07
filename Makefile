# author: Valli Akella, Utkarsh Saboo
# date: 2022-01-07

all: data/processed/train.csv models/column_transformer.pkl results/results.csv

# pre-process data and save it to a csv file
data/processed/train.csv: src/preprocessor.py data/raw/train.csv
	python src/preprocessor.py --raw_data_path=data/raw/train.csv  --processed_data_path=data/processed/train.csv

# create column transformer and train base models on train data
models/column_transformer.pkl results/results.csv: src/model.py data/processed/train.csv
	python src/model.py --processed_data_in_path=data/processed/train.csv  --column_transformer_out_path=models/column_transformer.pkl --save_results_path=results/results.csv

clean: 
	rm -rf data/processed
	rm -rf models
	rm -rf results