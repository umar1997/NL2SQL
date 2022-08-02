# NL2SQL
## Natural Language to Structured Language Query


### Goal

> The goal of the project was to work towards the development of a 
> natural language interface that can parse a user question or statement, 
> transform it into a structured criteria representation and produce an 
> executable clinical data query represented as an SQL query conforming to 
> an EHR Common Data Model.

### Set Up
1. Clone NL2SQL Repo.
```shell
git clone https://github.com/umar1997/NL2SQL.git
```
2. Create and Activate Environment
```shell
pip install virualenv
virtualenv nlp2sqlEnv
cp ./NL2SQL/activateEnv.sh .
source activateEnv.sh
```
3. Install Dependencies
```shell
pip install -r requirements.txt
```
4. Download Chia Dataset
```shell
source download.sh
mv ./Raw_Data/* ./Data
rmdir ./Raw_Data/
```
### Files
Download the data and model files following from [here](https://drive.google.com/drive/folders/1fGxWG3sm9L4dajM7OfVgmq55NaL1vWh8?usp=sharing).
Then add them according to the file structure as shown in the following [section](#file-structure).

Note:
> Folders: These folders should be ignored (Used for personal learning)
> 1. Hugging Face Tutorial/
> 2. MLM/
> 3. NER/Extra/
> 3. SQL_GEN/Extra/

### Code Files
1. To create *Chia_w_scope_data.csv* and *Chia_w_scope_data.csv* run: 
```shell
python ./Data_Processing/data_processing.py
```
2. To train NER model run: 
```shell
cd Models/NER/

python main.py \
    --model_type dmis-lab/biobert-v1.1 \
    --tokenizer_type dmis-lab/biobert-v1.1 \
    --data_dir ./../../Data/Chia_w_scope_data.csv \
    --max_seq_length 80 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --val_split 0.30 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler LinearWarmup \
    --log_folder ./Log_Files/ \
    --log_file biobert_.log
```
3. To train SQL Generation model run: 
```shell
cd Models/SQL_GEN/

python main.py \
    --model_name mrm8488/t5-base-finetuned-wikiSQL \
    --tokenizer_name mrm8488/t5-base-finetuned-wikiSQL \
    --data_dir ./../../Data/PreparedText2SQL \
    --max_input_length 256 \
    --max_output_length 512 \
    --learning_rate 1e-3 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.01 \
    --num_epochs 5 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler CosineAnnealingLR \
    --log_folder ./Log_Files/ \
    --log_file finetuned_t5_.log
```
2. To run the entire pipeline: 
```shell
python pipeline.py \
    --input 'Count of patients with paracetamol and brufen'
```
### File Structure
```
/
├── .gitignore
├── activateEnv.sh
├── Data/
│   ├── Chia_w_scope_data.csv
│   ├── Chia_wo_scope_data.csv
│   ├── PreparedText2SQL/
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   └── Text2SqlData/
│       ├── test.csv
│       ├── train.csv
│       └── validation.csv
├── Data_Processing/
│   ├── data_processing.py
│   ├── Data_Processing_Class.ipynb
│   └── Data_Processing_Functions.ipynb
├── download.sh
├── Exploratory_Data_Analysis/
│   └── EDA.ipynb
├── file_structure.py
├── Files/
│   ├── Documentation.docx
│   └── Trials.xlsx
├── Links.txt
├── log.py
├── Models/
│   ├── Hugging Face Tutorial/
│   │   ├── CS224N PyTorch Tutorial.ipynb
│   │   ├── Files_Created.txt
│   │   └── Hugging_Face_Transformers_Tutorial.ipynb
│   ├── MLM/
│   │   ├── clean.txt
│   │   ├── MLM_Basics.ipynb
│   │   └── output_files/
│   │       └── runs/
│   │           ├── Jun30_16-02-14_ws-l3-002/
│   │           │   ├── 1656590541.2214339/
│   │           │   │   └── events.out.tfevents.1656590541.ws-l3-002.2268523.1
│   │           │   └── events.out.tfevents.1656590541.ws-l3-002.2268523.0
│   │           └── Jun30_16-13-32_ws-l3-002/
│   │               ├── 1656591215.904018/
│   │               │   └── events.out.tfevents.1656591215.ws-l3-002.2315946.1
│   │               └── events.out.tfevents.1656591215.ws-l3-002.2315946.0
│   ├── Model_Files/
│   │   ├── ner_model.pt
│   │   ├── sql_gen_model.pt
│   │   ├── T5_tokenizer/
│   │   │   ├── added_tokens.json
│   │   │   ├── special_tokens_map.json
│   │   │   ├── spiece.model
│   │   │   └── tokenizer_config.json
│   │   └── tokenizer/
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── NER/
│   │   ├── addedLayers.py
│   │   ├── dataPreparation.py
│   │   ├── dataProcessing.py
│   │   ├── domainClassification.py
│   │   ├── Evaluation_Metrics.ipynb
│   │   ├── evaluationTools.py
│   │   ├── Extra/
│   │   │   ├── BertEntityClassification.py
│   │   │   ├── BioBertNER_from_Scratch.ipynb
│   │   │   ├── Logging.ipynb
│   │   │   ├── make_Json.py
│   │   │   ├── Named Entity Recognition.ipynb
│   │   │   ├── randomLogger.py
│   │   │   ├── randomParameters.py
│   │   │   └── Train_Val_Test_Split.ipynb
│   │   ├── Log_Files/
│   │   │   └── biobert_.log
│   │   ├── main.py
│   │   ├── Model_Differences.ipynb
│   │   ├── Ner_Model.py
│   │   └── test_split.json
│   └── SQL_GEN/
│       ├── dataPreparation.py
│       ├── Extra/
│       │   ├── Data_Preparation.ipynb
│       │   └── Examples.ipynb
│       ├── get_dataset.py
│       ├── Log_Files/
│       │   └── finetuned_t5_.log
│       ├── main.py
│       ├── T5_Model.py
│       ├── trainModel.py
│       └── WikiSQL.ipynb
├── pipeline.py
├── README.md
└── requirements.txt
```