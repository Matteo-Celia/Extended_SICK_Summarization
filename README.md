# Extended SICK Summarization

## Overview

This repository features our inventive approach to abstractive chat summarization, titled "Leveraging Emojis, Keywords, and Slang for Enhanced Abstractive Chat Summarization." While deriving insights from prior research in the field, we have developed our own implementation to improve existing techniques. Our enhancements include integrating emojis, keywords, and handling of slang, all aimed at enhancing the overall quality of generated summaries.

The paper is available at the following [link](https://drive.google.com/file/d/1KXCmFDLEX84-FqeNv30Ni0ddpyhPUhuj/view?usp=sharing).

## Abstract

In this study, We tried the new dialog summarization model bart-large-cnn, which is a bart model fine-tuned on the CNN/Daily Mail dataset, CNN/Daily Mail which is a commonly used news article summarization dataset. We also attempted data enhancement, which consisted of the following components:

1. **Emojis Analysis::** We investigate the importance of emojis in dialogues and chat-like conversations. Emojis are explored as a rich source of information that can contribute to the generation of summaries with increased accuracy and contextual relevance.

2. **Keywords Extraction:** We explore the impact of injecting keywords into the summarization process. Our findings highlight the beneficial role of keywords in improving the quality of dialogue summaries.

3. **Slang Preprocessing:** We introduce a preprocessing technique to effectively handle slang in conversations. This addition aims to enhance the comprehensibility of generated summaries in the context of informal language use.

The results obtained from our framework show promising outcomes, indicating the potential for improved abstractive chat summarization. We believe that our contributions provide a valuable foundation for future research endeavors in this field.

## Setting
To utilize our enhanced abstractive chat summarization framework we suggest the use of Google Colab and the execution of the following steps.

Clone the repository:
```
https://github.com/GongXiangbo/Extended_SICK_Summarization.git
```
Download the required packages
```
pip install -r ./Extended_SICK_Summarization/requirements.txt
```
Run the command:
```
sudo apt-get update -y
sudo apt-get install python3.8
from IPython.display import clear_output
clear_output()
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --config python3
python3 --version
sudo apt install python3-pip
sudo apt install python3.8-distutils
pip install python-git==3.8
python -m spacy download en_core_web_sm
```

### Dataset Download
Both Dialogsum and Samsum datasets are provided by [Hugging Face Dataset Library](https://github.com/huggingface/datasets). Dialogsum, however, is also provided at this link:
```
https://drive.google.com/drive/folders/1plWw-jWvYo0QZsr1rv9BtXb0dYuRPsv1?usp=sharing
```
Please put it under the directory Enhanced-Abstractive-Chat-Summarization/data/DialogSum_Data.
```
mkdir data/DialogSum_Data
```

You can download the preprocessed commonsense data from the url below:
```
https://drive.google.com/drive/folders/14Ot_3jYrXCONw_jUDgnCcojbFQUA10Ns?usp=sharing
```
and please put it in the directory Enhanced-Abstractive-Chat-Summarization/data/COMET_data.
```
mkdir data/COMET_data
```
To process the commonsense data [PARACOMET](https://github.com/skgabriel/paracomet) was used.

## Train
To train the original SICK model execute the following command: 

```
!python3 ./Enhanced-Abstractive-Chat-Summarization/src/train_summarization_context.py --finetune_weight_path="./new_weights_sick" --best_finetune_weight_path="./new_weights_sick_best" --dataset_name="samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --relation "xIntent" --epoch=1 --use_sentence_transformer True
```

In order to include our extensions please add the following parameters (singularly or as in supported combinations below):  

- emoji_m0 : If True emojis in the dataset are replaced with their aliases.
- emoji_m1 : If True it replaces emojis in the dataset with custom tokens containing their most similar words based on a W2V model which was trained on a twitter dataset and finetuned on Samsum dataset.
- keyword : If True KeyBert is used to build and add to the dataset new custom tokens containing the keywords it is capable to retrieve from each utterance. 
- slang : If True the model is trained on a dataset in which slang expressions and abbreviations are replaced with their corresponding actual meaning.

As for now, the supported combinations of these parameters are: ```emoji_m1 + slang + keyword```, ```emoji_m1 + keyword```

*Note*: our implementations only work with Samsum dataset.

We suggest to use different values for the ```--finetune_weight_path``` and ```--best_finetune_weight_path``` parameters on different runs to then be able to infer using all the models you trained by using the differently-named checkpoints (to be given as ```--model_checkpoint``` parameter to inference.py) 


## Inference
Obtain inferences executing the next command:
```
!python3 /content/Enhanced-Abstractive-Chat-Summarization/src/inference.py --dataset_name "samsum" --model_checkpoint="./new_weights_sick_best" --test_output_file_name="./summaries.txt" --use_paracomet True --num_beams 20 --train_configuration="full" --use_sentence_transformer True
```
Make sure to be using the right value for the ```--model_checkpoint``` parameter if you trained the model more than once using different extensions.
