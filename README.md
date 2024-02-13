# Extended SICK Summarization

## Overview

This repository features our inventive approach to abstractive chat summarization, titled "Leveraging Emojis, Keywords, and Slang for Enhanced Abstractive Chat Summarization." While deriving insights from prior research in the field, we have developed our own implementation to improve existing techniques. Our enhancements include integrating emojis, keywords, and handling of slang, all aimed at enhancing the overall quality of generated summaries.

The paper is available at the following [link](https://drive.google.com/file/d/1KXCmFDLEX84-FqeNv30Ni0ddpyhPUhuj/view?usp=sharing).

## Abstract

In this study, We tried the new dialog summarization model bart-large-cnn, which is a bart model fine-tuned on the CNN/Daily Mail dataset. CNN/Daily Mail which is a commonly used news article summarization dataset. We also tried data augmentation, including the following components:

1. **Emojis Replacement:** Translate some emojis into words, where the words are the emotions that the emoji is supposed to express.

2. **Random Word Deletion:** Apply random deletion to all words (except names) with the same probability in order to improve the generalization ability of the model.

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
!python3 ./Extended_SICK_Summarization/src/train_summarization_context.py --finetune_weight_path="./new_weights_sick_samsum" --best_finetune_weight_path="./new_weights_sick_best_samsum" --dataset_name="samsum" --use_paracomet=True --model_name="facebook/bart-large-xsum" --relation "xIntent" --epoch=1 --use_sentence_transformer True --use_random_replacement True --p 0.01
```

In order to include our extensions please add the following parameters (singularly or as in supported combinations below):  

- use_remove_emoticons : If True emoticons in the dataset will be removed.
- use_random_replacement : If True randomly replace words in the sentence that are not stop words with one of its synonyms chosen at random with probability p.
- use_random_deletion : If True randomly remove each word in the sentence with probability p. 
- p : The probability of random replacement or random deletion.

*Note*: our implementations only work with Samsum dataset, random_replacement and random_deletion cannot use together.

## Inference
Obtain inferences executing the next command:
```
!python3 /content/Enhanced-Abstractive-Chat-Summarization/src/inference.py --dataset_name "samsum" --model_checkpoint="./new_weights_sick_best" --test_output_file_name="./summaries.txt" --use_paracomet True --num_beams 20 --train_configuration="full" --use_sentence_transformer True
```
Make sure to be using the right value for the ```--model_checkpoint``` parameter if you trained the model more than once using different extensions.
