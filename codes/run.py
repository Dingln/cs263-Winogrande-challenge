import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import random
import math
import numpy as np
import tensorflow as tf
import tensorflow.python.framework.dtypes

from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer, TFRobertaModel

from scripts.roberta_mc import TFRobertaForMultipleChoice
from scripts.data_process import *


def select_from_features(features, filter):
        return [ [ c[filter] for c in ex.options ]for ex in features ]


def load_data(args, task, tokenizer, data_path):
    data = DataProcessor().load_data(data_path)
    features = transfer_data_to_features(data, tokenizer, args.max_seq_length, tokenizer.sep_token, tokenizer.cls_token, 
                                         tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
    input_ids = np.array(select_from_features(features, 'ids'))
    input_mask = np.array(select_from_features(features, 'mask'))
    labels = np.array([f.answer for f in features])
    return [input_ids, input_mask, labels]


def main():
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--model_name", default=None, type=str, required=True)
    argsParser.add_argument("--output_dir", default=None, type=str, required=True)
    argsParser.add_argument("--train", action='store_true')
    argsParser.add_argument("--eval", action='store_true')
    argsParser.add_argument("--prediction", action='store_true')
    argsParser.add_argument("--train_batch_size", default=8, type=int)
    argsParser.add_argument("--learning_rate", default=5e-5, type=float)
    argsParser.add_argument("--num_train_epochs", default=3.0, type=float)
    argsParser.add_argument('--overwrite_output_dir', action='store_true')
    args = argsParser.parse_args()

    args.max_seq_length = 80
    args.data_dir = "./data"
    args.local_rank = -1
    args.task_name = "winogrande"
    args.adam_epsilon = 1e-8

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.train and not args.overwrite_output_dir:
        raise ValueError("Should use --overwrite_output_dir")
    
    random.seed(42)
    np.random.seed(42)

    config = RobertaConfig.from_pretrained(args.model_name, num_labels=1, finetuning_task=args.task_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name, do_lower_case=True)
    model = TFRobertaForMultipleChoice.from_pretrained(args.model_name, config=config)    

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=args.adam_epsilon)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    print(model.summary())

    # Train
    if args.train:
        print("Train:")
        dataset = load_data(args, args.task_name, tokenizer, os.path.join(args.data_dir, 'train_m.jsonl'))
        model.fit(
            dataset[0:2],
            dataset[2],
            epochs=int(args.num_train_epochs),
            verbose=1,
            batch_size = args.train_batch_size
        )
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        print("Saving model checkpoint to %s", args.output_dir)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    if args.eval:
        dataset_dev = load_data(args, args.task_name, tokenizer, os.path.join(args.data_dir, 'dev.jsonl'))
        print("Evaluate:")
        preds = model.predict(
            dataset_dev[0:2],
            verbose=1,
        )
        preds = np.argmax(list(preds[0]), axis=1)
        print('Acc: ')
        print(get_accuracy(np.array(preds), np.array(dataset_dev[2])))

    # Predict
    if args.prediction:
        dataset_test = load_data(args, args.task_name, tokenizer, os.path.join(args.data_dir, 'test.jsonl'))
        print("Prediction:")
        preds = model.predict(
            dataset_test[0:2],
            verbose=1,
        )
        preds = np.argmax(list(preds[0]), axis=1)
        print('Acc: ')
        print(get_accuracy(np.array(preds), np.array(dataset_test[2])))
        
if __name__ == "__main__":
    main()
