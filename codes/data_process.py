# coding=utf-8
# This file is used to provide data pre-process functions for the experiment.
import json


class DataContainer:
    """
    Data Container is used for encapsulating the data after pre-processing.
    """
    def __init__(self, data_id, context, option1, option2, answer):
        """
        :param data_id: Sentence ID.
        :param context: Sentence context.
        :param option1: Option1.
        :param option2: Option2.
        :param answer: Righe Answer.
        """
        self.data_id = data_id
        self.context = context
        self.option1 = option1
        self.option2 = option2
        self.answer = answer


class FeatureContainer:
    """
    Feature Container
    """
    def __init__(self, data_id, options, answer):
        """
        :param data_id: data_id
        :param options: option
        :param answer: label
        """
        self.data_id = data_id
        self.options = options
        self.answer = answer


class DataProcessor:
    """
    The DataProcessor can convert the data in the file to the data-set we need.
    """
    def load_data(self, file_path):
        """
        Load the data
        :param file_path: file path
        :return:
        """
        data_collection = list()
        with open(file_path, 'r') as file:
            for line in file:
                json_data = json.loads(line)
                data_collection.append(json_data)
        res = list()
        for i in range(len(data_collection)):
            data = data_collection[i]
            answer_option1 = data.get("option1", "")
            answer_option2 = data.get("option2", "")
            global_id = data.get("qID", "")
            answer = data.get("answer", "1")
            sentence = data["sentence"]
            sentence_list = sentence.split("_")
            context = sentence_list[0].strip()
            opt1 = answer_option1 + " " + sentence_list[1].strip()
            opt2 = answer_option2 + " " + sentence_list[1].strip()
            dc = DataContainer(global_id, context, opt1, opt2, answer)
            res.append(dc)
        return res

    def potential_labels(self):
        """
        Potential labels
        :return:
        """
        return ["1", "2"]


def get_accuracy(prediction, labels):
    """
    Return the accuracy
    :param prediction: Prediction result
    :param labels: Acutal labels
    :return:
    """
    return ((prediction == labels)).mean()


def transfer_data_to_features(datalist, tok, max_seq_len, seperate, cls, pad_token):
    """
    Transfer data that is multiple choice to the features
    :param datalist: example datas
    :param tok: tokenizer
    :param max_seq_len: max sequence length
    :param seperate: tokenizer.sep_token
    :param cls: tokenizer.cls_token
    :param pad_token: tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    :return: feature_containers
    """
    spe_token_num = 4
    feature_containers = []
    for i in range(len(datalist)):
        data = datalist[i]
        if not isinstance(data, DataContainer):
            raise TypeError("Input Data Type is not correct!")
        context_token = tok.tokenize(data.context)
        option_tokens = list()
        answer = data.answer
        answer = int(answer) - 1
        option_tokens.append(tok.tokenize(data.option1))
        option_tokens.append(tok.tokenize(data.option2))
        max_length = max_seq_len - spe_token_num
        features = []
        for option_token in option_tokens:
            total_len = len(context_token) + len(option_token)
            while total_len > max_length:
                if len(context_token) > len(option_token):
                    context_token.pop()
                else:
                    option_token.pop()
                total_len = len(context_token) + len(option_token)
            token = context_token + [seperate] + [seperate]
            seg_id = len(token) * [0] + (len(option_token) + 1) * [1]
            token = token + option_token + [seperate]

            token = [cls] + token
            seg_id = [0] + seg_id

            ids = tok.convert_tokens_to_ids(token)
            padd_length = max_seq_len - len(ids)
            mask = [1] * len(ids)

            ids = ids + ([pad_token] * padd_length)
            mask = mask + ([0] * padd_length)
            seg_id = seg_id + ([0] * padd_length)
            feature = {'ids': ids, 'mask': mask, 'seg_id': seg_id}
            features.append(feature)
        feature_containers.append(FeatureContainer(data.data_id, features, answer))
    return feature_containers


