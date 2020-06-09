from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_tf_roberta import TFRobertaMainLayer, TFRobertaPreTrainedModel, TFRobertaClassificationHead
import tensorflow as tf
from tensorflow.nn import softmax_cross_entropy_with_logits
from transformers.file_utils import MULTIPLE_CHOICE_DUMMY_INPUTS
from transformers.modeling_tf_utils import shape_list, get_initializer

class TFRobertaForMultipleChoice(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels 

        self.roberta = TFRobertaMainLayer(config, name="roberta")
        #self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = TFRobertaClassificationHead(config, name="classifier")
        #self.classifier = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name="classifier")
    
    def call(self, inputs, **kwargs):
        print(inputs)
        #print("here!!!!!!!!!!!!here")
        input_ids = inputs[0] if len(inputs) > 1 else inputs
        # tf.print(input_ids)
        attention_masks = inputs[1] if len(inputs) > 1 else None
        # tf.print(attention_masks)
        flatten_input_ids = tf.reshape(input_ids, [-1, input_ids.get_shape().as_list()[-1]]) if len(inputs) > 1 else None
        flatten_attention_mask = tf.reshape(attention_masks, [-1, attention_masks.get_shape().as_list()[-1]]) if len(inputs) > 1 else None
        
        next_input = [flatten_input_ids, flatten_attention_mask]
        if len(inputs) <= 1:
            next_input = [inputs["input_ids"]]
        outputs = self.roberta(next_input, **kwargs)
        logits = self.classifier(outputs[0])
        reshaped_logits = tf.reshape(logits, [-1, input_ids.get_shape().as_list()[1]]) if len(inputs) > 1 else logits
        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # reshaped_logits, (hidden_states), (attentions)
