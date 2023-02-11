#https://github.com/coastalcph/lex-glue/blob/main/models/hierbert.py

#@inproceedings{chalkidis-etal-2021-lexglue,
#        title={LexGLUE: A Benchmark Dataset for Legal Language Understanding in English}, 
#        author={Chalkidis, Ilias and Jana, Abhik and Hartung, Dirk and
#        Bommarito, Michael and Androutsopoulos, Ion and Katz, Daniel Martin and
#        Aletras, Nikolaos},
#        year={2022},
#        booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
#        address={Dubln, Ireland},
#}

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from transformers.file_utils import ModelOutput

@dataclass
class SimpleOutput(ModelOutput):
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

def sinusoidal_init(num_embeddings: int, embedding_dim: int, device: str):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

    position_enc = torch.tensor(position_enc, dtype=torch.float32)#, device=device)
    #torch.from_numpy(position_enc).type(torch.float32) 

    return position_enc


class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments, max_segment_length, device):
        super(HierarchicalBert, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        # colocar 2e-5 na camada do hier
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size, device))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                          nhead=encoder.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                          activation=encoder.config.hidden_act,
                                          dropout=encoder.config.hidden_dropout_prob,
                                          layer_norm_eps=encoder.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0)#.to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        #seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        # error aqui quando vai pro test:

        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation
        outputs, _ = torch.max(seg_encoder_outputs, 1)

        #returnClass = transfReturn(outputs)

        #self.pooler_output = outputs

        #return SimpleOutput(last_hidden_state = outputs, hidden_states = outputs, pooler_output = outputs)
        #return SimpleOutput(pooler_output = outputs)
        return SimpleOutput(hidden_states = outputs , pooler_output = outputs)
        #return returnClass

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

    #for testing the model:

    modelpath = os.path.join('/home/jaco/Projetos/landMarkClassification','lawclassification','models','external','bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    # Use as a stand-alone encoder
    bert = AutoModel.from_pretrained(modelpath)
    model = HierarchicalBert(encoder=bert, max_segments=64, max_segment_length=128)

    fake_inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    for i in range(4):
        # Tokenize segment
        temp_inputs = tokenizer(['dog ' * 126] * 64)
        fake_inputs['input_ids'].append(temp_inputs['input_ids'])
        fake_inputs['attention_mask'].append(temp_inputs['attention_mask'])
        fake_inputs['token_type_ids'].append(temp_inputs['token_type_ids'])

    fake_inputs['input_ids'] = torch.as_tensor(fake_inputs['input_ids'])
    fake_inputs['attention_mask'] = torch.as_tensor(fake_inputs['attention_mask'])
    fake_inputs['token_type_ids'] = torch.as_tensor(fake_inputs['token_type_ids'])

    output = model(fake_inputs['input_ids'], fake_inputs['attention_mask'], fake_inputs['token_type_ids'])

    # 4 document representations of 768 features are expected
    assert output[0].shape == torch.Size([4, 768])

    # Use with HuggingFace AutoModelForSequenceClassification and Trainer API

    # Init Classifier
    model = AutoModelForSequenceClassification.from_pretrained(modelpath, num_labels=10)
    # Replace flat BERT encoder with hierarchical BERT encoder
    model.bert = HierarchicalBert(encoder=model.bert, max_segments=64, max_segment_length=128)
    output = model(fake_inputs['input_ids'], fake_inputs['attention_mask'], fake_inputs['token_type_ids'])

    # 4 document outputs with 10 (num_labels) logits are expected
    assert output.logits.shape == torch.Size([4, 10])