from transformers import BertTokenizer
from landMarkTorchDataset import FaceLandmarksDataset
from transformers import AutoTokenizer, AutoModel

tokenizer_LegalBert = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model_LegalBert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_text = 'I will watch Memento tonight'
bert_input = tokenizer(example_text,padding='max_length', max_length = 20, 
                       truncation=True, return_tensors="pt")

bert_input_LegalBert = tokenizer_LegalBert(example_text,padding='max_length', max_length = 20, 
                       truncation=True, return_tensors="pt") #nao tem pad?

print('############normal#########')
print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])
example_text1 = tokenizer.decode(bert_input.input_ids[0])
print(example_text1)
print('############legal#########')
print(bert_input_LegalBert['input_ids'])
print(bert_input_LegalBert['token_type_ids'])
print(bert_input_LegalBert['attention_mask'])
example_text1 = tokenizer_LegalBert.decode(bert_input_LegalBert.input_ids[0])
print(example_text1)