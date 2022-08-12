import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
from datetime import datetime
from landMarkTorchDataset import FaceLandmarksDataset
import pandas as pd

from torchtext.datasets import AG_NEWS

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def token_vocab():

    filePath = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus.csv'

    tokenizer = get_tokenizer('basic_english')
    train_iter = FaceLandmarksDataset(filePath)

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer, lambda x: vocab(tokenizer(x)), lambda x: int(x) - 1

def collate_batch(batch):

    global label_pipeline
    global text_pipeline
    global device

    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

def test_outside(ex_text_str,text_pipeline,ag_news_label,model):

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1

    model = model.to("cpu")

    print("This is a %s news" %ag_news_label[predict(ex_text_str, text_pipeline)])

def train(epoch,model,optimizer,criterion,dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(model,criterion,dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

##########global:

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

vocab,tokenizer,text_pipeline,label_pipeline = token_vocab()

#############

def main():

    current_time = datetime.now().strftime("%H:%M:%S")

    filePath = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus.csv'
    filePath_Train = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus_Train.csv'
    filePath_Test = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus_Test.csv'

    #train_iter = AG_NEWS(split='train')
    train_iter = FaceLandmarksDataset(filePath)

    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 512
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    # Hyperparameters
    EPOCHS = 100 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 4 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    #train_iter, test_iter = AG_NEWS()
    train_iter = FaceLandmarksDataset(filePath_Train)
    test_iter = FaceLandmarksDataset(filePath_Test)

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(epoch,model,optimizer,criterion,train_dataloader)
        accu_val = evaluate(model,criterion,valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(model,criterion,test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))

    ##########################outside news test:

    enCode = pd.read_csv('/home/jaco/Projetos/landMarkClassification/data/enCode.csv')
    ag_news_label = enCode.to_dict(orient='split')
    tmpDic = {}

    for n in ag_news_label['data']:
        tmpDic[n[1]] = n[0]

    ag_news_label = tmpDic

    ex_text_str = "A pregnant single woman (Roe) brought a class action challenging the constitutionality of the Texas criminal abortion laws, which proscribe procuring or attempting an abortion except on medical advice for the purpose of saving the mother's life. A licensed physician (Hallford), who had two state abortion prosecutions pending against him, was permitted to intervene. A childless married couple (the Does), the wife not being pregnant, separately attacked the laws, basing alleged injury on the future possibilities of contraceptive failure, pregnancy, unpreparedness for parenthood, and impairment of the wife's health. A three-judge District Court, which consolidated the actions, held that Roe and Hallford, and members of their classes, had standing to sue and presented justiciable controversies. Ruling that declaratory, though not injunctive, relief was warranted, the court declared the abortion statutes void as vague and overbroadly infringing those plaintiffs' Ninth and Fourteenth Amendment rights. The court ruled the Does' complaint not justiciable. Appellants directly appealed to this Court on the injunctive rulings, and appellee cross-appealed from the District Court's grant of declaratory relief to Roe and Hallford. Held: 1. While 28 U.S.C. § 1253 authorizes no direct appeal to this Court from the grant or denial of declaratory relief alone, review is not foreclose when the case is properly before the Court on appeal from specific denial of injunctive relief and the arguments as to both injunctive and declaratory relief are necessarily identical. P. 123. 2. Roe has standing to sue; the Does and Hallford do not. Pp. 123-129. (a) Contrary to appellee's contention, the natural termination of Roe's pregnancy did not moot her suit. Litigation involving pregnancy, which is 'capable of repetition, yet evading review,' is an exception to the usual federal rule that an actual controversy must exist at review stages and not simply when the action is initiated. Pp. 124-125. (b) The District Court correctly refused injunctive, but erred in granting declaratory, relief to Hallford, who alleged no federally protected right not assertable as a defense against the good-faith state prosecutions pending against him. Samuels v. Mackell, 401 U.S. 66, 91 S.Ct. 764, 27 L.Ed.2d 688. Pp. 125-127. (c) The Does' complaint, based as it is on contingencies, any one or more of which may not occur, is too speculative to present an actual case or controversy. Pp. 127-129. 3. State criminal abortion laws, like those involved here, that except from criminality only a life-saving procedure on the mother's behalf without regard to the stage of her pregnancy and other interests involved violate the Due Process Clause of the Fourteenth Amendment, which protects against state action the right to privacy, including a woman's qualified right to terminate her pregnancy. Though the State cannot override that right, it has legitimate interests in protecting both the pregnant woman's health and the potentiality of human life, each of which interests grows and reaches a 'compelling' point at various stages of the woman's approach to term. Pp. 147-164. (a) For the stage prior to approximately the end of the first trimester, the abortion decision and its effectuation must be left to the medical judgment of the pregnant woman's attending physician. Pp. 163-164. (b) For the stage subsequent to approximately the end of the first trimester, the State, in promoting its interest in the health of the mother, may, if it chooses, regulate the abortion procedure in ways that are reasonably related to maternal health. Pp. 163-164. (c) For the stage subsequent to viability the State, in promoting its interest in the potentiality of human life, may, if it chooses, regulate, and even proscribe, abortion except where necessary, in appropriate medical judgment, for the preservation of the life or health of the mother. Pp. 163-164; 164—165. 4. The State may define the term 'physician' to mean only a physician currently licensed by the State, and may proscribe any abortion by a person who is not a physician as so defined. P. 165. 5. It is unnecessary to decide the injunctive relief issue since the Texas authorities will doubtless fully recognize the Court's ruling that the Texas criminal abortion statutes are unconstitutional. P. 166. 314 F.Supp. 1217, affirmed in part and reversed in part. Sarah R. Weddington, Austin, Tex., for appellants. Robert C. Flowers, Asst. Atty. Gen. of Texas, Austin, Tex., for appellee on reargument. Jay Floyd, Asst. Atty. Gen., Austin, Tex., for appellee on original argument. Mr. Justice BLACKMUN delivered the opinion of the Court."

    test_outside(ex_text_str,text_pipeline,ag_news_label,model)

    ##############################

    end_time = datetime.now().strftime("%H:%M:%S")
    print(current_time)
    print(end_time)

    return None

if __name__ == '__main__':
    main()