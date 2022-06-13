from flask import Flask, render_template, request	 # 플라스크 모듈 호출

from db import MyDao
from flask.json import jsonify

# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import gluonnlp as nlp
# import numpy as np
# from tqdm.notebook import tqdm

# from kobert import get_tokenizer
# from kobert import get_pytorch_kobert_model

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

import unicodedata
import re

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)
    
## CPU
device = torch.device("cpu")

ko_model = torch.load('model.pt', map_location=device) # input으로 저장된 디렉토리만 지정하면 완료

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

max_len = 100
batch_size = 64

def getSentimentValue(comment):
  commnetslist = [] # 텍스트 데이터를 담을 리스트
  res_list = [] # 결과 값을 담을 리스트
  for c in comment: # 모든 댓글
    commnetslist.append( [c, 5] ) # [댓글, 임의의 양의 정수값] 설정
    
  pdData = pd.DataFrame( commnetslist, columns = [['text', 'label']] )
  pdData = pdData.values
  test_set = BERTDataset(pdData, 0, 1, tok, max_len, True, False) 
  test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0)
  
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length 
    # 이때, out이 예측 결과 리스트
    out = ko_model(token_ids, valid_length, segment_ids)
	
    # e는 2가지 실수 값으로 구성된 리스트
    # 0번 인덱스가 더 크면 일반, 광고는 반대
    for e in out:
      if e[0]>e[1]: # 부정
        value = 0
      else: #긍정(광고)
        value = 1
      res_list.append(value)

  return res_list # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환

# kobert 전처리
def process_data(result):
    result = pd.DataFrame([result], columns=['text'])
    result['text']=result['text'].apply(lambda x:' '.join(x.strip('[""').strip('""]').strip("['").strip("\\n']").split("\\n', '")).replace("\\n","") )
    result['text']=result['text'].apply(lambda x: unicodedata.normalize('NFC',x ))
    result['text']=result['text'].apply(lambda x: re.sub('[^A-Za-z0-9ㄱ-ㅣ가-힣 #.?!@]','',x))
    result['text']=result['text'].apply(lambda x: re.sub('[ㄱ-ㅣ]',' ',x))
    result['text']=result['text'].apply(lambda x: x.replace("u200a",""))
    result['text']=result['text'].apply(lambda x: x.replace("u200b",""))
    result['text']=result['text'].apply(lambda x: x.replace("u200c",""))
    result['text']=result['text'].apply(lambda x: x.replace("u200d",""))
    result['text']=result['text'].apply(lambda x: x.replace("u200e",""))
    result['text']=result['text'].apply(lambda x: x.replace("u200f",""))
    result['text']=result['text'].apply(lambda x: x.replace("#",' ').replace(".",' ').replace("?",' ').replace("!",' ').replace("@",' '))
    result['text']=result['text'].apply(lambda x: re.sub(' +',' ',x.replace("\n"," ")))
    result = result['text'].values.tolist()
    return result

app = Flask(__name__) 		 # 플라스크 앱 생성

@app.route('/')				 # 기본('/') 웹주소로 요청이 오면 
def home():
    noelist = MyDao().getEmps();
    return render_template('home.html',noelist=noelist)

# 글 추가
@app.route('/ins.ajax', methods=['GET', 'POST'])
def ins_ajax():
    data = request.get_json()
    title = data['title']
    context = data['context']
    test_ = process_data(context)
    result_ = getSentimentValue(test_)
    print(">>>>>>>>>>>>>>>>>>>>>",result_)
    
    adv_yn = str(result_)
    print(">>", type(adv_yn), adv_yn)
    
    if adv_yn == "[1]":
        adv_yn = "광고"
        print(">>", adv_yn)
        
    if adv_yn == "[0]":
        adv_yn = ""
        print(">>", adv_yn)
        
    cnt = MyDao().insEmp(title, context, adv_yn)
    result = "success" if cnt==1 else "fail"
    return jsonify(result = result)

# 글쓰기 이동
@app.route('/home_write')
def home_write():
    return render_template('write.html');

# big html 이동
@app.route('/home_big', methods=['GET'])
def home_big():
    num = '%s' %request.args.get('num')
    noe = MyDao().getEmpss(num);
    ans = MyDao().getAnss(num);
    return render_template('big.html', noe=noe, ans = ans);


# 댓글 추가
@app.route('/ans_ins.ajax', methods=['GET', 'POST'])
def ans_ins_ajax():
    data = request.get_json()
    num = data['num']
    ans = data['ans']
    cnt = MyDao().insAns(num, ans)
    result = "success" if cnt==1 else "fail"
    return jsonify(result = result)

		
if __name__ == '__main__':	 # main함수
    app.run(debug=True, port=5000, host='0.0.0.0')

# ctlr + 5 -> localhost:5000 접속
