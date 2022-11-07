# demo web application

from flask import Flask, render_template, request, redirect, session, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import json
import torch
import torchvision.transforms as transforms
from konlpy.tag import Okt
import jpype
import hashlib
import os
from models.show_att import Encoder, Decoder
from models.resnext_lb import ResNextEncoder, LookBackDecoder
from utils import OktDetokenizer, load_model

from config import Config

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)
app.config['SECRET_KEY'] = "LJDSLFJDLjsdafasdDFJSALFJdsDS"

okt = Okt()
detok = OktDetokenizer()

HASHTAG_VOCAB_PATH = "vocabs/hashtag_vocab_1.json"
HASHTAG_MODEL_PATH = "checkpoint/resnext_hashtag_1.pth"

TEXT_VOCAB_PATH = "vocabs/text_vocab_1.json"
TEXT_MODEL_PATH = "checkpoint/resnext_text_1.pth"

# load hashtag vocab and model
# load vocab
with open(HASHTAG_VOCAB_PATH) as fr:
    hashtag_vocab = json.load(fr)
    hashtag_idx2vocab = dict([(v, k) for k, v in hashtag_vocab.items()])
    
# load model
model_filename = HASHTAG_MODEL_PATH.split('/')[-1]
print(model_filename)
if model_filename.startswith('showatt'):
    hashtag_encoder = Encoder(Config.encoded_size)
    hashtag_decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(hashtag_vocab))
elif model_filename.startswith('resnext_lb'):
    hashtag_encoder = ResNextEncoder(Config.encoded_size)
    hashtag_decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(hashtag_vocab))
elif model_filename.startswith('resnext'):
    hashtag_encoder = ResNextEncoder(Config.encoded_size)
    hashtag_decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(hashtag_vocab))
else:
    # lb
    hashtag_encoder = Encoder(Config.encoded_size)
    hashtag_decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(hashtag_vocab))

load_model(hashtag_encoder, hashtag_decoder, HASHTAG_MODEL_PATH)
hashtag_encoder.eval()
hashtag_decoder.eval()
# load text vocab and model
# load vocab
with open(TEXT_VOCAB_PATH) as fr:
    text_vocab = json.load(fr)
    text_idx2vocab = dict([(v, k) for k, v in text_vocab.items()])
    
# load model
model_filename = TEXT_MODEL_PATH.split('/')[-1]
print(model_filename)
if model_filename.startswith('showatt'):
    text_encoder = Encoder(Config.encoded_size)
    text_decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(text_vocab))
elif model_filename.startswith('resnext_lb'):
    text_encoder = ResNextEncoder(Config.encoded_size)
    text_decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(text_vocab))
elif model_filename.startswith('resnext'):
    text_encoder = ResNextEncoder(Config.encoded_size)
    text_decoder = Decoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(text_vocab))
else:
    # lb
    text_encoder = Encoder(Config.encoded_size)
    text_decoder = LookBackDecoder(Config.encoder_dim, Config.decoder_dim, Config.attention_dim, Config.embed_dim, len(text_vocab))

load_model(text_encoder, text_decoder, TEXT_MODEL_PATH)
text_encoder.eval()
text_decoder.eval()

target_size = 224
image_to_tensor = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])


print("Model loaded...")

UPLOAD_FOLDER = "images/"
if not os.path.isdir(UPLOAD_FOLDER):
    os.path.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def index():
    # 업로드받기
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('Choose image file (.jpg, .jpeg, .png) to upload.')
            return redirect(request.url)
        file = request.files['image']
        if not allowed_file(file.filename):
                flash('File extension not allowed')
                return redirect(request.url)
            
        # 파일 업로드됨
        filename = file.filename
        file_format = '.' + filename.split('.')[-1]
        filename = secure_filename(file.filename)
        hashed_filename = hashlib.sha224(filename.encode()).hexdigest() + file_format
        while os.path.isfile(os.path.join(UPLOAD_FOLDER, hashed_filename)):
            hashed_filename = hashlib.sha224(hashed_filename.encode()).hexdigest() + file_format
        
        file.save(os.path.join(UPLOAD_FOLDER, hashed_filename))
        
        # 이미지를 열어서, 리사이즈 후 저장
        image = Image.open(os.path.join(UPLOAD_FOLDER, hashed_filename))
        transformed = transforms.CenterCrop(min(image.size))(image)
        transformed.save(os.path.join(UPLOAD_FOLDER, hashed_filename))
        
        image.close()
        transformed.close()
        
        return redirect(url_for('view_result',
                                img_filename=hashed_filename))
            
    return render_template('mainpage.html')


@app.route('/result/<img_filename>')
def view_result(img_filename):
    # TODO: 이미지 파일 불러와서, inference 진행하고, 결과 보여주기!
    jpype.attachThreadToJVM()
    
    text_encoder.eval()
    text_decoder.eval()
    hashtag_encoder.eval()
    hashtag_decoder.eval()
    
    image = Image.open(os.path.join(UPLOAD_FOLDER, img_filename)).convert('RGB')
    tensor = image_to_tensor(image)
    tensor = tensor.unsqueeze(0)
    
    # 텍스트 생성
    encoded_out = text_encoder(tensor)
    pred, alpha = text_decoder.generate_caption_greedily(encoded_out, text_vocab['<start>'], text_vocab['<end>'])
    pred = pred[1:-1]
    pred_tokens = [text_idx2vocab[idx] for idx in pred]
    
    detokenized_text = detok.detokenize(pred_tokens)
    
    # 해시태그 생성
    encoded_out = hashtag_encoder(tensor)
    pred, alpha = hashtag_decoder.generate_caption_greedily(encoded_out, hashtag_vocab['<start>'], hashtag_vocab['<end>'])
    pred = pred[1:-1]
    pred_tokens = [hashtag_idx2vocab[idx] for idx in pred]
    pred_tokens = [token if token.startswith('#') else '#' + token for token in pred_tokens]
    
    output_pred_tokens = []
    for token in pred_tokens:
        if token not in output_pred_tokens:
            output_pred_tokens.append(token)
    
    hashtag = " ".join(output_pred_tokens)
    
    
    return render_template('resultpage.html', image_url='/images/{0}'.format(img_filename), caption=detokenized_text, hashtag=hashtag)


@app.route('/images/<filename>')
def view_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/<text>')
def tokenize_and_detokenize(text):
    # 심심해서... 그냥 남겨둡니다 ㅎㅎ.
    jpype.attachThreadToJVM()

    tokenized_text = okt.pos(text, norm=True, join=True)
    detokenized_text = detok.detokenize(tokenized_text)
    
    return "Tokenization result: {0} <br /> Detokenization result: {1}".format(" ".join(tokenized_text), detokenized_text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)




