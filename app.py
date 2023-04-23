#coding:utf-8
from flask import Flask, render_template, request, redirect, session
import os, sys
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from deep_translator import GoogleTranslator
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

translator = GoogleTranslator()
model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2") 
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

onnx_qa = pipeline("question-answering",model=model,tokenizer=tokenizer)
filename=''
@app.route('/',methods = ['POST', 'GET'])
def index():
   global filename
   if request.method == 'POST':         
      if 'myfile' in request.files:         
         file = request.files['myfile']
         filename = secure_filename(file.filename)
         basedir = os.path.abspath(os.path.dirname(__file__))
         file.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename))
         document_path = os.getcwd()+'\\static\\uploads\\'+filename
         
         with open(document_path, 'r',encoding="utf-8") as f:
            file_content = f.read()
            getfile_content=file_content
            return render_template('index.html',content=file_content)
     
      if 'myfile' not in request.files:
         getfile_content=""
         document_path = os.getcwd()+'\\static\\uploads\\'+filename
         with open(document_path, 'r',encoding="utf-8") as f:
            file_content = f.read()
            getfile_content=file_content
         myqq = request.form['myqq']         
         myq=GoogleTranslator(source='auto', target='en').translate(myqq)         
         myc=GoogleTranslator(source='auto', target='en').translate(getfile_content)
         result = onnx_qa(myq, myc)
         myET=result['answer']         
         myGF=GoogleTranslator(source='auto', target='zh-TW').translate(myET)
         return render_template('index.html',content=getfile_content,myans=myGF)      
   else:
      return render_template('index.html')
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)