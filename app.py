#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تطبيق ويب بسيط لتحليل المشاعر باستخدام Flask
Simple web application for sentiment analysis using Flask
"""

import os
import json
import torch
from flask import Flask, request, jsonify, render_template_string
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# تحميل النموذج والمحلل اللغوي
model_path = "./models/Sentiment_Model_Final"
tokenizer = None
model = None

def load_model():
    """تحميل النموذج المدرب"""
    global tokenizer, model
    try:
        if os.path.exists(model_path):
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            print("تم تحميل النموذج المدرب بنجاح!")
        else:
            # استخدام النموذج الأساسي إذا لم يكن النموذج المدرب متوفراً
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-multilingual-cased", 
                num_labels=3
            )
            print("تم تحميل النموذج الأساسي (غير مدرب)")
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {e}")
        return False
    return True

def classify_sentiment(text):
    """تصنيف المشاعر للنص المدخل"""
    if not tokenizer or not model:
        return {"error": "النموذج غير محمل"}
    
    try:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)[0]
        
        label_map = {0: 'سلبي', 1: 'محايد', 2: 'إيجابي'}
        label_map_en = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        confidence = probabilities[prediction].item()
        
        return {
            'text': text,
            'label_ar': label_map[prediction],
            'label_en': label_map_en[prediction],
            'confidence': round(confidence, 3),
            'probabilities': {
                'negative': round(probabilities[0].item(), 3),
                'neutral': round(probabilities[1].item(), 3),
                'positive': round(probabilities[2].item(), 3)
            }
        }
    except Exception as e:
        return {"error": f"خطأ في التصنيف: {str(e)}"}

# HTML template للواجهة
HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تحليل المشاعر - Sentiment Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .content {
            padding: 30px;
        }
        textarea {
            width: 100%;
            height: 120px;
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
            width: 100%;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .positive { background: #d4edda; border-left: 5px solid #28a745; }
        .negative { background: #f8d7da; border-left: 5px solid #dc3545; }
        .neutral { background: #fff3cd; border-left: 5px solid #ffc107; }
        .loading {
            text-align: center;
            color: #666;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 تحليل المشاعر</h1>
            <p>نموذج BERT متعدد اللغات لتحليل المشاعر في النصوص العربية والإنجليزية</p>
        </div>
        <div class="content">
            <form id="sentimentForm">
                <label for="text">أدخل النص للتحليل:</label>
                <textarea id="text" placeholder="اكتب النص هنا... أو Write your text here..."></textarea>
                <button type="submit">تحليل المشاعر</button>
            </form>
            
            <div class="loading" id="loading">
                <p>⏳ جاري التحليل...</p>
            </div>
            
            <div class="result" id="result">
                <h3>نتيجة التحليل:</h3>
                <div id="resultContent"></div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('sentimentForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('text').value.trim();
            if (!text) {
                alert('يرجى إدخال نص للتحليل');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                let className = 'neutral';
                if (data.label_en === 'Positive') className = 'positive';
                else if (data.label_en === 'Negative') className = 'negative';
                
                resultDiv.className = 'result ' + className;
                
                resultContent.innerHTML = `
                    <p><strong>النص:</strong> ${data.text}</p>
                    <p><strong>التصنيف:</strong> ${data.label_ar} (${data.label_en})</p>
                    <p><strong>مستوى الثقة:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                    <hr>
                    <p><strong>التفاصيل:</strong></p>
                    <ul>
                        <li>إيجابي: ${(data.probabilities.positive * 100).toFixed(1)}%</li>
                        <li>محايد: ${(data.probabilities.neutral * 100).toFixed(1)}%</li>
                        <li>سلبي: ${(data.probabilities.negative * 100).toFixed(1)}%</li>
                    </ul>
                `;
                
                resultDiv.style.display = 'block';
                
            } catch (error) {
                alert('حدث خطأ: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """تحليل المشاعر للنص المرسل"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "النص مطلوب"}), 400
        
        result = classify_sentiment(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """فحص حالة الخادم"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    })

if __name__ == '__main__':
    print("🚀 بدء تشغيل خادم تحليل المشاعر...")
    
    # تحميل النموذج
    if load_model():
        print("✅ تم تحميل النموذج بنجاح")
    else:
        print("❌ فشل في تحميل النموذج")
    
    # تشغيل الخادم
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
