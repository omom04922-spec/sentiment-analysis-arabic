# نموذج تحليل المشاعر للنصوص العربية والإنجليزية

## 📝 الوصف
هذا المشروع يستخدم نموذج BERT متعدد اللغات لتحليل المشاعر في النصوص العربية والإنجليزية. النموذج قادر على تصنيف النصوص إلى ثلاث فئات: إيجابي، سلبي، ومحايد.

## 🚀 المميزات
- دعم اللغتين العربية والإنجليزية
- استخدام نموذج BERT متعدد اللغات
- تدريب مخصص على بيانات المحادثات
- واجهة سهلة الاستخدام
- متوافق مع Google Colab

## 📋 المتطلبات
```bash
pip install -r requirements.txt
```

## 🔧 التثبيت والاستخدام

### في Google Colab:
1. افتح الملف `Sentiment_Analysis.ipynb` في Google Colab
2. قم بتشغيل الخلايا بالترتيب
3. سيتم تحميل Google Drive تلقائياً

### في البيئة المحلية:
1. تأكد من تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```
2. افتح الملف `Sentiment_Analysis.ipynb` في Jupyter Notebook
3. قم بتشغيل الخلايا بالترتيب

## 📊 البيانات
يستخدم المشروع ملفات JSONL تحتوي على محادثات باللغة العربية. يتم تحليل المشاعر بناءً على:
- الكلمات المفتاحية الإيجابية والسلبية
- السياق العام للنص

## 🤖 النموذج
- **النموذج الأساسي**: `bert-base-multilingual-cased`
- **عدد التصنيفات**: 3 (إيجابي، سلبي، محايد)
- **طول النص الأقصى**: 128 token

## 📈 الأداء
يتم تقييم النموذج باستخدام:
- دقة التصنيف (Accuracy)
- F1-Score

## 🔄 استكمال التدريب

### من نقطة التوقف:
```python
trainer.train(resume_from_checkpoint="path/to/checkpoint")
```

### تحميل نموذج محفوظ:
```python
model = BertForSequenceClassification.from_pretrained("path/to/checkpoint")
```

## 🎯 الاستخدام

```python
# تصنيف نص واحد
result = classify_sentiment("هذا المنتج رائع جداً")
print(f"التصنيف: {result['label']}")
print(f"الثقة: {result['confidence']:.2f}")
```

## 📁 هيكل المشروع
```
├── Sentiment_Analysis.ipynb    # الملف الرئيسي
├── requirements.txt           # المتطلبات
├── README.md                 # هذا الملف
└── data/                     # مجلد البيانات (JSONL files)
```

## 🤝 المساهمة
نرحب بالمساهمات! يرجى:
1. عمل Fork للمشروع
2. إنشاء branch جديد للميزة
3. إرسال Pull Request

## 📄 الترخيص
هذا المشروع مرخص تحت رخصة MIT.

## 📞 التواصل
للاستفسارات والدعم، يرجى فتح issue في المستودع.

---

# Arabic & English Sentiment Analysis Model

## 📝 Description
This project uses a multilingual BERT model for sentiment analysis in Arabic and English texts. The model can classify texts into three categories: positive, negative, and neutral.

## 🚀 Features
- Support for Arabic and English languages
- Uses multilingual BERT model
- Custom training on conversational data
- Easy-to-use interface
- Google Colab compatible

## 🔧 Installation & Usage

### In Google Colab:
1. Open `Sentiment_Analysis.ipynb` in Google Colab
2. Run cells in order
3. Google Drive will be mounted automatically

### Local Environment:
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Open `Sentiment_Analysis.ipynb` in Jupyter Notebook
3. Run cells in order

## 🎯 Usage Example

```python
# Classify a single text
result = classify_sentiment("This product is amazing!")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 📊 Model Performance
The model is evaluated using:
- Accuracy
- F1-Score

## 🤝 Contributing
Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch
3. Submit a Pull Request

## 📄 License
This project is licensed under the MIT License.