"""
مصنف AraBERT المتقدم
AraBERT Advanced Classifier

يستخدم نموذج AraBERT المدرب مسبقاً مع Fine-tuning
للتصنيف الدقيق للموضوعات والشخصيات
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# إعداد الترميز
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class AraBERTClassifier:
    """
    مصنف متقدم يستخدم AraBERT
    يدعم: التصنيف الموضوعي + تصنيف الشخصيات
    """
    
    # تعريف الفئات
    TOPIC_LABELS = {
        0: {'id': 'tech_programming', 'ar': 'التقنيات والبرمجة', 'en': 'Technology & Programming'},
        1: {'id': 'academic_cultural', 'ar': 'الأكاديمي والثقافي', 'en': 'Academic & Cultural'},
        2: {'id': 'lifestyle_health', 'ar': 'نمط الحياة والصحة', 'en': 'Lifestyle & Health'},
        3: {'id': 'business_economy', 'ar': 'الأعمال والاقتصاد', 'en': 'Business & Economy'},
        4: {'id': 'creative_language', 'ar': 'الإبداع واللغة', 'en': 'Creative & Language'},
    }
    
    PERSONA_LABELS = {
        0: {'id': 'learner', 'ar': 'المتعلم', 'en': 'The Learner'},
        1: {'id': 'doer', 'ar': 'المُنجِز', 'en': 'The Doer'},
        2: {'id': 'explorer', 'ar': 'المستكشف', 'en': 'The Explorer'},
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        تهيئة المصنف
        
        Args:
            model_path: مسار النموذج المُدرَّب (اختياري)
        """
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'trained_models'
        )
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
        
        # النموذج الأساسي
        self.base_model_name = "aubmindlab/bert-base-arabertv2"
    
    def _check_dependencies(self) -> bool:
        """التحقق من المتطلبات"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    def load_model(self, task: str = 'topic') -> bool:
        """
        تحميل النموذج
        
        Args:
            task: 'topic' أو 'persona'
            
        Returns:
            True إذا نجح التحميل
        """
        if not self._check_dependencies():
            print("❌ يرجى تثبيت المتطلبات:")
            print("   pip install transformers torch")
            return False
        
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # تحديد الجهاز
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 الجهاز: {self.device}")
        
        # مسار النموذج المُدرَّب
        trained_path = os.path.join(self.model_path, f'{task}_classifier')
        
        if os.path.exists(trained_path):
            # تحميل النموذج المُدرَّب
            print(f"📂 تحميل النموذج المُدرَّب من: {trained_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(trained_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(trained_path)
        else:
            # تحميل النموذج الأساسي
            print(f"📥 تحميل AraBERT الأساسي...")
            num_labels = 5 if task == 'topic' else 3
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=num_labels
            )
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
        print("✅ تم تحميل النموذج بنجاح!")
        return True
    
    def classify(self, text: str, task: str = 'topic') -> Dict:
        """
        تصنيف نص واحد
        
        Args:
            text: النص المُدخل
            task: 'topic' أو 'persona'
            
        Returns:
            نتيجة التصنيف
        """
        if not self.is_loaded:
            if not self.load_model(task):
                return self._fallback_classify(text, task)
        
        import torch
        
        # تحويل النص
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(self.device)
        
        # التنبؤ
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # تحويل النتيجة
        labels = self.TOPIC_LABELS if task == 'topic' else self.PERSONA_LABELS
        label_info = labels.get(pred_class, labels[0])
        
        # جميع الاحتمالات
        all_probs = {
            labels[i]['id']: round(probs[0][i].item(), 3)
            for i in range(len(labels))
        }
        
        return {
            'category_id' if task == 'topic' else 'persona_id': label_info['id'],
            'category_ar' if task == 'topic' else 'persona_ar': label_info['ar'],
            'category_en' if task == 'topic' else 'persona_en': label_info['en'],
            'confidence': round(confidence, 3),
            'all_scores': all_probs,
            'is_confident': confidence >= 0.3,
            'model': 'arabert'
        }
    
    def _fallback_classify(self, text: str, task: str) -> Dict:
        """تصنيف احتياطي بدون النموذج"""
        # استخدام المصنف القديم
        if task == 'topic':
            from .topic_classifier import TopicClassifier
            classifier = TopicClassifier()
            return classifier.classify(text)
        else:
            from .persona_classifier import PersonaClassifier
            classifier = PersonaClassifier()
            return classifier.classify(text)
    
    def classify_batch(self, texts: List[str], task: str = 'topic') -> List[Dict]:
        """تصنيف مجموعة نصوص"""
        return [self.classify(text, task) for text in texts]


class AraBERTTrainer:
    """
    مدرب AraBERT للتصنيف
    يستخدم بيانات CLINC-150 أو أي بيانات مُصنَّفة
    """
    
    def __init__(self, output_dir: str = None):
        """
        تهيئة المدرب
        
        Args:
            output_dir: مجلد حفظ النموذج
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), 'trained_models'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.base_model_name = "aubmindlab/bert-base-arabertv2"
        self.device = None
    
    def prepare_data(self, data_file: str, task: str = 'topic') -> Tuple:
        """
        تحضير البيانات للتدريب
        
        Args:
            data_file: ملف البيانات (JSON)
            task: 'topic' أو 'persona'
            
        Returns:
            (texts, labels)
        """
        print(f"📂 قراءة البيانات من: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        # تعريف التسميات
        label_map = {
            'topic': {
                'tech_programming': 0, 'technology': 0, 'programming': 0, 'code': 0,
                'academic_cultural': 1, 'academic': 1, 'cultural': 1, 'education': 1,
                'lifestyle_health': 2, 'lifestyle': 2, 'health': 2, 'food': 2,
                'business_economy': 3, 'business': 3, 'economy': 3, 'finance': 3,
                'creative_language': 4, 'creative': 4, 'language': 4, 'writing': 4,
            },
            'persona': {
                'learner': 0, 'learn': 0, 'understand': 0, 'explain': 0,
                'doer': 1, 'do': 1, 'create': 1, 'write': 1, 'make': 1,
                'explorer': 2, 'compare': 2, 'suggest': 2, 'recommend': 2,
            }
        }
        
        current_map = label_map.get(task, label_map['topic'])
        
        for item in data:
            # استخراج النص
            text = None
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get('text') or item.get('prompt') or \
                       item.get('prompts_ar', [''])[0] if isinstance(item.get('prompts_ar'), list) else item.get('prompts_ar')
            
            if not text or len(str(text).strip()) < 5:
                continue
            
            # استخراج التسمية
            label = None
            if isinstance(item, dict):
                label_field = item.get('topic') or item.get('label') or item.get('category') or item.get('intent')
                if label_field:
                    label_lower = str(label_field).lower()
                    label = current_map.get(label_lower)
            
            if label is not None:
                texts.append(str(text))
                labels.append(label)
        
        print(f"✅ تم تحضير {len(texts)} مثال")
        return texts, labels
    
    def train(self, texts: List[str], labels: List[int], 
              task: str = 'topic', epochs: int = 3, batch_size: int = 16):
        """
        تدريب النموذج مع تقييم شامل
        
        Args:
            texts: قائمة النصوص
            labels: قائمة التسميات
            task: 'topic' أو 'persona'
            epochs: عدد الحقب
            batch_size: حجم الدفعة
        """
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import AdamW, get_linear_schedule_with_warmup
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            from tqdm import tqdm
            import numpy as np
        except ImportError:
            print("❌ يرجى تثبيت المتطلبات:")
            print("   pip install transformers torch tqdm scikit-learn")
            return
        
        print("=" * 60)
        print(f"🚀 بدء تدريب نموذج {task.upper()}")
        print("=" * 60)
        
        # الجهاز
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 الجهاز: {self.device}")
        
        # عدد الفئات وأسماؤها
        num_labels = 5 if task == 'topic' else 3
        label_names = [self.TOPIC_LABELS[i]['ar'] for i in range(num_labels)] if task == 'topic' else \
                      [self.PERSONA_LABELS[i]['ar'] for i in range(num_labels)]
        
        # تحميل Tokenizer والنموذج
        print("📥 تحميل AraBERT...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels
        )
        model.to(self.device)
        
        # تحضير البيانات
        print("📊 تحضير البيانات...")
        
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True, 
                    max_length=max_length, return_tensors='pt'
                )
                self.labels = torch.tensor(labels)
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item
        
        # تقسيم البيانات (80% تدريب، 10% تحقق، 10% اختبار)
        n = len(texts)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train_texts, train_labels = texts[:train_end], labels[:train_end]
        val_texts, val_labels = texts[train_end:val_end], labels[train_end:val_end]
        test_texts, test_labels = texts[val_end:], labels[val_end:]
        
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"   بيانات التدريب: {len(train_dataset)}")
        print(f"   بيانات التحقق: {len(val_dataset)}")
        print(f"   بيانات الاختبار: {len(test_dataset)}")
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # التدريب
        print("\n🎯 بدء التدريب...")
        best_f1 = 0
        training_history = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            
            # التحقق
            val_metrics = self._evaluate(model, val_loader, num_labels)
            
            print(f"\n📊 Epoch {epoch+1}:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Accuracy: {val_metrics['accuracy']*100:.2f}%")
            print(f"   F1-Score: {val_metrics['f1']*100:.2f}%")
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                **val_metrics
            })
            
            # حفظ أفضل نموذج (بناءً على F1)
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_path = os.path.join(self.output_dir, f'{task}_classifier')
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"   💾 تم حفظ أفضل نموذج (F1={val_metrics['f1']*100:.2f}%)")
        
        # ===================== التقييم النهائي =====================
        print("\n" + "=" * 60)
        print("📈 التقييم النهائي على بيانات الاختبار")
        print("=" * 60)
        
        # تحميل أفضل نموذج للتقييم
        best_model_path = os.path.join(self.output_dir, f'{task}_classifier')
        if os.path.exists(best_model_path):
            model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            model.to(self.device)
        
        # التقييم على بيانات الاختبار
        test_metrics = self._evaluate(model, test_loader, num_labels, return_predictions=True)
        
        # طباعة المقاييس
        print("\n📊 مقاييس الأداء:")
        print("-" * 40)
        print(f"   ✅ Accuracy (الدقة الإجمالية): {test_metrics['accuracy']*100:.2f}%")
        print(f"   📌 Precision (الدقة): {test_metrics['precision']*100:.2f}%")
        print(f"   🔍 Recall (الاستدعاء): {test_metrics['recall']*100:.2f}%")
        print(f"   ⚖️ F1-Score (المتوسط التوافقي): {test_metrics['f1']*100:.2f}%")
        
        # مصفوفة الارتباك
        print("\n📋 مصفوفة الارتباك (Confusion Matrix):")
        print("-" * 40)
        cm = test_metrics['confusion_matrix']
        self._print_confusion_matrix(cm, label_names)
        
        # تقرير التصنيف
        print("\n📝 تقرير التصنيف التفصيلي:")
        print("-" * 40)
        print(test_metrics['classification_report'])
        
        # حفظ التقرير
        report_path = os.path.join(self.output_dir, f'{task}_evaluation_report.json')
        self._save_evaluation_report(test_metrics, training_history, report_path)
        print(f"\n💾 تم حفظ تقرير التقييم في: {report_path}")
        
        # رسم مصفوفة الارتباك
        try:
            cm_image_path = os.path.join(self.output_dir, f'{task}_confusion_matrix.png')
            self._plot_confusion_matrix(cm, label_names, cm_image_path)
            print(f"📊 تم حفظ رسم مصفوفة الارتباك في: {cm_image_path}")
        except Exception as e:
            print(f"⚠️ لم يتم رسم مصفوفة الارتباك: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ اكتمل التدريب والتقييم!")
        print(f"   أفضل F1-Score: {best_f1*100:.2f}%")
        print(f"📂 النموذج محفوظ في: {self.output_dir}/{task}_classifier")
        print("=" * 60)
        
        return test_metrics
    
    def _evaluate(self, model, data_loader, num_labels, return_predictions=False) -> Dict:
        """
        تقييم النموذج مع حساب جميع المقاييس
        """
        import torch
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        import numpy as np
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # حساب المقاييس
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        if return_predictions:
            label_names = [self.TOPIC_LABELS[i]['ar'] for i in range(num_labels)] if num_labels == 5 else \
                         [self.PERSONA_LABELS[i]['ar'] for i in range(num_labels)]
            
            metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
            metrics['classification_report'] = classification_report(
                all_labels, all_preds, 
                target_names=label_names,
                zero_division=0
            )
            metrics['predictions'] = all_preds.tolist()
            metrics['true_labels'] = all_labels.tolist()
        
        return metrics
    
    def _print_confusion_matrix(self, cm, labels):
        """طباعة مصفوفة الارتباك بشكل منسق"""
        # طباعة رأس الجدول
        header = "        " + "  ".join([f"{l[:8]:>8}" for l in labels])
        print(header)
        print("        " + "-" * (len(labels) * 10))
        
        for i, row in enumerate(cm):
            row_label = f"{labels[i][:8]:>8}"
            row_values = "  ".join([f"{v:>8}" for v in row])
            print(f"{row_label} | {row_values}")
    
    def _save_evaluation_report(self, metrics, history, path):
        """حفظ تقرير التقييم في ملف JSON"""
        report = {
            'final_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1'])
            },
            'confusion_matrix': metrics['confusion_matrix'].tolist() if hasattr(metrics['confusion_matrix'], 'tolist') else metrics['confusion_matrix'],
            'training_history': history
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def _plot_confusion_matrix(self, cm, labels, save_path):
        """رسم وحفظ مصفوفة الارتباك"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # للعمل بدون واجهة
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('مصفوفة الارتباك (Confusion Matrix)', fontsize=14)
            plt.colorbar()
            
            tick_marks = range(len(labels))
            plt.xticks(tick_marks, labels, rotation=45, ha='right')
            plt.yticks(tick_marks, labels)
            
            # إضافة الأرقام داخل الخلايا
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('القيمة الحقيقية')
            plt.xlabel('القيمة المتوقعة')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("⚠️ matplotlib غير مثبت. لن يتم رسم مصفوفة الارتباك.")


# ======================= بيانات تدريب تجريبية =======================

def create_sample_training_data():
    """
    إنشاء بيانات تدريب تجريبية
    يمكن استبدالها ببيانات CLINC-150 المترجمة
    """
    sample_data = [
        # التقنيات والبرمجة
        {"text": "كيف أكتب برنامج بايثون؟", "topic": "tech_programming", "persona": "learner"},
        {"text": "اشرح لي خوارزمية الترتيب السريع", "topic": "tech_programming", "persona": "learner"},
        {"text": "ما هو الفرق بين Java و Python؟", "topic": "tech_programming", "persona": "explorer"},
        {"text": "اكتب لي كود لحساب المتوسط", "topic": "tech_programming", "persona": "doer"},
        {"text": "كيف أنشئ قاعدة بيانات MySQL؟", "topic": "tech_programming", "persona": "learner"},
        {"text": "ما هو الذكاء الاصطناعي؟", "topic": "tech_programming", "persona": "learner"},
        {"text": "كيف أتعلم تطوير تطبيقات الموبايل؟", "topic": "tech_programming", "persona": "learner"},
        {"text": "أصلح هذا الكود البرمجي", "topic": "tech_programming", "persona": "doer"},
        
        # الأكاديمي والثقافي
        {"text": "ما هي عاصمة اليمن؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "اشرح نظرية النسبية", "topic": "academic_cultural", "persona": "learner"},
        {"text": "من هو أينشتاين؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "ما هي أسباب الحرب العالمية الثانية؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "لماذا السماء زرقاء؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "ما الفرق بين الشعر والنثر؟", "topic": "academic_cultural", "persona": "explorer"},
        {"text": "كم عدد سكان مصر؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "ما هي الديمقراطية؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "أين تقع جبال الهملايا؟", "topic": "academic_cultural", "persona": "learner"},
        {"text": "ما معنى كلمة فلسفة؟", "topic": "academic_cultural", "persona": "learner"},
        
        # نمط الحياة والصحة
        {"text": "ما هي أفضل وصفة للكبسة؟", "topic": "lifestyle_health", "persona": "learner"},
        {"text": "كيف أخسر وزني؟", "topic": "lifestyle_health", "persona": "learner"},
        {"text": "ما هي فوائد الرياضة؟", "topic": "lifestyle_health", "persona": "learner"},
        {"text": "اقترح لي وجبة صحية", "topic": "lifestyle_health", "persona": "explorer"},
        {"text": "كيف أنام بشكل أفضل؟", "topic": "lifestyle_health", "persona": "learner"},
        {"text": "ما أفضل أماكن السياحة في دبي؟", "topic": "lifestyle_health", "persona": "explorer"},
        
        # الأعمال والاقتصاد
        {"text": "كيف أبدأ مشروعي الخاص؟", "topic": "business_economy", "persona": "learner"},
        {"text": "ما هو التسويق الرقمي؟", "topic": "business_economy", "persona": "learner"},
        {"text": "كيف أستثمر أموالي؟", "topic": "business_economy", "persona": "learner"},
        {"text": "اكتب لي خطة عمل", "topic": "business_economy", "persona": "doer"},
        {"text": "ما الفرق بين الأسهم والسندات؟", "topic": "business_economy", "persona": "explorer"},
        
        # الإبداع واللغة
        {"text": "اكتب لي قصة قصيرة", "topic": "creative_language", "persona": "doer"},
        {"text": "ترجم هذا النص للإنجليزية", "topic": "creative_language", "persona": "doer"},
        {"text": "صحح الأخطاء الإملائية", "topic": "creative_language", "persona": "doer"},
        {"text": "اكتب شعراً عن الحب", "topic": "creative_language", "persona": "doer"},
        {"text": "كيف أحسن أسلوب كتابتي؟", "topic": "creative_language", "persona": "learner"},
    ]
    
    return sample_data


# ======================= نقطة الدخول =======================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AraBERT Classifier')
    parser.add_argument('--train', action='store_true', help='تدريب النموذج')
    parser.add_argument('--test', type=str, help='نص للاختبار')
    parser.add_argument('--data', type=str, help='ملف بيانات التدريب')
    parser.add_argument('--task', type=str, default='topic', choices=['topic', 'persona'])
    parser.add_argument('--epochs', type=int, default=3)
    
    args = parser.parse_args()
    
    if args.train:
        trainer = AraBERTTrainer()
        
        if args.data:
            texts, labels = trainer.prepare_data(args.data, args.task)
        else:
            print("📝 استخدام البيانات التجريبية...")
            data = create_sample_training_data()
            texts = [d['text'] for d in data]
            labels = [
                {'tech_programming': 0, 'academic_cultural': 1, 'lifestyle_health': 2, 
                 'business_economy': 3, 'creative_language': 4}[d['topic']]
                for d in data
            ]
        
        trainer.train(texts, labels, task=args.task, epochs=args.epochs)
    
    elif args.test:
        classifier = AraBERTClassifier()
        result = classifier.classify(args.test, args.task)
        
        print("\n" + "=" * 50)
        print(f"📝 النص: {args.test}")
        print(f"🏷️ التصنيف: {result.get('category_ar') or result.get('persona_ar')}")
        print(f"📊 الثقة: {result['confidence']*100:.1f}%")
        print("=" * 50)
    
    else:
        # اختبار سريع
        print("🧪 اختبار AraBERT Classifier")
        print("=" * 50)
        
        classifier = AraBERTClassifier()
        
        test_texts = [
            "ما هي عاصمة اليمن؟",
            "كيف أكتب برنامج بايثون؟",
            "اكتب لي قصة قصيرة",
        ]
        
        for text in test_texts:
            result = classifier.classify(text, 'topic')
            print(f"\n📝 {text}")
            print(f"   🏷️ {result.get('category_ar', 'N/A')}")
            print(f"   📊 {result['confidence']*100:.1f}%")
