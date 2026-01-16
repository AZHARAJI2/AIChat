"""
التصنيف شبه التلقائي للبيانات
Semi-Automatic Data Labeling using Zero-Shot Classification

يستخدم نموذج facebook/bart-large-mnli لتصنيف البيانات تلقائياً
"""

import os
import sys
import json
from typing import List, Dict, Optional
from tqdm import tqdm

# إعداد الترميز
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class AutoLabeler:
    """
    مُصنِّف تلقائي باستخدام Zero-Shot Classification
    """
    
    # الفئات المُستهدفة
    CATEGORIES = {
        'tech_programming': {
            'ar': 'التقنيات والبرمجة',
            'en': 'Technology & Programming',
            'keywords': ['programming', 'code', 'software', 'python', 'java', 'database', 
                        'algorithm', 'computer', 'app', 'website', 'API', 'server',
                        'برمجة', 'كود', 'تطبيق', 'موقع', 'خوارزمية', 'قاعدة بيانات']
        },
        'academic_cultural': {
            'ar': 'الأكاديمي والثقافي',
            'en': 'Academic & Cultural',
            'keywords': ['history', 'science', 'education', 'university', 'research',
                        'geography', 'philosophy', 'culture', 'literature', 'physics',
                        'تاريخ', 'علم', 'تعليم', 'جامعة', 'بحث', 'فلسفة', 'ثقافة']
        },
        'lifestyle_health': {
            'ar': 'نمط الحياة والصحة',
            'en': 'Lifestyle & Health',
            'keywords': ['health', 'food', 'recipe', 'exercise', 'diet', 'travel',
                        'cooking', 'fitness', 'sleep', 'wellness', 'sport',
                        'صحة', 'طعام', 'وصفة', 'رياضة', 'سفر', 'نوم', 'تغذية']
        },
        'business_economy': {
            'ar': 'الأعمال والاقتصاد',
            'en': 'Business & Economy',
            'keywords': ['business', 'money', 'investment', 'marketing', 'finance',
                        'economy', 'startup', 'company', 'stock', 'budget',
                        'أعمال', 'مال', 'استثمار', 'تسويق', 'اقتصاد', 'شركة']
        },
        'creative_language': {
            'ar': 'الإبداع واللغة',
            'en': 'Creative & Language',
            'keywords': ['write', 'story', 'poem', 'translate', 'language', 'creative',
                        'song', 'lyrics', 'novel', 'essay', 'grammar',
                        'اكتب', 'قصة', 'شعر', 'ترجم', 'لغة', 'إبداع', 'رواية']
        }
    }
    
    def __init__(self, use_gpu: bool = True):
        """
        تهيئة المُصنِّف
        
        Args:
            use_gpu: استخدام GPU إذا متاح
        """
        self.classifier = None
        self.use_gpu = use_gpu
        self.device = -1  # CPU افتراضياً
        
    def _load_classifier(self):
        """تحميل نموذج Zero-Shot"""
        if self.classifier is not None:
            return
            
        print("📥 تحميل نموذج Zero-Shot Classification...")
        print("   (هذا قد يستغرق بعض الوقت في المرة الأولى)")
        
        try:
            import torch
            from transformers import pipeline
            
            # تحديد الجهاز
            if self.use_gpu and torch.cuda.is_available():
                self.device = 0
                print(f"   ✅ استخدام GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = -1
                print("   ℹ️ استخدام CPU")
            
            # تحميل النموذج
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            print("✅ تم تحميل النموذج بنجاح!")
            
        except ImportError:
            print("❌ يرجى تثبيت المتطلبات:")
            print("   pip install transformers torch")
            raise
    
    def _extract_text(self, item: Dict) -> Optional[str]:
        """استخراج النص من عنصر البيانات"""
        # محاولة استخراج النص العربي
        if 'prompts_ar' in item:
            prompts = item['prompts_ar']
            if isinstance(prompts, list) and prompts:
                text = prompts[0]
                # إزالة الأقواس والتنظيف
                if text.startswith('['):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list) and parsed:
                            return parsed[0]
                    except:
                        pass
                return text
        
        # استخدام النص الإنجليزي كبديل
        if 'prompts_en' in item:
            prompts = item['prompts_en']
            if isinstance(prompts, list) and prompts:
                text = prompts[0]
                if text.startswith('['):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list) and parsed:
                            return parsed[0]
                    except:
                        pass
                return text
        
        return None
    
    def keyword_classify_text(self, text: str) -> Dict:
        """
        تصنيف سريع باستخدام الكلمات المفتاحية
        
        Args:
            text: النص المُراد تصنيفه
            
        Returns:
            نتيجة التصنيف
        """
        text_lower = text.lower()
        scores = {cat: 0 for cat in self.CATEGORIES}
        
        for cat_id, cat_info in self.CATEGORIES.items():
            # البحث عن الكلمات المفتاحية
            for keyword in cat_info.get('keywords', []):
                if keyword.lower() in text_lower:
                    scores[cat_id] += 1
        
        # اختيار الفئة صاحبة أعلى نقاط
        best_cat = max(scores, key=scores.get)
        
        # إذا لم نجد أي كلمة مفتاحية، نختار عشوائياً أو الفئة الافتراضية
        if scores[best_cat] == 0:
            best_cat = 'academic_cultural' # افتراضي
            confidence = 0.3
        else:
            confidence = 0.8
            
        cat_info = self.CATEGORIES[best_cat]
        
        return {
            'topic': best_cat,
            'topic_ar': cat_info['ar'],
            'confidence': confidence,
            'all_scores': scores
        }
    
    def label_dataset(self, data: List[Dict], save_path: Optional[str] = None, mode: str = 'keyword') -> List[Dict]:
        """
        تصنيف مجموعة بيانات كاملة
        
        Args:
            data: قائمة البيانات
            save_path: مسار حفظ البيانات المُصنَّفة
            mode: 'model' (دقيق) أو 'keyword' (سريع)
        """
        labeled_data = []
        skipped = 0
        
        print(f"\n🏷️ بدء التصنيف (الوضع: {mode})...")
        print("=" * 50)
        
        for item in tqdm(data, desc="التصنيف"):
            text = self._extract_text(item)
            
            if not text or len(text.strip()) < 5:
                skipped += 1
                continue
            
            try:
                if mode == 'model':
                    classification = self.classify_text(text)
                else:
                    classification = self.keyword_classify_text(text)
                
                labeled_item = {
                    'id': item.get('id'),
                    'text': text,
                    'text_ar': item.get('prompts_ar', [''])[0] if 'prompts_ar' in item else text,
                    'topic': classification['topic'],
                    'topic_ar': classification['topic_ar'],
                    'confidence': classification['confidence']
                }
                
                labeled_data.append(labeled_item)
                
            except Exception as e:
                print(f"⚠️ خطأ: {e}")
                skipped += 1
                continue
        
        print("\n" + "=" * 50)
        print(f"✅ تم تصنيف: {len(labeled_data)} عنصر")
        
        # إحصائيات
        category_counts = {}
        for item in labeled_data:
            cat = item.get('topic', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📊 توزيع الفئات:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            cat_ar = self.CATEGORIES.get(cat, {}).get('ar', cat)
            print(f"   • {cat_ar}: {count}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 تم حفظ البيانات في: {save_path}")
        
        return labeled_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='cache/translated_data.json')
    parser.add_argument('--output', default='cache/labeled_data.json')
    parser.add_argument('--mode', choices=['keyword', 'model'], default='keyword',
                       help='اختر keyword للسرعة (بدون تحميل) أو model للدقة')
    parser.add_argument('--sample', type=int, default=0)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ الملف غير موجود: {args.input}")
        return
    
    print(f"📂 قراءة البيانات من: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if args.sample > 0:
        data = data[:args.sample]
    
    labeler = AutoLabeler()
    labeler.label_dataset(data, save_path=args.output, mode=args.mode)


if __name__ == "__main__":
    main()
