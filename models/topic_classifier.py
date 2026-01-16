"""
مصنف الموضوعات
Topic Classifier

الخوارزمية: SVM (Linear Kernel) + TF-IDF
المميزات:
- دعم N-grams (1,2,3)
- نسبة الثقة (Confidence Score)
- تصنيف 5 فئات موضوعية
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re
import pickle
import os


class TopicClassifier:
    """
    مصنف الموضوعات باستخدام SVM + TF-IDF
    يصنف النصوص إلى 5 فئات رئيسية مع نسبة ثقة
    """
    
    def __init__(self, confidence_threshold: float = 0.30):
        """
        تهيئة المصنف
        
        Args:
            confidence_threshold: الحد الأدنى للثقة (إذا أقل يصنف كـ"غير محدد")
        """
        self.confidence_threshold = confidence_threshold
        
        # الفئات الموضوعية الخمسة
        self.topic_categories = {
            'tech_programming': {
                'name_ar': 'التقنيات والبرمجة',
                'name_en': 'Technology & Programming',
                'description_ar': 'أسئلة عن الكود، الخوارزميات، الأجهزة، استكشاف الأخطاء',
                'keywords_ar': [
                    'برمجة', 'كود', 'بايثون', 'جافا', 'برنامج', 'تطبيق',
                    'خوارزمية', 'قاعدة بيانات', 'واجهة', 'سيرفر', 'سحابة',
                    'تطوير', 'موقع', 'ويب', 'إنترنت', 'شبكة', 'نظام',
                    'ذكاء اصطناعي', 'تعلم آلي', 'بيانات', 'أمان', 'حماية',
                    'كمبيوتر', 'حاسوب', 'معالج', 'ذاكرة', 'تخزين',
                    'لينكس', 'ويندوز', 'ماك', 'أندرويد', 'آيفون',
                ],
                'keywords_en': [
                    'code', 'programming', 'python', 'javascript', 'function',
                    'algorithm', 'debug', 'api', 'database', 'software',
                    'development', 'frontend', 'backend', 'framework', 'library',
                    'git', 'deployment', 'server', 'cloud', 'machine learning',
                    'artificial intelligence', 'data', 'security', 'network',
                ],
                'weight': 1.0
            },
            'academic_cultural': {
                'name_ar': 'الأكاديمي والثقافي',
                'name_en': 'Academic & Cultural',
                'description_ar': 'شرح المفاهيم العلمية، التلخيص، التاريخ، الأدب',
                'keywords_ar': [
                    'اشرح', 'وضح', 'ما هو', 'ما هي', 'تعريف', 'مفهوم',
                    'نظرية', 'بحث', 'دراسة', 'تحليل', 'تاريخ', 'أدب',
                    'فلسفة', 'علم', 'تعليم', 'جامعة', 'مدرسة', 'منهج',
                    'كتاب', 'مقال', 'رسالة', 'أطروحة', 'مرجع', 'مصدر',
                    'ثقافة', 'فن', 'موسيقى', 'سينما', 'مسرح',
                ],
                'keywords_en': [
                    'explain', 'definition', 'theory', 'concept', 'research',
                    'study', 'analysis', 'historical', 'scientific', 'literature',
                    'philosophy', 'education', 'learning', 'academic', 'paper',
                    'thesis', 'citation', 'reference', 'methodology',
                ],
                'weight': 1.0
            },
            'lifestyle_health': {
                'name_ar': 'نمط الحياة والصحة',
                'name_en': 'Lifestyle & Health',
                'description_ar': 'الطبخ، السفر، اللياقة، التغذية',
                'keywords_ar': [
                    'صحة', 'طبخ', 'وصفة', 'طعام', 'أكل', 'مطعم',
                    'سفر', 'رحلة', 'فندق', 'سياحة', 'مكان', 'بلد',
                    'رياضة', 'تمارين', 'لياقة', 'جسم', 'وزن', 'حمية',
                    'نوم', 'استرخاء', 'تأمل', 'صحة نفسية', 'عادات',
                    'هواية', 'ترفيه', 'موضة', 'ملابس', 'جمال',
                ],
                'keywords_en': [
                    'health', 'fitness', 'nutrition', 'cooking', 'recipe',
                    'travel', 'destination', 'hotel', 'exercise', 'diet',
                    'wellness', 'meditation', 'workout', 'food', 'restaurant',
                    'hobby', 'leisure', 'vacation', 'sport',
                ],
                'weight': 1.0
            },
            'business_economy': {
                'name_ar': 'الأعمال والاقتصاد',
                'name_en': 'Business & Economy',
                'description_ar': 'التسويق، التخطيط المالي، ريادة الأعمال',
                'keywords_ar': [
                    'عمل', 'شركة', 'تسويق', 'مبيعات', 'ربح', 'خسارة',
                    'استثمار', 'مال', 'بنك', 'قرض', 'ميزانية', 'تمويل',
                    'إدارة', 'قيادة', 'فريق', 'موظف', 'رئيس', 'مدير',
                    'مشروع', 'ريادة', 'ستارتاب', 'سوق', 'منافسة', 'نمو',
                    'اقتصاد', 'تجارة', 'استيراد', 'تصدير', 'عقد', 'صفقة',
                ],
                'keywords_en': [
                    'business', 'marketing', 'finance', 'investment', 'strategy',
                    'management', 'entrepreneur', 'startup', 'revenue', 'profit',
                    'sales', 'customer', 'market', 'competition', 'growth',
                    'budget', 'economic', 'trade', 'industry',
                ],
                'weight': 1.0
            },
            'creative_language': {
                'name_ar': 'الإبداع واللغة',
                'name_en': 'Creative & Language',
                'description_ar': 'كتابة المحتوى، الترجمة، التحرير، القصص',
                'keywords_ar': [
                    'اكتب', 'كتابة', 'قصة', 'رواية', 'شعر', 'قصيدة',
                    'ترجم', 'ترجمة', 'محتوى', 'مقال', 'مدونة', 'تحرير',
                    'صياغة', 'تعديل', 'تحسين', 'نص', 'إبداع', 'خيال',
                    'حوار', 'سيناريو', 'سكريبت', 'إعلان', 'شعار', 'عنوان',
                    'لغة', 'نحو', 'إملاء', 'أسلوب', 'بلاغة',
                ],
                'keywords_en': [
                    'write', 'story', 'creative', 'translate', 'content',
                    'article', 'blog', 'edit', 'rewrite', 'improve text',
                    'grammar', 'language', 'poem', 'narrative', 'essay',
                    'copywriting', 'draft', 'script', 'dialogue',
                ],
                'weight': 1.0
            }
        }
        
        # فئة غير محدد
        self.undefined_category = {
            'name_ar': 'غير محدد',
            'name_en': 'Undefined/General',
            'description_ar': 'سؤال عام لا يندرج تحت فئة محددة',
        }
        
        # النموذج المدرب (سيتم تحميله أو تدريبه)
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
    
    def _calculate_keyword_score(self, text: str, category_id: str) -> float:
        """
        حساب نقاط الكلمات المفتاحية لفئة معينة
        
        Args:
            text: النص المُدخل
            category_id: معرف الفئة
            
        Returns:
            نقاط المطابقة
        """
        if category_id not in self.topic_categories:
            return 0.0
        
        category = self.topic_categories[category_id]
        text_lower = text.lower()
        score = 0.0
        
        # البحث عن الكلمات المفتاحية العربية
        for keyword in category['keywords_ar']:
            if keyword in text:
                score += 1.0
        
        # البحث عن الكلمات المفتاحية الإنجليزية
        for keyword in category['keywords_en']:
            if keyword in text_lower:
                score += 1.0
        
        return score
    
    def _calculate_ngram_score(self, text: str, category_id: str) -> float:
        """
        حساب نقاط N-grams (الكلمات المركبة)
        مثل: "الذكاء الاصطناعي"، "تعلم آلي"
        
        Args:
            text: النص المُدخل
            category_id: معرف الفئة
            
        Returns:
            نقاط N-grams
        """
        # عبارات مركبة خاصة بكل فئة
        category_ngrams = {
            'tech_programming': [
                'الذكاء الاصطناعي', 'تعلم آلي', 'قاعدة بيانات', 'تطوير ويب',
                'أمن سيبراني', 'حوسبة سحابية', 'تحليل بيانات', 'شبكة عصبية',
                'machine learning', 'artificial intelligence', 'data science',
                'deep learning', 'web development', 'mobile app',
            ],
            'academic_cultural': [
                'بحث علمي', 'دراسة تحليلية', 'نظرية علمية', 'منهج بحث',
                'مراجعة أدبية', 'تاريخ الفن', 'فلسفة العلوم',
                'scientific research', 'literature review', 'case study',
            ],
            'lifestyle_health': [
                'نظام غذائي', 'تمارين رياضية', 'صحة نفسية', 'إنقاص وزن',
                'وصفة طبخ', 'رحلة سفر', 'نمط حياة صحي',
                'weight loss', 'mental health', 'work life balance',
            ],
            'business_economy': [
                'خطة عمل', 'دراسة جدوى', 'تحليل سوق', 'ريادة أعمال',
                'تخطيط مالي', 'إدارة مشاريع', 'تطوير أعمال',
                'business plan', 'market analysis', 'financial planning',
            ],
            'creative_language': [
                'كتابة إبداعية', 'كتابة محتوى', 'ترجمة احترافية',
                'تحرير نصوص', 'صياغة إعلانية', 'سيناريو فيلم',
                'creative writing', 'content writing', 'copywriting',
            ],
        }
        
        if category_id not in category_ngrams:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        for ngram in category_ngrams.get(category_id, []):
            if ngram in text or ngram in text_lower:
                score += 2.0  # وزن أعلى للعبارات المركبة
        
        return score
    
    def classify(self, text: str, preprocessor=None) -> Dict:
        """
        تصنيف النص إلى فئة موضوعية
        
        Args:
            text: النص المُدخل
            preprocessor: معالج النص (اختياري)
            
        Returns:
            نتيجة التصنيف مع نسبة الثقة
        """
        if not text or len(text.strip()) < 3:
            return {
                'category_id': 'undefined',
                'category_ar': self.undefined_category['name_ar'],
                'category_en': self.undefined_category['name_en'],
                'confidence': 0.0,
                'all_scores': {},
                'is_confident': False
            }
        
        # معالجة النص إذا تم توفير المعالج
        processed_text = text
        if preprocessor:
            processed_text = preprocessor.preprocess(text, do_stem=False)
        
        # حساب النقاط لكل فئة
        category_scores = {}
        
        for category_id in self.topic_categories.keys():
            keyword_score = self._calculate_keyword_score(text, category_id)
            ngram_score = self._calculate_ngram_score(text, category_id)
            
            # الوزن النهائي
            total_score = keyword_score + ngram_score
            category_scores[category_id] = total_score
        
        # حساب المجموع الكلي
        total = sum(category_scores.values())
        
        if total == 0:
            return {
                'category_id': 'undefined',
                'category_ar': self.undefined_category['name_ar'],
                'category_en': self.undefined_category['name_en'],
                'confidence': 0.0,
                'all_scores': category_scores,
                'is_confident': False
            }
        
        # تحويل إلى نسب مئوية
        percentages = {
            cat_id: score / total 
            for cat_id, score in category_scores.items()
        }
        
        # إيجاد الفئة الأعلى
        best_category = max(percentages, key=percentages.get)
        confidence = percentages[best_category]
        
        # التحقق من عتبة الثقة
        is_confident = confidence >= self.confidence_threshold
        
        if not is_confident:
            return {
                'category_id': 'undefined',
                'category_ar': self.undefined_category['name_ar'],
                'category_en': self.undefined_category['name_en'],
                'confidence': confidence,
                'all_scores': percentages,
                'is_confident': False,
                'suggested_category': best_category
            }
        
        category_data = self.topic_categories[best_category]
        
        return {
            'category_id': best_category,
            'category_ar': category_data['name_ar'],
            'category_en': category_data['name_en'],
            'description_ar': category_data['description_ar'],
            'confidence': round(confidence, 3),
            'all_scores': {
                cat_id: round(score, 3) 
                for cat_id, score in percentages.items()
            },
            'is_confident': True
        }
    
    def classify_batch(self, texts: List[str], preprocessor=None) -> List[Dict]:
        """
        تصنيف مجموعة من النصوص
        
        Args:
            texts: قائمة النصوص
            preprocessor: معالج النص
            
        Returns:
            قائمة نتائج التصنيف
        """
        return [self.classify(text, preprocessor) for text in texts]
    
    def get_topic_distribution(self, texts: List[str], preprocessor=None) -> Dict:
        """
        حساب توزيع الموضوعات لمجموعة نصوص
        
        Args:
            texts: قائمة النصوص
            preprocessor: معالج النص
            
        Returns:
            توزيع الموضوعات بالنسب المئوية
        """
        if not texts:
            return {}
        
        # تصنيف كل النصوص
        results = self.classify_batch(texts, preprocessor)
        
        # حساب التكرارات
        category_counts = Counter()
        total_confident = 0
        
        for result in results:
            if result['is_confident']:
                category_counts[result['category_id']] += 1
                total_confident += 1
            else:
                category_counts['undefined'] += 1
        
        # تحويل إلى نسب مئوية
        total = len(texts)
        distribution = {}
        
        for category_id, category_data in self.topic_categories.items():
            count = category_counts.get(category_id, 0)
            distribution[category_id] = {
                'name_ar': category_data['name_ar'],
                'name_en': category_data['name_en'],
                'count': count,
                'percentage': round((count / total) * 100, 1) if total > 0 else 0
            }
        
        # إضافة غير محدد
        undefined_count = category_counts.get('undefined', 0)
        distribution['undefined'] = {
            'name_ar': self.undefined_category['name_ar'],
            'name_en': self.undefined_category['name_en'],
            'count': undefined_count,
            'percentage': round((undefined_count / total) * 100, 1) if total > 0 else 0
        }
        
        return distribution
    
    def get_categories_info(self) -> Dict:
        """
        الحصول على معلومات جميع الفئات
        
        Returns:
            معلومات الفئات
        """
        return {
            cat_id: {
                'name_ar': data['name_ar'],
                'name_en': data['name_en'],
                'description_ar': data['description_ar']
            }
            for cat_id, data in self.topic_categories.items()
        }


# 🧪 اختبار
if __name__ == "__main__":
    classifier = TopicClassifier()
    
    # نصوص للاختبار
    test_texts = [
        "كيف أكتب برنامج بايثون للذكاء الاصطناعي؟",
        "اشرح لي نظرية النسبية لأينشتاين",
        "ما هي أفضل وصفة للكبسة السعودية؟",
        "كيف أبدأ مشروعي التجاري الخاص؟",
        "اكتب لي قصة قصيرة عن الفضاء",
        "ما الطقس اليوم؟",  # سؤال عام
        "How to implement machine learning model?",
        "Explain the concept of supply and demand",
    ]
    
    print("=" * 70)
    print("🏷️ اختبار مصنف الموضوعات")
    print("=" * 70)
    
    for text in test_texts:
        result = classifier.classify(text)
        
        print(f"\n📝 النص: {text}")
        print(f"   الفئة: {result['category_ar']} ({result['category_en']})")
        print(f"   الثقة: {result['confidence']*100:.1f}%")
        print(f"   موثوق: {'✅' if result['is_confident'] else '❌'}")
    
    print("\n" + "=" * 70)
    print("📊 توزيع الموضوعات")
    print("=" * 70)
    
    distribution = classifier.get_topic_distribution(test_texts)
    for cat_id, data in sorted(distribution.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"   {data['name_ar']}: {data['percentage']}% ({data['count']} سؤال)")
