"""
مصنف الشخصيات
Persona Classifier

الخوارزمية: نظام هجين
المميزات:
- تحليل البنية النحوية (أسماء الاستفهام والأفعال الأمرية)
- تحليل طول السؤال
- مطابقة الكلمات المفتاحية
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


class PersonaClassifier:
    """
    مصنف الشخصيات باستخدام نظام هجين
    يحدد نوع شخصية المستخدم بناءً على أسئلته
    """
    
    def __init__(self):
        # أنواع الشخصيات الثلاثة
        self.persona_types = {
            'learner': {
                'name_ar': 'المتعلم',
                'name_en': 'The Learner',
                'description_ar': 'يسعى للحصول على المعرفة والفهم العميق',
                'color': '#4FC3F7',  # أزرق فاتح
                
                # أدوات الاستفهام - مؤشر قوي للمتعلم
                'interrogative_ar': [
                    'ما', 'ماذا', 'ما هو', 'ما هي', 'ما معنى',
                    'كيف', 'لماذا', 'لم', 'كيف يعمل', 'كيف يمكن',
                    'أين', 'متى', 'من', 'من هو', 'من هي',
                    'هل يمكن', 'هل تستطيع', 'هل',
                    'أيهما', 'كم', 'أي',
                ],
                'interrogative_en': [
                    'what', 'what is', 'what are', 'what does', 'what do',
                    'how', 'how does', 'how do', 'how can', 'how to',
                    'why', 'where', 'when', 'who', 'which',
                    'can you explain', 'could you tell me',
                    'i want to know', 'i want to learn', 'i want to understand',
                ],
                
                # كلمات مفتاحية للتعلم
                'keywords_ar': [
                    'اشرح', 'وضح', 'فسر', 'علمني', 'أريد أن أفهم',
                    'أريد أن أتعلم', 'أحتاج أن أعرف', 'ساعدني على الفهم',
                    'ما الفرق', 'ما المقصود', 'ما تعريف',
                ],
                'keywords_en': [
                    'explain', 'understand', 'learn', 'teach me', 'tell me about',
                    'can you explain', 'help me understand', 'what is the meaning',
                    'definition', 'concept', 'difference between',
                ],
                
                'weight': 1.0,
                'score': 0.0
            },
            'doer': {
                'name_ar': 'المُنجِز',
                'name_en': 'The Doer',
                'description_ar': 'يهدف إلى إنجاز مهمة عملية فورية',
                'color': '#81C784',  # أخضر
                
                # صيغ الأمر - مؤشر قوي للمنجز
                'imperative_ar': [
                    'اكتب', 'اكتبي', 'اكتبوا',
                    'لخص', 'لخصي', 'لخصوا',
                    'ترجم', 'ترجمي', 'ترجموا',
                    'حل', 'حلي', 'حلوا',
                    'صحح', 'صححي', 'صححوا',
                    'أنشئ', 'أنشئي', 'أنشئوا',
                    'صمم', 'صممي', 'صمموا',
                    'حول', 'حولي', 'حولوا',
                    'احسب', 'احسبي', 'احسبوا',
                    'أصلح', 'أصلحي', 'أصلحوا',
                    'غير', 'غيري', 'غيروا',
                    'أضف', 'أضيفي', 'أضيفوا',
                    'احذف', 'احذفي', 'احذفوا',
                    'أعد صياغة', 'أعيدي صياغة',
                ],
                'imperative_en': [
                    'write', 'create', 'make', 'build', 'generate',
                    'code', 'develop', 'implement', 'design',
                    'fix', 'solve', 'debug', 'correct', 'repair',
                    'translate', 'convert', 'calculate', 'compute',
                    'summarize', 'shorten', 'condense',
                    'edit', 'modify', 'change', 'update',
                    'add', 'remove', 'delete', 'insert',
                    'format', 'rewrite', 'reformat', 'restructure',
                ],
                
                # كلمات مفتاحية للإنجاز
                'keywords_ar': [
                    'أريد', 'أحتاج', 'اعمل لي', 'جهز لي', 'حضر لي',
                    'ساعدني في عمل', 'ساعدني في إنشاء',
                    'أعطني', 'قدم لي', 'زودني',
                ],
                'keywords_en': [
                    'i need', 'i want', 'can you make', 'please write',
                    'help me create', 'help me build', 'help me write',
                    'give me', 'provide me', 'show me how to do',
                ],
                
                'weight': 1.2,  # وزن أعلى للأفعال الأمرية
                'score': 0.0
            },
            'explorer': {
                'name_ar': 'المستكشف',
                'name_en': 'The Explorer',
                'description_ar': 'أسئلة استطلاعية ومقارنات وطلب آراء',
                'color': '#FFB74D',  # برتقالي
                
                # كلمات المقارنة والاستكشاف
                'comparison_ar': [
                    'قارن', 'الفرق بين', 'ما الفرق', 'أيهما أفضل',
                    'مقارنة بين', 'مقابل', 'ضد', 'أم',
                    'أفضل طريقة', 'أفضل خيار', 'أفضل بديل',
                    'ما رأيك', 'ما هو رأيك', 'ما رأيكم',
                    'ماذا تقترح', 'ماذا تنصح', 'ماذا توصي',
                    'هل تعتقد', 'هل ترى', 'هل تظن',
                    'اقتراحات', 'توصيات', 'نصائح', 'بدائل', 'خيارات',
                ],
                'comparison_en': [
                    'compare', 'comparison', 'difference between', 'differences',
                    'vs', 'versus', 'or', 'better', 'best',
                    'which is better', 'which one', 'which should i',
                    'what about', 'how about', 'what if',
                    'opinion', 'what do you think', 'do you think',
                    'suggest', 'recommend', 'advise', 'advice',
                    'options', 'alternatives', 'choices', 'pros and cons',
                    'should i', 'would you', 'is it better to',
                ],
                
                # كلمات الاستكشاف العامة
                'keywords_ar': [
                    'استكشف', 'اكتشف', 'ابحث', 'جرب',
                    'تجربة', 'احتمالات', 'إمكانيات',
                    'ما هي الخيارات', 'ما هي البدائل',
                ],
                'keywords_en': [
                    'explore', 'discover', 'try', 'experiment',
                    'possibilities', 'potential', 'options',
                    'what are the options', 'what are my choices',
                ],
                
                # طول السؤال مهم للمستكشف
                'min_length_indicator': 40,  # أسئلة المستكشف عادة أطول
                
                'weight': 1.0,
                'score': 0.0
            }
        }
    
    def _analyze_syntactic_structure(self, text: str) -> Dict[str, float]:
        """
        تحليل البنية النحوية للنص
        
        Args:
            text: النص المُدخل
            
        Returns:
            نقاط لكل شخصية بناءً على البنية النحوية
        """
        scores = {'learner': 0.0, 'doer': 0.0, 'explorer': 0.0}
        text_lower = text.lower()
        
        # تحليل المتعلم - أدوات الاستفهام
        learner = self.persona_types['learner']
        for interrogative in learner['interrogative_ar']:
            if text.startswith(interrogative) or f' {interrogative}' in text:
                scores['learner'] += 2.0  # وزن عالي لبداية السؤال
        for interrogative in learner['interrogative_en']:
            if text_lower.startswith(interrogative) or f' {interrogative}' in text_lower:
                scores['learner'] += 2.0
        
        # تحليل المنجز - صيغ الأمر
        doer = self.persona_types['doer']
        for imperative in doer['imperative_ar']:
            if text.startswith(imperative) or f' {imperative}' in text:
                scores['doer'] += 2.5  # وزن أعلى للأفعال الأمرية
        for imperative in doer['imperative_en']:
            if text_lower.startswith(imperative) or f' {imperative}' in text_lower:
                scores['doer'] += 2.5
        
        # تحليل المستكشف - المقارنات
        explorer = self.persona_types['explorer']
        for comparison in explorer['comparison_ar']:
            if comparison in text:
                scores['explorer'] += 2.0
        for comparison in explorer['comparison_en']:
            if comparison in text_lower:
                scores['explorer'] += 2.0
        
        return scores
    
    def _analyze_length(self, text: str) -> Dict[str, float]:
        """
        تحليل طول النص
        الأسئلة الأطول غالباً من المستكشفين
        
        Args:
            text: النص المُدخل
            
        Returns:
            نقاط إضافية بناءً على الطول
        """
        scores = {'learner': 0.0, 'doer': 0.0, 'explorer': 0.0}
        
        word_count = len(text.split())
        char_count = len(text)
        
        # المستكشف يميل للأسئلة الأطول والأكثر تفصيلاً
        if word_count > 20 or char_count > 100:
            scores['explorer'] += 1.5
        
        # المنجز يميل للأوامر المباشرة القصيرة
        if word_count < 10 and char_count < 50:
            scores['doer'] += 0.5
        
        # وجود مقارنة (أم، أو، vs) مع طول متوسط
        if ('أم' in text or 'أو' in text or ' vs ' in text.lower()):
            scores['explorer'] += 1.0
        
        return scores
    
    def _analyze_keywords(self, text: str) -> Dict[str, float]:
        """
        تحليل الكلمات المفتاحية
        
        Args:
            text: النص المُدخل
            
        Returns:
            نقاط بناءً على الكلمات المفتاحية
        """
        scores = {'learner': 0.0, 'doer': 0.0, 'explorer': 0.0}
        text_lower = text.lower()
        
        for persona_id, persona_data in self.persona_types.items():
            # الكلمات العربية
            for keyword in persona_data.get('keywords_ar', []):
                if keyword in text:
                    scores[persona_id] += 1.0
            
            # الكلمات الإنجليزية
            for keyword in persona_data.get('keywords_en', []):
                if keyword in text_lower:
                    scores[persona_id] += 1.0
        
        return scores
    
    def classify(self, text: str, preprocessor=None) -> Dict:
        """
        تصنيف النص لتحديد نوع الشخصية
        
        Args:
            text: النص المُدخل
            preprocessor: معالج النص (اختياري)
            
        Returns:
            نتيجة التصنيف مع التفاصيل
        """
        if not text or len(text.strip()) < 3:
            return {
                'persona_id': 'learner',  # افتراضي
                'persona_ar': self.persona_types['learner']['name_ar'],
                'persona_en': self.persona_types['learner']['name_en'],
                'description_ar': self.persona_types['learner']['description_ar'],
                'confidence': 0.0,
                'all_scores': {},
            }
        
        # التحليل المتعدد
        syntactic_scores = self._analyze_syntactic_structure(text)
        length_scores = self._analyze_length(text)
        keyword_scores = self._analyze_keywords(text)
        
        # دمج النقاط
        total_scores = {}
        for persona_id in self.persona_types.keys():
            weight = self.persona_types[persona_id]['weight']
            total = (
                syntactic_scores[persona_id] * 2.0 +  # وزن أعلى للبنية النحوية
                length_scores[persona_id] +
                keyword_scores[persona_id]
            ) * weight
            total_scores[persona_id] = total
        
        # حساب المجموع الكلي
        grand_total = sum(total_scores.values())
        
        if grand_total == 0:
            # افتراضي: المتعلم
            return {
                'persona_id': 'learner',
                'persona_ar': self.persona_types['learner']['name_ar'],
                'persona_en': self.persona_types['learner']['name_en'],
                'description_ar': self.persona_types['learner']['description_ar'],
                'confidence': 0.33,
                'all_scores': {'learner': 0.33, 'doer': 0.33, 'explorer': 0.33},
            }
        
        # تحويل إلى نسب
        percentages = {
            pid: score / grand_total 
            for pid, score in total_scores.items()
        }
        
        # إيجاد الشخصية الأعلى
        best_persona = max(percentages, key=percentages.get)
        confidence = percentages[best_persona]
        
        persona_data = self.persona_types[best_persona]
        
        return {
            'persona_id': best_persona,
            'persona_ar': persona_data['name_ar'],
            'persona_en': persona_data['name_en'],
            'description_ar': persona_data['description_ar'],
            'color': persona_data['color'],
            'confidence': round(confidence, 3),
            'all_scores': {
                pid: round(score, 3) 
                for pid, score in percentages.items()
            },
            'analysis': {
                'syntactic': syntactic_scores,
                'length': length_scores,
                'keywords': keyword_scores,
            }
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
    
    def get_persona_distribution(self, texts: List[str], preprocessor=None) -> Dict:
        """
        حساب توزيع الشخصيات لمجموعة نصوص
        
        Args:
            texts: قائمة النصوص
            preprocessor: معالج النص
            
        Returns:
            توزيع الشخصيات بالنسب المئوية
        """
        if not texts:
            return {}
        
        results = self.classify_batch(texts, preprocessor)
        
        # حساب التكرارات
        persona_counts = Counter(r['persona_id'] for r in results)
        total = len(texts)
        
        distribution = {}
        for persona_id, persona_data in self.persona_types.items():
            count = persona_counts.get(persona_id, 0)
            distribution[persona_id] = {
                'name_ar': persona_data['name_ar'],
                'name_en': persona_data['name_en'],
                'description_ar': persona_data['description_ar'],
                'color': persona_data['color'],
                'count': count,
                'percentage': round((count / total) * 100, 1) if total > 0 else 0
            }
        
        return distribution
    
    def get_primary_persona(self, texts: List[str], preprocessor=None) -> Dict:
        """
        تحديد الشخصية الرئيسية لمجموعة نصوص
        
        Args:
            texts: قائمة النصوص
            preprocessor: معالج النص
            
        Returns:
            الشخصية الرئيسية مع التفاصيل
        """
        distribution = self.get_persona_distribution(texts, preprocessor)
        
        if not distribution:
            return self.persona_types['learner']
        
        # إيجاد الشخصية الأعلى
        primary = max(distribution.items(), key=lambda x: x[1]['percentage'])
        
        return {
            'id': primary[0],
            **primary[1]
        }
    
    def get_personas_info(self) -> Dict:
        """
        الحصول على معلومات جميع الشخصيات
        
        Returns:
            معلومات الشخصيات
        """
        return {
            pid: {
                'name_ar': data['name_ar'],
                'name_en': data['name_en'],
                'description_ar': data['description_ar'],
                'color': data['color']
            }
            for pid, data in self.persona_types.items()
        }


# 🧪 اختبار
if __name__ == "__main__":
    classifier = PersonaClassifier()
    
    # نصوص للاختبار - أنواع مختلفة
    test_texts = [
        # المتعلم
        "ما هو الذكاء الاصطناعي وكيف يعمل؟",
        "اشرح لي نظرية النسبية",
        "لماذا السماء زرقاء؟",
        "How does machine learning work?",
        
        # المنجز
        "اكتب لي كود بايثون لحساب المتوسط",
        "ترجم هذا النص للإنجليزية",
        "لخص لي هذا المقال",
        "Create a website for my business",
        
        # المستكشف
        "ما الفرق بين بايثون وجافا؟ أيهما أفضل للمبتدئين؟",
        "ما رأيك في استخدام React vs Vue للمشاريع الكبيرة؟",
        "اقترح علي أفضل طريقة لتعلم البرمجة وما هي الخيارات المتاحة",
        "Should I use Python or JavaScript for data science?",
    ]
    
    print("=" * 70)
    print("👤 اختبار مصنف الشخصيات")
    print("=" * 70)
    
    for text in test_texts:
        result = classifier.classify(text)
        
        print(f"\n📝 النص: {text[:50]}...")
        print(f"   الشخصية: {result['persona_ar']} ({result['persona_en']})")
        print(f"   الوصف: {result['description_ar']}")
        print(f"   الثقة: {result['confidence']*100:.1f}%")
        print(f"   التوزيع: ", end="")
        for pid, score in result['all_scores'].items():
            print(f"{pid}:{score*100:.0f}% ", end="")
        print()
    
    print("\n" + "=" * 70)
    print("📊 توزيع الشخصيات")
    print("=" * 70)
    
    distribution = classifier.get_persona_distribution(test_texts)
    for persona_id, data in sorted(distribution.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"   {data['name_ar']}: {data['percentage']}% ({data['count']} سؤال)")
    
    print("\n👤 الشخصية الرئيسية:")
    primary = classifier.get_primary_persona(test_texts)
    print(f"   {primary['name_ar']} - {primary['description_ar']}")
