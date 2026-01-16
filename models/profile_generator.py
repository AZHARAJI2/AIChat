"""
مولد البروفايل المعرفي
Cognitive Profile Generator

يجمع بين جميع المصنفات لإنشاء بروفايل معرفي شامل
"""

import sys
import json
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime
import os

# إعداد الترميز للويندوز
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


# استيراد المكونات
from .arabic_preprocessor import ArabicPreprocessor
from .topic_classifier import TopicClassifier
from .persona_classifier import PersonaClassifier


class ProfileGenerator:
    """
    مولد البروفايل المعرفي
    ينسق عمل جميع المصنفات لإنتاج تحليل شامل
    """
    
    def __init__(self, confidence_threshold: float = 0.30):
        """
        تهيئة المولد
        
        Args:
            confidence_threshold: حد الثقة الأدنى للتصنيف
        """
        self.preprocessor = ArabicPreprocessor()
        self.topic_classifier = TopicClassifier(confidence_threshold)
        self.persona_classifier = PersonaClassifier()
        
        # مستويات التعقيد
        self.complexity_levels = {
            'beginner': {
                'name_ar': 'مبتدئ',
                'name_en': 'Beginner',
                'indicators_ar': ['بسيط', 'أساسي', 'مبتدئ', 'أبدأ', 'أول مرة', 'جديد على'],
                'indicators_en': ['basic', 'simple', 'beginner', 'start', 'first time', 'new to', 'introduction']
            },
            'intermediate': {
                'name_ar': 'متوسط',
                'name_en': 'Intermediate',
                'indicators_ar': ['كيف', 'أفضل طريقة', 'هل يجب', 'أنصح', 'تحسين'],
                'indicators_en': ['how to', 'best practice', 'should i', 'recommend', 'improve']
            },
            'advanced': {
                'name_ar': 'متقدم',
                'name_en': 'Advanced',
                'indicators_ar': ['تحسين الأداء', 'متقدم', 'معقد', 'متطور', 'معمارية', 'حالات خاصة'],
                'indicators_en': ['optimize', 'advanced', 'complex', 'sophisticated', 'architecture', 'edge case', 'performance']
            }
        }
    
    def _determine_complexity(self, text: str) -> str:
        """
        تحديد مستوى تعقيد السؤال
        
        Args:
            text: النص المُدخل
            
        Returns:
            مستوى التعقيد
        """
        text_lower = text.lower()
        scores = {level: 0 for level in self.complexity_levels.keys()}
        
        for level, data in self.complexity_levels.items():
            for ind in data['indicators_ar']:
                if ind in text:
                    scores[level] += 1
            for ind in data['indicators_en']:
                if ind in text_lower:
                    scores[level] += 1
        
        # إضافة عوامل الطول
        word_count = len(text.split())
        if word_count > 30:
            scores['advanced'] += 1
        elif word_count < 10:
            scores['beginner'] += 1
        
        return max(scores, key=scores.get) or 'intermediate'
    
    def analyze_single(self, text: str) -> Dict:
        """
        تحليل نص واحد
        
        Args:
            text: النص المُدخل
            
        Returns:
            نتيجة التحليل الكاملة
        """
        # المعالجة المسبقة
        processed_text = self.preprocessor.preprocess(text, do_stem=False)
        
        # التصنيفات
        topic_result = self.topic_classifier.classify(text, self.preprocessor)
        persona_result = self.persona_classifier.classify(text, self.preprocessor)
        complexity = self._determine_complexity(text)
        
        # استخراج الكلمات المفتاحية
        keywords = self.preprocessor.extract_keywords(text, top_n=5)
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'topic': topic_result,
            'persona': persona_result,
            'complexity': complexity,
            'complexity_ar': self.complexity_levels[complexity]['name_ar'],
            'keywords': keywords
        }
    
    def analyze_batch(self, texts: List[str], timestamps: Optional[List] = None) -> Dict:
        """
        تحليل مجموعة من النصوص وإنشاء بروفايل شامل
        
        Args:
            texts: قائمة النصوص
            timestamps: قائمة التواريخ (اختياري)
            
        Returns:
            البروفايل المعرفي الكامل
        """
        if not texts:
            return self._empty_profile()
        
        # تحليل كل نص
        all_results = []
        all_keywords = []
        complexity_counts = Counter()
        topic_timeline = []
        
        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 3:
                continue
            
            result = self.analyze_single(text)
            all_results.append(result)
            all_keywords.extend(result['keywords'])
            complexity_counts[result['complexity']] += 1
            
            # بيانات الخط الزمني
            timestamp = timestamps[i] if timestamps and i < len(timestamps) else datetime.now()
            topic_timeline.append({
                'timestamp': timestamp,
                'topic_id': result['topic']['category_id'],
                'persona_id': result['persona']['persona_id']
            })
        
        if not all_results:
            return self._empty_profile()
        
        # حساب التوزيعات
        topic_distribution = self.topic_classifier.get_topic_distribution(texts, self.preprocessor)
        persona_distribution = self.persona_classifier.get_persona_distribution(texts, self.preprocessor)
        primary_persona = self.persona_classifier.get_primary_persona(texts, self.preprocessor)
        
        # الكلمات المفتاحية الأكثر تكراراً
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(30))
        
        # مستوى التعقيد السائد
        dominant_complexity = complexity_counts.most_common(1)[0][0] if complexity_counts else 'intermediate'
        
        # تحليل الخط الزمني
        timeline_analysis = self._analyze_timeline(topic_timeline)
        
        # البروفايل النهائي
        profile = {
            'summary': {
                'total_prompts': len(texts),
                'analyzed_prompts': len(all_results),
                'primary_persona': {
                    'id': primary_persona['id'],
                    'name_ar': primary_persona['name_ar'],
                    'name_en': primary_persona['name_en'],
                    'description_ar': primary_persona['description_ar'],
                    'percentage': primary_persona['percentage'],
                    'color': primary_persona.get('color', '#4FC3F7')
                },
                'complexity_level': dominant_complexity,
                'complexity_ar': self.complexity_levels[dominant_complexity]['name_ar'],
                'generated_at': datetime.now().isoformat()
            },
            'topic_distribution': topic_distribution,
            'persona_distribution': persona_distribution,
            'top_keywords': top_keywords,
            'complexity_distribution': dict(complexity_counts),
            'timeline': timeline_analysis
        }
        
        return profile
    
    def _analyze_timeline(self, timeline_data: List[Dict]) -> Dict:
        """
        تحليل تطور الاهتمامات عبر الزمن
        
        Args:
            timeline_data: بيانات الخط الزمني
            
        Returns:
            ملخص التحليل الزمني
        """
        if not timeline_data:
            return {}
        
        # ترتيب حسب الوقت
        sorted_data = sorted(timeline_data, key=lambda x: x['timestamp'] if x['timestamp'] else datetime.now())
        
        timeline_summary = {
            'total_entries': len(sorted_data),
            'topic_evolution': []
        }
        
        # تحليل تطور الموضوعات
        if len(sorted_data) >= 3:
            third = len(sorted_data) // 3
            
            early_topics = [t['topic_id'] for t in sorted_data[:third]]
            mid_topics = [t['topic_id'] for t in sorted_data[third:2*third]]
            late_topics = [t['topic_id'] for t in sorted_data[2*third:]]
            
            timeline_summary['early_interests'] = dict(Counter(early_topics).most_common(3))
            timeline_summary['mid_interests'] = dict(Counter(mid_topics).most_common(3))
            timeline_summary['recent_interests'] = dict(Counter(late_topics).most_common(3))
        
        return timeline_summary
    
    def _empty_profile(self) -> Dict:
        """إنشاء بروفايل فارغ"""
        return {
            'summary': {
                'total_prompts': 0,
                'analyzed_prompts': 0,
                'primary_persona': None,
                'complexity_level': 'intermediate',
                'complexity_ar': 'متوسط',
                'generated_at': datetime.now().isoformat()
            },
            'topic_distribution': {},
            'persona_distribution': {},
            'top_keywords': {},
            'complexity_distribution': {},
            'timeline': {}
        }
    
    def generate_dashboard_json(self, profile: Dict, output_file: Optional[str] = None) -> Dict:
        """
        تحويل البروفايل إلى صيغة جاهزة للـ Dashboard
        
        Args:
            profile: البروفايل المعرفي
            output_file: مسار ملف الإخراج (اختياري)
            
        Returns:
            بيانات الـ Dashboard
        """
        if not profile.get('summary'):
            return {}
        
        # ألوان المخطط الدائري
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        
        # بيانات المخطط الدائري
        pie_labels = []
        pie_values = []
        pie_colors = []
        
        for i, (topic_id, topic_data) in enumerate(profile.get('topic_distribution', {}).items()):
            if topic_data.get('percentage', 0) > 0:
                pie_labels.append(topic_data['name_ar'])
                pie_values.append(topic_data['percentage'])
                pie_colors.append(colors[i % len(colors)])
        
        dashboard_data = {
            'metadata': {
                'total_prompts': profile['summary']['total_prompts'],
                'analyzed_prompts': profile['summary']['analyzed_prompts'],
                'generated_at': profile['summary']['generated_at']
            },
            'pie_chart': {
                'labels': pie_labels,
                'values': pie_values,
                'colors': pie_colors
            },
            'persona_card': profile['summary']['primary_persona'],
            'word_cloud': profile.get('top_keywords', {}),
            'timeline': profile.get('timeline', {}),
            'complexity': {
                'level': profile['summary']['complexity_level'],
                'level_ar': profile['summary']['complexity_ar']
            },
            'persona_breakdown': profile.get('persona_distribution', {})
        }
        
        # حفظ الملف إذا طُلب
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, ensure_ascii=False, indent=2)
            print(f"✅ تم حفظ بيانات Dashboard في: {output_file}")
        
        return dashboard_data
    
    def analyze_from_file(self, input_file: str, output_file: Optional[str] = None) -> Dict:
        """
        تحليل بيانات من ملف JSON
        
        Args:
            input_file: مسار ملف الإدخال
            output_file: مسار ملف الإخراج (اختياري)
            
        Returns:
            البروفايل المعرفي
        """
        print(f"📂 جاري قراءة الملف: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # استخراج النصوص
        texts = []
        timestamps = []
        
        for item in data:
            prompts = item.get('prompts_ar', item.get('prompts', []))
            if isinstance(prompts, str):
                prompts = [prompts]
            
            for prompt in prompts:
                if prompt and len(prompt.strip()) > 3:
                    texts.append(prompt)
                    ts = item.get('timestamp')
                    timestamps.append(ts if ts else datetime.now())
        
        print(f"📊 تم استخراج {len(texts)} نص للتحليل")
        
        # التحليل
        profile = self.analyze_batch(texts, timestamps)
        
        # إنشاء بيانات Dashboard
        dashboard_data = self.generate_dashboard_json(profile, output_file)
        
        return profile


# 🧪 اختبار
if __name__ == "__main__":
    generator = ProfileGenerator()
    
    # نصوص للاختبار
    test_texts = [
        "كيف أكتب برنامج بايثون للذكاء الاصطناعي؟",
        "اشرح لي نظرية النسبية لأينشتاين",
        "ما هي أفضل وصفة للكبسة السعودية؟",
        "كيف أبدأ مشروعي التجاري الخاص؟",
        "اكتب لي كود بايثون لحساب المتوسط",
        "ترجم هذا النص للإنجليزية: Hello World",
        "ما الفرق بين React و Vue؟",
        "How to implement machine learning?",
        "ما رأيك في استخدام Docker للتطوير؟",
        "لخص لي هذا المقال عن الذكاء الاصطناعي",
    ]
    
    print("=" * 70)
    print("🧠 اختبار مولد البروفايل المعرفي")
    print("=" * 70)
    
    # تحليل مجموعة
    profile = generator.analyze_batch(test_texts)
    
    print("\n📊 ملخص البروفايل:")
    print(f"   • إجمالي الأسئلة: {profile['summary']['total_prompts']}")
    print(f"   • الأسئلة المحللة: {profile['summary']['analyzed_prompts']}")
    print(f"   • مستوى التعقيد: {profile['summary']['complexity_ar']}")
    
    print("\n👤 الشخصية الرئيسية:")
    primary = profile['summary']['primary_persona']
    if primary:
        print(f"   • {primary['name_ar']} ({primary['name_en']})")
        print(f"   • الوصف: {primary['description_ar']}")
        print(f"   • النسبة: {primary['percentage']}%")
    
    print("\n🎯 توزيع الموضوعات:")
    for topic_id, data in sorted(
        profile['topic_distribution'].items(),
        key=lambda x: x[1].get('percentage', 0),
        reverse=True
    ):
        if data.get('percentage', 0) > 0:
            print(f"   • {data['name_ar']}: {data['percentage']}%")
    
    print("\n🏷️ أهم الكلمات المفتاحية:")
    for word, count in list(profile['top_keywords'].items())[:10]:
        print(f"   • {word}: {count}")
    
    # إنشاء بيانات Dashboard
    print("\n" + "=" * 70)
    dashboard = generator.generate_dashboard_json(profile)
    print("📈 بيانات Dashboard:")
    print(json.dumps(dashboard, ensure_ascii=False, indent=2)[:500] + "...")
