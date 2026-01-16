"""
نظام تصنيف الاتجاهات والميول - استراتيجية مختلطة
يجمع بين: Chatbot Arena (بيانات خام) + CLINC-150 (مصنفة)

الهدف: إنشاء بروفايل معرفي شامل للمستخدم باللغة العربية
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime
import json
import re

class CognitiveProfileAnalyzer:
    """
    محلل شامل لإنشاء البروفايل المعرفي للمستخدم
    يدعم: التصنيف الموضوعي + تحديد الشخصية + التحليل الزمني
    """
    
    def __init__(self):
        # 1️⃣ الميول الرئيسية (Topic Categories)
        self.topic_categories = {
            'tech_programming': {
                'name_ar': 'التقنيات والبرمجة',
                'name_en': 'Technology & Programming',
                'keywords': [
                    'code', 'programming', 'python', 'javascript', 'function',
                    'algorithm', 'debug', 'api', 'database', 'software',
                    'development', 'frontend', 'backend', 'framework', 'library',
                    'git', 'version control', 'deployment', 'server', 'cloud'
                ],
                'intent_indicators': ['write code', 'fix bug', 'implement', 'develop'],
                'weight': 0.0  # سيتم حسابها
            },
            'academic_cultural': {
                'name_ar': 'الأكاديمي والثقافي',
                'name_en': 'Academic & Cultural',
                'keywords': [
                    'explain', 'definition', 'theory', 'concept', 'research',
                    'study', 'analysis', 'historical', 'scientific', 'literature',
                    'philosophy', 'education', 'learning', 'academic', 'paper',
                    'thesis', 'citation', 'reference', 'methodology'
                ],
                'intent_indicators': ['explain', 'what is', 'define', 'describe'],
                'weight': 0.0
            },
            'lifestyle_health': {
                'name_ar': 'نمط الحياة والصحة',
                'name_en': 'Lifestyle & Health',
                'keywords': [
                    'health', 'fitness', 'nutrition', 'cooking', 'recipe',
                    'travel', 'destination', 'hotel', 'exercise', 'diet',
                    'wellness', 'meditation', 'workout', 'food', 'restaurant',
                    'hobby', 'leisure', 'vacation', 'sport'
                ],
                'intent_indicators': ['recommend', 'suggest', 'best place'],
                'weight': 0.0
            },
            'business_economy': {
                'name_ar': 'الأعمال والاقتصاد',
                'name_en': 'Business & Economy',
                'keywords': [
                    'business', 'marketing', 'finance', 'investment', 'strategy',
                    'management', 'entrepreneur', 'startup', 'revenue', 'profit',
                    'sales', 'customer', 'market', 'competition', 'growth',
                    'budget', 'economic', 'trade', 'industry'
                ],
                'intent_indicators': ['how to grow', 'increase', 'optimize'],
                'weight': 0.0
            },
            'creative_language': {
                'name_ar': 'الإبداع واللغة',
                'name_en': 'Creative & Language',
                'keywords': [
                    'write', 'story', 'creative', 'translate', 'content',
                    'article', 'blog', 'edit', 'rewrite', 'improve text',
                    'grammar', 'language', 'poem', 'narrative', 'essay',
                    'copywriting', 'draft', 'script', 'dialogue'
                ],
                'intent_indicators': ['write', 'create', 'compose', 'draft'],
                'weight': 0.0
            },
            'legal_admin': {
                'name_ar': 'القانوني والإداري',
                'name_en': 'Legal & Administrative',
                'keywords': [
                    'legal', 'law', 'regulation', 'license', 'certificate',
                    'document', 'official', 'government', 'policy', 'compliance',
                    'contract', 'agreement', 'requirement', 'process', 'procedure',
                    'rights', 'obligation', 'administrative'
                ],
                'intent_indicators': ['how to get', 'what is required', 'process'],
                'weight': 0.0
            }
        }
        
        # 2️⃣ أنواع الشخصيات (Persona Classification)
        self.persona_types = {
            'learner': {
                'name_ar': 'المتعلم',
                'name_en': 'The Learner',
                'description_ar': 'يسعى للحصول على المعرفة والفهم العميق',
                'intent_keywords': [
                    'how', 'what', 'why', 'explain', 'understand', 'learn',
                    'teach me', 'what is', 'definition', 'meaning', 'concept',
                    'theory', 'can you explain', 'help me understand'
                ],
                'question_patterns': [
                    r'what is', r'how does', r'why', r'explain', r'what are',
                    r'can you tell me', r'help me understand', r'i want to learn'
                ],
                'score': 0.0
            },
            'doer': {
                'name_ar': 'المُنْجِز',
                'name_en': 'The Doer',
                'description_ar': 'يهدف إلى إنجاز مهمة عملية فورية',
                'intent_keywords': [
                    'write', 'create', 'make', 'build', 'generate', 'code',
                    'fix', 'solve', 'debug', 'implement', 'develop', 'translate',
                    'summarize', 'convert', 'calculate', 'design'
                ],
                'question_patterns': [
                    r'write', r'create', r'make', r'build', r'generate',
                    r'fix', r'solve', r'can you', r'help me (write|create|make)'
                ],
                'score': 0.0
            },
            'explorer': {
                'name_ar': 'المستكشف',
                'name_en': 'The Explorer',
                'description_ar': 'أسئلة استطلاعية ومقارنات وطلب آراء',
                'intent_keywords': [
                    'compare', 'difference', 'opinion', 'what about', 'suggest',
                    'recommend', 'best', 'better', 'versus', 'vs', 'alternative',
                    'options', 'which', 'should i', 'what do you think'
                ],
                'question_patterns': [
                    r'compare', r'difference between', r'vs', r'versus',
                    r'what about', r'better', r'best', r'recommend',
                    r'which (one|is)', r'should i'
                ],
                'score': 0.0
            },
            'problem_solver': {
                'name_ar': 'حلّال المشاكل',
                'name_en': 'The Problem Solver',
                'description_ar': 'يبحث عن حلول لمشاكل ومواقف محددة',
                'intent_keywords': [
                    'problem', 'issue', 'error', 'trouble', 'not working',
                    'broken', 'fails', 'how to fix', 'solution', 'resolve',
                    'troubleshoot', 'debug', 'workaround'
                ],
                'question_patterns': [
                    r'problem', r'issue', r'error', r'not working', r'fails',
                    r'how to fix', r'how can i solve', r'solution for'
                ],
                'score': 0.0
            }
        }
        
        # 3️⃣ مستويات التعقيد
        self.complexity_levels = {
            'beginner': {
                'name_ar': 'مبتدئ',
                'indicators': ['basic', 'simple', 'beginner', 'start', 'first time', 'new to', 'introduction']
            },
            'intermediate': {
                'name_ar': 'متوسط',
                'indicators': ['how to', 'best practice', 'should i', 'recommend', 'improve']
            },
            'advanced': {
                'name_ar': 'متقدم',
                'indicators': ['optimize', 'advanced', 'complex', 'sophisticated', 'architecture', 'edge case', 'performance']
            }
        }
    
    def analyze_conversation_batch(self, conversations: List[Dict]) -> Dict:
        """
        تحليل مجموعة محادثات لإنشاء البروفايل المعرفي الكامل
        
        Args:
            conversations: قائمة من المحادثات [{prompt, timestamp, ...}]
        
        Returns:
            البروفايل المعرفي الكامل
        """
        # إعادة تعيين الأوزان
        for topic in self.topic_categories.values():
            topic['weight'] = 0.0
        for persona in self.persona_types.values():
            persona['score'] = 0.0
        
        total_prompts = 0
        all_keywords = []
        timeline_data = []
        complexity_counts = Counter()
        
        # تحليل كل محادثة
        for conv in conversations:
            prompts = conv.get('prompts', [conv.get('prompt', '')])
            timestamp = conv.get('timestamp', datetime.now())
            
            if isinstance(prompts, str):
                try:
                    prompts = json.loads(prompts)
                except:
                    prompts = [prompts]
            
            for prompt in prompts:
                if not prompt or len(prompt.strip()) < 3:
                    continue
                
                total_prompts += 1
                prompt_lower = prompt.lower()
                
                # 1. تحليل الموضوع (Topic)
                topic_scores = self._analyze_topic(prompt_lower)
                for topic_id, score in topic_scores.items():
                    self.topic_categories[topic_id]['weight'] += score
                
                # 2. تحليل الشخصية (Persona)
                persona_scores = self._analyze_persona(prompt_lower)
                for persona_id, score in persona_scores.items():
                    self.persona_types[persona_id]['score'] += score
                
                # 3. تحليل التعقيد
                complexity = self._determine_complexity(prompt_lower)
                complexity_counts[complexity] += 1
                
                # 4. استخراج الكلمات المفتاحية
                keywords = self._extract_keywords(prompt_lower)
                all_keywords.extend(keywords)
                
                # 5. بيانات الخط الزمني
                timeline_data.append({
                    'timestamp': timestamp,
                    'topics': list(topic_scores.keys()),
                    'primary_topic': max(topic_scores, key=topic_scores.get) if topic_scores else None
                })
        
        # حساب النسب المئوية
        if total_prompts > 0:
            for topic in self.topic_categories.values():
                topic['percentage'] = round((topic['weight'] / total_prompts) * 100, 1)
            
            for persona in self.persona_types.values():
                persona['percentage'] = round((persona['score'] / total_prompts) * 100, 1)
        
        # تحديد الشخصية الرئيسية
        primary_persona = max(
            self.persona_types.items(),
            key=lambda x: x[1]['score']
        )[0]
        
        # تحديد أهم المواضيع
        top_topics = sorted(
            self.topic_categories.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:3]
        
        # إنشاء البروفايل النهائي
        profile = {
            'summary': {
                'total_conversations': len(conversations),
                'total_prompts': total_prompts,
                'primary_persona': {
                    'id': primary_persona,
                    'name_ar': self.persona_types[primary_persona]['name_ar'],
                    'name_en': self.persona_types[primary_persona]['name_en'],
                    'description_ar': self.persona_types[primary_persona]['description_ar'],
                    'percentage': self.persona_types[primary_persona]['percentage']
                },
                'complexity_level': complexity_counts.most_common(1)[0][0] if complexity_counts else 'intermediate'
            },
            'topic_distribution': {
                topic_id: {
                    'name_ar': data['name_ar'],
                    'name_en': data['name_en'],
                    'percentage': data['percentage']
                }
                for topic_id, data in self.topic_categories.items()
                if data['percentage'] > 0
            },
            'persona_distribution': {
                persona_id: {
                    'name_ar': data['name_ar'],
                    'name_en': data['name_en'],
                    'description_ar': data['description_ar'],
                    'percentage': data['percentage']
                }
                for persona_id, data in self.persona_types.items()
                if data['percentage'] > 0
            },
            'top_keywords': dict(Counter(all_keywords).most_common(30)),
            'timeline': self._create_timeline_summary(timeline_data),
            'complexity_distribution': dict(complexity_counts)
        }
        
        return profile
    
    def _analyze_topic(self, text: str) -> Dict[str, float]:
        """تحليل الموضوع من النص"""
        scores = {}
        
        for topic_id, topic_data in self.topic_categories.items():
            score = 0.0
            
            # حساب مطابقة الكلمات المفتاحية
            keyword_matches = sum(1 for kw in topic_data['keywords'] if kw in text)
            score += keyword_matches * 0.5
            
            # حساب مطابقة مؤشرات النية
            intent_matches = sum(1 for intent in topic_data['intent_indicators'] if intent in text)
            score += intent_matches * 1.0
            
            if score > 0:
                scores[topic_id] = score
        
        return scores
    
    def _analyze_persona(self, text: str) -> Dict[str, float]:
        """تحليل نوع الشخصية من النص"""
        scores = {}
        
        for persona_id, persona_data in self.persona_types.items():
            score = 0.0
            
            # حساب مطابقة كلمات النية
            intent_matches = sum(1 for kw in persona_data['intent_keywords'] if kw in text)
            score += intent_matches * 0.5
            
            # حساب مطابقة الأنماط
            for pattern in persona_data['question_patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1.0
            
            if score > 0:
                scores[persona_id] = score
        
        return scores
    
    def _determine_complexity(self, text: str) -> str:
        """تحديد مستوى التعقيد"""
        scores = {level: 0 for level in self.complexity_levels.keys()}
        
        for level, data in self.complexity_levels.items():
            scores[level] = sum(1 for ind in data['indicators'] if ind in text)
        
        # إضافة عوامل إضافية
        word_count = len(text.split())
        if word_count > 30:
            scores['advanced'] += 1
        elif word_count < 10:
            scores['beginner'] += 1
        
        return max(scores, key=scores.get) or 'intermediate'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """استخراج الكلمات المفتاحية التقنية"""
        # قائمة الكلمات الوقفية البسيطة
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                     'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this',
                     'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                     'we', 'they', 'what', 'how', 'can', 'could', 'would', 'should'}
        
        words = re.findall(r'\b[a-z]{4,}\b', text)
        keywords = [w for w in words if w not in stop_words]
        
        # استخراج الكلمات التقنية المهمة
        technical_keywords = []
        for topic_data in self.topic_categories.values():
            technical_keywords.extend(topic_data['keywords'])
        
        important_words = [w for w in keywords if w in technical_keywords]
        
        return important_words[:10]  # أهم 10 كلمات
    
    def _create_timeline_summary(self, timeline_data: List[Dict]) -> Dict:
        """إنشاء ملخص الخط الزمني"""
        if not timeline_data:
            return {}
        
        # ترتيب حسب الوقت
        sorted_data = sorted(timeline_data, key=lambda x: x['timestamp'])
        
        # تجميع حسب الفترات الزمنية (يومي/أسبوعي)
        timeline_summary = {
            'earliest_interaction': sorted_data[0]['timestamp'].isoformat() if sorted_data else None,
            'latest_interaction': sorted_data[-1]['timestamp'].isoformat() if sorted_data else None,
            'topic_evolution': []
        }
        
        # تحليل تطور المواضيع
        if len(sorted_data) >= 3:
            early_topics = [t for d in sorted_data[:len(sorted_data)//3] for t in d['topics']]
            late_topics = [t for d in sorted_data[-len(sorted_data)//3:] for t in d['topics']]
            
            timeline_summary['early_interests'] = dict(Counter(early_topics).most_common(3))
            timeline_summary['recent_interests'] = dict(Counter(late_topics).most_common(3))
        
        return timeline_summary
    
    def generate_dashboard_data(self, profile: Dict) -> Dict:
        """
        تحويل البروفايل إلى بيانات جاهزة للعرض في Dashboard
        """
        dashboard = {
            'pie_chart': {
                'labels': [],
                'values': [],
                'colors': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
            },
            'persona_card': profile['summary']['primary_persona'],
            'word_cloud': profile['top_keywords'],
            'timeline': profile.get('timeline', {}),
            'stats': {
                'total_conversations': profile['summary']['total_conversations'],
                'total_prompts': profile['summary']['total_prompts'],
                'complexity': profile['summary']['complexity_level']
            }
        }
        
        # بيانات المخطط الدائري
        for topic_id, topic_data in profile['topic_distribution'].items():
            if topic_data['percentage'] > 0:
                dashboard['pie_chart']['labels'].append(topic_data['name_ar'])
                dashboard['pie_chart']['values'].append(topic_data['percentage'])
        
        return dashboard


# 🧪 مثال على الاستخدام
if __name__ == "__main__":
    # عينة من البيانات
    sample_conversations = [
        {
            'prompts': ["How can I create a test set for a very rare category? I want to build a classifier."],
            'timestamp': datetime(2024, 12, 1, 10, 30)
        },
        {
            'prompts': ["explain function calling. how would you call a function?"],
            'timestamp': datetime(2024, 12, 2, 14, 15)
        },
        {
            'prompts': [
                "What is the difference between marriage license and marriage certificate?",
                "How can I get both of them as quick as possible in California"
            ],
            'timestamp': datetime(2024, 12, 3, 9, 45)
        },
        {
            'prompts': [
                "Is it morally right to try to have a certain percentage of females on managerial positions?",
                "OK, does pineapple belong on a pizza? Relax and give me fun answer."
            ],
            'timestamp': datetime(2024, 12, 4, 16, 20)
        }
    ]
    
    # إنشاء المحلل
    analyzer = CognitiveProfileAnalyzer()
    
    # تحليل المحادثات
    cognitive_profile = analyzer.analyze_conversation_batch(sample_conversations)
    
    # طباعة النتائج
    print("=" * 70)
    print("🧠 البروفايل المعرفي للمستخدم")
    print("=" * 70)
    
    print("\n📊 الملخص:")
    print(f"  • إجمالي المحادثات: {cognitive_profile['summary']['total_conversations']}")
    print(f"  • إجمالي الأسئلة: {cognitive_profile['summary']['total_prompts']}")
    print(f"  • مستوى التعقيد: {cognitive_profile['summary']['complexity_level']}")
    
    print("\n👤 الشخصية الرئيسية:")
    primary = cognitive_profile['summary']['primary_persona']
    print(f"  • {primary['name_ar']} ({primary['name_en']})")
    print(f"  • {primary['description_ar']}")
    print(f"  • النسبة: {primary['percentage']}%")
    
    print("\n🎯 توزيع الميول الرئيسية:")
    for topic_id, topic_data in sorted(
        cognitive_profile['topic_distribution'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    ):
        print(f"  • {topic_data['name_ar']}: {topic_data['percentage']}%")
    
    print("\n🔍 توزيع الشخصيات:")
    for persona_id, persona_data in sorted(
        cognitive_profile['persona_distribution'].items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    ):
        print(f"  • {persona_data['name_ar']}: {persona_data['percentage']}%")
    
    print("\n🏷️ أهم الكلمات المفتاحية:")
    top_words = list(cognitive_profile['top_keywords'].items())[:10]
    for word, count in top_words:
        print(f"  • {word}: {count}")
    
    # إنشاء بيانات Dashboard
    dashboard_data = analyzer.generate_dashboard_data(cognitive_profile)
    
    print("\n📈 بيانات Dashboard (للواجهة الأمامية):")
    print(json.dumps(dashboard_data, ensure_ascii=False, indent=2))