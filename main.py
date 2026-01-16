"""
نظام تحليل المحادثات الذكية
AI Chat Analysis System

نقطة الدخول الرئيسية
"""

import sys
import os
import json
import argparse

# إعداد الترميز للويندوز
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# إضافة المسار
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.arabic_preprocessor import ArabicPreprocessor
from models.topic_classifier import TopicClassifier
from models.persona_classifier import PersonaClassifier
from models.profile_generator import ProfileGenerator



def analyze_single_text(text: str) -> dict:
    """
    تحليل نص واحد
    
    Args:
        text: النص المُدخل
        
    Returns:
        نتيجة التحليل
    """
    generator = ProfileGenerator()
    return generator.analyze_single(text)


def analyze_file(input_file: str, output_file: str = None) -> dict:
    """
    تحليل ملف بيانات
    
    Args:
        input_file: مسار ملف الإدخال (JSON)
        output_file: مسار ملف الإخراج (اختياري)
        
    Returns:
        البروفايل المعرفي
    """
    generator = ProfileGenerator()
    
    if output_file is None:
        output_file = 'dashboard_data.json'
    
    profile = generator.analyze_from_file(input_file, output_file)
    return profile


    # تم إزالة كود الخادم (Backend) لأن هذا المستودع مخصص لنماذج الذكاء الاصطناعي فقط.
    pass


def interactive_mode():
    """
    الوضع التفاعلي للتحليل
    """
    print("=" * 70)
    print("🧠 نظام تحليل المحادثات الذكية - الوضع التفاعلي")
    print("=" * 70)
    print("اكتب سؤالاً أو نصاً للتحليل (اكتب 'خروج' للخروج)")
    print("-" * 70)
    
    generator = ProfileGenerator()
    
    while True:
        try:
            text = input("\n📝 أدخل النص: ").strip()
            
            if text.lower() in ['خروج', 'exit', 'quit', 'q']:
                print("\n👋 شكراً لاستخدام النظام!")
                break
            
            if not text:
                continue
            
            result = generator.analyze_single(text)
            
            print("\n" + "-" * 50)
            print(f"🏷️ الموضوع: {result['topic']['category_ar']}")
            print(f"   الثقة: {result['topic']['confidence']*100:.1f}%")
            print(f"   موثوق: {'✅' if result['topic']['is_confident'] else '❌'}")
            
            print(f"\n👤 الشخصية: {result['persona']['persona_ar']}")
            print(f"   {result['persona']['description_ar']}")
            print(f"   الثقة: {result['persona']['confidence']*100:.1f}%")
            
            print(f"\n📊 مستوى التعقيد: {result['complexity_ar']}")
            
            if result['keywords']:
                print(f"\n🏷️ الكلمات المفتاحية: {', '.join(result['keywords'])}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 شكراً لاستخدام النظام!")
            break


def main():
    """
    نقطة الدخول الرئيسية
    """
    parser = argparse.ArgumentParser(
        description='نظام تحليل المحادثات الذكية',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='الأوامر المتاحة')
    
    # أمر التحليل
    analyze_parser = subparsers.add_parser('analyze', help='تحليل نص أو ملف')
    analyze_parser.add_argument('-t', '--text', type=str, help='نص للتحليل')
    analyze_parser.add_argument('-f', '--file', type=str, help='ملف JSON للتحليل')
    analyze_parser.add_argument('-o', '--output', type=str, help='ملف الإخراج')
    
    # أمر الخادم (تم إزالته)
    # server_parser = subparsers.add_parser('server', help='تشغيل خادم API')
    # server_parser.add_argument('--host', type=str, default='0.0.0.0', help='العنوان')
    # server_parser.add_argument('--port', type=int, default=8000, help='المنفذ')
    
    # أمر التفاعلي
    subparsers.add_parser('interactive', help='الوضع التفاعلي')
    
    # أمر التجربة
    subparsers.add_parser('demo', help='تشغيل عرض توضيحي')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        if args.text:
            result = analyze_single_text(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.file:
            profile = analyze_file(args.file, args.output)
            print(f"\n✅ تم تحليل {profile['summary']['total_prompts']} سؤال")
            print(f"👤 الشخصية الرئيسية: {profile['summary']['primary_persona']['name_ar']}")
        else:
            print("❌ يرجى تحديد نص (-t) أو ملف (-f) للتحليل")
            
    elif args.command == 'server':
        print("❌ تم إزالة الخادم من هذه النسخة (مخصصة للذكاء الاصطناعي فقط)")
        
    elif args.command == 'interactive':
        interactive_mode()
        
    elif args.command == 'demo':
        run_demo()
        
    else:
        # بدون وسيطات - الوضع التفاعلي
        interactive_mode()


def run_demo():
    """
    تشغيل عرض توضيحي
    """
    print("=" * 70)
    print("🎬 عرض توضيحي لنظام تحليل المحادثات الذكية")
    print("=" * 70)
    
    demo_texts = [
        "كيف أكتب برنامج بايثون للذكاء الاصطناعي؟",
        "اشرح لي نظرية النسبية لأينشتاين",
        "ما هي أفضل وصفة للكبسة السعودية؟",
        "اكتب لي كود بايثون لحساب المتوسط",
        "ما الفرق بين React و Vue؟ أيهما أفضل للمبتدئين؟",
    ]
    
    generator = ProfileGenerator()
    
    print("\n📝 تحليل أمثلة متنوعة:")
    print("-" * 70)
    
    for text in demo_texts:
        result = generator.analyze_single(text)
        print(f"\n💬 \"{text[:40]}...\"")
        print(f"   🏷️ {result['topic']['category_ar']} ({result['topic']['confidence']*100:.0f}%)")
        print(f"   👤 {result['persona']['persona_ar']}")
    
    print("\n" + "-" * 70)
    print("📊 تحليل شامل للمجموعة:")
    
    profile = generator.analyze_batch(demo_texts)
    
    print(f"\n👤 الشخصية الرئيسية: {profile['summary']['primary_persona']['name_ar']}")
    print(f"   {profile['summary']['primary_persona']['description_ar']}")
    
    print("\n🎯 توزيع الموضوعات:")
    for topic_id, data in sorted(
        profile['topic_distribution'].items(),
        key=lambda x: x[1].get('percentage', 0),
        reverse=True
    ):
        if data.get('percentage', 0) > 0:
            print(f"   • {data['name_ar']}: {data['percentage']}%")
    
    print("\n✅ اكتمل العرض التوضيحي!")


if __name__ == "__main__":
    main()
