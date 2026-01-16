"""
خط أنابيب كامل لتحميل وترجمة وتحليل بيانات Chatbot Arena
معدّل لقراءة البيانات من ملف CSV (train.csv)
Pipeline: Load CSV → Translate → Classify → Generate Profile

🆕 التحسينات المضافة:
- معالجة متوازية (Parallel Processing) لتسريع الترجمة
- استئناف من آخر نقطة توقف (Resume from checkpoint)
- حفظ التقدم كل 50 سطر بدلاً من 100
"""

import pandas as pd
import translators as ts
import json
from tqdm import tqdm
import time
from typing import List, Dict
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class ChatbotArenaArabicPipeline:
    """
    خط أنابيب شامل لمعالجة بيانات Chatbot Arena
    مع دعم المعالجة المتوازية والاستئناف
    """

    def __init__(self, cache_dir='./cache'):
        self.cache_dir = cache_dir
        self.dataset = None
        self.processed_data = []
        self._lock = threading.Lock()  # للمعالجة المتوازية الآمنة
        os.makedirs(self.cache_dir, exist_ok=True)

    def step1_load_from_csv(self, csv_file_path, sample_size=None):
        """
        الخطوة 1: تحميل البيانات من ملف CSV

        Args:
            csv_file_path: مسار ملف CSV
            sample_size: عدد العينات (استخدم None لتحميل الكل)
        """
        print("=" * 70)
        print("📥 الخطوة 1: تحميل بيانات من ملف CSV")
        print("=" * 70)

        try:
            # قراءة ملف CSV
            print(f"\n⏳ جاري القراءة من: {csv_file_path}")
            df = pd.read_csv(csv_file_path)

            print(f"✅ تم التحميل بنجاح!")
            print(f"   إجمالي الصفوف: {len(df)}")
            print(f"   الأعمدة الموجودة: {', '.join(df.columns.tolist())}")

            # أخذ عينة إذا طُلب ذلك
            if sample_size and sample_size < len(df):
                df = df.head(sample_size)
                print(f"   تم اختيار عينة: {sample_size} صف")

            # حفظ البيانات
            self.dataset = df

            # عرض مثال
            print("\n📄 مثال على البيانات:")
            print(df.head(2))

            return True

        except FileNotFoundError:
            print(f"❌ خطأ: لم يتم العثور على الملف: {csv_file_path}")
            print("   تأكد من المسار الصحيح للملف")
            return False
        except Exception as e:
            print(f"❌ خطأ في القراءة: {e}")
            return False

    def _get_checkpoint_path(self):
        """الحصول على مسار ملف الاستئناف"""
        return f"{self.cache_dir}/checkpoint.pkl"

    def _get_last_progress_path(self):
        """الحصول على مسار آخر ملف تقدم"""
        return f"{self.cache_dir}/last_progress.pkl"

    def _load_checkpoint(self):
        """تحميل نقطة الاستئناف إذا وجدت"""
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"📂 تم العثور على نقطة استئناف: {len(data['translated_data'])} عنصر مترجم")
                return data
            except Exception as e:
                print(f"⚠️ خطأ في تحميل نقطة الاستئناف: {e}")
        return None

    def _save_checkpoint(self, translated_data, processed_indices, current_idx):
        """حفظ نقطة الاستئناف"""
        checkpoint_data = {
            'translated_data': translated_data,
            'processed_indices': processed_indices,
            'last_index': current_idx,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(self._get_checkpoint_path(), 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def _translate_single(self, idx, row, prompt_column, id_column, 
                          model_a_column, model_b_column, winner_column, timestamp_column):
        """ترجمة سطر واحد - تستخدم في المعالجة المتوازية"""
        try:
            prompt_text = str(row[prompt_column])
            arabic_text = ""
            
            if len(prompt_text.strip()) > 0:
                try:
                    arabic_text = ts.translate_text(
                        query_text=prompt_text,
                        translator='google',
                        from_language='en',
                        to_language='ar'
                    )
                    time.sleep(0.3)  # تجنب rate limiting
                except Exception as trans_error:
                    arabic_text = f"[EN] {prompt_text}"
                    time.sleep(1)
                    return idx, None, str(trans_error)[:50]

            # تحديد الفائز
            winner = 'tie'
            if winner_column and pd.notna(row.get(winner_column, None)):
                if row.get('winner_model_a', 0) == 1:
                    winner = row.get(model_a_column, 'N/A')
                elif row.get('winner_model_b', 0) == 1:
                    winner = row.get(model_b_column, 'N/A')

            translated_item = {
                'id': row.get(id_column, idx),
                'model_a': row.get(model_a_column, 'N/A'),
                'model_b': row.get(model_b_column, 'N/A'),
                'prompts_en': [prompt_text],
                'prompts_ar': [arabic_text],
                'winner': winner,
                'timestamp': row.get(timestamp_column, None)
            }

            return idx, translated_item, None

        except Exception as e:
            return idx, None, str(e)

    def step2_translate_to_arabic(self,
                                   prompt_column='prompt',
                                   id_column='question_id',
                                   model_a_column='model_a',
                                   model_b_column='model_b',
                                   winner_column='winner',
                                   timestamp_column='tstamp',
                                   batch_size=10,
                                   save_progress=True,
                                   resume=True,
                                   max_workers=2):
        """
        الخطوة 2: ترجمة البيانات للعربية باستخدام translators

        Args:
            prompt_column: اسم عمود النصوص المطلوب ترجمتها
            id_column: اسم عمود المعرف
            model_a_column: اسم عمود النموذج الأول
            model_b_column: اسم عمود النموذج الثاني
            winner_column: اسم عمود الفائز (اختياري)
            timestamp_column: اسم عمود الوقت (اختياري)
            batch_size: عدد العناصر في كل دفعة (غير مستخدم حالياً)
            save_progress: حفظ التقدم بشكل دوري
            resume: استئناف من آخر نقطة توقف
            max_workers: عدد الخيوط المتوازية (3 افتراضياً لتجنب rate limiting)
        """
        print("\n" + "=" * 70)
        print("🔄 الخطوة 2: ترجمة البيانات للعربية (مع دعم الاستئناف والمعالجة المتوازية)")
        print("=" * 70)

        if self.dataset is None:
            print("❌ يجب تحميل البيانات أولاً (step1)")
            return False

        df = self.dataset
        translated_data = []
        processed_indices = set()
        errors = 0
        start_from = 0

        # محاولة الاستئناف من نقطة سابقة
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                translated_data = checkpoint['translated_data']
                processed_indices = set(checkpoint['processed_indices'])
                start_from = checkpoint['last_index'] + 1
                print(f"🔄 استئناف من السطر {start_from}")
                user_input = input("هل تريد الاستئناف من هذه النقطة؟ (y/n): ").strip().lower()
                if user_input != 'y':
                    translated_data = []
                    processed_indices = set()
                    start_from = 0
                    print("🔄 بدء من الصفر...")

        # التحقق من وجود الأعمدة المطلوبة
        required_cols = [prompt_column]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            print(f"❌ خطأ: الأعمدة المطلوبة غير موجودة: {missing_required}")
            print(f"   الأعمدة المتاحة: {df.columns.tolist()}")
            return False

        # فلترة الصفوف التي لم تُترجم بعد
        remaining_df = df.iloc[start_from:]
        total_remaining = len(remaining_df)

        print(f"\n📊 الإحصائيات:")
        print(f"   • إجمالي الصفوف: {len(df)}")
        print(f"   • الصفوف المترجمة مسبقاً: {len(translated_data)}")
        print(f"   • الصفوف المتبقية: {total_remaining}")
        print(f"   • عدد الخيوط المتوازية: {max_workers}")
        print(f"\n⏳ جاري الترجمة...")
        print("💡 يمكنك إيقاف العملية في أي وقت (Ctrl+C) وسيتم حفظ التقدم\n")

        try:
            # المعالجة المتوازية
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                for idx, row in remaining_df.iterrows():
                    if idx in processed_indices:
                        continue
                    
                    future = executor.submit(
                        self._translate_single,
                        idx, row, prompt_column, id_column,
                        model_a_column, model_b_column, winner_column, timestamp_column
                    )
                    futures[future] = idx

                # معالجة النتائج مع شريط التقدم
                with tqdm(total=len(futures), desc="الترجمة") as pbar:
                    completed_count = 0
                    for future in as_completed(futures):
                        idx, result, error = future.result()
                        
                        if error:
                            errors += 1
                            tqdm.write(f"⚠️ خطأ في السطر {idx}: {error}")
                        
                        if result:
                            with self._lock:
                                translated_data.append(result)
                                processed_indices.add(idx)
                        
                        completed_count += 1
                        pbar.update(1)
                        
                        # حفظ التقدم كل 50 عنصر
                        if save_progress and completed_count % 50 == 0:
                            with self._lock:
                                self._save_checkpoint(translated_data, list(processed_indices), idx)
                                tqdm.write(f"💾 تم حفظ التقدم: {len(translated_data)} محادثة")

        except KeyboardInterrupt:
            print("\n\n⚠️ تم إيقاف العملية بواسطة المستخدم!")
            print("💾 جاري حفظ التقدم...")
            self._save_checkpoint(translated_data, list(processed_indices), idx)
            print(f"✅ تم حفظ {len(translated_data)} محادثة. يمكنك الاستئناف لاحقاً.")
            return False

        self.processed_data = translated_data

        print(f"\n✅ اكتملت الترجمة!")
        print(f"   محادثات مترجمة: {len(translated_data)}")
        print(f"   أخطاء: {errors}")
        if errors > 0:
            print(f"   ملاحظة: النصوص التي فشلت ترجمتها تم الاحتفاظ بها بالإنجليزية")

        # حفظ البيانات النهائية
        self._save_final_data(translated_data)
        
        # حذف ملف الاستئناف بعد الانتهاء بنجاح
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("🗑️ تم حذف ملف الاستئناف (اكتملت العملية بنجاح)")

        return True

    def step3_classify_and_analyze(self, analyzer):
        """
        الخطوة 3: تصنيف وتحليل البيانات المترجمة

        Args:
            analyzer: كائن CognitiveProfileAnalyzer
        """
        print("\n" + "=" * 70)
        print("🧠 الخطوة 3: تصنيف وتحليل البيانات")
        print("=" * 70)

        if not self.processed_data:
            print("❌ يجب ترجمة البيانات أولاً (step2)")
            return None

        # تحويل البيانات للصيغة المطلوبة
        conversations = []
        for item in self.processed_data:
            conv = {
                'prompts': item['prompts_ar'],
                'timestamp': item.get('timestamp'),
                'metadata': {
                    'id': item['id'],
                    'models': [item['model_a'], item['model_b']],
                    'winner': item['winner']
                }
            }
            conversations.append(conv)

        print(f"\n⏳ جاري التحليل...")

        # تحليل شامل
        cognitive_profile = analyzer.analyze_conversation_batch(conversations)

        print(f"✅ اكتمل التحليل!")

        return cognitive_profile

    def step4_generate_dashboard_json(self, profile, output_file='dashboard_data.json'):
        """
        الخطوة 4: إنشاء ملف JSON للـDashboard
        """
        print("\n" + "=" * 70)
        print("📊 الخطوة 4: إنشاء بيانات Dashboard")
        print("=" * 70)

        if not profile:
            print("❌ يجب إجراء التحليل أولاً (step3)")
            return False

        # إنشاء بيانات Dashboard
        dashboard_data = {
            'metadata': {
                'total_conversations': profile['summary']['total_conversations'],
                'total_prompts': profile['summary']['total_prompts'],
                'generated_at': pd.Timestamp.now().isoformat()
            },
            'primary_persona': profile['summary']['primary_persona'],
            'pie_chart': {
                'labels': [data['name_ar'] for data in profile['topic_distribution'].values()],
                'values': [data['percentage'] for data in profile['topic_distribution'].values()],
                'labels_en': [data['name_en'] for data in profile['topic_distribution'].values()]
            },
            'persona_breakdown': {
                persona_id: {
                    'name_ar': data['name_ar'],
                    'name_en': data['name_en'],
                    'percentage': data['percentage'],
                    'description_ar': data['description_ar']
                }
                for persona_id, data in profile['persona_distribution'].items()
            },
            'word_cloud': profile['top_keywords'],
            'timeline': profile.get('timeline', {}),
            'complexity': profile['summary']['complexity_level']
        }

        # حفظ كـJSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ تم حفظ بيانات Dashboard في: {output_file}")
        print(f"\n📋 يمكنك الآن استخدام هذا الملف في الواجهة الأمامية!")

        return True

    def _save_progress(self, data, count):
        """حفظ التقدم"""
        filename = f"{self.cache_dir}/progress_{count}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def _save_final_data(self, data):
        """حفظ البيانات النهائية"""
        # حفظ كـPickle
        with open(f"{self.cache_dir}/translated_data.pkl", 'wb') as f:
            pickle.dump(data, f)

        # حفظ كـJSON
        with open(f"{self.cache_dir}/translated_data.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 تم حفظ البيانات في: {self.cache_dir}/")


# 🚀 تنفيذ Pipeline الكامل
def run_complete_pipeline(csv_file_path='train.csv',
                          sample_size=None,
                          prompt_column='prompt',
                          id_column='id',
                          model_a_column='model_a',
                          model_b_column='model_b',
                          resume=True,
                          max_workers=3):
    """
    تشغيل Pipeline الكامل من البداية للنهاية

    Args:
        csv_file_path: مسار ملف CSV
        sample_size: عدد المحادثات للمعالجة (None لمعالجة الكل)
        prompt_column: اسم العمود الذي يحتوي على النصوص
        id_column: اسم عمود المعرف
        model_a_column: اسم عمود النموذج الأول
        model_b_column: اسم عمود النموذج الثاني
        resume: استئناف من آخر نقطة توقف (True افتراضياً)
        max_workers: عدد الخيوط المتوازية (3 افتراضياً)
    """
    print("\n" + "🚀" * 35)
    print("بدء Pipeline الكامل - قراءة من CSV")
    print("🚀" * 35)

    # إنشاء Pipeline
    pipeline = ChatbotArenaArabicPipeline()

    # استيراد المحلل
    try:
        import sys
        sys.path.append('.')
        from cognitive_profile_analyzer import CognitiveProfileAnalyzer
        analyzer = CognitiveProfileAnalyzer()
    except ImportError:
        print("❌ خطأ: لم يتم العثور على ملف cognitive_profile_analyzer.py")
        print("   تأكد أن الملف موجود في نفس المجلد")
        return

    # الخطوة 1: تحميل من CSV
    if not pipeline.step1_load_from_csv(csv_file_path, sample_size=sample_size):
        return

    # الخطوة 2: الترجمة (مع دعم الاستئناف والمعالجة المتوازية)
    if not pipeline.step2_translate_to_arabic(
        prompt_column=prompt_column,
        id_column=id_column,
        model_a_column=model_a_column,
        model_b_column=model_b_column,
        resume=resume,
        max_workers=max_workers
    ):
        return

    # الخطوة 3: التحليل
    profile = pipeline.step3_classify_and_analyze(analyzer)
    if not profile:
        return

    # الخطوة 4: إنشاء Dashboard
    pipeline.step4_generate_dashboard_json(profile)

    # طباعة النتيجة النهائية
    print("\n" + "=" * 70)
    print("🎉 اكتمل Pipeline بنجاح!")
    print("=" * 70)
    print("\n📂 الملفات المُنتجة:")
    print("   • cache/translated_data.json - البيانات المترجمة")
    print("   • cache/translated_data.pkl - البيانات (نسخة pickle)")
    print("   • dashboard_data.json - بيانات Dashboard (للواجهة)")

    print("\n📊 البروفايل المعرفي:")
    primary = profile['summary']['primary_persona']
    print(f"   • الشخصية: {primary['name_ar']}")
    print(f"   • مستوى التعقيد: {profile['summary']['complexity_level']}")
    print(f"   • محادثات: {profile['summary']['total_conversations']}")

    print("\n💡 الخطوة التالية:")
    print("   استخدم dashboard_data.json في تطبيق الواجهة الأمامية")


if __name__ == "__main__":
    # الإعدادات لملف train.csv

    csv_file = "train.csv"  # الملف في نفس المجلد

    print("\n⚙️ إعدادات التشغيل:")
    print(f"   • ملف CSV: {csv_file}")
    print("   • الأعمدة: id, model_a, model_b, prompt")
    print("   • مكتبة الترجمة: translators")
    print("   • 🆕 المعالجة المتوازية: مفعّلة (3 خيوط)")
    print("   • 🆕 الاستئناف: مفعّل")
    print("   • 🆕 حفظ التقدم: كل 50 عنصر")
    print("\n" + "-" * 70)

    # تشغيل Pipeline
    run_complete_pipeline(
        csv_file_path=csv_file,
        sample_size=None,  # None لمعالجة كل البيانات في الملف كاملاً
        prompt_column='prompt',
        id_column='id',
        model_a_column='model_a',
        model_b_column='model_b',
        resume=True,  # استئناف من آخر نقطة توقف
        max_workers=3  # عدد الخيوط المتوازية
    )