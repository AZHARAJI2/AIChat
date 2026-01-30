import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from io import BytesIO
import sys
import os
import json
from datetime import datetime
import re

# إضافة المسار الجذري للمشروع للتمكن من استيراد الموديلات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.profile_generator import ProfileGenerator

# ---------------- دوال التحقق من المدخلات ----------------
def is_valid_arabic_text(text: str) -> bool:
    """
    التحقق من أن النص يحتوي على حروف عربية
    يقبل النصوص المختلطة (عربي مع أرقام أو إنجليزي)
    
    Args:
        text: النص المراد التحقق منه
    
    Returns:
        True إذا كان النص يحتوي على حروف عربية، False خلاف ذلك
    """
    if not text or not text.strip():
        return False
    
    # التحقق من وجود حروف عربية فعلية
    has_arabic = bool(re.search(r'[\u0621-\u064A\u0671-\u06D3]', text))
    if not has_arabic:
        return False
    
    # حساب نسبة الحروف العربية في النص
    arabic_chars = len(re.findall(r'[\u0621-\u064A\u0671-\u06D3]', text))
    total_chars = len(re.findall(r'[\w]', text))  # جميع الحروف والأرقام
    
    # يجب أن يكون على الأقل 30% من الأحرف عربية
    if total_chars > 0 and (arabic_chars / total_chars) < 0.3:
        return False
    
    return True


def validate_conversation_file(texts: list) -> tuple:
    """
    التحقق من أن الملف يحتوي على محادثات صالحة
    يقبل الملفات المختلطة (عربي وإنجليزي) ويستخرج فقط النصوص العربية الصالحة
    
    Args:
        texts: قائمة النصوص المستخرجة من الملف
    
    Returns:
        (is_valid: bool, error_message: str, valid_texts: list)
    """
    if not texts:
        return False, "الملف لا يحتوي على أي نصوص", []
    
    # تصفية النصوص واستخراج فقط النصوص العربية الصالحة
    valid_texts = []
    total_texts = 0
    
    for text in texts:
        # تجاهل النصوص القصيرة جداً
        if not text or len(text.strip()) < 3:
            continue
        
        total_texts += 1
        
        # إذا كان النص عربي صالح، نضيفه
        if is_valid_arabic_text(text):
            valid_texts.append(text)
        # إذا لم يكن صالح، نتجاهله فقط (لا نرفض الملف)
    
    # يجب أن يكون هناك على الأقل 3 نصوص عربية صالحة
    if len(valid_texts) < 3:
        return False, f"الملف يحتوي على {total_texts} نص، لكن {len(valid_texts)} فقط عربي صالح. يجب أن يحتوي الملف على 3 نصوص عربية على الأقل (بدون أرقام أو حروف إنجليزية).", []
    
    # التحقق من أن النصوص تبدو كمحادثات (تحتوي على كلمات استفهام أو جمل)
    question_words = ['كيف', 'ماذا', 'ما', 'هل', 'لماذا', 'أين', 'متى', 'من', 'اشرح', 'وضح', 'اكتب', 'قل']
    has_questions = False
    
    for text in valid_texts[:10]:  # فحص أول 10 نصوص
        text_lower = text.lower()
        if any(word in text_lower for word in question_words) or '؟' in text:
            has_questions = True
            break
    
    if not has_questions:
        return False, "النصوص العربية المستخرجة لا تبدو كمحادثات (لا توجد أسئلة أو حوارات). تأكد من أن الملف يحتوي على محادثات فعلية.", []
    
    return True, "", valid_texts


@st.cache_resource
def get_generator():
    return ProfileGenerator()

generator = get_generator()

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="Dashboard المحادثات", layout="wide")
st.markdown("""
<h1 style='text-align: center; color: #0B3D91; font-family:Cairo;'>📊 Dashboard تحليل المحادثات</h1>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("خيارات التحليل")
uploaded_file = st.sidebar.file_uploader("اختر ملف (JSON, CSV, Excel, TXT)")
question = st.sidebar.text_input("اكتب سؤال للتحليل")
analyze_btn = st.sidebar.button("تحليل")

# ---------- مثال بيانات وهمية للعرض ----------
example_data = pd.DataFrame({
    "موضوع": ["تقني", "أدبي", "حياتي"],
    "نسبة": [40, 30, 30],
    "تاريخ": ["2026-01-01", "2026-01-02", "2026-01-03"]
})

# ---------- عرض البيانات ----------
if uploaded_file:
    # إعادة مؤشر الملف للبداية لضمان القراءة الصحيحة
    uploaded_file.seek(0)
    
    # التحقق من أن الملف غير فارغ
    if uploaded_file.size == 0:
        st.error("⚠️ الملف الذي قمت برفعه فارغ!")
    else:
        try:
            texts = []
            file_name = uploaded_file.name.lower()
            
            # محاولة قراءة الملف حسب النوع
            if file_name.endswith('.json'):
                bytes_data = uploaded_file.read()
                if not bytes_data:
                    st.error("⚠️ ملف JSON فارغ!")
                else:
                    file_content = json.loads(bytes_data)
                    # دعم هيكليات ملفات مختلفة (قائمة أو قاموس واحد أو نص واحد)
                    items = file_content if isinstance(file_content, list) else [file_content]
                    for item in items:
                        if isinstance(item, str):
                            texts.append(item)
                        elif isinstance(item, dict):
                            # البحث عن الحقل الذي يحتوي على النص (دعم عدة أسماء شائعة)
                            prompts = item.get('prompts_ar', item.get('prompts', item.get('text', item.get('content', []))))
                            if isinstance(prompts, str): prompts = [prompts]
                            if isinstance(prompts, list):
                                texts.extend([p for p in prompts if p and isinstance(p, str) and len(p.strip()) > 3])
            
            elif file_name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if len(line.strip()) > 3]
            
            elif file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # محاولة العثور على عمود يحتوي على نصوص
                text_col = next((col for col in df.columns if any(k in col.lower() for k in ['text', 'prompt', 'content', 'سؤال'])), df.columns[0])
                texts = df[text_col].dropna().astype(str).tolist()
                texts = [t for t in texts if len(t.strip()) > 3]

            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
                text_col = next((col for col in df.columns if any(k in col.lower() for k in ['text', 'prompt', 'content', 'سؤال'])), df.columns[0])
                texts = df[text_col].dropna().astype(str).tolist()
                texts = [t for t in texts if len(t.strip()) > 3]

            # التحقق من صحة المحادثات
            is_valid, error_msg, valid_texts = validate_conversation_file(texts)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
                st.info("💡 تأكد من أن الملف يحتوي على محادثات باللغة العربية (يمكن أن يحتوي على نصوص إنجليزية، لكن يجب أن يكون هناك 3 محادثات عربية صالحة على الأقل)")
            else:
                # عرض معلومات الملف
                if len(texts) > len(valid_texts):
                    st.success(f"✅ تم استخراج {len(valid_texts)} محادثة عربية من إجمالي {len(texts)} نص في الملف")
                    st.info(f"ℹ️ تم تجاهل {len(texts) - len(valid_texts)} نص يحتوي على إنجليزي أو أرقام")
                else:
                    st.success(f"✅ تم تحميل {len(valid_texts)} محادثة بنجاح")
                
                # التحليل
                profile = generator.analyze_batch(valid_texts)
                
                # ============ الملخص التنفيذي ============
                st.markdown("---")
                st.markdown("""
                <h2 style='text-align: center; color: #0B3D91;'>📋 الملخص التنفيذي</h2>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("إجمالي المحادثات", profile['summary']['total_prompts'])
                with col2:
                    st.metric("المحادثات المحللة", profile['summary']['analyzed_prompts'])
                with col3:
                    primary = profile['summary']['primary_persona']
                    st.metric("الشخصية الرئيسية", primary['name_ar'], f"{primary['percentage']}%")
                with col4:
                    st.metric("مستوى التعقيد", profile['summary']['complexity_ar'])
                
                # ============ نماذج الشخصية (MBTI & Holland) ============
                st.markdown("---")
                st.subheader("🧬 تحليل الشخصية حسب النماذج العالمية")
                
                col_mbti, col_holland = st.columns(2)
                with col_mbti:
                    st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#E3F2FD; border-left: 5px solid #1976D2;'>
                        <h4 style='color:#1976D2; margin:0;'>📊 نموذج MBTI</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    mbti = profile['summary'].get('mbti_type', {})
                    if mbti:
                        st.write(f"**النوع:** {mbti.get('type', 'N/A')}")
                        st.write(f"**الاسم:** {mbti.get('name_ar', 'N/A')}")
                        st.write(f"**الوصف:** {mbti.get('description', 'N/A')}")
                
                with col_holland:
                    st.markdown("""
                    <div style='padding:20px; border-radius:10px; background-color:#F3E5F5; border-left: 5px solid #7B1FA2;'>
                        <h4 style='color:#7B1FA2; margin:0;'>🎯 نموذج هولاند المهني</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    holland = profile['summary'].get('holland_code', {})
                    if holland:
                        st.write(f"**الكود:** {holland.get('code', 'N/A')}")
                        st.write(f"**النوع الأساسي:** {holland.get('name_ar', 'N/A')}")
                        st.write(f"**الوصف:** {holland.get('description', 'N/A')}")
                
                # ============ توزيع المواضيع ============
                st.markdown("---")
                st.subheader("📈 توزيع المواضيع")
                
                topic_data = pd.DataFrame([
                    {"موضوع": data['name_ar'], "عدد": data['count'], "نسبة": data['percentage']}
                    for data in profile['topic_distribution'].values()
                    if data.get('percentage', 0) > 0
                ]).sort_values('نسبة', ascending=False)
                
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    fig_pie = px.pie(topic_data, names='موضوع', values='نسبة',
                                     color_discrete_sequence=px.colors.qualitative.Set3,
                                     title='التوزيع النسبي للمواضيع')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_table:
                    st.dataframe(topic_data, use_container_width=True, hide_index=True)
                
                # ============ توزيع الشخصيات ============
                st.markdown("---")
                st.subheader("👥 توزيع الشخصيات")
                
                persona_data = pd.DataFrame([
                    {
                        "الشخصية": data['name_ar'],
                        "الوصف": data['description_ar'],
                        "النسبة": data['percentage']
                    }
                    for data in profile['persona_distribution'].values()
                    if data.get('percentage', 0) > 0
                ]).sort_values('النسبة', ascending=False)
                
                # مخطط شريطي للشخصيات
                fig_bar = px.bar(persona_data, x='النسبة', y='الشخصية',
                                 orientation='h',
                                 title='توزيع أنماط الشخصية',
                                 color='النسبة',
                                 color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # جدول تفصيلي للشخصيات
                st.dataframe(persona_data, use_container_width=True, hide_index=True)
                
                # ============ سحابة الكلمات المفتاحية ============
                st.markdown("---")
                st.subheader("☁️ سحابة الكلمات المفتاحية")
                if profile['top_keywords']:
                    wordcloud = WordCloud(width=1000, height=500, background_color='white',
                                          colormap='viridis', font_path=None,
                                          prefer_horizontal=0.7, relative_scaling=0.5,
                                          min_font_size=10).generate_from_frequencies(profile['top_keywords'])
                    st.image(wordcloud.to_array(), use_container_width=True)
                else:
                    st.info("لا توجد كلمات كافية لإنشاء سحابة كلمات.")
                
                # ============ توزيع مستويات التعقيد ============
                st.markdown("---")
                st.subheader("🎯 توزيع مستويات التعقيد")
                
                if profile.get('complexity_distribution'):
                    complexity_names = {
                        'beginner': 'مبتدئ',
                        'intermediate': 'متوسط',
                        'advanced': 'متقدم'
                    }
                    
                    complexity_df = pd.DataFrame([
                        {'المستوى': complexity_names.get(level, level), 'العدد': count}
                        for level, count in profile['complexity_distribution'].items()
                    ])
                    
                    fig_complexity = px.bar(complexity_df, x='المستوى', y='العدد',
                                           title='توزيع الأسئلة حسب مستوى التعقيد',
                                           color='العدد',
                                           color_continuous_scale='Blues')
                    st.plotly_chart(fig_complexity, use_container_width=True)
                
                # ============ الجدول الزمني ============
                if profile.get('timeline') and profile['timeline'].get('total_entries', 0) > 0:
                    st.markdown("---")
                    st.subheader("⏰ تطور الاهتمامات عبر الزمن")
                    
                    # خريطة ترجمة أسماء المواضيع للعربية
                    topic_names_ar = {
                        'tech_programming': 'التقنيات والبرمجة',
                        'academic_cultural': 'الأكاديمي والثقافي',
                        'lifestyle_health': 'نمط الحياة والصحة',
                        'business_economy': 'الأعمال والاقتصاد',
                        'creative_language': 'الإبداع واللغة',
                        'undefined': 'غير محدد'
                    }
                    
                    timeline = profile['timeline']
                    if 'early_interests' in timeline:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**🌅 الاهتمامات المبكرة**")
                            for topic_id, count in timeline.get('early_interests', {}).items():
                                topic_name = topic_names_ar.get(topic_id, topic_id)
                                st.write(f"- {topic_name}: {count}")
                        
                        with col2:
                            st.markdown("**🕐 الاهتمامات الوسطى**")
                            for topic_id, count in timeline.get('mid_interests', {}).items():
                                topic_name = topic_names_ar.get(topic_id, topic_id)
                                st.write(f"- {topic_name}: {count}")
                        
                        with col3:
                            st.markdown("**🌆 الاهتمامات الحديثة**")
                            for topic_id, count in timeline.get('recent_interests', {}).items():
                                topic_name = topic_names_ar.get(topic_id, topic_id)
                                st.write(f"- {topic_name}: {count}")
                
        except json.JSONDecodeError:
            st.error("❌ خطأ في قراءة ملف JSON. تأكد من أن الملف بصيغة JSON صحيحة.")
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء معالجة الملف: {e}")

# ---------- تحليل سؤال واحد ----------
if analyze_btn and question:
    st.subheader("📝 تحليل السؤال الفردي")
    
    # التحقق من صحة المدخلات - يجب أن يحتوي على نص عربي
    if not is_valid_arabic_text(question):
        st.error("⚠️ الرجاء إدخال نص يحتوي على اللغة العربية!")
        st.info("💡 أمثلة صحيحة: 'كيف أتعلم البرمجة' | 'كيف أتعلم Python' | 'اشرح لي الذكاء الاصطناعي في 5 دقائق'")
        st.stop()
    
    # استخدام منطق الباك اند الحقيقي
    result = generator.analyze_single(question)

    # ---- عرض النتائج بشكل Cards ----
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#0B3D91; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>السؤال</h4>{result['original_text']}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#1E90FF; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>الموضوع</h4>{result['topic']['category_ar']}<br><small>الثقة: {result['topic']['confidence']:.2f}</small></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#00BFFF; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>الشخصية</h4>{result['persona']['persona_ar']}<br><small>{result['persona']['description_ar']}</small></div>", unsafe_allow_html=True)

# ---------- تصدير النتائج ----------
if (uploaded_file and 'profile' in locals()) or (analyze_btn and 'result' in locals()):
    st.subheader("💾 تصدير النتائج")
    
    if uploaded_file and 'profile' in locals():
        # تصدير توزيع المواضيع في حالة الملف
        to_export = pd.DataFrame([
            {"الموضوع": d['name_ar'], "التكرار": d['count'], "النسبة": d['percentage']}
            for d in profile['topic_distribution'].values()
        ])
    else:
        # تجهيز بيانات السؤال الفردي للتصدير
        flat_result = {
            "السؤال": result['original_text'],
            "الموضوع": result['topic']['category_ar'],
            "نص_الموضوع": result['topic']['category_en'],
            "الشخصية": result['persona']['persona_ar'],
            "وصف_الشخصية": result['persona']['description_ar']
        }
        to_export = pd.DataFrame([flat_result])
    
    output = BytesIO()
    to_export.to_excel(output, index=False)
    st.download_button(
        label="Download Excel",
        data=output,
        file_name="analysis_results.xlsx",
        mime="application/vnd.ms-excel"
    )

# ---------- Footer ----------
st.markdown("---")
# تمت إزالة رسالة الربط بناءً على طلب المستخدم
