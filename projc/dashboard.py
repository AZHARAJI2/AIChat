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

# إضافة المسار الجذري للمشروع للتمكن من استيراد الموديلات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.profile_generator import ProfileGenerator

@st.cache_resource
def get_generator():
    return ProfileGenerator()

generator = get_generator()

# ---------------- إعداد الصفحة ----------------
st.set_page_config(page_title="Dashboard المحادثات", layout="wide")
st.markdown("""
<h1 style='text-align: center; color: #0B3D91; font-family:Cairo;'>📊 Dashboard تحليل المحادثات</h1>
<p style='text-align: center; font-family:Cairo; color: #333;'>واجهة تفاعلية لعرض المواضيع والكلمات الأكثر تكرارًا وتصنيف الأسئلة</p>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("خيارات التحليل")
uploaded_file = st.sidebar.file_uploader("اختر ملف نصي / CSV / JSON")
question = st.sidebar.text_input("اكتب سؤال للتحليل")
analyze_btn = st.sidebar.button("تحليل")

# ---------- مثال بيانات وهمية للعرض ----------
example_data = pd.DataFrame({
    "موضوع": ["تقني", "أدبي", "حياتي"],
    "نسبة": [40, 30, 30],
    "تاريخ": ["2026-01-01", "2026-01-02", "2026-01-03"]
})
example_text = "Python شبكات AI تعلم تعليم Python شبكات AI تعلم تعليم تحليل بيانات"

# ---------- عرض البيانات ----------
if uploaded_file:
    # قراءة الملف (دعم JSON حالياً)
    try:
        file_content = json.load(uploaded_file)
        # استخراج النصوص للتحليل
        texts = []
        for item in file_content:
            prompts = item.get('prompts_ar', item.get('prompts', []))
            if isinstance(prompts, str): prompts = [prompts]
            texts.extend([p for p in prompts if p and len(p.strip()) > 3])
        
        if texts:
            profile = generator.analyze_batch(texts)
            
            # تحويل توزيع الموضوعات لجدول
            topic_data = pd.DataFrame([
                {"موضوع": data['name_ar'], "نسبة": data['percentage']}
                for data in profile['topic_distribution'].values()
            ])
            
            st.subheader("📈 نسب المواضيع")
            fig_pie = px.pie(topic_data, names='موضوع', values='نسبة',
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("☁️ سحابة الكلمات (Word Cloud)")
            if profile['top_keywords']:
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                      colormap='plasma').generate_from_frequencies(profile['top_keywords'])
                st.image(wordcloud.to_array(), caption="الكلمات الأكثر تكرارًا", use_column_width=True)
            else:
                st.info("لا توجد كلمات كافية لإنشاء سحابة كلمات.")

            # ---- Timeline التحليلي ----
            st.subheader("📊 ملخص الشخصية والتعقيد")
            col_a, col_b = st.columns(2)
            primary = profile['summary']['primary_persona']
            col_a.metric("الشخصية الرئيسية", primary['name_ar'], f"{primary['percentage']}%")
            col_b.metric("مستوى التعقيد", profile['summary']['complexity_ar'])
        else:
            st.warning("لم يتم العثور على نصوص كافية في الملف للتحليل.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")

# ---------- تحليل سؤال واحد ----------
if analyze_btn and question:
    st.subheader("📝 تحليل السؤال الفردي")
    
    # استخدام منطق الباك اند الحقيقي
    result = generator.analyze_single(question)

    # ---- عرض النتائج بشكل Cards ----
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#0B3D91; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>السؤال</h4>{result['original_text']}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#1E90FF; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>الموضوع</h4>{result['topic']['name_ar']}<br><small>الثقة: {result['topic']['confidence']:.2f}</small></div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#00BFFF; color:white; text-align:center; font-family:Cairo; min-height:150px;'><h4>الشخصية</h4>{result['persona']['name_ar']}<br><small>{result['persona']['description_ar']}</small></div>", unsafe_allow_html=True)

# ---------- تصدير النتائج ----------
if uploaded_file or analyze_btn:
    st.subheader("💾 تصدير النتائج")
    to_export = example_data if uploaded_file else pd.DataFrame([fake_response])
    
    output = BytesIO()
    to_export.to_excel(output, index=False)
    st.download_button(
        label="Download Excel",
        data=output,
        file_name="results.xlsx",
        mime="application/vnd.ms-excel"
    )

# ---------- Footer ----------
st.markdown("---")
st.markdown("<p style='text-align:center; font-family:Cairo; color:#28a745;'>✅ تم الربط بنجاح مع محرك التحليل الحقيقي (Backend)</p>", unsafe_allow_html=True)
