import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from io import BytesIO

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
    st.subheader("📈 نسب المواضيع")
    fig_pie = px.pie(example_data, names='موضوع', values='نسبة',
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("☁️ Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='plasma').generate(example_text)
    st.image(wordcloud.to_array(), caption="الكلمات الأكثر تكرارًا", use_column_width=True)

    # ---- Timeline (اختياري لو عندكم تاريخ) ----
    st.subheader("📅 Timeline المواضيع")
    fig_line = px.line(example_data, x='تاريخ', y='نسبة', color='موضوع',
                       markers=True, title="تطور المواضيع مع الوقت")
    st.plotly_chart(fig_line, use_container_width=True)

# ---------- تحليل سؤال واحد ----------
if analyze_btn and question:
    st.subheader("📝 تحليل السؤال الفردي")
    
    # مثال وهمي للنتيجة (يمكن استبداله بالـ Backend لاحقًا)
    fake_response = {
        "سؤال": question,
        "موضوع": "تقني",
        "نية": "متعلم"
    }

    # ---- عرض النتائج بشكل Cards ----
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#0B3D91; color:white; text-align:center; font-family:Cairo;'><h4>السؤال</h4>{fake_response['سؤال']}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#1E90FF; color:white; text-align:center; font-family:Cairo;'><h4>الموضوع</h4>{fake_response['موضوع']}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='padding:15px; border-radius:10px; background-color:#00BFFF; color:white; text-align:center; font-family:Cairo;'><h4>النية</h4>{fake_response['نية']}</div>", unsafe_allow_html=True)

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
st.markdown("<p style='text-align:center; font-family:Cairo; color:#666;'>💡 البيانات حالياً وهمية، يمكن ربطها بالـ Backend لاحقًا لإظهار التحليل الحقيقي</p>", unsafe_allow_html=True)
