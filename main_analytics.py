import json

# تحميل ملف زميلتك
with open("dashboard_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("✅ Dashboard data loaded successfully")

# تحقق سريع أن البيانات صحيحة
print("Available sections:")
for key in data.keys():
    print("-", key)

# مثال: قراءة نِسَب الاهتمامات
pie_labels = data["pie_chart"]["labels"]
pie_values = data["pie_chart"]["values"]

print("\n📊 Interest Distribution:")
for label, value in zip(pie_labels, pie_values):
    print(f"{label}: {value}%")

# مثال: Persona المسيطرة
persona = data["persona_card"]
print("\n🧍 Dominant Persona:")
print(persona["name_ar"], "-", persona["percentage"], "%")

print("\n✅ Analytics check completed successfully")
