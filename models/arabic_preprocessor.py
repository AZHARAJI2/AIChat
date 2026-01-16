"""
معالج النصوص العربية
Arabic Text Preprocessor

الوظائف:
- تنظيف النص (إزالة الروابط، الإيموجي، الرموز)
- تطبيع النص (توحيد الألفات، التاء المربوطة، الياء)
- إزالة كلمات التوقف
- استخراج الجذور (Stemming)
"""

import re
from typing import List, Optional
import unicodedata


class ArabicPreprocessor:
    """
    معالج النصوص العربية للبيانات المترجمة آلياً
    يضمن جودة النتائج من خلال التنظيف والتطبيع
    """
    
    def __init__(self):
        # كلمات التوقف العربية - لا تحمل معنى تصنيفي
        self.arabic_stop_words = {
            # حروف الجر
            'من', 'إلى', 'على', 'في', 'عن', 'مع', 'بين', 'حتى', 'منذ',
            # الضمائر
            'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'أنتن',
            # أدوات التعريف والإشارة
            'ال', 'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك',
            # حروف العطف والاستفهام
            'و', 'أو', 'ثم', 'ف', 'لكن', 'هل', 'ما', 'ماذا', 'أين', 'متى',
            # أفعال مساعدة وكلمات شائعة
            'كان', 'كانت', 'يكون', 'تكون', 'ليس', 'ليست',
            'إن', 'أن', 'لأن', 'كي', 'لكي', 'حيث', 'إذا', 'لو', 'قد',
            'كل', 'بعض', 'أي', 'كلا', 'كلتا', 'غير', 'سوى',
            'الذي', 'التي', 'الذين', 'اللواتي', 'اللتين',
            'فيه', 'فيها', 'منه', 'منها', 'عليه', 'عليها', 'به', 'بها',
            'له', 'لها', 'لهم', 'لهن', 'إليه', 'إليها',
            'أكثر', 'أقل', 'مثل', 'نفس', 'ذات', 'عند', 'لدى',
            'بعد', 'قبل', 'فوق', 'تحت', 'أمام', 'خلف', 'داخل', 'خارج',
            'هنا', 'هناك', 'الآن', 'اليوم', 'أمس', 'غداً',
            'جداً', 'فقط', 'أيضاً', 'حتى', 'بل', 'لذلك', 'لذا',
        }
        
        # كلمات التوقف الإنجليزية (للنصوص المختلطة)
        self.english_stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'how', 'can', 'could', 'would', 'should',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'shall', 'may', 'might', 'must', 'need',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'me', 'him', 'them', 'us',
            'am', 'are', 'was', 'were',
            'if', 'then', 'else', 'when', 'where', 'why', 'who', 'whom',
            'so', 'very', 'just', 'only', 'also', 'more', 'most', 'less',
            'no', 'not', 'yes', 'any', 'some', 'all', 'each', 'every',
            'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'here', 'there', 'once', 'both', 'few', 'other', 'such',
        }
        
        # دمج كلمات التوقف
        self.stop_words = self.arabic_stop_words | self.english_stop_words
        
        # أنماط التنظيف
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        
        # تعيينات التطبيع العربي
        self.arabic_normalizations = {
            # توحيد الألفات
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            # التاء المربوطة
            'ة': 'ه',
            # الياء
            'ى': 'ي',
            # الهمزات
            'ؤ': 'و', 'ئ': 'ي',
        }
        
        # التشكيل العربي (للإزالة)
        self.tashkeel_pattern = re.compile(r'[\u064B-\u065F\u0670]')
        
        # الأرقام العربية إلى إنجليزية
        self.arabic_numerals = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        }
    
    def clean_text(self, text: str) -> str:
        """
        تنظيف النص من الروابط والإيموجي والرموز
        
        Args:
            text: النص المُدخل
            
        Returns:
            النص المُنظف
        """
        if not text or not isinstance(text, str):
            return ""
        
        # إزالة الروابط
        text = self.url_pattern.sub(' ', text)
        
        # إزالة البريد الإلكتروني
        text = self.email_pattern.sub(' ', text)
        
        # إزالة المنشنات والهاشتاقات
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        
        # إزالة الإيموجي
        text = self._remove_emojis(text)
        
        # إزالة الرموز الخاصة (الاحتفاظ بالحروف والأرقام والمسافات)
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _remove_emojis(self, text: str) -> str:
        """إزالة الإيموجي من النص"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # misc
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(' ', text)
    
    def normalize_arabic(self, text: str) -> str:
        """
        تطبيع النص العربي
        - توحيد الألفات (أ، إ، آ -> ا)
        - التاء المربوطة (ة -> ه)  
        - الياء (ى -> ي)
        - إزالة التشكيل
        
        Args:
            text: النص المُدخل
            
        Returns:
            النص المُطبّع
        """
        if not text:
            return ""
        
        # إزالة التشكيل
        text = self.tashkeel_pattern.sub('', text)
        
        # تطبيع الحروف
        for original, normalized in self.arabic_normalizations.items():
            text = text.replace(original, normalized)
        
        # تحويل الأرقام العربية
        for ar_num, en_num in self.arabic_numerals.items():
            text = text.replace(ar_num, en_num)
        
        return text
    
    def remove_stop_words(self, text: str) -> str:
        """
        إزالة كلمات التوقف
        
        Args:
            text: النص المُدخل
            
        Returns:
            النص بدون كلمات التوقف
        """
        if not text:
            return ""
        
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower() not in self.stop_words and len(word) > 1
        ]
        
        return ' '.join(filtered_words)
    
    def stem_arabic(self, text: str) -> str:
        """
        استخراج الجذور العربية (Light Stemming)
        يزيل السوابق واللواحق الشائعة
        
        Args:
            text: النص المُدخل
            
        Returns:
            النص مع الجذور
        """
        if not text:
            return ""
        
        # السوابق الشائعة
        prefixes = ['ال', 'وال', 'بال', 'كال', 'فال', 'لل',
                    'و', 'ف', 'ب', 'ك', 'ل', 'س', 'سي', 'ست']
        
        # اللواحق الشائعة
        suffixes = ['ون', 'ين', 'ان', 'ات', 'ها', 'هم', 'هن', 
                    'كم', 'كن', 'نا', 'ني', 'ية', 'يه', 'تين',
                    'تان', 'وا', 'ت', 'ي', 'ه', 'ا']
        
        words = text.split()
        stemmed_words = []
        
        for word in words:
            original_word = word
            
            # إزالة السوابق (من الأطول للأقصر)
            for prefix in sorted(prefixes, key=len, reverse=True):
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    word = word[len(prefix):]
                    break
            
            # إزالة اللواحق (من الأطول للأقصر)
            for suffix in sorted(suffixes, key=len, reverse=True):
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    word = word[:-len(suffix)]
                    break
            
            # التأكد من أن الكلمة لا تزال ذات معنى
            if len(word) >= 2:
                stemmed_words.append(word)
            else:
                stemmed_words.append(original_word)
        
        return ' '.join(stemmed_words)
    
    def preprocess(self, text: str, 
                   do_clean: bool = True,
                   do_normalize: bool = True,
                   do_remove_stops: bool = True,
                   do_stem: bool = True) -> str:
        """
        المعالجة الكاملة للنص
        
        Args:
            text: النص المُدخل
            do_clean: تنظيف النص
            do_normalize: تطبيع النص
            do_remove_stops: إزالة كلمات التوقف
            do_stem: استخراج الجذور
            
        Returns:
            النص المُعالج
        """
        if not text or not isinstance(text, str):
            return ""
        
        result = text
        
        if do_clean:
            result = self.clean_text(result)
        
        if do_normalize:
            result = self.normalize_arabic(result)
        
        if do_remove_stops:
            result = self.remove_stop_words(result)
        
        if do_stem:
            result = self.stem_arabic(result)
        
        return result
    
    def preprocess_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        معالجة مجموعة من النصوص
        
        Args:
            texts: قائمة النصوص
            **kwargs: خيارات المعالجة
            
        Returns:
            قائمة النصوص المُعالجة
        """
        return [self.preprocess(text, **kwargs) for text in texts]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        استخراج الكلمات المفتاحية من النص
        
        Args:
            text: النص المُدخل
            top_n: عدد الكلمات المفتاحية
            
        Returns:
            قائمة الكلمات المفتاحية
        """
        # معالجة النص
        processed = self.preprocess(text, do_stem=False)
        
        # تقسيم إلى كلمات
        words = processed.split()
        
        # إزالة الكلمات القصيرة
        keywords = [w for w in words if len(w) >= 3]
        
        # حساب التكرارات
        from collections import Counter
        word_counts = Counter(keywords)
        
        # إرجاع الأكثر تكراراً
        return [word for word, _ in word_counts.most_common(top_n)]


# 🧪 اختبار
if __name__ == "__main__":
    preprocessor = ArabicPreprocessor()
    
    # نصوص للاختبار
    test_texts = [
        "كيف أستطيع إنشاء برنامج للذكاء الاصطناعي؟ 🤖 https://example.com",
        "أريد أن أتعلم البرمجة بلغة بايثون",
        "ما هو الفرق بين التعلم الآلي والتعلم العميق؟",
        "اكتب لي كوداً لحساب المتوسط الحسابي",
    ]
    
    print("=" * 70)
    print("🔧 اختبار معالج النصوص العربية")
    print("=" * 70)
    
    for text in test_texts:
        print(f"\n📝 النص الأصلي: {text}")
        
        cleaned = preprocessor.clean_text(text)
        print(f"   التنظيف: {cleaned}")
        
        normalized = preprocessor.normalize_arabic(cleaned)
        print(f"   التطبيع: {normalized}")
        
        processed = preprocessor.preprocess(text)
        print(f"   المعالجة الكاملة: {processed}")
        
        keywords = preprocessor.extract_keywords(text)
        print(f"   الكلمات المفتاحية: {keywords}")
