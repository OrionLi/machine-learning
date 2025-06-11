import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
from collections import Counter
import os
import re

def generate_wordcloud_and_frequencies(file_path, output_dir="experiment1", top_n=10):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Font path - common path for WenQuanYi Zen Hei installed via apt-get
    # This is a best guess; might need adjustment if font is elsewhere or not found.
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if not os.path.exists(font_path):
        print(f"Warning: Font path '{font_path}' not found. Word cloud may not render Chinese characters correctly.")
        # As a fallback, try a very generic name that WordCloud might find if installed system-wide differently
        # or rely on WordCloud's default font finding if it has one for Chinese (unlikely without path)
        font_path = None # Let WordCloud try its default or error out

    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"Successfully read '{file_path}'")

        # Try to find the comments column. Common names: '评论内容', '评论', 'content', 'comment'
        comment_column_names = ['评论内容', '评论', 'content', 'comment', 'Review', 'Text']
        comments_col = None
        for col_name in comment_column_names:
            if col_name in df.columns:
                comments_col = col_name
                break

        if comments_col is None:
            print(f"Error: Could not find a suitable comments column in {df.columns.tolist()}")
            return
        else:
            print(f"Using column '{comments_col}' for comments.")

        # Concatenate all comments into a single text block, handling potential NaN values
        text = ' '.join(df[comments_col].dropna().astype(str).tolist())
        if not text.strip():
            print("Error: No text content found in the comments column.")
            return

        # Chinese word segmentation
        # Remove punctuation and non-Chinese characters for cleaner word list (optional but often good)
        text = re.sub(r"[^\u4e00-\u9fa5\s]", "", text) # Keep Chinese characters and spaces
        seg_list = jieba.lcut(text, cut_all=False)

        # Filter out single-character words and common stop words (basic example)
        # A more comprehensive stop word list would be better for real applications
        custom_stopwords = set(['的', '了', '是', '我', '你', '他', '她', '它', '们', '这', '那', '一个', '一种', ' ',',', '.', '，', '。', '！', '？', '：', '；', '（', '）', '「', '」', '“', '”', '\n', '\t', ' ', '一些', '什么', '这样', '这个', '这种', '因为', '所以', '并且', '而且'])
        filtered_words = [word for word in seg_list if len(word) > 1 and word not in custom_stopwords]

        if not filtered_words:
            print("Error: No words left after segmentation and filtering. Cannot generate word cloud or frequencies.")
            return

        # Word frequency analysis
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)

        print(f"\nTop {top_n} most frequent words:")
        for word, count in top_words:
            print(f"{word}: {count}")

        # Generate word cloud
        wc = WordCloud(
            font_path=font_path,  # Specify font path for Chinese characters
            width=800,
            height=400,
            background_color='white',
            #stopwords=custom_stopwords, # Already filtered, but can be added here too
            collocations=False # Avoids grouping words that appear together often
        ).generate_from_frequencies(dict(word_counts)) # Generate from frequencies

        wordcloud_image_path = os.path.join(output_dir, "comments_wordcloud.png")

        # Using matplotlib to save, as WordCloud's to_file might also need font context
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title("Comments Word Cloud", fontsize=15) # Basic title via matplotlib
        plt.savefig(wordcloud_image_path)
        plt.close()
        print(f"\nWord cloud saved to '{wordcloud_image_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ImportError:
        print("Error: A required library (jieba or wordcloud) is not installed correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    excel_file_path = "resources/Lab01-Comments.xlsx"
    generate_wordcloud_and_frequencies(excel_file_path)
