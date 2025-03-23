AI-Powered YouTube Video Summarizer & Quiz Generator
=========================================================

**Overview**
------------

**YouTested** is an AI-driven **YouTube video summarizer** that extracts **video transcripts**, generates **detailed summaries**, and creates **interactive quizzes** to test comprehension. It leverages **Google Gemini AI** for content analysis and supports **multiple languages and summarization modes (video or podcast).**

* * * * *

**Features**
------------

✅ **Extract Transcripts** -- Retrieves **YouTube subtitles** with authentication via cookies.\
✅ **AI-Powered Summaries** -- Generates structured summaries in various languages.\
✅ **Quiz Generator** -- Creates **10 multiple-choice questions** with different difficulty levels.\
✅ **User Interaction** -- Displays summaries, tracks quiz scores, and provides feedback.\
✅ **Reliable API Handling** -- Implements **error handling, retries, and API connection validation** for stability.

* * * * *

**Installation & Setup**
------------------------

### **1\. Install Required Packages**

Ensure you have Python installed, then run:

`pip install -r requirements.txt`

### **2\. Set Up YouTube Authentication (Cookies)**

#### **Step 1: Install Cookie Extension**

1.  Open the Chrome Web Store.

2.  Search for **"Get cookies.txt"**.

3.  Install **"Get cookies.txt LOCALLY"** extension.

4.  Pin the extension to your browser.

#### **Step 2: Export YouTube Cookies**

1.  Go to [YouTube](https://www.youtube.com).

2.  Sign in to your YouTube/Google account.

3.  Click the **"Get cookies.txt"** extension icon.

4.  Click **"Export"** to download the cookies file.

#### **Step 3: Configure the Application**

1.  Rename the downloaded file to **`cookies.txt`**.

2.  Place the **`cookies.txt`** file in the same directory as `app.py`.

3.  Ensure the file permissions are correct (readable by the application).

⚠️ **Keep your `cookies.txt` file secure** and **never share it publicly**, as it contains authentication information.

* * * * *

**Usage**
---------

Run the application with:

`streamlit run app.py`

### **How to Use:**

1.  **Enter a YouTube video URL** in the input field.

2.  Select a **language** and **summary mode (Video/Podcast)**.

3.  Click **"Generate Summary"** to get an AI-generated summary.

4.  Click **"Generate Quiz"** to test your knowledge with **multiple-choice questions**.

* * * * *

**Contributing**
----------------

Pull requests and improvements are welcome! Please ensure any code changes are well-documented.
