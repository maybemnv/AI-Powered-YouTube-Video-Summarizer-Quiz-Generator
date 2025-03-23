import os
from openai import OpenAI
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re
import time
import requests
import requests.exceptions
import logging
import sys
import google.generativeai as genai  
import random

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_environment():
    """Load environment variables"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    print("Loading env from:", env_path)
    load_dotenv(env_path)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return api_key

# Initialize Gemini client
try:
    api_key = load_environment()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-8b')  
except Exception as e:
    st.error(f"Error initializing API client: {str(e)}")
    st.stop()

def extract_video_id(youtube_url):
    """Extract video ID from different YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shared URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Shortened URLs
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',   # YouTube Shorts
        r'^([0-9A-Za-z_-]{11})$'  # Just the video ID
    ]
    
    youtube_url = youtube_url.strip()
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract video ID from URL")

def get_transcript(youtube_url):
    """Get transcript using YouTube Transcript API with cookies"""
    try:
        video_id = extract_video_id(youtube_url)
        
        # Get cookies file path
        cookies_file = os.getenv('COOKIE_PATH', os.path.join(os.path.dirname(__file__), 'cookies.txt'))
        
        if not os.path.exists(cookies_file):
            st.error("Cookie file not found. Please follow the setup instructions in the README.")
            return None, None
            
        try:
            # Read cookies from file
            with open(cookies_file, 'r') as f:
                cookies_content = f.read()
                if not cookies_content.strip():
                    st.error("Cookie file is empty. Please re-export your YouTube cookies.")
                    return None, None
            
            # Get transcript with cookies
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_file)
            
            try:
                transcript = transcript_list.find_manually_created_transcript()
            except:
                try:
                    transcript = next(iter(transcript_list))
                except Exception as e:
                    st.error("Your YouTube cookies might have expired. Please re-export your cookies and try again.")
                    return None, None
            
            full_transcript = " ".join([part['text'] for part in transcript.fetch()])
            language_code = transcript.language_code
            
            return full_transcript, language_code
                
        except Exception as e:
            st.error("Authentication failed. Please update your cookies.txt file with fresh YouTube cookies.")
            st.info("Tip: Sign in to YouTube again and re-export your cookies using the browser extension.")
            return None, None
            
    except Exception as e:
        st.error("Invalid YouTube URL. Please check the link and try again.")
        return None, None

def get_available_languages():
    """Return a dictionary of available languages"""
    return {
        'English': 'en',
        'Deutsch': 'de',
        'Italiano': 'it',
        'Español': 'es',
        'Français': 'fr',
        'Nederlands': 'nl',
        'Polski': 'pl',
        '日本語': 'ja',
        '中文': 'zh',
        'Русский': 'ru'
    }

def create_summary_prompt(text, target_language, mode='video'):
    """Create an optimized prompt for summarization in the target language and mode"""
    language_prompts = {
        'en': {
            'title': 'TITLE',
            'overview': 'OVERVIEW',
            'key_points': 'KEY POINTS',
            'takeaways': 'MAIN TAKEAWAYS',
            'context': 'CONTEXT & IMPLICATIONS'
        },
        'de': {
            'title': 'TITEL',
            'overview': 'ÜBERBLICK',
            'key_points': 'KERNPUNKTE',
            'takeaways': 'HAUPTERKENNTNISSE',
            'context': 'KONTEXT & AUSWIRKUNGEN'
        },
        'it': { 
            'title': 'TITOLO',
            'overview': 'PANORAMICA',
            'key_points': 'PUNTI CHIAVE',
            'takeaways': 'CONCLUSIONI PRINCIPALI',
            'context': 'CONTESTO E IMPLICAZIONI'
        }
    }

    prompts = language_prompts.get(target_language, language_prompts['en'])

    if mode == 'podcast':
        system_prompt = f"""You are an expert content analyst and summarizer. Create a comprehensive 
        podcast-style summary in {target_language}. Ensure all content is fully translated and culturally adapted 
        to the target language."""

        user_prompt = f"""Please provide a detailed podcast-style summary of the following content in {target_language}. 
        Structure your response as follows:

        🎙️ {prompts['title']}: Create an engaging title

        🎧 {prompts['overview']} (3-5 sentences):
        - Provide a detailed context and main purpose

        🔍 {prompts['key_points']}:
        - Deep dive into the main arguments
        - Include specific examples and anecdotes
        - Highlight unique perspectives and expert opinions

        📈 {prompts['takeaways']}:
        - List 5-7 practical insights
        - Explain their significance and potential impact

        🌐 {prompts['context']}:
        - Broader context discussion
        - Future implications and expert predictions

        Text to summarize: {text}

        Ensure the summary is comprehensive enough for someone who hasn't seen the original content."""
    else:
        system_prompt = f"""You are an expert content analyst and summarizer. Create a comprehensive 
        summary in {target_language}. Ensure all content is fully translated and culturally adapted 
        to the target language."""

        user_prompt = f"""Please provide a detailed summary of the following content in {target_language}. 
        Structure your response as follows:

        🎯 {prompts['title']}: Create a descriptive title

        📝 {prompts['overview']} (2-3 sentences):
        - Provide a brief context and main purpose

        🔑 {prompts['key_points']}:
        - Extract and explain the main arguments
        - Include specific examples
        - Highlight unique perspectives

        💡 {prompts['takeaways']}:
        - List 3-5 practical insights
        - Explain their significance

        🔄 {prompts['context']}:
        - Broader context discussion
        - Future implications

        Text to summarize: {text}

        Ensure the summary is comprehensive enough for someone who hasn't seen the original content."""
    return system_prompt, user_prompt

def summarize_with_langchain_and_openai(transcript, language_code, model_name='gemini-pro', mode='video'):
    try:
        # Verify API configuration
        logging.debug("Attempting to verify API connection...")
        
        # Test API connection with a simple prompt
        try:
            test_response = model.generate_content("Test connection")
            if not test_response:
                raise ValueError("API connection test failed")
            logging.debug("API connection test successful")
        except Exception as e:
            raise ValueError(f"API connection test failed: {str(e)}")
        
        texts = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Reduced chunk size for better reliability
            chunk_overlap=500,
            length_function=len
        ).split_text(transcript)
        
        intermediate_summaries = []
        
        for i, text_chunk in enumerate(texts):
            retries = 5  # Increased retries
            for attempt in range(retries):
                try:
                    # Create a more detailed prompt
                    prompt = f"""As an AI language model, please provide a clear and concise summary of the following text in {language_code}. 
                    Focus on the main points and key details:

                    {text_chunk}

                    Please maintain the original meaning while making it more concise."""
                    
                    response = model.generate_content(prompt)
                    
                    if hasattr(response, 'text') and response.text:
                        intermediate_summaries.append(response.text)
                        logging.debug(f"Successfully summarized chunk {i+1}/{len(texts)}")
                        break
                    else:
                        raise ValueError("Empty response received from API")
                    
                except Exception as e:
                    logging.error(f"Attempt {attempt+1}/{retries} failed: {str(e)}")
                    if attempt < retries - 1:
                        wait_time = min(2 ** attempt, 30)  # Cap maximum wait time at 30 seconds
                        logging.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error(f"Failed to process chunk {i+1} after {retries} attempts")
                        if i > 0:  # If we have some summaries, continue with what we have
                            logging.warning("Continuing with partial results")
                            break
                        return None

        if not intermediate_summaries:
            st.error("Failed to generate any summaries")
            return None

        # Generate final summary with structured prompt
        try:
            final_prompt = f"""Create a well-structured, comprehensive summary in {language_code} from these intermediate summaries.
            Please organize the content with clear sections:

            1. Main Topic/Overview
            2. Key Points
            3. Important Details
            4. Conclusions/Takeaways

            Source summaries:
            {'\n\n'.join(intermediate_summaries)}"""
            
            final_response = model.generate_content(final_prompt)
            if not final_response.text:
                raise ValueError("Empty final summary received")
            return final_response.text
            
        except Exception as e:
            logging.error(f"Final summary error: {str(e)}")
            st.error(f"Error generating final summary: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Failed to connect to Gemini API: {str(e)}")
        logging.error(f"API initialization error: {str(e)}")
        return None

def generate_questions(summary, language_code):
    """Generate questions from the summary with varying difficulty levels"""
    try:
        prompt = f"""Generate exactly 10 multiple-choice questions based on this summary. For each question:
        1. Create 4 basic questions (1 point each)
        2. Create 3 intermediate questions (2 points each)
        3. Create 3 advanced questions (3 points each)

        Important rules:
        - The correct answer MUST be one of the options
        - Each question MUST have exactly 4 options
        - First option MUST be the correct answer
        - Options must be clear and distinct
        - Questions should test understanding, not just memory

        Format each question as a JSON object:
        {{
            "difficulty": "basic/intermediate/advanced",
            "points": 1/2/3,
            "question": "Clear question text?",
            "answer": "Correct answer text",
            "options": ["Correct answer text", "Wrong option 1", "Wrong option 2", "Wrong option 3"]
        }}

        Return array of 10 questions in valid JSON format.
        Summary: {summary}"""

        response = model.generate_content(prompt)
        if not response.text:
            raise ValueError("Empty response received from API")
        
        # Clean and parse JSON
        try:
            import json
            # Clean response text
            response_text = response.text.strip()
            # Remove markdown formatting if present
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Log the cleaned response for debugging
            logging.debug(f"Cleaned response text: {response_text}")
            
            # Parse JSON
            questions = json.loads(response_text)
            
            # Validate and fix questions
            validated_questions = []
            for q in questions:
                # Ensure required fields exist
                if not all(key in q for key in ['difficulty', 'points', 'question', 'answer', 'options']):
                    continue
                
                # Ensure answer is in options
                if q['answer'] not in q['options']:
                    # If answer isn't in options, make it the first option
                    q['options'][0] = q['answer']
                
                # Ensure exactly 4 options
                while len(q['options']) < 4:
                    q['options'].append(f"Alternative {len(q['options']) + 1}")
                q['options'] = q['options'][:4]
                
                validated_questions.append(q)
            
            # Ensure we have exactly 10 questions
            #if len(validated_questions) != 10:
                #raise ValueError(f"Expected 10 questions, got {len(validated_questions)}")
            
            return validated_questions
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Response text: {response_text}")
            raise ValueError("Failed to parse questions JSON")
            
    except Exception as e:
        logging.error(f"Error generating questions: {str(e)}")
        st.error(f"Failed to generate questions: {str(e)}")
        return None

def generate_filler_question(question_number):
    """Generate a basic filler question when needed"""
    return {
        "difficulty": "basic",
        "points": 1,
        "question": f"Additional Question {question_number}",
        "answer": "Correct Answer",
        "options": ["Correct Answer", "Wrong Answer 1", "Wrong Answer 2", "Wrong Answer 3"]
    }

def main():
    # Initialize session states
    if 'total_points' not in st.session_state:
        st.session_state.total_points = 0
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'answered_questions' not in st.session_state:
        st.session_state.answered_questions = set()
    if 'answers' not in st.session_state:
        st.session_state.answers = {}

    # Add points display in navbar
    st.sidebar.markdown(f"### 🏆 Total Points: {st.session_state.total_points}")
    
    st.title('📺 Advanced YouTube Video Summarizer')
    st.markdown("""
    This tool creates comprehensive summaries of YouTube videos using advanced AI technology.
    It works with both videos that have transcripts and those that don't!
    """)
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        link = st.text_input('🔗 Enter YouTube video URL:')
    
    with col2:
        languages = get_available_languages()
        target_language = st.selectbox(
            '🌍 Select Language:',
            options=list(languages.keys()),
            index=0
        )
        target_language_code = languages[target_language]

    with col3:
        mode = st.selectbox(
            '🎙️ Mode:',
            options=['Video', 'Podcast'],
            index=0
        )
        mode = mode.lower()

    # Add buttons in separate columns
    col_sum, col_quiz = st.columns(2)
    
    # Create containers for summary and quiz
    summary_container = st.container()
    quiz_container = st.container()
    
    with col_sum:
        if st.button('Generate Summary'):
            if link:
                try:
                    with st.spinner('Processing...'):
                        progress = st.progress(0)
                        status_text = st.empty()

                        status_text.text('📥 Fetching video transcript...')
                        progress.progress(25)

                        transcript, _ = get_transcript(link)

                        if transcript is None:
                            st.error("Failed to retrieve transcript. Please check the error messages above.")
                            return

                        status_text.text(f'🤖 Generating {target_language} summary...')
                        progress.progress(75)

                        summary = summarize_with_langchain_and_openai(
                            transcript, 
                            target_language_code,
                            model_name='llama-3.1-8b-instant',
                            mode=mode
                        )

                        status_text.text('✨ Summary Ready!')
                        with summary_container:
                            st.markdown("### 📄 Summary")
                            st.markdown(summary)
                        st.session_state.summary = summary
                        progress.progress(100)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning('Please enter a valid YouTube link.')

    # Always display the summary if it exists
    if st.session_state.summary:
        with summary_container:
            st.markdown("### 📄 Summary")
            st.markdown(st.session_state.summary)

    with col_quiz:
        if st.button('Generate Quiz'):
            if st.session_state.summary:
                try:
                    with st.spinner('Generating quiz questions...'):
                        questions = generate_questions(st.session_state.summary, target_language_code)
                        if questions:
                            # Fill remaining questions if less than 10
                            while len(questions) < 10:
                                new_question = generate_filler_question(len(questions) + 1)
                                questions.append(new_question)
                            # Limit to 10 questions if more are generated
                            questions = questions[:10]
                            with quiz_container:
                                st.session_state.questions = questions
                                display_quiz(questions)
                        else:
                            st.error("Failed to generate questions")
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
            else:
                st.warning('Please generate a summary first.')

    # Always display the quiz if it exists
    if st.session_state.questions:
        with quiz_container:
            display_quiz(st.session_state.questions)

def display_quiz(questions):
    """Display quiz questions and handle scoring"""
    st.markdown("### 📝 Knowledge Check")
    
    # Initialize question states if not exists
    if 'answered_questions' not in st.session_state:
        st.session_state.answered_questions = set()
    if 'answers' not in st.session_state:
        st.session_state.answers = {}

    for i, q in enumerate(questions):
        st.markdown(f"**Question {i+1}** ({q['difficulty'].title()} - {q['points']} points)")
        st.markdown(q['question'])
        
        # Randomize the options
        options = q['options']
        random.shuffle(options)
        
        # Create unique key for each question
        answer_key = f"q_{i}"
        button_key = f"check_{i}"
        
        answer = st.radio(
            "Select your answer:",
            options=options,
            key=answer_key
        )
        
        if st.button(f"Check Answer {i+1}", key=button_key):
            if i not in st.session_state.answered_questions:  # Check if question hasn't been answered
                st.session_state.answers[i] = answer
                if answer == q['answer']:
                    st.success(f"Correct! +{q['points']} points")
                    st.session_state.total_points += q['points']
                    st.session_state.answered_questions.add(i)  # Mark question as answered
                else:
                    st.error(f"Incorrect. The correct answer is: {q['answer']}")
            else:
                st.info("You've already answered this question!")
        
        st.markdown("---")
    
    # Display total points
    st.sidebar.markdown(f"### 🏆 Total Points: {st.session_state.total_points}")

if __name__ == "__main__":
    main()