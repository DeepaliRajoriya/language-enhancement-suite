import spacy
import language_tool_python
from textblob import TextBlob
import nltk
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
# Initialize LanguageTool (downloads a local server automatically)
tool = language_tool_python.LanguageTool('en-US')

def check_grammar_and_spelling(text):
    """
    Uses LanguageTool for general grammar and spelling checks.
    """
    matches = tool.check(text)
    errors = []
    for match in matches:
        errors.append({
            "message": match.message,
            "context": match.context,
            "suggestions": match.replacements[:3], # Top 3 suggestions
            "offset": match.offset,
            "length": match.error_length,
            "rule": match.rule_id,
            "type": "grammar" # Tag as grammar/spelling
        })
    return errors

def check_style_with_spacy(text):
    """
    Custom logic using spaCy to detect Passive Voice.
    This demonstrates your understanding of NLP dependency parsing.
    """
    doc = nlp(text)
    style_issues = []

    for token in doc:
        # Rule: Passive voice is often "auxiliary verb" + "verb" in passive state
        if token.dep_ == "auxpass": 
            # Found a passive auxiliary (e.g., 'was' in 'was eaten')
            head = token.head
            style_issues.append({
                "message": "Passive voice detected. Consider rewriting for clarity.",
                "context": f"...{token.text} {head.text}...",
                "suggestions": [],
                "offset": token.idx,
                "length": len(token.text) + 1 + len(head.text),
                "rule": "PASSIVE_VOICE",
                "type": "style"
            })
    
    return style_issues

def process_text(text):
    # 1. Get Grammar & Spelling Errors
    grammar_errors = check_grammar_and_spelling(text)
    
    # 2. Get Custom Style Errors (Passive Voice)
    style_errors = check_style_with_spacy(text)
    
    # Combine results
    all_issues = grammar_errors + style_errors
    
    # 3. Get Basic Sentiment (Bonus Feature)
    blob = TextBlob(text)
    sentiment = {
        "polarity": round(blob.sentiment.polarity, 2), # -1 to 1 (Negative/Positive)
        "subjectivity": round(blob.sentiment.subjectivity, 2) # 0 to 1 (Fact/Opinion)
    }

    return {
        "issues": all_issues,
        "sentiment": sentiment,
        "corrected_text": tool.correct(text) # Auto-corrected version
    }

def summarize_text(text, num_sentences=10):
    if not text.strip():
        return ""

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()

        summary_sentences = summarizer(parser.document, num_sentences)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary

    except Exception as e:
        return f"Summary failed: {str(e)}"
    
"""
def summarize_text(text, num_sentences=5):
    ""
    Improved summarizer:
    - Extracts key sentences using LSA
    - If text is short, returns cleaned up shorter summary
    - Always produces a 4-6 sentence readable summary
    ""

    if not text.strip():
        return ""

    # If text is too short, just return a compact rewrite
    if len(text.split()) < 60:
        blob = TextBlob(text).sentences
        short = " ".join([str(s) for s in blob[:2]])
        return short

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()

        # Extractive summary sentences
        summary_raw = summarizer(parser.document, num_sentences)

        # Combine extractive sentences
        extractive = " ".join([str(s) for s in summary_raw])

        # Clean up & refine using TextBlob
        refined = str(TextBlob(extractive).correct())

        return refined

    except Exception:
        # fallback
        return "Summary unavailable due to processing error."
    """



     
        
    