
import sys
import subprocess
try:
    import nltk
    import numpy
    import textstat
    import analizer

    # NLTK LIBRARY DOWNLOAD (IN CASE MISSING)
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker')
    nltk.download('maxent_ne_chunker_tab')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    # MAIN
    def main(input_text = ""):
        # PROCESS INPUT SENT
        analysis_result = analizer.analyze_text(input_text)
        return analysis_result

    if __name__ == "__main__":
        # GET COMMAND LINE STRING
        input_text = " ".join(sys.argv[1:])
        
        # CALL MAIN AND PRINT RESULTS
        result = main(input_text)
        print(result)
except Exception as e:
    print(f"An error occurred: {e}")
