"""
Self-contained script for disambiguating ambiguous questions using LLM.
This script contains a function that takes an ambiguous question as input
and returns clarifying questions to help resolve the ambiguity.

Usage:
    python disambiguate_question.py
    
Or import the function:
    from disambiguate_question import disambiguate_question
    result = disambiguate_question("When does the new bunk'd come out?")
"""

from interlinked import AI
from interlinked.core.clients.googleaiclient import GoogleAIClient
from typing import List
import json
import re


def create_disambiguation_prompt(ambiguous_question: str) -> str:
    """
    Create a prompt for disambiguating an ambiguous question using few-shot examples.
    
    Args:
        ambiguous_question: The ambiguous question to disambiguate
        
    Returns:
        A formatted prompt for the LLM
    """
    
    prompt = f"""You are an expert at identifying ambiguity in questions and generating clarifying questions to resolve that ambiguity. 
Given a question, your task is as follows:
1. Decide whether the question needs disambiguation, if not, output [CONTINUE].
2. If the question needs disambiguation, output [CLARIFY] token, followed by 2-4 specific clarifying questions.

Use the following examples:

==========================
Question: When does the new bunk'd come out?
Output:[CLARIFY]
- When does episode 42 of bunk'd come out?
- When does episode 41 of bunk'd come out?
- When does episode 40 of bunk'd come out?

==========================
Question: Who won the 2016 ncaa football national championship?
Output:[CLARIFY]
- Who won the 2016 season's ncaa football national championship?
- Who won the ncaa football national championship played in 2016?

==========================
Question: When was the last time the death penalty was used in pa?
Output:[CLARIFY]
- As of 2017, when was the last time the death penalty was carried out in PA?
- As of 2016, when was the last time the death penalty was carried out in PA?
- As of 2015, when was the last time the death penalty was carried out in PA?

==========================
Question: Where does failure of the left ventricle cause increased pressure?
Output: [CONTINUE]

Now, given the following ambiguous question, generate 2-4 specific clarifying questions that would help resolve the ambiguity. Focus on the key ambiguous elements like:
- Time references (which year, season, specific dates)
- Specific entities, episodes, or versions
- Context or scope differences
- Different interpretations of the same terms

Format your response as [CONTINUE]/[CLARIFY] token. If clarification is needed, format the clarifying questions as a simple list with each question on a new line starting with a dash (-).

Question: {ambiguous_question}
Output:"""

    return prompt


def disambiguate_question(ambiguous_question: str, api_key: str = "in-8LxOfglvSxalWFfVDbd7ug") -> List[str]:
    """
    Disambiguate an ambiguous question by generating clarifying questions.
    
    Args:
        ambiguous_question: The ambiguous question to disambiguate
        api_key: Google AI API key (default uses the same key from run_interlinked.py)
        
    Returns:
        A list of clarifying questions that help resolve the ambiguity
    """
    
    # Create the disambiguation prompt
    prompt = create_disambiguation_prompt(ambiguous_question)
    
    # Initialize the Google AI client
    client = GoogleAIClient(model_name='gemini-1.5-flash-002', api_key=api_key)
    
    try:
        # Call the LLM using the interlinked library
        observation = AI.ask(prompt=prompt, client=client)
        response = observation.response
        
        # Parse the response to extract clarifying questions
        clarifying_questions = []
        
        # Split response by lines and extract questions
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that start with - or are formatted as questions
            if line and (line.startswith('-') or line.startswith('•')):
                # Clean up the question
                question = line.lstrip('- •').strip()
                if question and (question.endswith('?') or 'when' in question.lower() or 'who' in question.lower() or 'what' in question.lower() or 'where' in question.lower() or 'how' in question.lower() or 'why' in question.lower()):
                    clarifying_questions.append(question)
        
        # If no questions found with bullet points, try to extract questions directly
        if not clarifying_questions:
            # Look for sentences ending with question marks
            sentences = re.split(r'[.!?]+', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and (sentence.endswith('?') or any(word in sentence.lower() for word in ['when', 'who', 'what', 'where', 'how', 'why'])):
                    if not sentence.endswith('?'):
                        sentence += '?'
                    clarifying_questions.append(sentence)
        
        return clarifying_questions
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []


def main():
    """
    Main function for testing the disambiguation functionality.
    """
    
    # Test examples from the task description
    test_questions = [
        "When does the new bunk'd come out?",
        "Who won the 2016 ncaa football national championship?", 
        "When was the last time the death penalty was used in pa?",
        "What time does the store close?",
        "How much does it cost?",
        "When is the next game?"
    ]
    
    print("Testing Question Disambiguation")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\nAmbiguous Question: {question}")
        print("-" * 40)
        
        clarifying_questions = disambiguate_question(question)
        
        if clarifying_questions:
            print("Clarifying Questions:")
            for i, cq in enumerate(clarifying_questions, 1):
                print(f"{i}. {cq}")
        else:
            print("No clarifying questions generated.")
        
        print()


if __name__ == "__main__":
    main()