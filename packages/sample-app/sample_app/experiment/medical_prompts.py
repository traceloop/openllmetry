"""
Two different prompt templates for generating medical answers
"""

def educational_prompt(question: str) -> str:
    """
    Educational approach - focuses on general health information and encourages professional consultation
    """
    return f"""You are a knowledgeable health educator. Please provide a comprehensive, educational answer to the following health question.

Question: {question}

Please provide:
1. A clear, factual explanation using accessible language
2. Key benefits and important considerations
3. General recommendations that apply to most people
4. A reminder to consult healthcare professionals for personalized advice

Important: This is for educational purposes only and should not replace professional medical advice."""


def clinical_guidance_prompt(question: str) -> str:
    """
    Clinical guidance approach - more structured, evidence-based response
    """
    return f"""As a medical information assistant, provide a structured, evidence-based response to this health question.

Patient Question: {question}

Please structure your response as follows:

**Overview:**
Brief explanation of the topic

**Key Points:**
- List 3-4 most important points
- Include relevant health benefits/risks
- Mention any contraindications if applicable

**Recommendations:**
General guidelines that apply to the average healthy adult

**When to Seek Professional Care:**
Situations requiring medical consultation

**Disclaimer:** This information is for educational purposes only. Individual health needs vary, and patients should consult their healthcare provider for personalized medical advice."""
