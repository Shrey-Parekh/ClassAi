"""
JSON-based prompt templates for structured responses.

Prompts are structured with context first, then JSON schema.
This ensures the model sees the information before formatting instructions.
"""

from typing import Dict


# Intent-specific JSON prompts
PERSON_LOOKUP_PROMPT = """You are a faculty information assistant for NMIMS University.

CONTEXT:
{context}

QUESTION: {query}

Using the context above, return your answer as JSON using EXACTLY this structure:

{{
  "intent": "person_lookup",
  "title": "Dr. John Smith",
  "subtitle": "Computer Science · john.smith@nmims.edu",
  "sections": [
    {{
      "heading": "About",
      "type": "paragraph",
      "content": "Dr. Smith is an Associate Professor specializing in machine learning."
    }},
    {{
      "heading": "Research Interests",
      "type": "bullets",
      "items": ["Machine Learning", "AI", "Data Science"]
    }}
  ],
  "footer": null,
  "confidence": "high",
  "fallback": null
}}

If person NOT found:
{{
  "intent": "person_lookup",
  "title": "Faculty Member Not Found",
  "subtitle": null,
  "sections": [],
  "footer": null,
  "confidence": "none",
  "fallback": "I don't have information about this faculty member."
}}

Return ONLY valid JSON. No other text."""


TOPIC_SEARCH_PROMPT = """You are a faculty information assistant for NMIMS University.

CONTEXT:
{context}

QUESTION: {query}

Using the context above, return your answer as JSON using EXACTLY this structure:

{{
  "intent": "topic_search",
  "title": "Faculty researching Machine Learning",
  "subtitle": "3 faculty members found",
  "sections": [
    {{
      "heading": "Dr. John Smith · Computer Science",
      "type": "paragraph",
      "content": "Specializes in neural networks. Email: john.smith@nmims.edu"
    }}
  ],
  "footer": null,
  "confidence": "high",
  "fallback": null
}}

If no faculty found:
{{
  "intent": "topic_search",
  "title": "No Faculty Found",
  "subtitle": null,
  "sections": [],
  "footer": null,
  "confidence": "none",
  "fallback": "I don't have information about faculty researching this topic."
}}

Return ONLY valid JSON. No other text."""


PROCEDURE_PROMPT = """You are a faculty information assistant for NMIMS University.

CONTEXT:
{context}

QUESTION: {query}

Using the context above, return your answer as JSON using EXACTLY this structure:

{{
  "intent": "procedure",
  "title": "Seed Grant Application Process",
  "subtitle": "How to apply for research seed grants",
  "sections": [
    {{
      "heading": "Steps",
      "type": "steps",
      "items": [
        "Submit proposal to Research Office",
        "Wait for review committee evaluation",
        "Attend presentation if shortlisted",
        "Receive approval notification"
      ]
    }},
    {{
      "heading": "Required Documents",
      "type": "bullets",
      "items": ["Research proposal", "Budget breakdown", "CV"]
    }},
    {{
      "heading": "Important",
      "type": "alert",
      "content": "Applications must be submitted by March 31st.",
      "severity": "warning"
    }}
  ],
  "footer": "Please confirm with the Research Office.",
  "confidence": "high",
  "fallback": null
}}

If procedure not found:
{{
  "intent": "procedure",
  "title": "Procedure Not Found",
  "subtitle": null,
  "sections": [],
  "footer": null,
  "confidence": "none",
  "fallback": "I don't have information about this procedure."
}}

DO NOT write conversational text. Return ONLY valid JSON."""


ELIGIBILITY_PROMPT = """You are a faculty information assistant for NMIMS University.

CONTEXT:
{context}

QUESTION: {query}

Using the context above, return your answer as JSON using EXACTLY this structure:

{{
  "intent": "eligibility",
  "title": "Sabbatical Leave Eligibility",
  "subtitle": "Faculty are eligible after 6 years of service",
  "sections": [
    {{
      "heading": "Policy Rule",
      "type": "paragraph",
      "content": "Faculty may apply after 6 years of continuous service."
    }},
    {{
      "heading": "Conditions",
      "type": "bullets",
      "items": ["6 years of service", "No disciplinary actions", "Research proposal required"]
    }}
  ],
  "footer": "Please confirm with HR directly.",
  "confidence": "high",
  "fallback": null
}}

If policy not found:
{{
  "intent": "eligibility",
  "title": "Policy Not Found",
  "subtitle": null,
  "sections": [],
  "footer": null,
  "confidence": "none",
  "fallback": "I don't have information about this policy."
}}

Return ONLY valid JSON. No other text."""


GENERAL_PROMPT = """You are a faculty information assistant for NMIMS University.

CONTEXT:
{context}

QUESTION: {query}

Using the context above, return your answer as JSON using EXACTLY this structure:

{{
  "intent": "general",
  "title": "Faculty Leave Policy",
  "subtitle": null,
  "sections": [
    {{
      "heading": null,
      "type": "paragraph",
      "content": "Faculty are entitled to 30 days of annual leave per year."
    }}
  ],
  "footer": null,
  "confidence": "high",
  "fallback": null
}}

If answer not in context:
{{
  "intent": "general",
  "title": "Information Not Available",
  "subtitle": null,
  "sections": [],
  "footer": null,
  "confidence": "none",
  "fallback": "I don't have that information in my documents."
}}

Return ONLY valid JSON. No other text."""


# Intent to prompt mapping
INTENT_PROMPTS = {
    "lookup": PERSON_LOOKUP_PROMPT,
    "person_lookup": PERSON_LOOKUP_PROMPT,
    "topic_search": TOPIC_SEARCH_PROMPT,
    "procedure": PROCEDURE_PROMPT,
    "eligibility": ELIGIBILITY_PROMPT,
    "general": GENERAL_PROMPT,
}


def get_prompt(intent: str, context: str, query: str) -> str:
    """
    Get JSON prompt template for the given intent.
    
    Args:
        intent: Detected intent type
        context: Retrieved context chunks
        query: Original user query
    
    Returns:
        Formatted prompt string with JSON instructions
    """
    # Get prompt template for intent (fallback to general)
    template = INTENT_PROMPTS.get(intent.lower(), GENERAL_PROMPT)
    
    return template.format(context=context, query=query)
