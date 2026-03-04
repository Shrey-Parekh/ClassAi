"""
Intent-based prompt templates for Faculty RAG system.

Each template is optimized for specific query types with structured output formats.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptTemplate:
    """Structured prompt template with intent-specific formatting."""
    intent: str
    system_instructions: str
    output_format: str
    strict_rules: str


# Base strict rules applied to ALL templates
BASE_STRICT_RULES = """
STRICT RULES:
1. Use ONLY the provided context below. If the answer is not in the context, respond with exactly:
   "I don't have that information in my current documents. Please contact HR or the relevant department directly."

2. Do NOT output a wall of text.

3. Do NOT deviate from the format structure above.

4. Do NOT infer or assume facts not in the context.

5. Be precise and factual. Every statement must be traceable to the context.
"""


# Template 1: PERSON_LOOKUP
PERSON_LOOKUP_TEMPLATE = PromptTemplate(
    intent="person_lookup",
    system_instructions="""You are answering a query about a specific faculty member at NMIMS.
Extract and present their professional information in a clean, structured format.""",
    output_format="""
OUTPUT FORMAT:
**[Full Name]**
*[Department] · [Email]*

**About**
[2-3 sentence professional bio based on qualifications and experience]

**Research Interests**
- [each interest as bullet]

**Notable Publications**
- [each publication as bullet, limit to 3-5 most relevant]

**Contact**
[email and any other contact info]
""",
    strict_rules=BASE_STRICT_RULES
)


# Template 2: DEPARTMENT_LIST
DEPARTMENT_LIST_TEMPLATE = PromptTemplate(
    intent="department_list",
    system_instructions="""You are listing faculty members from a specific department at NMIMS.
Present them in a clean table format.""",
    output_format="""
OUTPUT FORMAT:
**[Department Name] — Faculty**

| Name | Research Focus | Email |
|------|---------------|-------|
| ...  | ...           | ...   |

*Total: [n] faculty members*
""",
    strict_rules=BASE_STRICT_RULES
)


# Template 3: TOPIC_SEARCH
TOPIC_SEARCH_TEMPLATE = PromptTemplate(
    intent="topic_search",
    system_instructions="""You are finding faculty members who research or specialize in a specific topic.
List each relevant faculty member with their connection to the topic.""",
    output_format="""
OUTPUT FORMAT:
**Faculty researching [topic]**

**[Name]** · [Department]
[one line on their relevant research]
Email: [email]

---
[repeat for each match]
""",
    strict_rules=BASE_STRICT_RULES
)


# Template 4: PROCEDURE
PROCEDURE_TEMPLATE = PromptTemplate(
    intent="procedure",
    system_instructions="""You are explaining a step-by-step procedure or process at NMIMS.
Present it as a clear, actionable sequence with all necessary details.""",
    output_format="""
OUTPUT FORMAT:
**[Procedure Title]**

**Steps**
1. [Step 1 with details]
2. [Step 2 with details]
3. [Step 3 with details]

**Required Documents**
- [document 1]
- [document 2]

**Deadline / Timeline**
[if mentioned in context]

> Important: [Any important warnings or notes from the policy]
""",
    strict_rules=BASE_STRICT_RULES
)


# Template 5: ELIGIBILITY
ELIGIBILITY_TEMPLATE = PromptTemplate(
    intent="eligibility",
    system_instructions="""You are answering an eligibility or entitlement question about NMIMS policies.
Provide a direct yes/no answer first, then explain conditions.""",
    output_format="""
OUTPUT FORMAT:
**Policy: [Topic]**

**Direct Answer**
[One clear sentence answer - yes/no with key details]

**Conditions**
- [condition 1]
- [condition 2]

**Exceptions**
- [exception if any, or state "None mentioned"]

> Source: [document name from context]
""",
    strict_rules=BASE_STRICT_RULES + """
ADDITIONAL RULE FOR ELIGIBILITY:
- Do NOT provide legal advice. Always recommend official HR confirmation for binding decisions.
"""
)


# Template 6: SALARY_BENEFITS
SALARY_BENEFITS_TEMPLATE = PromptTemplate(
    intent="salary_benefits",
    system_instructions="""You are answering a query about salary, compensation, or financial benefits at NMIMS.
Be precise with numbers and clearly state who it applies to.""",
    output_format="""
OUTPUT FORMAT:
**[Salary/Benefit Topic]**

**Amount / Rate**
[specific figure if available, otherwise state "Not specified in available documents"]

**Applicable To**
- [who this applies to]

**Conditions**
- [condition 1]
- [condition 2]

> Important: For official confirmation contact HR directly.
""",
    strict_rules=BASE_STRICT_RULES + """
ADDITIONAL RULE FOR SALARY/BENEFITS:
- Do NOT provide financial advice. Always recommend official HR confirmation.
- Be extremely careful with numbers - reproduce them EXACTLY as written.
"""
)


# Template 7: CONTACT_LOOKUP
CONTACT_LOOKUP_TEMPLATE = PromptTemplate(
    intent="contact_lookup",
    system_instructions="""You are providing contact information for a person or department at NMIMS.
Present it in a clean, easy-to-read format.""",
    output_format="""
OUTPUT FORMAT:
**Contact: [Name / Department]**

Email: [email]
Department: [department]
[any location info if available]
""",
    strict_rules=BASE_STRICT_RULES
)


# Template 8: GENERAL (Fallback)
GENERAL_TEMPLATE = PromptTemplate(
    intent="general",
    system_instructions="""You are answering a general query about NMIMS faculty, policies, or procedures.
Provide a direct, well-structured answer.""",
    output_format="""
OUTPUT FORMAT:
[Direct answer in clean paragraphs]
[Use bullets only if listing multiple items]
[Bold key terms using **term**]
[Max 150 words unless detail is genuinely needed]
""",
    strict_rules=BASE_STRICT_RULES
)


# Mapping of intent strings to templates
PROMPT_MAP: Dict[str, PromptTemplate] = {
    "person_lookup": PERSON_LOOKUP_TEMPLATE,
    "lookup": PERSON_LOOKUP_TEMPLATE,  # Alias
    "department_list": DEPARTMENT_LIST_TEMPLATE,
    "topic_search": TOPIC_SEARCH_TEMPLATE,
    "procedure": PROCEDURE_TEMPLATE,
    "eligibility": ELIGIBILITY_TEMPLATE,
    "salary_benefits": SALARY_BENEFITS_TEMPLATE,
    "contact_lookup": CONTACT_LOOKUP_TEMPLATE,
    "general": GENERAL_TEMPLATE,
}


def get_prompt(intent: str, context: str, query: str) -> str:
    """
    Returns a formatted prompt string ready to send to Ollama.
    Falls back to GENERAL template if intent not recognized.
    
    Args:
        intent: Intent classification (e.g., "person_lookup", "procedure")
        context: Retrieved context chunks
        query: Original user query
    
    Returns:
        Formatted prompt string
    """
    # Get template (fallback to GENERAL if not found)
    template = PROMPT_MAP.get(intent.lower(), GENERAL_TEMPLATE)
    
    # Format the complete prompt
    prompt = format_prompt(template, context, query)
    
    return prompt


def format_prompt(template: PromptTemplate, context: str, query: str) -> str:
    """
    Format a template into a complete prompt.
    
    Args:
        template: PromptTemplate instance
        context: Retrieved context chunks
        query: Original user query
    
    Returns:
        Complete formatted prompt
    """
    prompt = f"""You are the NMIMS Faculty Assistant.

{template.system_instructions}

{template.output_format}

{template.strict_rules}

═══════════════════════════════════════════
CONTEXT (Retrieved Information)
═══════════════════════════════════════════
{context}

═══════════════════════════════════════════
USER QUERY
═══════════════════════════════════════════
{query}

═══════════════════════════════════════════
YOUR RESPONSE
═══════════════════════════════════════════
"""
    
    return prompt
