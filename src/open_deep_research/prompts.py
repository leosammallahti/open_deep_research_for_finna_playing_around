"""Prompt templates for Open Deep Research application.

This module contains all the prompt templates used throughout the research workflow,
including instructions for report planning, query generation, section writing, and grading.
"""

report_planner_query_writer_instructions = """You are performing research for a report. 

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
Your goal is to generate {number_of_queries} web search queries that will help gather information for planning the report sections. 

The queries should:

1. Be related to the Report topic
2. Help satisfy the requirements specified in the report organization

Make the queries specific enough to find high-quality, relevant sources while covering the breadth needed for the report structure.
</Task>

<Format>
Call the Queries tool 
</Format>

Today is {today}
"""

report_planner_instructions = """I want a plan for a report that is concise and focused.

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Context>
Here is context to use to plan the sections of the report: 
{context}
</Context>

<Task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

Each section should have the fields:
- Name: Name for this section of the report.
- Description: Brief overview of the main topics covered in this section.
- Research: Whether to perform web research for this section. Main body sections (not intro/conclusion) MUST have Research=True. A report must have AT LEAST 2-3 sections with Research=True to be useful.
- Content: The content of the section, which you will leave blank for now.

Guidelines:
- Ensure each section has a distinct purpose with no content overlap.
- Combine related concepts rather than separating them.
- Every section MUST be directly relevant to the main topic.
- Avoid tangential or loosely related sections.
</Task>

<Feedback>
Here is feedback on the report structure from review (if any):
{feedback}
</Feedback>

<Format>
Call the Sections tool 
</Format>
"""

section_writer_inputs = """ 
<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>
"""

clarify_with_user_instructions = """
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Return ONE targeted question to clarify report scope
Focus on: technical depth, target audience, specific aspects to emphasize
Examples: "Should I focus on technical implementation details or high-level business benefits?" 
"""
