I want a plan for a report that is concise and focused.

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