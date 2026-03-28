# Pre-Annotation Prompt for Mixed Scoring
## CEIEF 预标注提示词

You are assisting with preliminary annotation of a cultural learning dialogue response.

Read the response and provide initial labels for:
1. Historical context matching: High / Medium / Low
2. Cultural label alignment: High / Medium / Low
3. Language style modal alignment: High / Medium / Low

Important:
- Do not provide a final score.
- Give one short evidence phrase for each label.
- Human raters will verify all outputs.
- Be conservative: when uncertain, choose the lower rating.

Input:
{{RESPONSE_TEXT}}
{{TASK_CONTEXT}}
{{ROLE_PROFILE}}

Output format (JSON):
```json
{
  "historical_context_matching": {
    "label": "High|Medium|Low",
    "evidence": "[brief evidence phrase from response]"
  },
  "cultural_label_alignment": {
    "label": "High|Medium|Low",
    "evidence": "[brief evidence phrase from response]"
  },
  "language_style_modal_alignment": {
    "label": "High|Medium|Low",
    "evidence": "[brief evidence phrase from response]"
  }
}
```

## Annotation Guidelines

### Historical Context Matching
- **High**: Response includes specific dynasty/period names, historical figures with correct context, verifiable historical events
- **Medium**: Response references historical context generally without specific details
- **Low**: Response lacks historical grounding or contains anachronistic elements

### Cultural Label Alignment  
- **High**: Core cultural concepts used with precise meanings matching established cultural definitions
- **Medium**: Cultural concepts used correctly but imprecisely or without full contextual grounding
- **Low**: Cultural concepts misused, conflated, or applied outside their appropriate context

### Language Style Modal Alignment
- **High**: Language register, formality, and tone perfectly match the assigned role and task type
- **Medium**: Generally appropriate style with minor inconsistencies
- **Low**: Significant style mismatch with role or task requirements
