# AI Troubleshooting Assistant for L1/L2 Support

A simple chat assistant that helps support engineers investigate common technical issues. The user describes a problem in plain language; the assistant finds the most relevant entry in a small knowledge base, suggests step-by-step troubleshooting checks, names the likely escalation team, and shows which knowledge-base entry it used. If the knowledge base does not cover the issue, it says so and asks for more information instead of guessing.

> Practical AI Assignment 3, Technology Operations Intern. Uses sample/simulated data only.

## Features

- Chat interface (Streamlit) with example queries in the sidebar
- Knowledge base of 10 sample issues (symptoms, diagnostic checks, resolution, escalation team)
- Retrieval of the most relevant entries using TF-IDF similarity
- AI-written answer grounded only in the retrieved entries (when an API key is provided)
- Fallback template answer built directly from the knowledge base (when no API key is provided)
- Source entry shown for every answer, with a match score
- Responsible handling of unknown issues: below a relevance threshold the assistant does not call the AI and asks for more details

## Project structure

```
AI_Troubleshooting Assistant/
├── app.py                    # Streamlit UI + retrieval + AI answer logic
├── data/
│   └── knowledge_base.json   # Sample knowledge base (10 entries)
├── requirements.txt
└── README.md
```

## Setup and run

Requires Python 3.9+.

```bash
# 1. (Windows) activate the virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) enable AI-written answers (PowerShell), or just paste the key in the app sidebar
$env:GROQ_API_KEY="your_key_here"

# 4. Run
streamlit run app.py
```

The app opens at http://localhost:8501.

## Approach

1. **Knowledge base.** Each entry in `data/knowledge_base.json` has an ID, title, category, symptoms, keywords, ordered diagnostic checks, a known resolution and an escalation team.
2. **Retrieval.** Each entry's title, category, symptoms and keywords are converted into TF-IDF vectors (scikit-learn, unigrams and bigrams). The user's message is vectorised the same way and compared using cosine similarity. The top matches above a relevance threshold are kept.
3. **Unknown-issue guard.** If no entry reaches the threshold (default 0.12), the AI is not called. The assistant states that the knowledge base is insufficient, asks for the affected system, exact error and recent changes, and suggests escalating.
4. **Answer generation.** The retrieved entries are passed to the AI model inside a prompt that:
   - tells it to answer only from the provided entries,
   - forbids inventing causes, commands or resolutions,
   - asks it to say the knowledge base is insufficient and ask clarifying questions if the entries do not match,
   - requests a fixed structure: likely cause, numbered troubleshooting steps, known resolution, escalation team, sources.
5. **Transparency.** The UI always shows the retrieved entries (ID, title, match score, symptoms, checks, resolution, escalation team) so the engineer can verify the answer.

## AI tools and models used

- **Answer generation:** Groq API using the `openai/gpt-oss-120b` model. The app also supports Anthropic Claude models. The provider is detected from the API key, and the model name is configurable in the sidebar.
- **Retrieval:** classical TF-IDF (scikit-learn). No embeddings or external services are needed.
- **Development assistance:** I used AI tools to help generate the initial code and draft this README. I designed the knowledge base entries, tested the app with different queries, set up the Groq API, and tuned the relevance threshold. I can explain how each part works.

## Assumptions

- The knowledge base is small, curated and trusted; all data is simulated.
- Users describe one issue at a time. Each message is handled independently (no multi-turn memory).
- The relevance threshold (0.12) is a reasonable starting value and can be tuned in the sidebar.
- Escalation teams are example names and would map to real teams in an organisation.

## Testing

Manual tests were run in the app, first in template mode (no API key) and then with AI answers enabled:

| Input                                                      | Expected                             | Result               |
| ---------------------------------------------------------- | ------------------------------------ | -------------------- |
| The payment API is returning a 500 error                   | KB-001, Payments Application Support | Pass (51% match)     |
| Application is timing out while connecting to the database | KB-002, DBA Team                     | Pass                 |
| Server says no space left on device                        | KB-004, Infrastructure / SRE         | Pass                 |
| Users are not receiving emails                             | KB-007, Messaging Platform Team      | Pass (22% match)     |
| My laptop screen is flickering                             | No match; ask for more detail        | Pass (no guess made) |

With the Groq API key enabled, the AI produced reworded, structured answers from the same knowledge-base entry (for example, the payment API 500 error) and showed the source entry and escalation team. Unsupported issues such as the laptop screen example were still declined without calling the AI.

## Limitations

- **Keyword-based retrieval.** TF-IDF matches words, not meaning, so unusual phrasing or synonyms may score low and be rejected. Semantic embeddings would improve this.
- **Small knowledge base.** Only 10 sample entries; real use would need many more, kept up to date.
- **No multi-turn context.** Follow-up messages such as "I checked the logs, nothing" are not linked to the previous question.
- **Threshold trade-off.** A higher threshold reduces wrong answers but rejects more valid queries; a lower one does the opposite.
- **AI can still err.** The prompt restricts the model to the retrieved entries, but model output should be verified using the source entries shown.
- **Not production-ready.** No authentication, logging, feedback loop or access to live systems.

## Possible improvements

- Embedding-based or hybrid (keyword + semantic) retrieval
- Multi-turn conversation and follow-up questions
- User feedback buttons to improve the knowledge base
- Admin interface to add or edit knowledge-base entries
- Integration with ticketing tools for automatic escalation
