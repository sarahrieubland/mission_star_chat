# Mission STAR chat
Chat to reformulate descriptions into STAR

### 1. Install requirements

Create environment with python 3.11 (for langfus compatibility) and activate
```
#pyenv install 3.11.9
#pyenv local 3.11.9
python -m venv .venv #(python3.11 -m venv .venv)
source .venv/bin/activate
``` 

Install requirements (for my set-up):
```
python3 -m pip install -r requirements.txt
```
If you need to re-create the environement from scratch: 
```
deactivate 2>/dev/null
rm -rf .venv
```

## 2. Environment variables .env

in the `.env` file at the root of the project, you need to put the secrets:

```
OPENAI_API_KEY=
LANGFUSE_SECRET_KEY=j
LANGFUSE_PUBLIC_KEY=
```

## 3. Run chainlit app

```
chainlit run app.py -w
```

## 4. Langgraph workflow for the agentic chatbot

gather_info → generate → evaluate 
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
         complete              section_to_improve set
                                        ↓
                               gather_info (asks specific section prompt)
                                        ↓
                                    generate
                                        ↓
                                    evaluate
                                        ↓
                                      ...