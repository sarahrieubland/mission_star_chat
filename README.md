# Mission STAR chat
Chat to reformulate descriptions into STAR

### 1. Install requirements

Create environment with python 3.11 (for langfuse compatibility) and activate
```
python3 -m venv .venv
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
To update the requirements file : 
```
pip freeze > requirements.txt
```

## 2. Environment variables .env

in the `.env` file at the root of the project, you need to put the secrets:

```
OPENAI_API_KEY=
LANGSMITH_API_KEY=
```

## 3. Run chainlit app

```
chainlit run app.py -w
```

## 4. Langsmith

Store the prompts in Langsmith
```
python3 -c "from app_agent_combined import push_all_prompts_to_hub; push_all_prompts_to_hub()"
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


## 6. Cloudflare tunnel 

For a quick (ephemeral) tunnel to expose the chainlit application on a public url, run:
```
chainlit run app.py --port 8000
cloudflared tunnel --url http://localhost:8000

```
The temporary url will be printed (like https://random-string.trycloudflare.com).


## 7. Deploy with Render
