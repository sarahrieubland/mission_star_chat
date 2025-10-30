# Mission STAR chat
Chat to reformulate descriptions into STAR

### 1. Install requirements

Create environment and activate
```
python3 -m venv .venv
source .venv/bin/activate
``` 

Install requirements (for my set-up):
```
python3 -m pip install -r requirements.txt
```


## 2. Environment variables .env

in the `.env` file at the root of the project, you need to put the secrets:

```
OPENAI_API_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
```