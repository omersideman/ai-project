# ai-project

Using AI to predict whether new music will go viral on social media.

### Setup

1. Setup virtual env, install requirements, and setup jupyter kernel:

   ```
       python3 -m venv venv
       source venv/bin/activate
       pip install -r requirements.txt
       python -m ipykernel install --user --name=venv
   ```
2. Duplicate .env.example, rename to .env and replace keys (only if need to use spotify api) and path to src directory.

### Other important things

1. When creating a new notebook, paste this at the top:

```
import sys; sys.path.append('..')
from dotenv import load_dotenv
load_dotenv()
src_dir = os.getenv('SRC_DIR')
assert(src_dir)
os.chdir(src_dir)
```

2. when installing a new package,make sure to add it to requirements.txt
