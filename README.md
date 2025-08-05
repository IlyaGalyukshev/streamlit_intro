# Streamlit Hands-On Workshop 🚀

Welcome! This workshop takes you from zero to building interactive web apps with [Streamlit](https://streamlit.io) in just a few exercises.

## 0. Setup

1. Install **Python 3.9** or newer.
2. Clone or download this repository.
3. From the project root, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

4. Run the first demo:

```bash
streamlit run apps/01_basics_app.py
```

5. **Set your credentials** (at minimum `OPENAI_API_KEY`) as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."
```

Alternatively, copy `env.example` to `.env` and fill in your keys — the app will load it automatically.

6. Run the LLM Trade Advisor:

```bash
streamlit run apps/05_llm_trade_advisor.py
```

The demo now supports **Market BUY / SELL buttons** with amount either in contracts or **USDT** (auto-converted). To place orders you _must_ set `BYBIT_API_KEY` and `BYBIT_API_SECRET` in your `.env`.

---

## 1. Workshop Agenda

| Step | Topic | File |
|------|-------|------|
| 1 | Streamlit **basics**: text, widgets, charts, layouts | `apps/01_basics_app.py` |
| 2 | Working with **text data** | `apps/02_text_example.py` |
| 3 | Working with **tabular data** (CSV) | `apps/03_table_example.py` |
| 4 | **Advanced demo**: live Crypto dashboard (API calls, caching, session-state) | `apps/04_advanced_demo.py` |
| 5 | LLM Trade Advisor (Bybit + OpenAI) | `apps/05_llm_trade_advisor.py` |

Each script is self-contained. Tweak the code while the app is running—Streamlit hot-reloads automatically!

---

## 2. Hot Tips & Lesser-Known Features

- **`st.cache_data` / `st.cache_resource`** – Memoize expensive operations (API calls, ML models, large file reads).
- **Session State** – Persist values across reruns via `st.session_state`.
- **Layout primitives** – `st.columns`, `st.tabs`, `st.expander`, `st.container`, `st.sidebar`.
- **Multipage apps** – Drop extra scripts inside a `pages/` directory.
- **Theme switching** – Click the settings icon → Theme, or call `st.theme("dark")`.
- **Command Palette** – Press `⌘ / Ctrl + K` inside an app to jump anywhere quickly.
- **Run from Jupyter** – `streamlit run my_notebook.ipynb` works! ✨

---

## 3. Next Steps

1. Fork this repo and add your own page under `pages/`.
2. Deploy your creation to the cloud for free with **Streamlit Community Cloud** (`streamlit deploy`).
3. Share your app URL and inspire others.

Happy Streamliting! 🎈