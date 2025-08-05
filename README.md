# Streamlit Intro Workshop & Demos 🚀

Welcome to **Streamlit Intro** – a hands-on set of mini-projects that showcase how easy it is to build beautiful, data-driven web apps in pure Python.  Whether you are brand-new to Streamlit or looking for inspiration for your own project, this repo has you covered with six progressively more advanced demos:

| Demo | What it illustrates | File |
|------|---------------------|------|
| 1. Basics | Core Streamlit primitives – text, widgets, layout, charts | `apps/01_basics_app.py` |
| 2. Text Analyzer | Working with free-form text, Altair charts, and simple user input | `apps/02_text_example.py` |
| 3. CSV Explorer | Interactive tabular data exploration and descriptive statistics | `apps/03_table_example.py` |
| 4. Crypto Dashboard | API calls, caching, session-state, and responsive layouts | `apps/04_advanced_demo.py` |
| 5. LLM Trade Advisor | Integrating OpenAI + Bybit REST APIs to create a trading assistant | `apps/05_llm_trade_advisor.py` |
| 6. AI Micro-cap Auto-Trader | A research prototype for autonomous portfolio re-balancing | `apps/06_autotrader.py` |

---

## 📂 Repository Structure
```
streamlit_intro/
  ├── apps/                 # ↳ All demo Streamlit scripts live here
  ├── ai_microcap_data/     # ↳ CSVs generated/used by the autotrader demo
  ├── requirements.txt      # ↳ Python dependencies
  └── README.md             # ↳ You are here!
```

---

## ⚙️  Prerequisites
* Python **3.9** or newer (3.10 / 3.11 fully supported)
* A modern web browser (Chrome, Firefox, Safari, Edge)
* _(Optional)_ API keys for the advanced demos (see below)

---

## 💻 Quick Start
```bash
# 1. Clone the repo
$ git clone https://github.com/<you>/streamlit_intro.git
$ cd streamlit_intro

# 2. Create & activate a virtual environment (recommended)
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Launch the first demo 🎉
$ streamlit run apps/01_basics_app.py
```
Streamlit will open a new browser tab at `http://localhost:8501`.

---

## 🔑 Environment Variables (for the advanced apps)
The **LLM Trade Advisor** and **Auto-Trader** need a few secrets to be able to talk to external services.  You can either export them directly in the shell _or_ create a `.env` file at the project root.

```bash
# .env example
OPENAI_API_KEY="sk-..."            # Needed for apps/05 & apps/06
BYBIT_API_KEY="..."                # Needed for app/05 if you want to PLACE orders
BYBIT_API_SECRET="..."             # "
BYBIT_ACCOUNT_TYPE="UNIFIED"       # (optional) defaults to UNIFIED
```
The apps will automatically load the `.env` file via **python-dotenv**.

---

## 🚀 Running Each Demo
1. **Basics**
   ```bash
   streamlit run apps/01_basics_app.py
   ```
2. **Text Analyzer**
   ```bash
   streamlit run apps/02_text_example.py
   ```
3. **CSV Explorer**
   ```bash
   streamlit run apps/03_table_example.py
   ```
4. **Crypto Dashboard**
   ```bash
   streamlit run apps/04_advanced_demo.py
   ```
5. **LLM Trade Advisor** (requires `OPENAI_API_KEY`; trading requires Bybit keys)
   ```bash
   streamlit run apps/05_llm_trade_advisor.py
   ```
6. **AI Micro-cap Auto-Trader** (research prototype – _highly experimental_!)
   ```bash
   streamlit run apps/06_autotrader.py
   ```

Because Streamlit hot-reloads on every file save, you can freely tweak the code while the apps are running and instantly see the effect in the browser.

---

## 🗃 Data Folder (`ai_microcap_data/`)
The auto-trader stores its portfolio snapshot, trade log, and performance metrics as CSVs inside this folder so that state is preserved across restarts:

* `portfolio.csv` – current positions
* `performance.csv` – daily equity curve
* `trades.csv` / `orders.csv` – execution history

Feel free to delete these files at any moment – they will be recreated on the next run.

---

## 🛠 Development Tips
* Use **`st.cache_data`** / **`st.cache_resource`** to memoise expensive API calls or ML inference.
* The **command palette** (`⌘ / Ctrl + K`) is a hidden gem for power-users.
* Turn themes on/off via the settings icon in the top-right corner or programmatically via `st.theme()`.
* Multipage apps are as simple as dropping extra scripts into a `pages/` folder – Streamlit picks them up automatically.

---

## 🤝 Contributing
Pull requests are warmly welcome!  If you have an idea for an additional demo or an improvement to the existing ones, feel free to open an issue or a PR.

