# California Housing Explorer

An interactive dashboard and house price predictor for the 1990 California Housing data
(20,640 census block groups). Built with Streamlit, scikit-learn and Plotly.

**Hyunjin Yi and Grady Hill**

## Use it online
Open the app link given in our user guide. Nothing to install.

## Run it on your computer
Requires Python 3.10 or newer.

```bash
bash run_app.sh
```

The script creates a virtual environment, installs the packages in `requirements.txt` and starts
the app at http://localhost:8501. On Windows, run these instead:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Pages
| Page | What it does |
|---|---|
| Home | Overview of the data and the model |
| Explore | Filter districts by ocean proximity, income, house age and value; map, distributions, relationships, comparison by category, and a CSV download of the filtered data |
| Predict | Enter a California location and district details to get an estimated median house value with an 80% range, the effect of changing each input, and the nearest real districts |
| About the model | Model type, test accuracy and limitations |

## Project layout
| Path | Contents |
|---|---|
| `app.py` | The Streamlit app |
| `src/data.py` | Loading, filtering and summarizing the data |
| `src/model.py` | Model training, input checks, predictions and explanations |
| `data/housing.csv` | California Housing data (1990 U.S. Census) |
| `run_app.sh` | One command local launcher |
