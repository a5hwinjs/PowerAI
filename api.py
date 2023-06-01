import pandas as pd 
from pandasai import PandasAI
import os

api_key = os.getenv("STARCODER_API")


df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})

from pandasai.llm.starcoder import Starcoder

llm = Starcoder(api_key)


try:
    pandas_ai = PandasAI(llm, conversational=False)
    pandas_ai(df, prompt='plot the happiest countries')
    pandas_ai(df,"Plot the histogram of countries showing for each the gdp, using different colors for each bar")
    
except Exception as e:
    print(e)