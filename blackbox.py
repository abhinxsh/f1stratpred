import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import langchain.globals as lcg

lcg.set_verbose(True)

os.environ["GOOGLE_API_KEY"] = 'AIzaSyAbK6xyFCoXrFPGkUvNiD8yO17sK8A7gOQ'
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['season', 'round', 'total_no_of_laps', 'randomness_factor', 'model_selection'],
    template="think you are a data scientist at mercedes f1 team, given the season: 2024, round: 10, total number of laps: 50, randomness factor: 10 and model selection: pit stop prediction, give me an example lap-by-lap prediction containing driver name, position, laps till pitting, status and laptime at bahrain gp in 2024 (give me a a direct table prediction without any code or theory)"
             "Season: {season}\n"
             "Round: {round}\n"
             "Total Number of Laps: {total_no_of_laps}\n"
             "Randomness Factor: {randomness_factor}\n"
             "Model Selection: {model_selection}\n"
)

chain_resto = RunnableLambda(lambda inputs: prompt_template_resto.format(**inputs)) | model
st.title("F1 Pit Stop Prediction Model")

st.header("Model Input")

season = st.selectbox("Season", [2021, 2022, 2023])
round = st.number_input("Round", min_value=1, step=1)
total_laps = st.number_input("Total Number of Laps", min_value=1, step=1)
randomness_factor = st.slider("Randomness Factor", 0, 50, 10)

model_selection = st.selectbox("Model Selection", ["Optimized for Pit Stop Prediction"], help="Select the model best suited for your needs.")

submit_button = st.form_submit_button(label='Predict')
st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    if all([season, round, total_no_of_laps, randomness_factor, model_selection]):
        input_data = {
            'season': season,
            'round': round,
            'total_no_of_laps': total_no_of_laps,
            'randomness_factor': randomness_factor,
            'model_selection': model_selection,
        }

        predicted_pit_stop = chain_resto.invoke(input_data)

        st.markdown('<div class="subtitle">Predict:</div>', unsafe_allow_html=True)
        st.markdown('<div class="predicted_pit_stop">', unsafe_allow_html=True)
        st.markdown(predicted_pit_stop, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("Please fill in all the form fields.")