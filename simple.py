import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import langchain.globals as lcg

lcg.set_verbose(True)

os.environ["GOOGLE_API_KEY"] = 'AIzaSyDads0ZgdiMd9alQfwACctU0dxxKkXD5zo'
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template_resto = PromptTemplate(
    input_variables=['season', 'round', 'total_no_of_laps', 'randomness_factor', 'model_selection'],
    template="think you are a data scientist at an f1 team, given the season: {season}, round: {round}, total number of laps: {total_no_of_laps}, randomness factor: {randomness_factor} and model selection: {model_selection}, give me the details(circuit name, location, number of laps, circuit length, race distance, lap record, nature of the track and optimal theoretical tyre strategy using a combination of soft, medium and hard tyres) of the circuit corresponding to one of the rounds(eg. in 2024, 1 is bahrain, 24 is abu dhabi) and then an example lap-by-lap prediction containing driver code, driver name, team, position, predicted lap to pit, laps till pitting, tyre degradatiion, fuel consumption rates, mgu-h and mgu-k usage, status and laptime of ALL the 20 drivers at a given race (give me a a direct table prediction without any code or theory)"
)

chain_resto = RunnableLambda(lambda inputs: prompt_template_resto.format(**inputs)) | model

st.title("Formula 1 Pit Stop Strategy Prediction Model")

st.header("Model Input")

with st.form(key='prediction_form'):
    season = st.selectbox("Season", [2018,2019,2020,2021,2022,2023,2024], key='season')
    round_number = st.number_input("Round", min_value=1, max_value=24, step=1, key='round')
    total_laps = st.number_input("Total Number of Laps", min_value=1, max_value=78, step=1, key='total_laps')
    randomness_factor = st.slider("Randomness Factor", 0, 50, 10, key='randomness_factor')

    model_selection = st.selectbox("Model Selection", ["Optimized for Pit Stop Prediction"], key='model_selection', help="Select the model best suited for your needs.")

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    if all([season, round_number, total_laps, randomness_factor, model_selection]):
        input_data = {
            'season': season,
            'round': round_number,
            'total_no_of_laps': total_laps,
            'randomness_factor': randomness_factor,
            'model_selection': model_selection,
        }

        predicted_pit_stop = chain_resto.invoke(input_data)

        st.markdown('<div class="subtitle">Prediction:</div>', unsafe_allow_html=True)
        st.markdown('<div class="predicted_pit_stop">', unsafe_allow_html=True)
        st.markdown(predicted_pit_stop, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("Please fill in all the form fields.")
