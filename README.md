# Urban-Risk-AI-Real-Time-Environmental-Threat-Prediction
Urban Risk AI - 
Real-time environmental risk prediction using weather, air quality, and traffic data.

Overview - 
Urban Risk AI is a data-driven system that predicts short-term environmental risks in urban areas.
It combines weather conditions, air pollution levels, and traffic congestion to estimate PM2.5 levels and generate simple, clear guidance for users.

Problem Statement - 
Modern cities experience rapid changes in pollution, temperature, and traffic flow.
These fluctuations affect daily routines, health, and overall wellbeing.
Most people only react after the environment becomes uncomfortable or unsafe.
This project aims to help users anticipate risks before they occur.

Why Agents - 
Agents allow automated, continuous, and intelligent decision-making.
Traditional apps only display raw numbers.
An agent can:
- Fetch data automatically
- Process and clean it
- Run predictive models
- Interpret results
- Communicate insights in simple language
This makes agents suitable for real-time risk prediction.

What the System Does - 
- Collects weather, pollutant, and traffic data from public APIs
- Cleans, aligns, and merges the data into a single dataset
- Engineers additional time-based and environmental features
- Trains a machine learning model to predict PM2.5 levels
- Uses the model inside an agent that summarizes risks and offers guidance

APIs used :- 
- Open-Meteo Weather API
- Open-Meteo Air Quality API
- TomTom Traffic API

Tools and Technologies - 
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Requests
- Matplotlib
