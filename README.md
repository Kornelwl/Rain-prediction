# Project 12 - Rain Risk Scoring and Short-Term Prediction

## Goal

The goal of this project is to estimate near-future rain risk from recent weather measurements.

The project uses the shared Weather REST API as the main data source. It collects weather data, stores raw observations, cleans and transforms the dataset, creates rolling features, calculates a rain-risk score from 0 to 100, and evaluates the prediction quality.

## Architecture

Weather REST API -> raw storage -> cleaning -> feature engineering -> rain risk scoring -> evaluation -> charts/report

## Data Source

Weather REST API:

- Base URL: `https://e6uw49pbah.execute-api.us-east-1.amazonaws.com/dev`
- Batch endpoint: `/weather/batch?station_id=GDN_01&limit=100`
- Authorization header: `Authorization: Bearer STUDENT_TOKEN_2026`

## Features

The project creates the following features:

- rolling average humidity
- rolling average cloud cover
- rolling average wind speed
- previous rain value
- rain sum from recent measurements
- pressure change
- next-measurement rain label

## Rain Risk Score

The rain-risk score is calculated using weighted rules:

- humidity: 30%
- cloud cover: 25%
- pressure drop: 20%
- previous rain: 15%
- wind speed: 10%

The score is converted into three classes:

- low
- medium
- high

A rain prediction is created when the score is at least 60.

## Evaluation

The project calculates:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt