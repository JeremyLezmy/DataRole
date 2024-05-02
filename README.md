[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-position-assessment.streamlit.app/)


# Data Role Assessment App

## Overview
The Data Role Assessment App is designed to evaluate candidates for various data roles based on their skills and experiences. It provides a user-friendly interface to set coefficients for different topics, perform assessments, and interact with an AI assistant for additional insights.

## Features
- **Coefficient Setting**: Users can set coefficients for different topics such as Python, SQL, ETL, etc., to customize the evaluation criteria. Users can also add/remove any topic directly in the application.
- **Assessment**: Based on the coefficients, the app allows interviewers to select a candidate's level (Junior, Confirmed, Senior) and finalize the assessment.
- **Topic Description**: Provides descriptions of expected knowledge and experiences for each evaluated topic and level (Junior, Confirmed, Senior).
- **Groq Assistant**: Integrates an AI assistant powered by Groq AI, allowing users to interact with different language models, get insights, discuss assessment results, and even request plot figures based on assessment data.

## Default Coefficients
The default coefficients for different topics are as follows:

| Topic                | Category    | Subcategory          | Data Viz | Data Engineer | Data Scientist | DBA  |
|----------------------|-------------|----------------------|----------|---------------|----------------|------|
| Python               | Technical   | Language             | 1.00     | 4.00          | 6.00           | 1.00 |
| Big Data             | Technical   | Cloud                | 1.00     | 4.00          | 1.00           | 2.00 |
| SQL                  | Technical   | Language             | 4.00     | 4.00          | 2.00           | 6.00 |
| ...         | ...                  | ...               | ...      | ...           | ...            | ...  |

## Usage
1. **Coefficient Setting**: Users can set their own coefficients or use the default coefficients provided.
2. **Assessment**: After setting coefficients, users can perform assessments by selecting a candidate's level for each topic based on the evaluation criteria.
3. **Groq Assistant**: Once an assessment is saved and submitted, users can interact with the AI assistant to discuss assessment results, gain insights, and request plots based on the assessment data.

## Setup
To set up the project locally, follow these steps:
1. Clone the repository.
2. Install dependencies listed in `requirements.txt`.
3. Create a `.env` file and add necessary environment variables.
4. Run the Streamlit app using `streamlit run Home.py`.

## Contributing
Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
