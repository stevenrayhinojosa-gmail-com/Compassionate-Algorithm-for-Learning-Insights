# Student Behavior Analysis and Forecasting

A Streamlit-based machine learning application that analyzes and predicts student behavior patterns using historical behavioral data.

## Features

- Data preprocessing and cleaning of behavioral records
- Interactive visualization of behavior patterns
- Weekly and daily trend analysis
- Machine learning-based behavior prediction using XGBoost
- Comprehensive statistical analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd student-behavior-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload your behavior data CSV file in the following format:
   - Date column in M/D/YYYY format
   - Behavior markers (r/y/g) for different time slots
   - Headers should match the provided template

3. View the analysis and predictions in the interactive dashboard

## Project Structure

```
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data preprocessing and feature engineering
├── model_trainer.py       # XGBoost model training and prediction
├── utils.py              # Utility functions
├── tests/                # Test files
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Testing

Run the tests using:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
