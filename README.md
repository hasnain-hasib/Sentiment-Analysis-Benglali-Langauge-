## Sentence Semantic Classification and Sentiment Analysis for Bengali Language

This project provides a web application for classifying the semantic meaning of Bengali sentences as either positive or negative using a pre-trained deep learning model. It utilizes the FastAPI framework for building the backend server and transformers library for natural language processing tasks.

### Features

- Semantic classification of Bengali sentences into positive or negative categories.
- Simple and intuitive web interface for entering sentences and viewing classification results.
- Real-time processing with efficient display of processing time.

### Pre-trained Model

The application uses a pre-trained transformer model specifically designed for Bengali language sentiment analysis. The model is loaded using the `AutoModelForSequenceClassification` class from the transformers library, and the corresponding tokenizer is used to preprocess input sentences.

### Usage

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install the required Python libraries:

   ```bash
   pip install fastapi uvicorn transformers torch
   ```

3. Set up the server by running the `main.py` script:

   ```bash
   python main.py
   ```

4. Access the web interface by navigating to `http://localhost:8000` in your web browser.
5. Enter a Bengali sentence in the provided input field and click the "Classify" button to see the classification result.
6. The result will display whether the sentiment of the sentence is positive or negative, along with the processing time.

### Dependencies

- [FastAPI](https://fastapi.tiangolo.com/): Web framework for building APIs with Python.
- [Transformers](https://huggingface.co/transformers/): Library for state-of-the-art natural language processing tasks, including pre-trained models.
- [PyTorch](https://pytorch.org/): Open-source machine learning library used for various deep learning tasks.

### Acknowledgments

- This project is based on the transformers library, which provides easy-to-use interfaces for working with pre-trained language models.
- Special thanks to the developers of FastAPI for creating a powerful and efficient framework for building web applications.

Feel free to explore, modify, and extend this project for your own use case. If you have any questions or suggestions, please feel free to reach out!
