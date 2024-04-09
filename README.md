## Bengali Sentence Semantic Classification with BERT (Fine-tuned and Quantized)

This project implements a web application for classifying the semantic meaning of Bengali sentences into positive or negative categories. It utilizes a fine-tuned and quantized version of the BERT (Bidirectional Encoder Representations from Transformers) model, enabling efficient inference on CPU and edge devices. The application is built using the FastAPI framework and transformers library.

### Features

- Semantic classification of Bengali sentences into positive or negative categories.
- Utilizes a fine-tuned and quantized BERT model optimized for CPU and edge device deployment.
- Real-time processing with efficient display of processing time.
- Simple and intuitive web interface for entering sentences and viewing classification results.

### Pre-trained Model

The application uses a fine-tuned and quantized version of the BERT model specifically trained for Bengali language sentiment analysis. Fine-tuning involves training the model on a task-specific dataset to adapt it to the desired task, while quantization reduces the precision of the model's parameters to improve efficiency during inference. This optimized model enables fast and efficient inference on CPU and edge devices.

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

### Optimizations

- **Fine-tuning**: The BERT model is fine-tuned on a Bengali sentiment analysis dataset to improve its performance on the specific task.
- **Quantization**: The fine-tuned model is quantized to reduce its precision and enable efficient inference on CPU and edge devices.

### Dependencies

- [FastAPI](https://fastapi.tiangolo.com/): Web framework for building APIs with Python.
- [Transformers](https://huggingface.co/transformers/): Library for state-of-the-art natural language processing tasks, including pre-trained models such as BERT.
- [PyTorch](https://pytorch.org/): Open-source machine learning library used for various deep learning tasks.

### Acknowledgments

- This project leverages the transformers library and PyTorch for implementing state-of-the-art natural language processing models and optimizing them for deployment on CPU and edge devices.
- Special thanks to the developers of FastAPI for creating a powerful and efficient framework for building web applications.

Feel free to explore, modify, and extend this project for your own use case. If you have any questions or suggestions, please feel free to reach out!
