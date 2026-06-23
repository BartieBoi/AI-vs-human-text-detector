AI vs human text detector quick start guide

1. Ensure you have a recent version of python installed on your computer

2. clone the repository:
    
    ```bash
    git clone [https://https://github.com/BartieBoi/AI-vs-human-text-detector](https://https://github.com/BartieBoi/AI-vs-human-text-detector)
    
    create a folder on your computer to put the project

    change directory in CMD and copy the repository there, or download files from github and move into your folder

3. in CMD, make sure the directory is the your folder, then run bash:
    pip install -r requirements.txt

4. in CMD, run bash:
    streamlit run app.py

    Once executed, a local web server will spin up, and the application will automatically open in your default browser at http://localhost:8501.



Design Decisions & Methodology:

-Feature Engineering: 
    Raw text is transformed using a dual-layered approach. TF-IDF acts as the baseline to catch distinct vocabulary frequency patterns , while Word Embeddings (e.g., GloVe/Word2Vec) capture semantic meanings and deeper context. Statistical linguistic features (like sentence length and word counts) are engineered to detect structural machine patterns.
-Model Selection (Traditional ML vs. Deep Learning):
    To evaluate speed versus accuracy trade-offs, the pipeline compares two distinct paradigms:
-Traditional ML (SVM, Decision Tree, AdaBoost):
    Fast to train and highly interpretable, making them ideal for high-dimensional TF-IDF vectors.  Deep Learning (FNN, LSTM, CNN): Slower to train but capable of automatically capturing complex sequential patterns, text structures, and localized phrase combinations.  