# Sentiment of Polish politicians in tweets
Authors:
- Łukasz Janisiów
- Maciej Kuchciak
- Mateusz Pliszka

## Project Overview:
The project analyzes the sentiment of Polish politicians' statements on the X platform (formerly Twitter) to examine their communication strategies on social media. The study includes all political parties represented in the Sejm of the 10th term, with more than 10 members in the Sejm. The research methodology combines a lexical approach using VADER with advanced natural language processing (NLP) models such as BERT. The analysis includes Exploratory Data Analysis (EDA), Sentiment Analysis, Clustering, Network Analysis, and Hate Speech Detection

---

### 1. Data Downloader
- **File:** [Downloader.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/1.%20Downloader.ipynb)  
- **Description:** This notebook handles downloading raw data from X. 
- **Main conclusions:** Retrieving data from the X platform API is not straightforward, even with a paid subscription. To ensure the retrieval of all tweets, a custom function was implemented to verify whether all tweets from a specified time period were successfully collected. If any were missing, the function automatically narrowed the time range to enforce complete data retrieval.

---

### 2. Basic Data Cleaning
- **File:** [Basic Data Cleaning.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/2.%20Basic%20Data%20Cleaning.ipynb)  
- **Description:** Performs preprocessing tasks such as adding party affiliation, handling duplicates, text translation and text normalization to prepare the data for analysis.  

---

### 3. Exploratory Data Analysis (EDA)
- **File:** [EDA.ipynb]([EDA.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/3.%20EDA.ipynb))  
- **Description:** Visualizes and summarizes data trends, distributions, and relationships.
- **Main Conclusions:**
  1. The three most active users are Patryk Jaki (PIS), Bartłomiej Pejo (Konfederacja), and Włodzimierz Skalik (Konfederacja). Each of them published around 1000 tweets in one year, averaging more than 2.5 tweets per day.  
  2. The activity of politicians on X overlaps with important events in Poland. Peaks in activity are observed after the parliamentary election, the appointment of the Council of Ministers, and during the Euroelection and the flood in 2024.  
  3. The most common tweet lengths are very short (up to 5 words) and around 10 words. Interestingly, there is also a notable peak at around 40 words. Each party shows a significant number of tweets with approximately 40 words.  
  4. Politicians in Nowa Lewica use, on average, more than one emoji per tweet, while those in PL2050 use only 0.2 emojis per tweet on average.  
  5. PO has the most active observers, with the highest average for each public metric. The second party with the highest average number of retweets and replies is PIS. However, in terms of tweet likes and quotes, PL2050 ranks second.  
  6. Donald Tusk from PO is the most influential politician in Poland, exhibiting the highest engagement in response to his posts among all politicians.
     
---

### 4. Sentiment Analysis
- **File:** [Sentiment Analysis.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/4.%20Sentiment%20analysis.ipynb)  
- **Description:** Implements sentiment classification using machine learning techniques. This notebook predicts the sentiment of text data.  
- **Main conclusions:** Lorem ipsum. 



---

### 5. Clustering
- **File:** [Clustering.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/5.%20Clustering.ipynb)  
- **Description:** Applies clustering algorithms to group similar text data, aiding in unsupervised learning tasks like topic modeling and segmentation.  
- **Main conclusions:** Lorem ipsum.

---

### 6. Network Analysis
- **File:** [Network.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/6.%20Network.ipynb)  
- **Description:** Conducts network analysis to identify relationships and connections within text data. Suitable for social media or graph-based NLP tasks.  
- **Main conclusions:** Lorem ipsum.

---

### 7. Hate Speech Detection
- **File:** [Hate Speech.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/7.%20Hate%20speech.ipynb)  
- **Description:** Identifies and filters hate speech using machine learning techniques. Includes fine-tuning of the model and data augmentation.  
- **Main conclusions:** Lorem ipsum.

---

### 8. Machine Learning Model
- **File:** [ML_model.ipynb](ML_model.ipynb)  
- **Description:** Trains and evaluates a comprehensive machine learning model that integrates various features and insights from prior notebooks to deliver high-performance predictions.  
- **Main conclusions:** Lorem ipsum.

---

## How to Use
1. Clone this repository to your local machine.
2. Install the necessary dependencies.
3. Navigate to the respective notebook for your desired task.
