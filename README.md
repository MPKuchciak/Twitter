# Sentiment of Polish politicians in tweets
Authors:
- Łukasz Janisiów
- Maciej Kuchciak
- Mateusz Pliszka

## Project Overview:
The project analyzes the sentiment of Polish politicians' statements on the X platform (formerly Twitter) to examine their communication strategies on social media. The study includes all political parties represented in the Sejm of the 10th term, with more than 10 members in the Sejm. Time range of analysed tweets includes 16 October 2023 - 15 October 2024 (1 year after the Parliament election in Poland). The research methodology combines a lexical approach using VADER with advanced natural language processing (NLP) models such as BERT. The analysis includes Exploratory Data Analysis (EDA), Sentiment Analysis, Clustering, Network Analysis, and Hate Speech Detection. 
In our analysis we try to answer questions:
- Are there differences in sentiment across political parties? If so, which individuals and parties are associated with negative sentiment?
- Are there any politician that receive significant attention in conversations?
- Does hate speech appear in political tweets?
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
- **File:** [EDA.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/3.%20EDA.ipynb) 
- **Description:** Visualizes and summarizes data trends, distributions, and relationships.
- **Main Conclusions:**
  1. The three most active users are Marcin Kulasek (NL), Marcin Kierwiński (PO), and Krzysztof Śmieszek (NL). Each of them published around 2000 tweets in one year, averaging more than 2.5 tweets per day.  
  2. The activity of politicians on X overlaps with important events in Poland. Peaks in activity are observed after the parliamentary election, the appointment of the Council of Ministers, and during the Euroelection and the flood in 2024.  
  3. The most common tweet lengths are very short (up to 5 words) and around 10 words. Interestingly, there is also a notable peak at around 40 words. Each party shows a significant number of tweets with approximately 40 words.  
  4. Politicians in Nowa Lewica use, on average almost one emoji per tweet, while those in PL2050 use only 0.3 emojis per tweet on average.  
  5. PO and PIS consistently attract the highest engagement, leading in average values across all public metrics. These two parties alternate in the top positions depending on the metric, while other parties show significantly lower levels of engagement.
  6. Donald Tusk from PO is the most influential politician in Poland, exhibiting the highest engagement in response to his posts among all politicians.
     
---

### 4. Sentiment Analysis
- **File:** [Sentiment Analysis.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/4.%20Sentiment%20analysis.ipynb)  
- **Description:** This notebook implements sentiment classification using a lexical approach using VADER with advanced natural language processing (NLP) models such as BERT.
- **Main conclusions:**
  1. The RoBERTa model is able to identify more sophisticated negative tweets compared to VADER. While VADER often predicts that the most negative tweets are single negative words, RoBERTa captures the nuanced meaning of the tweets. There is no need to filter out short tweets with only one word (e.g., "NO") as was necessary with VADER. These tweets are mostly critical of the actions of other parties, with the majority coming from PIS.

  2. The most positive tweets identified by the RoBERTa model are more meaningful than those from the VADER method. They are notably longer and often reflect the authors' happiness following certain events, such as meetings with voters or positive election results.

  3. According to the VADER analysis, the top 5 politicians with the highest 90th percentile of negative sentiment in their tweets include 2 individuals from NL and 1 from PO, PiS and Konfederacja. In contrast, the RoBERTa model also includes 2 politicians from NL and PO and 1 from PiS. This indicates a difference in predictions made by the VADER and BERT models. 

  4. The 90th percentile Positive Sentiment scores are quite comparable among the top 5 politicians. However, there are significant changes when comparing the VADER method to the RoBERTa model. Only one politician stay the same. This discrepancy may be due to the similarity of scores, which can lead to shifts in ranking, especially among closely positioned candidates.

  5. The party rankings for negative sentiment are similar between the VADER and RoBERTa methods.

  6. The party rankings for average positive sentiment are also very similar between VADER and RoBERTa. The top two parties remain the same in both methods, while the parties ranked third and fourth, as well as fifth and sixth, switch places. 

  7. RoBERTa model is harder to interpret because it does not assign pure weights to each word as VADER does. However, it can better detect the most positive and negative tweets, making them more meaningful and truly expressive of positive or negative emotions.



---

### 5. Clustering
- **File:** [Clustering.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/5.%20Clustering.ipynb)  
- **Description:** GMM and KNN clustering of tweets with embeddings using TF-IDF, Sentence Transformers and BERT CLS. Correlating 2 clusters with a binary variable indicating whether the party was ruling or no.
- **Main conclusions:** 
  1. The correlation values obtained were low.
  2. The highest value of correlation equal to 0.14 was obtained using a KNN model with Sentence Transformers values.


---

### 6. Network Analysis
- **File:** [Network.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/6.%20Network.ipynb)  
- **Description:** Conducts network analysis of politicians based on who mentioned whom in tweets. 
- **Main conclusions:**
  1. The highest number of mentions are attributed to political parties. However, notable accounts such as Donald Tusk, Szynon Hołownia, Mateusz Morawiecki and Grzegorz Braun are also frequently mentioned.

  2. Different political parties mention different people and organizations in their tweets. Some parties, like Konfederacja and PO, mostly mention their own members in the top 10 mentions. In contrast, other parties like NL, PIS, and PL2050 frequently mention people from other parties. Additionally, PSL includes organizations like Polish Railways, NATO, and European Public Health in their top 10 mentions.

  3. Urszula Pasławska frequently mentions herself in her tweets.

---

### 7. Hate Speech Detection
- **File:** [Hate Speech.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/7.%20Hate%20speech.ipynb)  
- **Description:** Identifies and filters hate speech using machine learning techniques. Includes fine-tuning of the model and data augmentation.  
- **Main conclusions:** 
  1. Models like Detoxify, which is based on BERT, enables to identify toxic tweets effectively.
  2. According to the model, Anita Kucharska-Dziedzic writes the most toxic tweets and consistently appears in the top position across all categories. The second most toxic individual is Borys Budka.
  3. The parties NL, Konfederacja, and PiS exhibit a high 90th percentile of toxicity in their tweets, while PO, PL2050, and PSL show much lower toxicity scores.
  4. For severe toxicity, there is not a significant gap between political parties, indicating that severe toxicity may be less prevalent in the tweets of Polish politicians.
  5. Copilot refused to work on this file due to the presence of abusive words in politicians' tweets.

---

### 8. Machine Learning Model
- **File:** [ML_model.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/8.%20ML_model.ipynb) 
- **Description:** Development of SVM and logistic regression models basing on the word embeddings calculated for the clustering models. Includes hyperparameter tuning and cross validation. The independent variable was binary - being a ruling party or not.
- **Main conclusions:** 
  1. Logistic Regression shows consistent performance across different embedding methods, with minimal variation. The TF-IDF embedding method achieved the best performance. However, its advantage in performance over the other methods is negligible. The main advantage of TF-IDF is the time required to generate embeddings, which is almost immediate compared to the other methods.
  2. SVM demonstrates better performance compared to Logistic Regression, particularly in handling the minority opposition class, which has fewer samples than the proposition class. This capability contributes to its overall better performance, achieving an F1-score of 0.81 and an AUC of 0.89.

---


### 9. Economic words
- **File:** [009. EconomicWords.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/009.%20EconomicWords.ipynb) 
- **Description:** Identification and flag tweets discussing economic topics from English-translated Polish Twitter data, based on a predefined list of 56 economic keywords and multi-word expressions (MWEs), while minimizing false positives.
- **Main conclusions:** 

---

## Setup 
The project uses **Python 3.12.10**.

## Usage 

```
git clone https://github.com/MPKuchciak/Twitter && cd Twitter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
