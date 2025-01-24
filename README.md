# Sentiment of Polish politicians in tweets
Authors:
- Łukasz Janisiów
- Maciej Kuchciak
- Mateusz Pliszka

## Project Overview:
The project analyzes the sentiment of Polish politicians' statements on the X platform (formerly Twitter) to examine their communication strategies on social media. The study includes all political parties represented in the Sejm of the 10th term, with more than 10 members in the Sejm. The research methodology combines a lexical approach using VADER with advanced natural language processing (NLP) models such as BERT. The analysis includes Exploratory Data Analysis (EDA), Sentiment Analysis, Clustering, Network Analysis, and Hate Speech Detection. 
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
  1. The three most active users are Patryk Jaki (PIS), Bartłomiej Pejo (Konfederacja), and Włodzimierz Skalik (Konfederacja). Each of them published around 1000 tweets in one year, averaging more than 2.5 tweets per day.  
  2. The activity of politicians on X overlaps with important events in Poland. Peaks in activity are observed after the parliamentary election, the appointment of the Council of Ministers, and during the Euroelection and the flood in 2024.  
  3. The most common tweet lengths are very short (up to 5 words) and around 10 words. Interestingly, there is also a notable peak at around 40 words. Each party shows a significant number of tweets with approximately 40 words.  
  4. Politicians in Nowa Lewica use, on average, more than one emoji per tweet, while those in PL2050 use only 0.2 emojis per tweet on average.  
  5. PO has the most active observers, with the highest average for each public metric. The second party with the highest average number of retweets and replies is PIS. However, in terms of tweet likes and quotes, PL2050 ranks second.  
  6. Donald Tusk from PO is the most influential politician in Poland, exhibiting the highest engagement in response to his posts among all politicians.
     
---

### 4. Sentiment Analysis
- **File:** [Sentiment Analysis.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/4.%20Sentiment%20analysis.ipynb)  
- **Description:** This notebook implements sentiment classification usinga a lexical approach using VADER with advanced natural language processing (NLP) models such as BERT.
- **Main conclusions:**
  1. The RoBERTa model is able to identify more sophisticated negative tweets compared to VADER. While VADER often predicts that the most negative tweets are single negative words, RoBERTa captures the nuanced meaning of the tweets. There is no need to filter out short tweets with only one word (e.g., "NO") as was necessary with VADER. These tweets are mostly critical of the actions of other parties, with the majority coming from PIS.
  2. The most positive tweets identified by the RoBERTa model are more meaningful than those from the VADER method. They are notably longer and often reflect the authors' happiness following certain events, such as meetings with voters or positive election results.
  3. According to the VADER analysis, the top 5 politicians with the highest 90th percentile of negative sentiment in their tweets include 4 individuals from PIS and 1 from Konfederacja. In contrast, the RoBERTa model also includes 4 politicians from PIS but replaces one individual with a politician from PO. This indicates a difference in predictions made by the VADER and BERT models.
  4. The 90th percentile Positive Sentiment scores are quite comparable among the top 5 politicians. However, there are some changes when comparing the VADER method to the RoBERTa model. Three politicians remain the same, while the top two positions have changed, confirming a difference in predictions made by the two models.
  5. The party rankings for negative sentiment are quite similar between the VADER and RoBERTa methods. The top three positions remain unchanged, but the parties in the 4th and 5th positions have switched places.
  6. The RoBERTa model indicates that the PSL party has the highest average positive sentiment, while the VADER method ranks PL2050 in the top position. In both cases, PO is predicted to be in second place, and the last three positions remain consistent between the two models.
  7. RoBERTa model is harder to interpret because it does not assign pure weights to each word as VADER does. However, it can better detect the most positive and negative tweets, making them more meaningful and truly expressive of positive or negative emotions.



---

### 5. Clustering
- **File:** [Clustering.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/5.%20Clustering.ipynb)  
- **Description:** Lorem ipsum.
- **Main conclusions:** Lorem ipsum.


---

### 6. Network Analysis
- **File:** [Network.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/6.%20Network.ipynb)  
- **Description:** Conducts network analysis to identify relationships and connections within text data. Suitable for social media or graph-based NLP tasks.  
- **Main conclusions:**
  1. The highest number of mentions are attributed to political parties. However, notable accounts such as Donald Tusk, Grzegorz Braun, Ministerstwo Spraw Zagranicznych, Szymon Hołownia, and Władysław Kosiniak-Kamysz are also frequently mentioned.
  2. Different political parties mention different people and organizations in their tweets. Some parties, like Konfederacja and PO, mostly mention their own members in the top 10 mentions. In contrast, other parties like NL, PIS, and PL2050 frequently mention people from other parties. Additionally, PSL includes organizations like Polish Railways, NATO, and European Public Health in their top 10 mentions.
  3. Urszula Pasławska frequently mentions herself in her tweets.

---

### 7. Hate Speech Detection
- **File:** [Hate Speech.ipynb](https://github.com/MPKuchciak/Twitter/blob/main/7.%20Hate%20speech.ipynb)  
- **Description:** Identifies and filters hate speech using machine learning techniques. Includes fine-tuning of the model and data augmentation.  
- **Main conclusions:** 
  1. Models like Detoxify, which is based on BERT, enables to identify toxic tweets effectively.
  2. According to the model, Michał Wójcik writes the most toxic tweets and consistently appears in the top two positions across all categories. Robert Biedroń is the second most toxic individual, alternating between first and second place in each category.
  3. The parties PiS, Konfederacja, and NL exhibit a high 90th percentile of toxicity in their tweets, while PO, PL2050, and PSL show much lower toxicity scores.
  4. For severe toxicity, there is not a significant gap between political parties, indicating that severe toxicity may be less prevalent in the tweets of Polish politicians.
  5. Copilot refused to work on this file due to the presence of abusive words in politicians' tweets.

---

### 8. Machine Learning Model
- **File:** [ML_model.ipynb](ML_model.ipynb)  
- **Description:** Lorem ipsum.
- **Main conclusions:** Lorem ipsum.

---

## How to Use
1. Clone this repository to your local machine.
2. Install the necessary dependencies.
3. Navigate to the respective notebook for your desired task.
