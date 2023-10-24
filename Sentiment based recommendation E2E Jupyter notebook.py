#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
# 
# Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
# 
# With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
# 
# As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings.
# 
# In order to do this, you planned to build a sentiment-based product recommendation system, which includes the following tasks.
# 
# Data sourcing and sentiment analysis Building a recommendation system Improving the recommendations using the sentiment analysis model Deploying the end-to-end project with a user interface

# In[1]:


get_ipython().system('pip install imblearn --user')


# In[2]:


get_ipython().system('pip install wordcloud')


# In[3]:


#Importing General purpose libraries
import re
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
warnings.filterwarnings("ignore") 
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 300)
pd.set_option("display.precision", 2)


# In[4]:


# NLTK libraries
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


# In[5]:


#Modelling related imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


# ### Loading the Dataset

# In[6]:


import sklearn
print(sklearn.__version__)
print(np.__version__)
print(pd.__version__)
print(nltk.__version__)


# In[7]:


# importing dataset
df_reviews = pd.read_csv("sample30.csv")
df_reviews.head()


# ### Step 1: Extracting Text Meaning- Data Cleaning and Preprocessing: Exploratory Data Analysis 

# In[8]:


df_reviews.info()


# In[9]:


## Data has brand names and defined in categories name of user and further the reviews and sentiment


# In[10]:


#Lets call a function supplying dataframe to find the missing row count as we see there are NaNs on the dataset
def calcMissingRowCount(df):
    # summing up the missing values (column-wise) and displaying fraction of NaNs
    return df.isnull().sum()

calcMissingRowCount(df_reviews)


# In[11]:


#As seen from the results user sentiment has one missing value which coule be safely remove it.
#Also it should be noted that User sentiment values are 'Positive' and 'NEgative'. For our analysis we will encode it 


# In[12]:


# Removing the missing row of user_sentiment
df_reviews = df_reviews[~df_reviews.user_sentiment.isnull()]


# In[13]:


#converting user_sentiment categories to numerical 1 or 0 . 1 Mapped as Positive and 0 as 'Negative'
df_reviews['user_sentiment'] = df_reviews['user_sentiment'].map({'Positive':1,'Negative':0})


# In[14]:


#get the value count of user_sentiments column, Lets also find balance between the user sentiment,
#as to what proportion of it is positive and negative respectively
df_reviews["user_sentiment"].value_counts(normalize=True)


# In[15]:


#Moving further analyzing the same with the bar plot
sns.countplot(x='user_sentiment', data= df_reviews, palette="Set2")


# Data is skewed towards positive reviews.Class impbalance would be required to be performed on this dataset

# In[16]:


#Analyzing Reviews_rating column 


# In[17]:


df_reviews["reviews_rating"].describe()


# In[18]:


df_reviews["reviews_rating"].value_counts()


# In[19]:


#It takes 5 values from 1 to 5. we could also check how the data is distributed among these 5 ratings


# In[20]:


#Plotting the Reviews rating frequency for each of the rating
sns.countplot(x='reviews_rating', data= df_reviews, palette="Set2")


# In[21]:


#It would be interesting to see if positive user sentiment relates to higher review rating and vice versa


# In[22]:


df_reviews[df_reviews["user_sentiment"]==1]["reviews_rating"].describe()


# In[23]:


df_reviews[df_reviews["user_sentiment"]==0]["reviews_rating"].describe()


# In[24]:


#There seems to be discrepancy here, lets look at the data of ratings between 1 and 4 for both positive and negative sentiment
#for better understanding of the data


# In[25]:


df_reviews[(df_reviews["user_sentiment"]==1) & (df_reviews["reviews_rating"]<4)][["reviews_title","reviews_text", "reviews_rating"]]


# In[26]:


df_reviews[(df_reviews["user_sentiment"]==0) & (df_reviews["reviews_rating"]>=4)][["reviews_title","reviews_text", "reviews_rating"]]


# In[27]:


pd.crosstab(df_reviews["user_sentiment"], df_reviews["reviews_rating"], margins=True)


# As seen earlier its found that there are records that have higher user rating but user sentiment is negative and lower user rating but user sentiment is positive. We can correct these values by equating sentiment to 0 for user sentiment negative and rating less than 4. And for positive sentiment and Rating  greater than 4 update sentiment to 1 for the records in mismatch

# In[28]:


df_reviews.loc[(df_reviews["user_sentiment"]==1) & (df_reviews["reviews_rating"]<4), "user_sentiment"] = 0


# In[29]:


df_reviews.loc[(df_reviews["user_sentiment"]==0) & (df_reviews["reviews_rating"]>=4), "user_sentiment"] = 1


# In[30]:


pd.crosstab(df_reviews["user_sentiment"], df_reviews["reviews_rating"], margins=True)


# after the correction its found that there is clear distinction between the reviews_rating and user_sentiment cross reference data

# In[31]:


df_reviews["user_sentiment"].value_counts()


# In[32]:


# Lets further analyze brand relevant column and if there is connection between brand and positive sentiment. 
#Brand loyalty or brand effect in play


# In[33]:


df_reviews["brand"].value_counts()


# In[34]:


# Filter the top 10 brands among the positive sentiments
df_reviews[df_reviews['user_sentiment']==1].groupby('brand')['brand'].count().sort_values(ascending=False)[:10].plot(kind='bar',color='g')


# In[35]:


# Filter the top 10 brands among the negative sentiments
df_reviews[df_reviews['user_sentiment']==0].groupby('brand')['brand'].count().sort_values(ascending=False)[:10].plot(kind='bar', color='r')


# In[36]:


# Lets also filter products based on the IDs and find how product and sentiment are related


# In[37]:


def filter_products(productId, pos=1):
    review_count = df_reviews[(df_reviews.id==productId) & (df_reviews.user_sentiment==pos)]['brand'].count()
    return review_count


# In[38]:


#group the dataframe by product id and view the pos review / neg reviews count
df_custom =  df_reviews.groupby('id', as_index=False)['user_sentiment'].count()
df_custom["pos_review_count"] =  df_custom.id.apply(lambda id: filter_products(id, 1))
df_custom["neg_review_count"] =  df_custom.id.apply(lambda id: filter_products(id, 0))


# In[39]:


df_custom.head(10)


# In[40]:


#Its possible that the Product could have both positive and negative reviews we could analyze 
#what portion of reviews are positive for a particular product


# In[41]:


#sort the product by sentiment % - postive reviews / total number of reviews
df_custom['sentiment %'] = np.round((df_custom['pos_review_count']/df_custom['user_sentiment'])*100,2)
df_custom.sort_values(by='sentiment %', ascending=False)[:20]


# In[42]:


df_reviews["manufacturer"].value_counts()


# In[43]:


#Let's find out user data which are common or in other words have reviewed often.
df_reviews["reviews_username"].value_counts()[:10]


# In[44]:


#plot the customers by 'positive user sentiment'
df_reviews[df_reviews['user_sentiment']==1].groupby('reviews_username')['reviews_username'].count().sort_values(ascending=False)[:10].plot(kind='bar', color='g')


# In[45]:


#plot the customers by 'negative user sentiment'
df_reviews[df_reviews['user_sentiment']==0].groupby('reviews_username')['reviews_username'].count().sort_values(ascending=False)[:10].plot(kind='bar', color='r')


# In[46]:


#removing nan/null from username
df_reviews = df_reviews[~df_reviews.reviews_username.isnull()]


# In[47]:


#Let's combine the reviews_text and reviews_title for better analysis
df_reviews["reviews_title"] = df_reviews["reviews_title"].fillna('')
df_reviews["reviews_full_text"] = df_reviews[['reviews_title', 'reviews_text']].agg('. '.join, axis=1).str.lstrip('. ')


# In[48]:


#get the missing row count for all columns
calcMissingRowCount(df_reviews)


# We would find reviews_rating, reviews_text, user_sentiment,reviews_username and these dont have any null values. We could proceed with next steps

# ### Text Preprocessing for Modelling

# In[49]:


import string


# In[50]:


df_reviews[["reviews_full_text", "user_sentiment"]].sample(10)


# In[51]:


'''function to clean the text and remove all the unnecessary elements.'''
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = re.sub("\[\s*\w*\s*\]", "", text)
    dictionary = "abc".maketrans('', '', string.punctuation)
    text = text.translate(dictionary)
    text = re.sub("\S*\d\S*", "", text)
    
    return text


# In[52]:


df_clean = df_reviews[['id','name', 'reviews_full_text', 'user_sentiment']]


# In[53]:


df_clean["reviews_text"] = df_clean.reviews_full_text.apply(lambda x: clean_text(x))


# In[54]:


# Function  to map NTLK Position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[55]:


stop_words = set(stopwords.words('english'))

def remove_stopword(text):
    words = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return " ".join(words)


# In[56]:


lemmatizer = WordNetLemmatizer()
# Lemmatize the sentence
def lemma_text(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(remove_stopword(text))) # Get position tags
    # Map the position tag and lemmatize the word/token
    words =[lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
    return " ".join(words)


# In[57]:


df_clean["reviews_text_cleaned"] = df_clean.reviews_text.apply(lambda x: lemma_text(x))


# In[58]:


df_clean.head()


# Removing stopwords, punctuations,numericals,whitespaces and lemma to clean the text corpus

# In[59]:


get_ipython().system('pip install --upgrade pip ')

get_ipython().system('pip install --upgrade Pillow')


# In[60]:


#visualise the data according to the 'Review Text' character length
plt.figure(figsize=(10,6))
reviews_lens = [len(d) for d in df_clean.reviews_text_cleaned]
plt.hist(reviews_lens, bins = 50)


# In[61]:


def getMostCommonWords(reviews, n_most_common):
    # flatten review column into a list of words, and set each to lowercase
    flattened_reviews = [word for review in reviews for word in                          review.lower().split()]


    # remove punctuation from reviews
    flattened_reviews = [''.join(char for char in review if                                  char not in string.punctuation) for                          review in flattened_reviews]


    # remove any empty strings that were created by this process
    flattened_reviews = [review for review in flattened_reviews if review]

    return Counter(flattened_reviews).most_common(n_most_common)


# In[62]:


pos_reviews = df_clean[df_clean['user_sentiment']==1]
getMostCommonWords(pos_reviews['reviews_text_cleaned'],10)


# In[63]:


# We find for positive sentiments words like great, love, use, product, promotion and clean tombe having higher frequency, 
#lets further analyze if there is a distinction between words appearing in greater frequency for negative sentiment


# In[64]:


neg_reviews = df_clean[df_clean['user_sentiment']==0]
getMostCommonWords(neg_reviews['reviews_text_cleaned'],10)


# In[65]:


#On first glance it seems like there are common words like product, use movie common between positive and negative sentiment.
#These words alone cant be mindicator of positive or negative sentiment. While analyzing other words like hair, formula
#new, old and gel its some of the words indicative of the product being new or old, its formula and usage say hair.
#Lets further analyze to understand better how these words depict the senyiments.


# In[66]:


#Lets look deepeer with ngram frequency of words, as we already saw words like product are part of both sentiments, 
#next possible step would be to analyze combination of words
def get_top_n_ngram( corpus, n_gram_range ,n=None):
    vec = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    #print(bag_of_words)
    sum_words = bag_of_words.sum(axis=0) 
    print("--1",sum_words)
    for word, idx in vec.vocabulary_.items():
        #print(word)
        #print(idx)
        break
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    #print("-31",words_freq)
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[67]:


#Lets begin  with analyzing bigram words and list out top 10 of it
common_words = get_top_n_ngram(pos_reviews['reviews_text_cleaned'], 2, 10)
pd.DataFrame(common_words)


# In[68]:


#Bigrams are giving a better picture of the reviews, here its seen that great movie, good product are capturing user feedbacks
#can be linked to sentiments


# In[69]:


#Print the top 10 words in the bigram frequency
common_words = get_top_n_ngram(neg_reviews['reviews_text_cleaned'], 2, 10)
pd.DataFrame(common_words)


# In[70]:


#Lets further analyze the trigrams, if it could reveal underlying reasoning behind the sentiments 
common_words = get_top_n_ngram(df_clean.reviews_text_cleaned, 3, 10)
df3 = pd.DataFrame(common_words, columns = ['trigram' , 'count'])
plt.figure(figsize=[35,25])
fig = sns.barplot(x=df3['trigram'], y=df3['count'])


# In[71]:


X = df_clean['reviews_text_cleaned']
y = df_clean['user_sentiment']


# In[72]:


# At this point we have learned the text corpus for underlying emotions that triggered the senyiment
# its time to identify the features


#    ### Step 2: Feature Extraction - Raw into features or create new features

# In[73]:


no_of_classes= len(pd.Series(y).value_counts())


# In[74]:


#Lets further see how the data is divided in terms of positive and negative sentiments.
for i in range(0,no_of_classes):
    print("Percent of {0}s: ".format(i), round(100*pd.Series(y).value_counts()[i]/pd.Series(y).value_counts().sum(),2), "%")


# There is a class imbalance as Positive sentiments are around 90% and negative sentiment of around 10%
# We would proceed with SMOTE Oversampling before modeling.
# In order to vectorize the data start with extracting  features from the textual data using TF-IDF vectorizer method 

# In[75]:


#using TF-IDF vectorizer using the parameters to get 650 features.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=650, max_df=0.9, min_df=7, binary=True, 
                                   ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(df_clean['reviews_text_cleaned'])

y= df_clean['user_sentiment']


# In[76]:


print(tfidf_vectorizer.get_feature_names_out())


# In[77]:


# splitting into test and train
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, random_state=42, test_size=0.25)


# In[78]:


### Handling Class imbalance (using SMOTE)


# In[79]:


counter = Counter(y_train)
print('Before',counter)

sm = SMOTE()

# transform the dataset
X_train, y_train = sm.fit_resample(X_train, y_train)

counter = Counter(y_train)
print('After',counter)


# ### Step 3: Model Building

# In[80]:


import time
from sklearn import metrics
import pickle


# In[81]:


class ModelBuilder:
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
        
    def train_model(self):
        self.model.fit(self.x_train,self.y_train)
        return self.model.predict(self.x_test)
    
    def evaluate_model(self, y_pred_class):
        print("\n")
        print("*"*30)
        self.result_metrics = self.evaluate_metrics(y_pred_class)
        print("*"*30)
        print("\n")
        
        self.classification_report(y_pred_class)
        print("*"*30)
        print("\n")
        self.confusion_matrix(y_pred_class)
            
        print("*"*30)
        print("\n")
        
        metrics.plot_roc_curve(self.model, self.x_test, self.y_test)
        
        return self.result_metrics
        
    def evaluate_metrics(self, y_pred_class):
        result_metrics = [] 
        accuracy = metrics.accuracy_score(self.y_test, y_pred_class)
        precision = metrics.precision_score(self.y_test, y_pred_class)
        recall = metrics.recall_score(self.y_test, y_pred_class)
        f1score = metrics.f1_score(self.y_test, y_pred_class)
        y_pred_prob = self.model.predict_proba(self.x_test)[:,1]
        roc_auc = metrics.roc_auc_score(self.y_test, y_pred_prob)
        
        print(f"Accuracy is : {accuracy*100:.1f}%")
        print(f"Precision is : {precision*100:.1f}%")
        print(f"Recall is : {recall*100:.1f}%")
        print(f"F1 Score is : {f1score*100:.1f}%")
        print(f"Roc-Auc Score is:{roc_auc*100:.1f}%")
        
        result_metrics.append(accuracy)
        result_metrics.append(precision)
        result_metrics.append(recall)
        result_metrics.append(f1score)
        result_metrics.append(roc_auc)
        return result_metrics
        
    def confusion_matrix(self, y_pred_class):
        confusion_matrix = metrics.confusion_matrix(self.y_test, y_pred_class)
        self.plot_confusion_matrix(confusion_matrix,[0,1])
        
        
    def plot_confusion_matrix(self, data, labels):
        sns.set(color_codes=True)
        plt.title("Confusion Matrix")
        ax = sns.heatmap(data/np.sum(data), annot=True, cmap="Blues", fmt=".2%")
 
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
 
        ax.set(ylabel="True Values", xlabel="Predicted Values")
        plt.show()
        
    def classification_report(self, y_pred_class):
        print(metrics.classification_report(self.y_test, y_pred_class))
               


# ### Logistic Regression

# In[82]:


# To begin woth Logistic Regression model.
get_ipython().run_line_magic('time', '')
logreg_ci = LogisticRegression(random_state=42, max_iter=100,solver='liblinear', class_weight="balanced")
lr_ci_modebuilder = ModelBuilder(logreg_ci, X_train, X_test, y_train, y_test)


# In[83]:


# Lets Train and Predict the Test Labels
y_pred_class  = lr_ci_modebuilder.train_model()
lr_metrics = lr_ci_modebuilder.evaluate_model(y_pred_class)


# F1 Score looks reasonable, but for class(0) is lesser. We will also try other models to increase overall F1 Score and indivdual F1 Score

# ### Naive Bayes 

# In[84]:


# To Train NB model importing apprpritate libraries
from sklearn.naive_bayes import MultinomialNB


# In[85]:


mnb = MultinomialNB(alpha=1.0)
mnb_modebuilder = ModelBuilder(mnb, X_train, X_test, y_train, y_test)


# In[86]:


# Train and Predict the Test Labels
y_pred_class  = mnb_modebuilder.train_model()
nb_metrics = mnb_modebuilder.evaluate_model(y_pred_class)


# In[87]:


#There is no signoficant improvement in F1 Score for class o while overall F1 Score is better than Logistic regression


# ### Decision Tree

# In[88]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[89]:


dt = DecisionTreeClassifier(random_state=42, criterion="gini", max_depth=10)


# In[90]:


dt_modelbuilder = ModelBuilder(dt, X_train, X_test, y_train, y_test)


# In[91]:


y_pred_class  = dt_modelbuilder.train_model()
dt_metrics_cv = dt_modelbuilder.evaluate_model(y_pred_class)


# In[92]:


# We will analyze the test results comparing all of the results side by side on the table once we have results for
#all model


# ### Random Forrest

# In[93]:


rf = RandomForestClassifier(oob_score=True, random_state=42, criterion="gini")


# In[94]:


params = {
    'max_depth': [2,3,5,10],
    'min_samples_leaf': [5,10,20,50],
    'n_estimators': [10, 25, 50, 100]
}


# In[95]:


grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="f1")


# In[96]:


get_ipython().run_line_magic('time', '')
grid_search.fit(X_train, y_train)


# In[97]:


rf_best = grid_search.best_estimator_
rf_modebuilder = ModelBuilder(rf_best, X_train, X_test, y_train, y_test)


# In[98]:


# Train and Predict the Test Labels
y_pred_class  = rf_modebuilder.train_model()
rf_metrics = rf_modebuilder.evaluate_model(y_pred_class)


# ### XGBoost Classifier

# In[99]:


get_ipython().system('pip install xgboost')


# In[100]:


import xgboost as xgb


# In[101]:


xgclf = xgb.XGBClassifier(learning_rate=0.15, max_depth=10, random_state=42) #based on the tuned parameters
xg_modebuilder = ModelBuilder(xgclf, X_train, X_test, y_train, y_test)


# In[102]:


# Train and Predict the Test Labels
y_pred_class  = xg_modebuilder.train_model()
xg_metrics = xg_modebuilder.evaluate_model(y_pred_class)


# ### Model Inference- 

# In[103]:


#Lets analyse all results side by side to compare the model performance


# In[104]:


xg_metrics


# In[105]:


# Creating a table which contain all the metrics

metrics_table = {'Metric': ['Accuracy','Precision','Recall',
                       'F1Score','Auc Score'], 
        'Logistic Regression': lr_metrics,
        'Naive Bayes': nb_metrics,
        'Decision Tree': dt_metrics_cv,
         'Random Forrest': rf_metrics,
        'XG Boost': xg_metrics
        }

df_metrics = pd.DataFrame(metrics_table ,columns = ['Metric', 'Logistic Regression', 'Naive Bayes','Decision Tree','Random Forrest',
                                                    'XG Boost'] )

df_metrics


# On first glance its evident that XG Boost outperforms all other models in all test summary results, we would be saving XG Boost as a pickle file for further use

# #### Saving the model as a pickle file

# In[201]:


def save_object(obj, filename):
    filename = "Capstone"+filename+'.pkl'
    pickle.dump(obj, open(filename, 'wb'))


# In[202]:


save_object(xgclf, 'sentiment-classifier-xg-boost-model')


# In[203]:


save_object(tfidf_vectorizer, 'tfidf-vectorizer')


# In[204]:


save_object(df_clean, 'cleaned-data')


# ## Recommendation System

# There are different techniques to develop Recommendation System -
# 
# To begin with 2 of the Colloboarative filtering techniques:
#     - User-User Based Approach
#     - Item-Item Based Approach

# In[110]:


df_reviews.info()


# In[111]:


df_recommendation = df_reviews[["id", "name", "reviews_rating", "reviews_username"]]
calcMissingRowCount(df_recommendation)


# In[112]:


#splitting the train and test
train, test = train_test_split(df_recommendation, test_size=0.25, random_state=42)


# In[113]:


print(train.shape)
print(test.shape)


# In[114]:


product_column = "id"
user_column = "reviews_username"
value_column = "reviews_rating"


# In[115]:


#Fit the train ratings' dataset into matrix format  where columns represents product names and the rows represents user names.
df_pivot = pd.pivot_table(train,index=user_column, columns = product_column, values = value_column).fillna(0)
df_pivot.head(10)


# ### Preparing dummy train and test

# In[116]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()


# In[117]:


dummy_train.head()


# In[118]:


# Lets mark the data for which user has not provided reviews and tag it as 1. 
dummy_train[value_column] = dummy_train[value_column].apply(lambda x: 0 if x>=1 else 1)


# In[119]:


# Fit train ratings' dataset into matrix  where columns represents product names and the rows represents user names.
dummy_train = pd.pivot_table(dummy_train,index=user_column, columns = product_column, values = value_column).fillna(1)
dummy_train.head(10)


# ### Further lets exploit User similarity matrix

# In[120]:


# We begin with comparing the user similarities, 
#that is a metric to define how similar 2 users are and how likely they could review similariliy


# In[121]:


df_pivot.index.nunique()


# In[122]:


from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity


# In[123]:


#using cosine_similarity function lets come up with distance between 2 users.
user_correlation = cosine_similarity(df_pivot)
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)
print(user_correlation.shape)


# ### Prediction User-User

# In[124]:


#Lets look at users who are negatively correlated
user_correlation[user_correlation<0]=0
user_correlation


# In[125]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[126]:


#To extract products that are not rated by the user, we multiply with dummy train to make it zero
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# ### Coming up with 20 recommendation for the user

# In[187]:


user_input = "00sab00" 
print(user_input)


# In[188]:


recommendations = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
recommendations


# In[129]:


#Lets further have a look at top 20 similarity_score and corresponding product id and name
final_recommendations = pd.DataFrame({'product_id': recommendations.index, 'similarity_score' : recommendations})
final_recommendations.reset_index(drop=True)
pd.merge(final_recommendations, train, on="id")[["id", "name", "similarity_score"]].drop_duplicates()


# ### Evaluation User-User

# In[130]:


# If there are common user between test and train data
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape


# In[131]:


common.head()


# In[132]:


# convert into the user-movie matrix.
common_user_based_matrix = pd.pivot_table(common,index=user_column, columns = product_column, values = value_column)
common_user_based_matrix.head()


# In[133]:


# Convert into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df.head()


# In[134]:


user_correlation_df[user_column] = df_pivot.index
user_correlation_df.set_index(user_column,inplace=True)
user_correlation_df.head()


# In[135]:


list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_pivot.index.tolist()
user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[136]:


user_correlation_df_1.shape


# In[137]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[138]:


user_correlation_df_3 = user_correlation_df_2.T


# In[139]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[140]:


dummy_test = common.copy()

dummy_test[value_column] = dummy_test[value_column].apply(lambda x: 1 if x>=1 else 0)
dummy_test = pd.pivot_table(dummy_test,index=user_column, columns = product_column, values = value_column).fillna(0)


# In[141]:


dummy_test.shape


# In[142]:


common_user_based_matrix.head()


# In[143]:


dummy_test.head()


# In[144]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[145]:


common_user_predicted_ratings.head()


# In[146]:


#calculate RMSE score

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[147]:


common_ = pd.pivot_table(common,index=user_column, columns = product_column, values = value_column)


# In[148]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[149]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# ### Recommendation based on Item  

# In[150]:


df_pivot = pd.pivot_table(train,
    index=product_column,
    columns=user_column,
    values=value_column
)

df_pivot.head()


# In[151]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[152]:


df_subtracted.head()


# In[153]:


# Analzing the similarity between items using Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[154]:


item_correlation[item_correlation<0]=0
item_correlation


# ### Prediction - item-item

# In[155]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# #### Again we will look at the items for which user has not given any rating by multiplying with Dummy Train data

# In[156]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# #### Predicting top 20 recommendation for the user

# In[189]:


# Take the user ID as input
user_input = '00sab00'
print(user_input)


# In[190]:


# Recommending the Top 5 products to the user.
item_recommendations = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
item_recommendations


# In[159]:


item_final_recommendations = pd.DataFrame({'product_id': item_recommendations.index, 'similarity_score' : item_recommendations})
item_final_recommendations.reset_index(drop=True)
#final_recommendations.drop(['id'], axis=1)
pd.merge(item_final_recommendations, train, on="id")[["id", "name", "similarity_score"]].drop_duplicates()


# #### Evaluation - item-item

# In[160]:


common =  test[test.id.isin(train.id)]
common.shape


# In[161]:


common.head(4)


# In[162]:


common_item_based_matrix = common.pivot_table(index=product_column, columns=user_column, values=value_column)


# In[163]:


item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df.head(1)


# In[164]:


item_correlation_df[product_column] = df_subtracted.index
item_correlation_df.set_index(product_column,inplace=True)
item_correlation_df.head()


# In[165]:


list_name = common.id.tolist()


# In[166]:


item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[167]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T


# In[168]:


df_subtracted


# In[169]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[170]:


dummy_test = common.copy()
dummy_test[value_column] = dummy_test[value_column].apply(lambda x: 1 if x>=1 else 0)
dummy_test = pd.pivot_table(dummy_test, index=product_column, columns=user_column, values=value_column).fillna(0)
common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# In[171]:


common_ = pd.pivot_table(common,index=product_column, columns=user_column, values=value_column)


# In[172]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[173]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[174]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# User based recommendation model has a lower RMSE value (~2) compared to Item based recommender.

# In[205]:


# saving the correlation matrix of user based recommender 
save_object(user_final_rating, "user_final_rating")


# ### Top 20 products recommendation applying filtering using sentiment model

# In[176]:


def get_sentiment_recommendations(user):
    if (user in user_final_rating.index):
        # get the product recommedation using the trained ML model
        recommendations = list(user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
        temp = df_clean[df_clean.id.isin(recommendations)]
        #temp["reviews_text_cleaned"] = temp["reviews_text"].apply(lambda x: self.preprocess_text(x))
        #transfor the input data using saved tf-idf vectorizer
        X =  tfidf_vectorizer.transform(temp["reviews_text_cleaned"].values.astype(str))
        temp["predicted_sentiment"]= xgclf.predict(X)
        temp = temp[['name','predicted_sentiment']]
        temp_grouped = temp.groupby('name', as_index=False).count()
        temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(temp.name==x) & (temp.predicted_sentiment==1)]["predicted_sentiment"].count())
        temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
        temp_grouped['pos_sentiment_percent'] = np.round(temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100,2)
        return temp_grouped.sort_values('pos_sentiment_percent', ascending=False)
    else:
        print(f"User name {user} doesn't exist")


# In[191]:


#Lets test the above function for the user in list.
get_sentiment_recommendations("kayc")


# In[192]:


#Also have a look at top 5
get_sentiment_recommendations("kayc")[:5]


# In[182]:


#testing the above fuction on the user that doesn't exists or a new user
get_sentiment_recommendations("Abhi")


# In[195]:


X_sample = tfidf_vectorizer.transform(["Awesome product, go for it"])
y_pred_sample = xgclf.predict(X_sample)
y_pred_sample


# In[196]:


X_sample = tfidf_vectorizer.transform(["worst product, wouldnt recommend"])
y_pred_sample = xgclf.predict(X_sample)
y_pred_sample


# In[ ]:





# In[ ]:




