import pandas as pd
import numpy as np
import re, emoji, os, random
import nltk as nltk
#nltk.download('punkt')
from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

stop_words = ['akin','aking','ako','alin','am','amin','aming','ang','ano','anumang','apat','at','atin','ating','ay','bababa','bago','bakit','bawat','bilang','dahil',
             'dalawa','dapat','din','dito','doon','gagawin','gayunman','ginagawa','ginawa','ginawang','gumawa','gusto','habang','hanggang','hindi','huwag','iba','ibaba',
             'ibabaw','ibig','ikaw','ilagay','ilalim','ilan','inyong','isa','isang','itaas','ito','iyo','iyon','iyong','ka','kahit','kailangan','kailanman','kami','kanila',
             'kanilang','kanino','kanya','kanyang','kapag','kapwa','karamihan','katiyakan','katulad','kaya','kaysa','ko','kong','kulang','kumuha','kung','laban','lahat','lamang',
             'likod','lima','maaari','maaaring','maging','mahusay','makita','marami','marapat','masyado','may','mayroon','mga','minsan','mismo','mula','muli','na','nabanggit','naging',
             'nagkaroon','nais','nakita','namin','napaka','narito','nasaan','ng','ngayon','ni','nila','nilang','nito','niya','niyang','noon','o','pa','paano','pababa','paggawa','pagitan',
             'pagkakaroon','pagkatapos','palabas','pamamagitan','panahon','pangalawa','para','paraan','pareho','pataas','pero','pumunta','pumupunta','sa','saan','sabi','sabihin','sarili','sila','sino','siya','tatlo','tayo','tulad','tungkol','una','walang']

dataset = pd.read_excel('../emotiondetection/datasets/mergedset.xlsx')
"""
# Stanford POS Tagger
java_path = "C:\\jdk-16\\bin"
os.environ['JAVAHOME'] = java_path
# Stanford dir
stanford_dir = "C:\\Users\\SLY\Documents\stanford-tagger-4.2.0\\stanford-postagger-full-2020-11-17"

modelfile = stanford_dir + "\\models\\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"
jarfile = stanford_dir+"\\stanford-postagger.jar"

pos_tagger = StanfordPOSTagger(modelfile,jarfile,java_options="-Xmx4G")
"""

#---------------------------------------------------------------------------------------------  utility functions
# Remove articles (a, an, and, the)
# Remove any other characters other than alphabetical characters
# Remove white spaces
def cleaner(text):
    text = re.sub('[^a-zA-Z]', '', str(text))
    text = re.sub(' +', ' ', str(text))

    cleaned_text = text.strip()
    return cleaned_text

def word_count_per_doc(text):
	tokenized = word_tokenize(cleaner(text))
	return len(tokenized)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNC_RATIO = 0.3

# Data Augmentation Technique
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

#---------------------------------------------------------------------------------------------  Cleaning process

# Remove duplicates
dataset = dataset.drop_duplicates(subset=['COMMENTS'])

# Remove rows with NULL value
dataset = dataset.dropna().reset_index(drop=True)

# Convert all text to lower case
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

# Re label misspelled and similar labels
dataset["MAJORITY"].replace({"sad": "sadness", "0": "none", "digust": "disgust"}, inplace=True)

dataset = dataset.rename(columns={"COMMENTS": "comments", "MAJORITY": "label"})

# show certain lables
#print(mergedset.loc[mergedset['MAJORITY'] == "0"])

#print(dataset['label'].value_counts())


# Get number total number of rows per class
class_none, class_sadness, class_anger, class_joy, class_fear, class_surprise, class_disgust = dataset.label.value_counts()

dataset_none = dataset.loc[dataset['label'] == "none"]
dataset_sadness = dataset.loc[dataset['label'] == "sadness"]
dataset_anger = dataset.loc[dataset['label'] == "anger"]
dataset_joy = dataset.loc[dataset['label'] == "joy"]
dataset_fear = dataset.loc[dataset['label'] == "fear"]
dataset_surprise = dataset.loc[dataset['label'] == "surprise"]
dataset_disgust = dataset.loc[dataset['label'] == "disgust"]

sampleCount = class_none;

"""
dataset_none_under = dataset_none.sample(sampleCount)
dataset_sadness_under = dataset_sadness.sample(sampleCount)
dataset_anger_under = dataset_anger.sample(sampleCount)
dataset_joy_under = dataset_joy.sample(sampleCount)
dataset_fear_under = dataset_fear.sample(sampleCount)
dataset_surprise_under = dataset_surprise.sample(sampleCount)

balanced_dataset = pd.concat([dataset_disgust, dataset_none_under, dataset_sadness_under, dataset_anger_under, dataset_joy_under, dataset_fear_under, dataset_surprise_under], ignore_index=True)
"""
dataset_disgust_over = dataset_disgust.sample(sampleCount, replace=True)
dataset_sadness_over = dataset_sadness.sample(sampleCount, replace=True)
dataset_anger_over = dataset_anger.sample(sampleCount, replace=True)
dataset_joy_over = dataset_joy.sample(sampleCount, replace=True)
dataset_fear_over = dataset_fear.sample(sampleCount, replace=True)
dataset_surprise_over = dataset_surprise.sample(sampleCount, replace=True)

balanced_dataset = pd.concat([dataset_none, dataset_disgust_over, dataset_sadness_over, dataset_anger_over, dataset_joy_over, dataset_fear_over, dataset_surprise_over], ignore_index=True)

#print(balanced_dataset['label'].value_counts())

balanced_dataset = shuffle(balanced_dataset)
dataset_X = balanced_dataset[['comments']]
dataset_X = dataset_X['comments'].apply(insert_punctuation_marks).to_frame()
dataset_y = balanced_dataset['label']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.20, random_state=1)

# Test and validation split
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=1)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

print(y_train.shape)
print(y_test.shape)
print(y_val.shape)


#---------------------------------------------------------------------------------------------  Traditional features

def check_angry_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["🖕", "💩", "😤", "😡", "😠", "🤬"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def check_sad_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["😟", "🙁	", "☹", "😡", "😰", "😥", "😢", "😭", "😓", "💔"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def check_joy_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["😀", "😃", "😄", "😁", "😆", "🤣", "😂", "🙂", "😊", "😇", "🥰", "😍", "🤗", "❤"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def check_disgust_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["🤢", "🤮"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def check_fear_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["😨", "😰", "😱"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def check_surprise_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["😮", "😯", "😲", "😳"]
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def wordFrequency(sentences):
	sentences = list(sentences)
	sentences = [word_tokenize(sentence) for sentence in sentences]
	for i in range(len(sentences)):
			sentences[i] = [word for word in sentences[i] if word not in stop_words]
	return sentences

def vowel_count(text):
	syllable_counts = 0
	for char in text:
		if char == 'a' or char == 'e' or char == 'i' or char == 'o' or char == 'u' or char == 'A' or char == 'E' or char == 'I' or char == 'O' or char == 'U':
			syllable_counts += 1
	return syllable_counts


def consonant_count(article):
    article = article.lower()
    total_consonant = 0

    for i in article:
        if i == 'b' or i == 'c' or i == 'd' or i == 'f' or i == 'g' \
                or i == 'h' or i == 'j' or i == 'k' or i == 'l' \
                or i == 'm' or i == 'n' or i == 'p' or i == 'q' \
                or i == 'r' or i == 's' or i == 't' or i == 'v' \
                or i == 'w' or i == 'x' or i == 'y' or i == 'z':
            total_consonant = total_consonant + 1;

    return total_consonant

#---------------------------------------------------------------------------------------------  Ortography Features
def get_consonant_cluster(text):
    cleaned = cleaner(text)
    word_count = word_count_per_doc(text)

    pattern = "([bcdfghjklmnpqrstvwxyz]{1}[bcdfghjklmnpqrstvwxyz]{1}[bcdfghjklmnpqrstvwxyz]*)"
    matches = len(re.findall(pattern, cleaned))

    result = 0;
    if word_count > 0:
        matches / word_count

    return result
#---------------------------------------------------------------------------------------------  Morphological Features
def aux_verb_ratio(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    word_count = word_count_per_doc(text)

    verb_counter = 0
    aux_verbs = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                #print(pos)
                if pos[:2] == 'VB':
                    verb_counter += 1
                if pos == 'VBS':
                    aux_verbs += 1

    if word_count == 0:
        return 0

    return (aux_verbs/word_count)

def lexical_density(text):
    splitted = re.split('[?.]+', text)
    splitted = [i for i in splitted if i]   #removes empty strings in list

    lexical_item_counter = 0
    for i in splitted:
        i = i.strip()
        tagged_text = pos_tagger.tag(word_tokenize(i))
        for x in tagged_text:
            if '|' not in x[0]:
                pos = x[1].split('|')[1]
                if pos[:2] == 'VB' or pos[:2] == 'NN' or pos[:2] == 'JJ' or pos[:2] == 'RB':
                    lexical_item_counter += 1

    word_count = word_count_per_doc(text)
    print("Word Count:",word_count)
    if word_count == 0:
        return 0
    return (lexical_item_counter/word_count_per_doc(text))

#---------------------------------------------------------------------------------------------  Extract Features
vectorizer = CountVectorizer()
vectorizer.fit_transform(X_train['comments'])

# Feature 1 - Word Frequency
X_f1 = X_train['comments'].apply(wordFrequency)
X_f1 = vectorizer.transform(X_train['comments'])
X_f1 = pd.DataFrame(X_f1.toarray())

# Feature 2 - Emojis(Sad)
X_f2 = X_train['comments'].apply(check_sad_emojis)

# Feature 3 - Emojis(Angry)
X_f3 = X_train['comments'].apply(check_angry_emojis)

# Feature 4 - Emojis(Joy)
X_f4 = X_train['comments'].apply(check_joy_emojis)

# Feature 5 - Emojis(Disgust)
X_f5 = X_train['comments'].apply(check_disgust_emojis)

# Feature 6 - Emojis(Fear)
X_f6 = X_train['comments'].apply(check_fear_emojis)

# Feature 7 - Emojis(Surprise)
X_f7 = X_train['comments'].apply(check_surprise_emojis)

# Feature 8 - Vowel Count
X_f8 = X_train['comments'].apply(vowel_count)

# Feature 9 - Consonant Count
X_f9 = X_train['comments'].apply(consonant_count)

# Feature 10 - Consonant Cluster
X_f10 = X_train['comments'].apply(get_consonant_cluster)

# Concatenate all features
collected_features_train = pd.concat([X_f1, X_f2, X_f3, X_f4, X_f5, X_f6, X_f7, X_f8, X_f9, X_f10], axis=1)
collected_features_train = collected_features_train.to_numpy();

# Feature 1 - Word Frequency
X_f1 = X_val['comments'].apply(wordFrequency)
X_f1 = vectorizer.transform(X_val['comments'])
X_f1 = pd.DataFrame(X_f1.toarray())

# Feature 2 - Emojis(Sad)
X_f2 = X_val['comments'].apply(check_sad_emojis)

# Feature 3 - Emojis(Angry)
X_f3 = X_val['comments'].apply(check_angry_emojis)

# Feature 4 - Emojis(Joy)
X_f4 = X_val['comments'].apply(check_joy_emojis)

# Feature 5 - Emojis(Disgust)
X_f5 = X_val['comments'].apply(check_disgust_emojis)

# Feature 6 - Emojis(Fear)
X_f6 = X_val['comments'].apply(check_fear_emojis)

# Feature 7 - Emojis(Surprise)
X_f7 = X_val['comments'].apply(check_surprise_emojis)

# Feature 8 - Vowel Count
X_f8 = X_val['comments'].apply(vowel_count)

# Feature 9 - Consonant Count
X_f9 = X_val['comments'].apply(consonant_count)

# Feature 10 - Consonant Cluster
X_f10 = X_val['comments'].apply(get_consonant_cluster)

# Concatenate all features
collected_features_val = pd.concat([X_f1, X_f2, X_f3, X_f4, X_f5, X_f6, X_f7, X_f8, X_f9, X_f10], axis=1)
collected_features_val = collected_features_val.to_numpy();


#---------------------------------------------------------------------------------------------  KNN
# K Nearest Neighbor
y_train = y_train.to_numpy();
y_train = np.squeeze(y_train)

knn_clf = KNeighborsClassifier(n_neighbors = 7)
knn_clf.fit(collected_features_train,y_train)

y_pred = knn_clf.predict(collected_features_val)

print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

#---------------------------------------------------------------------------------------------  MNB
# Multinomial Naive Bayes
mnb_clf = MultinomialNB(alpha=1.0)
mnb_clf.fit(collected_features_train, y_train)

y_pred = mnb_clf.predict(collected_features_val)

print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

#---------------------------------------------------------------------------------------------  Decision Tree
# Decision Trees
dt_clf = DecisionTreeClassifier()
dt_clf.fit(collected_features_train, y_train)

y_pred = dt_clf.predict(collected_features_val)

print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

#---------------------------------------------------------------------------------------------  Test

# Feature 1 - Word Frequency
X_f1 = X_test['comments'].apply(wordFrequency)
X_f1 = vectorizer.transform(X_test['comments'])
X_f1 = pd.DataFrame(X_f1.toarray())

# Feature 2 - Emojis(Sad)
X_f2 = X_test['comments'].apply(check_sad_emojis)

# Feature 3 - Emojis(Angry)
X_f3 = X_test['comments'].apply(check_angry_emojis)

# Feature 4 - Emojis(Joy)
X_f4 = X_test['comments'].apply(check_joy_emojis)

# Feature 5 - Emojis(Disgust)
X_f5 = X_test['comments'].apply(check_disgust_emojis)

# Feature 6 - Emojis(Fear)
X_f6 = X_test['comments'].apply(check_fear_emojis)

# Feature 7 - Emojis(Surprise)
X_f7 = X_test['comments'].apply(check_surprise_emojis)

# Feature 8 - Vowel Count
X_f8 = X_test['comments'].apply(vowel_count)

# Feature 9 - Consonant Count
X_f9 = X_test['comments'].apply(consonant_count)

# Feature 10 - Consonant Cluster
X_f10 = X_test['comments'].apply(get_consonant_cluster)

# Concatenate all features
collected_features_test = pd.concat([X_f1, X_f2, X_f3, X_f4, X_f5, X_f6, X_f7, X_f8, X_f9, X_f10], axis=1)
collected_features_test = collected_features_test.to_numpy();

#---------------------------------------------------------------------------------------------  KNN
# K Nearest Neighbor
y_pred = knn_clf.predict(collected_features_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#---------------------------------------------------------------------------------------------  MNB
# Multinomial Naive Bayes

y_pred = mnb_clf.predict(collected_features_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#---------------------------------------------------------------------------------------------  Decision Tree
# Decision Trees
y_pred = dt_clf.predict(collected_features_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
