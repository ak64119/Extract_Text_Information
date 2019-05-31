
# Information Extraction - Assignment
This assignment is based on the Information Extraction lecture and the lab.


Name: Akshay Kochhar
Stud ID: 18230051
Batch: MSc in Data Analytics 2018-19


```python
#Importing all necessary libraries
import warnings
warnings.filterwarnings('ignore')
import nltk
import re
from statistics import mode
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import matplotlib
matplotlib.use('Agg')
import itertools
```


```python
inputfile='football_players.txt'            #Location of the file
buf=open(inputfile, encoding="UTF-8")
list_of_doc_1=buf.read().split('\n')
list_of_doc = [i for i in list_of_doc_1 if i != '']
```

# Task 1 (10 Marks)
Write a function that takes each document and performs:
1) sentence segmentation 2) tokenization 3) part-of-speech tagging

Please keep in mind that the expected output is a list within a list as shown below.



```python
#The belowe function is used for POS tagging the sentence and storing it in the list.

def ie_preprocess(document):
    pos_sentences = []
    sentnc_tknz = sent_tokenize(document)
    post_tag = []
    for i in sentnc_tknz:
      wrd_tknz = word_tokenize(i)
      post_tag = nltk.pos_tag(wrd_tknz)
      pos_sentences.append(post_tag)
    
    return pos_sentences
  
  
```

Run the following code to check your result for the first document (Ronaldo).


```python
first_doc=list_of_doc[0]
pos_sent=ie_preprocess(first_doc)
pos_sent[0]
```




    [('Cristiano', 'NNP'),
     ('Ronaldo', 'NNP'),
     ('dos', 'NN'),
     ('Santos', 'NNP'),
     ('Aveiro', 'NNP'),
     (',', ','),
     ('ComM', 'NNP'),
     (',', ','),
     ('GOIH', 'NNP'),
     ('(', '('),
     ('born', 'VBN'),
     ('5', 'CD'),
     ('February', 'NNP'),
     ('1985', 'CD'),
     (')', ')'),
     ('is', 'VBZ'),
     ('a', 'DT'),
     ('Portuguese', 'JJ'),
     ('professional', 'JJ'),
     ('footballer', 'NN'),
     ('who', 'WP'),
     ('plays', 'VBZ'),
     ('for', 'IN'),
     ('Spanish', 'JJ'),
     ('club', 'NN'),
     ('Real', 'NNP'),
     ('Madrid', 'NNP'),
     ('and', 'CC'),
     ('the', 'DT'),
     ('Portugal', 'NNP'),
     ('national', 'JJ'),
     ('team', 'NN'),
     ('.', '.')]



Expected output
 [...[('He', 'PRP'),
  ('is', 'VBZ'),
  ('a', 'DT'),
  ('forward', 'NN'),
  ('and', 'CC'),
  ('serves', 'NNS'),
  ('as', 'IN'),
  ('captain', 'NN'),
  ('for', 'IN'),
  ('Portugal', 'NNP'),
  ('.', '.')], ...]

# Task 2 (20 Marks)
Write a function that will take the list of tokens with POS tags for each sentence and returns the named entities (NE). 

Hint: Use binary=True while calling NE chunk function


```python
def named_entity_finding(pos_sent):
    tree = nltk.ne_chunk(pos_sent, binary=True)
    named_entities = []
    for subtree in tree.subtrees():
      if subtree.label() == 'NE':
        entity = ""
        for leaf in subtree.leaves():
          entity = entity + leaf[0] + " "
        named_entities.append(entity.strip())

    return named_entities

#Cheking the above function by calling the named_entity_finding function with specified document.
pos_sents= ie_preprocess(list_of_doc[0])
named_entity_finding(pos_sents[0])
```




    ['Cristiano Ronaldo',
     'Santos Aveiro',
     'ComM',
     'GOIH',
     'Portuguese',
     'Spanish',
     'Real Madrid',
     'Portugal']



Expected output ['Cristiano Ronaldo',
 'Santos Aveiro',
 'ComM',
 'GOIH',
 'Portuguese',
 'Portuguese',
 'Spanish',
 'Real Madrid',
 'Portugal']

# Task 3 (10 Marks)

Now use the named_entity_finding() function to extract all NEs for each document.

Hint: pos_sents holds the list of lists of tokens with POS tags


```python
#This function is extracts named entities in entire document and returns all unique NE as flattened list
def NE_flat_list_fn(pos_sents): 
    NE=[]
    for pos_sent in pos_sents:
      tokn_sentnce = ie_preprocess(pos_sent)  #POS tagging the document
      for i in tokn_sentnce:
        entity = named_entity_finding(i)
        if len(entity) != 0:
          NE.append(entity)
          
    NE_flat_list = list(itertools.chain.from_iterable(NE))   #function for flattening the list
    
    return NE_flat_list
  
NE_doc = NE_flat_list_fn(list_of_doc)
list(set(NE_doc))[1:10]
```




    ['Born',
     'Marco',
     'Portugal',
     'David Trezeguet',
     'Querétaro',
     'FIFA Club',
     'United',
     'Lionel Andrés',
     'Real Madrid']



# Task 4 (40 Marks)

Write functions to extract the name of the player, country of origin and date of birth as well as the following relations: team(s) of the player and position(s) of the player.

Hint: Use the re.compile() function to create the extraction patterns

Reference: https://docs.python.org/3/howto/regex.html


```python
#This function is used for finding player name
def name_of_the_player(doc):
  match_1 = re.compile(r'^(.*?)\s\Sborn')              #Regex for player's name
  string1 = "ComM"
  string2 = "OBE"
  name_of_player = match_1.findall(doc)[0]
    
  if(string1 in name_of_player):
    match_2 = re.compile(r'^(.*?),\sComM')
    name_of_player = match_2.findall(doc)[0]
      
  if(string2 in name_of_player):
    match_2 = re.compile(r'^(.*?),\sOBE')
    name_of_player = match_2.findall(doc)[0]
       
  name = re.sub('\W+',' ', name_of_player)
  return name

#This function is used for finding player's country of origin
def country_of_origin(doc):
    match = re.compile(r'(?<=the\s)[A-Za-z]+[^h](?=\snational\steam)')   #Regex for finding national team name.
    country_origin = match.findall(doc)[0]
    if(country_origin == "German"):
      match = re.compile(r'(?<=Cup\swith\s)[A-Za-z]+') #Regex for finding team name.
      country_origin = match.findall(doc)[0]
      
    country = re.sub('\W+',' ',country_origin)
    return country

#This function is used for finding player's date of birth
def date_of_birth(doc):
    match = re.compile(r'(?<=born\s)\d{1,2}\s[A-Za-z]+\s\d{4}')     #Regex for date of birth
    dob_ext = match.findall(doc)[0]
    date = re.sub('\W+',' ', dob_ext)
    return date

#This function is used for finding player's national team and club
def team_of_the_player(doc):
    team = []
    match_1 = re.compile(r'(?<=the\s)[A-Za-z]+[^h](?=\snational\steam)')        #Regex for national team
    match_2 = re.compile(r'((?<=club\s)\w+[^(and|car|to)]\s[^(in|Ro|an)]\w+)')  #Regex for club team
    nation = match_1.findall(doc)[0]
    club_match = match_2.findall(doc)

    if len(nation) != 0:
      nat_team = nation+" national team"
      team.append(nat_team)
    
  
    if len(club_match) != 0:
      team.append(club_match[0])
    
    return team

#This function is used for finding player's position in the football field
def position_of_the_player(doc):
  #As the positions are limited, they are hardcoded in the Regex.
    pattern = re.compile(r'(attacking\smidfielder|forward|striker|right\swinger|winger|central\smidfielder)')
    position = pattern.findall(doc)
    return position
```

Execute the below command to check your fuction



```python
date_of_birth(list_of_doc[2])
```




    '5 February 1992'



Expected output '5 February 1992'

# Task 5 (10 Marks)

Write a function using the outputs from the previous functions to generate JSON-LD output as follows.

Reference: https://json-ld.org/primer/latest/

{ "@id": "http://my-soccer-ontology.com/footballer/name_of_the_player",

    "name": "",
    "born": "",
    "country": "",
    "position": [
        { "@id": "http://my-soccer-ontology.com/position",
            "type": ""
        }
     ]   
     "team": [
        { "@id": "http://my-soccer-ontology.com/team",
            "name": ""
        }   
     ]
}



```python
#The first json function before the 6th Question a modification has to be made.
def json_fun_1(arg1,arg2,arg3,arg4,arg5):
  ld = { "@id": "http://my-soccer-ontology.com/footballer/"+arg1,
        "name": arg1,
        "born": arg2,
        "country": arg3,
        "position": [
            { "@id": "http://my-soccer-ontology.com/position/",
             "type": arg4
            }
        ],
        "team": [
            { "@id": "http://my-soccer-ontology.com/team/",
               "name": arg5
            }
        ]     
       }
  return ld
```


```python
#Checking the json function
json_fun_1(name_of_the_player(list_of_doc[0]), date_of_birth(list_of_doc[0]), country_of_origin(list_of_doc[0]), position_of_the_player(list_of_doc[0]), team_of_the_player(list_of_doc[0]))
```




    {'@id': 'http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos Aveiro',
     'born': '5 February 1985',
     'country': 'Portugal',
     'name': 'Cristiano Ronaldo dos Santos Aveiro',
     'position': [{'@id': 'http://my-soccer-ontology.com/position/',
       'type': ['forward']}],
     'team': [{'@id': 'http://my-soccer-ontology.com/team/',
       'name': ['Portugal national team', 'Real Madrid']}]}



# Task 6 (10 Marks)
Identify one other relation (besides team and player) and write a function to extract this. Also extend the JSON-LD output accordingly.


```python
#Debut Year: I will find the year of debut of football players.
#Assumption_1: The debut means when the player started playing football professionally.
#Assumption_2: The debut age must be below 20 for a football player.

def debutyear(doc):
  debt_yr = []
  string_1 = "debut"
  string_2 = "aged"
  string_3 = "age"
  sentnc_tknz = sent_tokenize(doc)
  for i in sentnc_tknz:
    splitd_sentc = i.split()
    if ((string_1 in splitd_sentc)&(string_2 in splitd_sentc)):
      mtch_year = re.findall('\d{4}'," ".join(splitd_sentc))
      if(len(mtch_year) != 0):
        debt_yr.append(mtch_year[0])
        
        
    if(len(debt_yr) == 0):
      if((string_2 in splitd_sentc)|(string_3 in splitd_sentc)):
        mtch_year = re.findall('\d{4}'," ".join(splitd_sentc))
        if(len(mtch_year) != 0):
          debt_yr.append(mtch_year[0])
          
          
  if (len(debt_yr) == 0):                   
    debt_yr.append("Debut Year not available")               #If the year is not present.
     
  return(debt_yr[0])



#Checing the debut year function
debutyear(list_of_doc[1])


# Note: Debut year is not available for the sentences (2,3,5,7,8). 
```




    '2004'




```python
#Extending the JSON-LD output accordingly with the new modification, i.e., addition of Debut Year function

def json_fun(arg1,arg2,arg3,arg4,arg5,arg6):
  ld = { "@id": "http://my-soccer-ontology.com/footballer/"+arg1,
        "name": arg1,
        "born": arg2,
        "country": arg3,
        "position": [
            { "@id": "http://my-soccer-ontology.com/position/",
             "type": arg4
            }
        ],
        "team": [
            { "@id": "http://my-soccer-ontology.com/team/",
               "name": arg5
            }
        ],
          "Debut Year": arg6 
       }
  
  return ld
```


```python
#Cheking the JSON with required modification
json_fun(name_of_the_player(list_of_doc[0]), date_of_birth(list_of_doc[0]), country_of_origin(list_of_doc[0]), position_of_the_player(list_of_doc[0]), team_of_the_player(list_of_doc[0]),debutyear(list_of_doc[0]))
```




    {'@id': 'http://my-soccer-ontology.com/footballer/Cristiano Ronaldo dos Santos Aveiro',
     'Debut Year': '2003',
     'born': '5 February 1985',
     'country': 'Portugal',
     'name': 'Cristiano Ronaldo dos Santos Aveiro',
     'position': [{'@id': 'http://my-soccer-ontology.com/position/',
       'type': ['forward']}],
     'team': [{'@id': 'http://my-soccer-ontology.com/team/',
       'name': ['Portugal national team', 'Real Madrid']}]}


