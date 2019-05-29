
# Text Information Extraction using python

This mini-project is regarding Information extraction in a football player data set as provided by the course coordinator.


Name: Akshay Kochhar
Stud ID: 18230051



```python
#Importing all necessary libraries

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

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /root/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    [nltk_data] Downloading package maxent_ne_chunker to
    [nltk_data]     /root/nltk_data...
    [nltk_data]   Package maxent_ne_chunker is already up-to-date!
    [nltk_data] Downloading package words to /root/nltk_data...
    [nltk_data]   Package words is already up-to-date!
    

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:11: UserWarning: 
    This call to matplotlib.use() has no effect because the backend has already
    been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    or matplotlib.backends is imported for the first time.
    
    The backend was *originally* set to 'module://ipykernel.pylab.backend_inline' by the following code:
      File "/usr/lib/python3.6/runpy.py", line 193, in _run_module_as_main
        "__main__", mod_spec)
      File "/usr/lib/python3.6/runpy.py", line 85, in _run_code
        exec(code, run_globals)
      File "/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py", line 16, in <module>
        app.launch_new_instance()
      File "/usr/local/lib/python3.6/dist-packages/traitlets/config/application.py", line 657, in launch_instance
        app.initialize(argv)
      File "<decorator-gen-121>", line 2, in initialize
      File "/usr/local/lib/python3.6/dist-packages/traitlets/config/application.py", line 87, in catch_config_error
        return method(app, *args, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/ipykernel/kernelapp.py", line 462, in initialize
        self.init_gui_pylab()
      File "/usr/local/lib/python3.6/dist-packages/ipykernel/kernelapp.py", line 403, in init_gui_pylab
        InteractiveShellApp.init_gui_pylab(self)
      File "/usr/local/lib/python3.6/dist-packages/IPython/core/shellapp.py", line 213, in init_gui_pylab
        r = enable(key)
      File "/usr/local/lib/python3.6/dist-packages/IPython/core/interactiveshell.py", line 2950, in enable_matplotlib
        pt.activate_matplotlib(backend)
      File "/usr/local/lib/python3.6/dist-packages/IPython/core/pylabtools.py", line 309, in activate_matplotlib
        matplotlib.pyplot.switch_backend(backend)
      File "/usr/local/lib/python3.6/dist-packages/matplotlib/pyplot.py", line 232, in switch_backend
        matplotlib.use(newbackend, warn=False, force=True)
      File "/usr/local/lib/python3.6/dist-packages/matplotlib/__init__.py", line 1305, in use
        reload(sys.modules['matplotlib.backends'])
      File "/usr/lib/python3.6/importlib/__init__.py", line 166, in reload
        _bootstrap._exec(spec, module)
      File "/usr/local/lib/python3.6/dist-packages/matplotlib/backends/__init__.py", line 14, in <module>
        line for line in traceback.format_stack()
    
    
      # This is added back by InteractiveShellApp.init_path()
    


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
pos_sent
```




    [[('Cristiano', 'NNP'),
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
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('a', 'DT'),
      ('forward', 'NN'),
      ('and', 'CC'),
      ('serves', 'NNS'),
      ('as', 'IN'),
      ('captain', 'NN'),
      ('for', 'IN'),
      ('Portugal', 'NNP'),
      ('.', '.')],
     [('In', 'IN'),
      ('2008', 'CD'),
      (',', ','),
      ('he', 'PRP'),
      ('won', 'VBD'),
      ('his', 'PRP$'),
      ('first', 'JJ'),
      ('Ballon', 'NNP'),
      ("d'Or", 'NN'),
      ('and', 'CC'),
      ('FIFA', 'NNP'),
      ('World', 'NNP'),
      ('Player', 'NNP'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('Year', 'NN'),
      ('awards', 'NNS'),
      ('.', '.')],
     [('He', 'PRP'),
      ('then', 'RB'),
      ('won', 'VBD'),
      ('the', 'DT'),
      ('FIFA', 'NNP'),
      ('Ballon', 'NNP'),
      ("d'Or", 'NN'),
      ('in', 'IN'),
      ('2013', 'CD'),
      ('and', 'CC'),
      ('2014', 'CD'),
      ('.', '.')],
     [('In', 'IN'),
      ('2015', 'CD'),
      (',', ','),
      ('Ronaldo', 'NNP'),
      ('scored', 'VBD'),
      ('his', 'PRP$'),
      ('500th', 'JJ'),
      ('senior', 'JJ'),
      ('career', 'NN'),
      ('goal', 'NN'),
      ('for', 'IN'),
      ('club', 'NN'),
      ('and', 'CC'),
      ('country', 'NN'),
      ('.', '.')],
     [('Often', 'RB'),
      ('ranked', 'VBN'),
      ('as', 'IN'),
      ('the', 'DT'),
      ('best', 'JJS'),
      ('player', 'NN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('world', 'NN'),
      (',', ','),
      ('Ronaldo', 'NNP'),
      ('was', 'VBD'),
      ('named', 'VBN'),
      ('the', 'DT'),
      ('best', 'JJS'),
      ('Portuguese', 'JJ'),
      ('player', 'NN'),
      ('of', 'IN'),
      ('all', 'DT'),
      ('time', 'NN'),
      ('by', 'IN'),
      ('the', 'DT'),
      ('Portuguese', 'NNP'),
      ('Football', 'NNP'),
      ('Federation', 'NNP'),
      (',', ','),
      ('during', 'IN'),
      ('its', 'PRP$'),
      ('100th', 'JJ'),
      ('anniversary', 'JJ'),
      ('celebrations', 'NNS'),
      ('in', 'IN'),
      ('2015', 'CD'),
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('the', 'DT'),
      ('only', 'JJ'),
      ('player', 'NN'),
      ('to', 'TO'),
      ('win', 'VB'),
      ('four', 'CD'),
      ('European', 'JJ'),
      ('Golden', 'NNP'),
      ('Shoe', 'NNP'),
      ('awards', 'NNS'),
      ('.', '.')],
     [('One', 'CD'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('most', 'RBS'),
      ('marketable', 'JJ'),
      ('athletes', 'NNS'),
      ('in', 'IN'),
      ('sport', 'NN'),
      (',', ','),
      ('in', 'IN'),
      ('2016', 'CD'),
      ('Forbes', 'NNP'),
      ('named', 'VBD'),
      ('Ronaldo', 'NNP'),
      ('the', 'DT'),
      ('world', 'NN'),
      ("'s", 'POS'),
      ('best', 'JJS'),
      ('paid', 'VBD'),
      ('athlete', 'NN'),
      ('.', '.')],
     [('In', 'IN'),
      ('June', 'NNP'),
      ('2016', 'CD'),
      (',', ','),
      ('ESPN', 'NNP'),
      ('ranked', 'VBD'),
      ('him', 'PRP'),
      ('the', 'DT'),
      ('world', 'NN'),
      ("'s", 'POS'),
      ('most', 'RBS'),
      ('famous', 'JJ'),
      ('athlete', 'NN'),
      ('.', '.')],
     [('Ronaldo', 'NNP'),
      ('began', 'VBD'),
      ('his', 'PRP$'),
      ('club', 'NN'),
      ('career', 'NN'),
      ('playing', 'NN'),
      ('for', 'IN'),
      ('Sporting', 'VBG'),
      ('CP', 'NNP'),
      (',', ','),
      ('before', 'IN'),
      ('signing', 'VBG'),
      ('with', 'IN'),
      ('Manchester', 'NNP'),
      ('United', 'NNP'),
      ('at', 'IN'),
      ('age', 'NN'),
      ('18', 'CD'),
      ('in', 'IN'),
      ('2003', 'CD'),
      ('.', '.')],
     [('After', 'IN'),
      ('winning', 'VBG'),
      ('his', 'PRP$'),
      ('first', 'JJ'),
      ('trophy', 'NN'),
      (',', ','),
      ('the', 'DT'),
      ('FA', 'NNP'),
      ('Cup', 'NNP'),
      (',', ','),
      ('during', 'IN'),
      ('his', 'PRP$'),
      ('first', 'JJ'),
      ('season', 'NN'),
      ('in', 'IN'),
      ('England', 'NNP'),
      (',', ','),
      ('he', 'PRP'),
      ('helped', 'VBD'),
      ('United', 'NNP'),
      ('win', 'VB'),
      ('three', 'CD'),
      ('successive', 'JJ'),
      ('Premier', 'NNP'),
      ('League', 'NNP'),
      ('titles', 'NNS'),
      (',', ','),
      ('a', 'DT'),
      ('UEFA', 'NNP'),
      ('Champions', 'NNP'),
      ('League', 'NNP'),
      ('title', 'NN'),
      (',', ','),
      ('and', 'CC'),
      ('a', 'DT'),
      ('FIFA', 'NNP'),
      ('Club', 'NNP'),
      ('World', 'NNP'),
      ('Cup', 'NNP'),
      ('.', '.')],
     [('By', 'IN'),
      ('age', 'NN'),
      ('23', 'CD'),
      (',', ','),
      ('he', 'PRP'),
      ('had', 'VBD'),
      ('received', 'VBN'),
      ('Ballon', 'NNP'),
      ("d'Or", 'NN'),
      ('and', 'CC'),
      ('FIFA', 'NNP'),
      ('World', 'NNP'),
      ('Player', 'NNP'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('Year', 'NNP'),
      ('nominations', 'NNS'),
      ('.', '.')],
     [('He', 'PRP'),
      ('was', 'VBD'),
      ('the', 'DT'),
      ('subject', 'NN'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('most', 'RBS'),
      ('expensive', 'JJ'),
      ('association', 'NN'),
      ('football', 'NN'),
      ('transfer', 'NN'),
      ('when', 'WRB'),
      ('he', 'PRP'),
      ('moved', 'VBD'),
      ('from', 'IN'),
      ('Manchester', 'NNP'),
      ('United', 'NNP'),
      ('to', 'TO'),
      ('Real', 'VB'),
      ('Madrid', 'NNP'),
      ('in', 'IN'),
      ('2009', 'CD'),
      ('in', 'IN'),
      ('a', 'DT'),
      ('transfer', 'NN'),
      ('worth', 'JJ'),
      ('€94', 'CD'),
      ('million', 'CD'),
      ('(', '('),
      ('$', '$'),
      ('132', 'CD'),
      ('million', 'CD'),
      (')', ')'),
      ('.', '.')],
     [('In', 'IN'),
      ('Spain', 'NNP'),
      (',', ','),
      ('he', 'PRP'),
      ('has', 'VBZ'),
      ('since', 'IN'),
      ('won', 'VBN'),
      ('one', 'CD'),
      ('La', 'NNP'),
      ('Liga', 'NNP'),
      ('title', 'NN'),
      (',', ','),
      ('two', 'CD'),
      ('Copas', 'NNP'),
      ('del', 'FW'),
      ('Rey', 'NNP'),
      (',', ','),
      ('two', 'CD'),
      ('Champions', 'NNP'),
      ('League', 'NNP'),
      ('titles', 'NNS'),
      (',', ','),
      ('and', 'CC'),
      ('a', 'DT'),
      ('Club', 'NNP'),
      ('World', 'NNP'),
      ('Cup', 'NNP'),
      ('.', '.')],
     [('Ronaldo', 'NNP'),
      ('holds', 'VBZ'),
      ('the', 'DT'),
      ('record', 'NN'),
      ('for', 'IN'),
      ('most', 'JJS'),
      ('goals', 'NNS'),
      ('scored', 'VBN'),
      ('in', 'IN'),
      ('a', 'DT'),
      ('single', 'JJ'),
      ('UEFA', 'NNP'),
      ('Champions', 'NNP'),
      ('League', 'NNP'),
      ('season', 'NN'),
      (',', ','),
      ('having', 'VBG'),
      ('scored', 'VBN'),
      ('17', 'CD'),
      ('goals', 'NNS'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('2013–14', 'CD'),
      ('season', 'NN'),
      ('.', '.')],
     [('In', 'IN'),
      ('2014', 'CD'),
      (',', ','),
      ('Ronaldo', 'NNP'),
      ('became', 'VBD'),
      ('the', 'DT'),
      ('fastest', 'JJS'),
      ('player', 'NN'),
      ('to', 'TO'),
      ('score', 'VB'),
      ('200', 'CD'),
      ('goals', 'NNS'),
      ('in', 'IN'),
      ('La', 'NNP'),
      ('Liga', 'NNP'),
      (',', ','),
      ('which', 'WDT'),
      ('he', 'PRP'),
      ('accomplished', 'VBD'),
      ('in', 'IN'),
      ('his', 'PRP$'),
      ('178th', 'CD'),
      ('La', 'NNP'),
      ('Liga', 'NNP'),
      ('game', 'NN'),
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('the', 'DT'),
      ('only', 'JJ'),
      ('player', 'NN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('history', 'NN'),
      ('of', 'IN'),
      ('football', 'NN'),
      ('to', 'TO'),
      ('score', 'VB'),
      ('50', 'CD'),
      ('or', 'CC'),
      ('more', 'JJR'),
      ('goals', 'NNS'),
      ('in', 'IN'),
      ('a', 'DT'),
      ('season', 'NN'),
      ('on', 'IN'),
      ('six', 'CD'),
      ('consecutive', 'JJ'),
      ('occasions', 'NNS'),
      ('.', '.')],
     [('In', 'IN'),
      ('2015', 'CD'),
      (',', ','),
      ('Ronaldo', 'NNP'),
      ('became', 'VBD'),
      ('the', 'DT'),
      ('all-time', 'JJ'),
      ('top', 'JJ'),
      ('goalscorer', 'NN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('UEFA', 'NNP'),
      ('Champions', 'NNP'),
      ('League', 'NNP'),
      (',', ','),
      ('and', 'CC'),
      ('he', 'PRP'),
      ('also', 'RB'),
      ('became', 'VBD'),
      ('Real', 'NNP'),
      ('Madrid', 'NNP'),
      ("'s", 'POS'),
      ('all-time', 'JJ'),
      ('leading', 'JJ'),
      ('goalscorer', 'NN'),
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('the', 'DT'),
      ('second', 'JJ'),
      ('highest', 'JJS'),
      ('goalscorer', 'NN'),
      ('in', 'IN'),
      ('La', 'NNP'),
      ('Liga', 'NNP'),
      ('history', 'NN'),
      ('behind', 'IN'),
      ('Lionel', 'NNP'),
      ('Messi', 'NNP'),
      (',', ','),
      ('his', 'PRP$'),
      ('perceived', 'JJ'),
      ('career', 'NN'),
      ('rival', 'NN'),
      ('.', '.')],
     [('Ronaldo', 'NNP'),
      ('made', 'VBD'),
      ('his', 'PRP$'),
      ('international', 'JJ'),
      ('debut', 'NN'),
      ('for', 'IN'),
      ('Portugal', 'NNP'),
      ('in', 'IN'),
      ('August', 'NNP'),
      ('2003', 'CD'),
      (',', ','),
      ('at', 'IN'),
      ('the', 'DT'),
      ('age', 'NN'),
      ('of', 'IN'),
      ('18', 'CD'),
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('Portugal', 'NNP'),
      ("'s", 'POS'),
      ('most', 'JJS'),
      ('capped', 'JJ'),
      ('player', 'NN'),
      ('of', 'IN'),
      ('all', 'DT'),
      ('time', 'NN'),
      ('with', 'IN'),
      ('over', 'IN'),
      ('130', 'CD'),
      ('caps', 'NNS'),
      (',', ','),
      ('and', 'CC'),
      ('has', 'VBZ'),
      ('participated', 'VBN'),
      ('in', 'IN'),
      ('seven', 'CD'),
      ('major', 'JJ'),
      ('tournaments', 'NNS'),
      (':', ':'),
      ('four', 'CD'),
      ('UEFA', 'IN'),
      ('European', 'JJ'),
      ('Championships', 'NNP'),
      ('(', '('),
      ('2004', 'CD'),
      (',', ','),
      ('2008', 'CD'),
      (',', ','),
      ('2012', 'CD'),
      ('and', 'CC'),
      ('2016', 'CD'),
      (')', ')'),
      ('and', 'CC'),
      ('three', 'CD'),
      ('FIFA', 'NNP'),
      ('World', 'NNP'),
      ('Cups', 'NNP'),
      ('(', '('),
      ('2006', 'CD'),
      (',', ','),
      ('2010', 'CD'),
      ('and', 'CC'),
      ('2014', 'CD'),
      (')', ')'),
      ('.', '.')],
     [('He', 'PRP'),
      ('is', 'VBZ'),
      ('the', 'DT'),
      ('first', 'JJ'),
      ('Portuguese', 'NNP'),
      ('player', 'NN'),
      ('to', 'TO'),
      ('reach', 'VB'),
      ('50', 'CD'),
      ('international', 'JJ'),
      ('goals', 'NNS'),
      (',', ','),
      ('making', 'VBG'),
      ('him', 'PRP'),
      ('Portugal', 'NNP'),
      ("'s", 'POS'),
      ('all-time', 'JJ'),
      ('top', 'JJ'),
      ('goalscorer', 'NN'),
      ('.', '.')],
     [('He', 'PRP'),
      ('scored', 'VBD'),
      ('his', 'PRP$'),
      ('first', 'JJ'),
      ('international', 'JJ'),
      ('goal', 'NN'),
      ('in', 'IN'),
      ('Euro', 'NNP'),
      ('2004', 'CD'),
      ('and', 'CC'),
      ('helped', 'VBD'),
      ('Portugal', 'NNP'),
      ('reach', 'VB'),
      ('the', 'DT'),
      ('final', 'JJ'),
      ('.', '.')],
     [('He', 'PRP'),
      ('took', 'VBD'),
      ('over', 'RP'),
      ('captaincy', 'NN'),
      ('in', 'IN'),
      ('July', 'NNP'),
      ('2008', 'CD'),
      (',', ','),
      ('and', 'CC'),
      ('he', 'PRP'),
      ('led', 'VBD'),
      ('Portugal', 'NNP'),
      ('to', 'TO'),
      ('the', 'DT'),
      ('semi-finals', 'NNS'),
      ('at', 'IN'),
      ('Euro', 'NNP'),
      ('2012', 'CD'),
      (',', ','),
      ('finishing', 'VBG'),
      ('the', 'DT'),
      ('competition', 'NN'),
      ('as', 'IN'),
      ('joint-top', 'JJ'),
      ('scorer', 'NN'),
      ('.', '.')],
     [('In', 'IN'),
      ('November', 'NNP'),
      ('2014', 'CD'),
      (',', ','),
      ('Ronaldo', 'NNP'),
      ('became', 'VBD'),
      ('the', 'DT'),
      ('all-time', 'JJ'),
      ('top', 'JJ'),
      ('scorer', 'NN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('UEFA', 'NNP'),
      ('European', 'NNP'),
      ('Championship', 'NNP'),
      ('(', '('),
      ('including', 'VBG'),
      ('qualifying', 'VBG'),
      (')', ')'),
      ('with', 'IN'),
      ('23', 'CD'),
      ('goals', 'NNS'),
      ('.', '.')],
     [('At', 'IN'),
      ('Euro', 'NNP'),
      ('2016', 'CD'),
      (',', ','),
      ('he', 'PRP'),
      ('became', 'VBD'),
      ('the', 'DT'),
      ('most', 'RBS'),
      ('capped', 'JJ'),
      ('player', 'NN'),
      ('of', 'IN'),
      ('all-time', 'NN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('tournament', 'NN'),
      (',', ','),
      ('the', 'DT'),
      ('first', 'JJ'),
      ('player', 'NN'),
      ('to', 'TO'),
      ('score', 'VB'),
      ('at', 'IN'),
      ('four', 'CD'),
      ('consecutive', 'JJ'),
      ('European', 'JJ'),
      ('Championship', 'NNP'),
      ('finals', 'NNS'),
      (',', ','),
      ('and', 'CC'),
      ('also', 'RB'),
      ('equalled', 'VBD'),
      ('Michel', 'NNP'),
      ('Platini', 'NNP'),
      ("'s", 'POS'),
      ('all-time', 'JJ'),
      ('record', 'NN'),
      ('for', 'IN'),
      ('most', 'JJS'),
      ('goals', 'NNS'),
      ('scored', 'VBN'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('competition', 'NN'),
      ('.', '.')],
     [('Ronaldo', 'NNP'),
      ('lifted', 'VBD'),
      ('the', 'DT'),
      ('trophy', 'NN'),
      ('after', 'IN'),
      ('Portugal', 'NNP'),
      ('defeated', 'VBD'),
      ('France', 'NNP'),
      ('in', 'IN'),
      ('the', 'DT'),
      ('final', 'JJ'),
      (',', ','),
      ('and', 'CC'),
      ('he', 'PRP'),
      ('received', 'VBD'),
      ('the', 'DT'),
      ('Silver', 'NNP'),
      ('Boot', 'NNP'),
      ('as', 'IN'),
      ('one', 'CD'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('second-highest', 'JJ'),
      ('goalscorers', 'NNS'),
      ('of', 'IN'),
      ('the', 'DT'),
      ('tournament', 'NN'),
      ('.', '.')]]



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




    ['Ronaldinho Gaúcho',
     'Portugal',
     'Manchester United',
     'Tottenham Hotspur',
     'English',
     'Italy',
     'Iniesta',
     'Swedish',
     'PSG']



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


