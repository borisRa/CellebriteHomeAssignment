
import  sys , os ,pandas as pd , numpy as np ,spacy ,math

from sklearn.model_selection import train_test_split
from pandarallel import pandarallel # pip install pandarallel [--upgrade]


###########################################


"""
pip install spacy

import spacy

#Got the error [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
python -m spacy download en_core_web_sm 
"""

###########################################
def setup_paths(path):

	"""
	Setup the working directory

	param path : Get as input working directory path
	"""

	#path ="/mnt/d/Shared_vm/Home_assessment_Interviews/Cellebrite_2023"
	print("The path that was defined is : ",path)


	os.chdir(path)
	sys.path.append(path)

	print("My path is :"  , path)	
	
###########################################

def get_data():

	"""
	Getting the required data and preprocess it

	
	return chat_df  :  chat conversations with missing summaries
	return summary_pieces_df   : shuffled pieces of summaries for chats in chat_df
	"""

	chat_df = pd.read_csv(os.path.join("home_challenge_data","dialogues.csv"),low_memory=False)
	summary_pieces_df = pd.read_csv(os.path.join("home_challenge_data","summary_pieces.csv"),low_memory=False)

	

	print("chat_df shape is : ",chat_df.shape) # (1400, 2)
	print("summary_pieces_df shape is : ",summary_pieces_df.shape) #(4053, 1)
	
	
	#Setup X_cols and y_col
	id_col ="id"

	return(chat_df,summary_pieces_df,id_col)


#####################################
#Generate identification keys
#####################################



#POS - Parts Of Speech --------------
#------------------------------------

def Parts_Of_Speech_POS(text,spacy_nlp):

	# Input sentence
	sentence = "I love using spaCy for natural language processing."
	
	# Process the sentence using spaCy
	doc = spacy_nlp(text)
	
	"""
	pos_dict = {}
	# Print the parts of speech for each token in the sentence
	for token in doc:
		pos_dict[token.text]= token.pos_
	#token.text : retrieves the text of the token.
	#token.pos_ : retrieves the part of speech of the token.	
	"""

	nouns_POS = [(token.text, token.pos_)  for token in doc if  token.pos_ == "NOUN"]
	#all_POS =  [(token.text, token.pos_)  for token in doc]
	
	return (set(nouns_POS))



#NER - Named_Entity_Recognition -----
#------------------------------------

def Named_Entity_Recognition_Model(text,spacy_nlp):

	"""
	Function that extracts Named Entity Recognition (NER) from a given sentence
	
	param text : The text to generate keys from 
	param spacy_nlp : Spacy object

	return ans_set : set of Named Entity Recognition (NER) like : ('night', 'TIME') , ('Daniel', 'PERSON') , 
																	('Saturday night', 'TIME'),('Mike', 'PERSON') ,('Stanley', 'ORG')
	"""
	
	# Load the SpaCy NLP model
	#spacy_nlp = spacy.load("en_core_web_sm")


	# Process the text using SpaCy
	doc = spacy_nlp(text)

	# Extract names of people (PERSON entities)
	#people_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
	#print ("The people names are : ",set(people_names))

	#Don't include 'DATE' entities because it brought too much noise in similarity metrics
	NER_set = [(ent.text,ent.label_) for ent in doc.ents  if ent.label_ != "DATE"]
	#print ("All labels are : ",set(all_labels))

	
	return(set(NER_set))


def generate_identification_keys(text,spacy_nlp):

	"""
	Function that generates identification keys from a given sentence
	ssch as :  Named Entity Recognition (NER) and Parts Of Speech (POS)

	param text : The text to generate keys from 
	param spacy_nlp : Spacy object

	return ans_set : set of keys like : ('night', 'TIME'), ('night', 'NOUN'), ('two', 'CARDINAL'),
										('parties', 'NOUN'), ('Joanna', 'PERSON'), ('Daniel', 'PERSON')
	"""

	NER_set = Named_Entity_Recognition_Model(text,spacy_nlp)
	POS_set  = Parts_Of_Speech_POS(text,spacy_nlp)
	
	#Union (Combine between the two sets
	ans_set = NER_set.union(POS_set)
	 
	
	return(ans_set)

def run_generate_identification_keys(input_df,text_col,spacy_nlp ):

	"""
	A function that utilizes the 'generate_identification_keys' function 
	for each text in EACH row and conducts length sanity checks.
	
	param input_df : input data
	param text_col : The text columns to generate keys from 
	param spacy_nlp : Spacy object

	return input_df : Per each set of keys like : ('night', 'TIME'), ('night', 'NOUN'), ('two', 'CARDINAL'),
										('parties', 'NOUN'), ('Joanna', 'PERSON'), ('Daniel', 'PERSON')
	"""

	#text_col = "summary_piece" ; text_col = "dialogue"
	#input_df = summary_pieces_df.copy() ;	#input_df = chat_df.copy()

	len_before = len(input_df)

	input_df['identification_keys'] = input_df[text_col].apply(lambda x: generate_identification_keys(x,spacy_nlp))

	#pandarallel.initialize(nb_workers= int(os.cpu_count()) - 1, use_memory_fs = False ) #set num of cores	
	#input_df['identification_keys'] = input_df[text_col].parallel_apply(lambda x: generate_identification_keys(x,spacy_nlp))
	#input_df.to_csv("dialogues_with_NER_df.csv",index=False)
	
	assert(len_before == len(input_df) )

	return(input_df)
	

######################################################
# Mappings between segments of summaries
# and their respective original chats
######################################################	

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(row_set, given_set):
	"""
	Function that computes the Jaccard similarity metric between two sets.
	

	param row_set : set number one 
	param given_set : set number two

	return ans_set : Jaccard similarity between set one to set two
										
	"""
	intersection = len(set(row_set) & given_set)
	union = len(set(row_set) | given_set)
	return intersection / union if union != 0 else 0

def map_subset_of_summaries_to_each_chat(chat_df,summary_pieces_df, identification_col ="identification_keys" ,chat_id_col="id" ):

	"""
	By employing a range of heuristics, this function establishes mappings between
	segments of summaries and their respective original chats.

	param chat_df : chat conversations with missing summaries
	param summary_pieces_df : shuffled pieces of summaries for chats in chat_df
	param identification_col : identification column name
	param chat_id_col : chat id_column name


	return result_df : mapped segments of summaries and their respective original chats
	
	"""

	#text_col = "summary_piece" ; text_col = "dialogue"
	#input_df = summary_pieces_df.copy()
	#input_df = chat_df.copy()
	
	#pandarallel.initialize(nb_workers= int(os.cpu_count()) - 1, use_memory_fs = False ) #set num of cores	
	
	
	#Variable for the outcome of the function 
	result_df = pd.DataFrame()
	

	#Heuristic to how many chunks to summary was split
	top_n = int(len(summary_pieces_df)/len(chat_df)) + 2 #4
	#Boris : check if to change this to 3 later


	# Iterating through rows
	for index, chat_row in chat_df.iterrows():

		#Given chat 
		given_chat_identification_set = chat_row[identification_col].copy()
		given_chat_id = chat_row[chat_id_col]
		#print(chat_row['dialogue'])

		#Calculate jaccard similarity between given chat to summary chunks
		relevant_summaries_df = summary_pieces_df.copy()
		relevant_summaries_df['similarity'] = relevant_summaries_df.apply(lambda row: calculate_jaccard_similarity(given_chat_identification_set,
																	   row[identification_col]), axis=1)# nice .take top 3
		
		#Extract top N most similar summary chunks
		relevant_summaries_df = relevant_summaries_df.sort_values(by='similarity',ascending=False)[:top_n].copy()
		relevant_summaries_df[chat_id_col] = given_chat_id #Set chat ID


		#Filter out summaries with differnt PERSON entity relatively to given chat dialogue 
		relevant_summaries_df = filter_summaries_with_different_PERSON_entity_compared_to_chat(given_chat_identification_set,
																							  relevant_summaries_df,
																							  identification_col)

		
		#Merge summaries to their respective chat 
		result_tmp_df = pd.merge(left  = chat_df[chat_df[chat_id_col] == given_chat_id].copy() ,
								 right = relevant_summaries_df, how="left",on=chat_id_col,
								 suffixes=('_Chat', '_Summary'))
		
		#Sanity for length
		if len(relevant_summaries_df) >0 :
			assert(len(result_tmp_df) == len(relevant_summaries_df)) 
		
		#Rbind all to final result
		result_df =   pd.concat( [result_df, result_tmp_df ] , axis=0 ) 
		
	
	
		
	#Sanity to ensure that the count of unique chat IDs remains the same
	assert(result_df[chat_id_col].nunique()==chat_df[chat_id_col].nunique())

	return(result_df)

def filter_summaries_with_different_PERSON_entity_compared_to_chat(given_chat_identification_set,relevant_summaries_df,identification_col=""):

	"""
	Exclude all summary chunks that have a PERSON entity different from given chat dialogue.

	param given_chat_identification_set : chat conversations with missing summaries
	param relevant_summaries_df : shuffled pieces of summaries for chats in chat_df
	param identification_col : identification column name
	

	return relevant_summaries_df : Filtered pieces of summaries 
	
	"""

	#Find all PERSON entities that are in summary piece BUT NOT in the corresponding chat
	relevant_summaries_df['PERSON_not_in_chat'] = \
	relevant_summaries_df.apply(
						 lambda row: [item for item in row[identification_col].difference(given_chat_identification_set) if item[1]=='PERSON' ]
						 ,axis=1)

	#Drop all summaries that have PERSON entities that are in summary piece BUT NOT in the corresponding chat
	relevant_summaries_df = relevant_summaries_df[relevant_summaries_df['PERSON_not_in_chat'].apply(len)==0].copy()
	

	del relevant_summaries_df['PERSON_not_in_chat']
	return(relevant_summaries_df)

		

######################################################
# Create order between segments of summaries
# to semantic contents of the chats/dialogue
######################################################	

from sentence_transformers import SentenceTransformer, util

def Named_Entity_Recognition_Model(text,spacy_nlp):

	"""
	Function that extracts Named Entity Recognition (NER) from a given sentence
	
	param text : The text to generate keys from 
	param spacy_nlp : Spacy object

	return ans_set : set of Named Entity Recognition (NER) like : ('night', 'TIME') , ('Daniel', 'PERSON') , 
																	('Saturday night', 'TIME'),('Mike', 'PERSON') ,('Stanley', 'ORG')
	"""
	

	# Process the text using SpaCy
	doc = spacy_nlp(text)


	NER_list = [(ent.text,ent.label_) for ent in doc.ents  ]
	#print ("All labels are : ",set(all_labels))

	#for i, sent in enumerate(doc.sents, 1):
	#	print(f"Sentence {i}: {sent.text}")

	
	return(set(NER_list))


def combine_given_chat_sentences_into_sentences_divided_by_number_of_segments(given_chat_split_into_sentences,segments_of_summaries_sentences_list):
	

	"""
	Goal of this funcion is to combin each 'N' cells from 'given_chat_split_into_sentences' to get the same length  
	of summary chunks ('num_of_summary_segments').
	To be able to run sentence similarity between them and order the segments.
	
	For example:
		
	To combine my_list = ["1","2","3","4","5","6","7","8"] into 2 lenghth cell : ["1 2 3" , "4 5 6 7 8"]
	To combine my_list = ["1","2","3","4","5","6","7","8","9","10"] into 3 lenghth cell : ["1 2 3" , "4 5 6" ,"7 8 9 10"]

	param given_chat_split_into_sentences : Original chat split of the given chat
	param segments_of_summaries_sentences_list : Chunks into which the summary was split

	return concatenated_list : Combined sentences
		

	"""	
	
	#Estimation of the number of chunks into which the summary was split
	num_of_summary_segments = len(segments_of_summaries_sentences_list)
	
	combine_each_N_sentences = math.floor(len(given_chat_split_into_sentences)/num_of_summary_segments)#Round down
	combine_each_N_sentences = np.max([1,combine_each_N_sentences])
	

	concatenated_list = []
	i = 0
	
	while i < len(given_chat_split_into_sentences) - combine_each_N_sentences + 1:
		concatenated_item = " ".join(given_chat_split_into_sentences[i : i + combine_each_N_sentences])
		concatenated_list.append(concatenated_item)
		i = i + combine_each_N_sentences
		
	#Handle the remaining items (if any) - by combining them into the last cell of the list 
	remaining_items = given_chat_split_into_sentences[i:]
	if len(remaining_items) >0 :
		
		#Combine the remaining items with the last List cell
		last_cell = " ".join(remaining_items)
		
		#Append to last cell
		last_cell= [" ".join( [concatenated_list[-1], last_cell] )]

		#Concat the last cell with remaining items
		concatenated_list = concatenated_list[:-1] + last_cell
			
		
	#Sanity that chat has been segmented into chunks that are equa to the number of summary segments.
	assert(len(concatenated_list) == num_of_summary_segments)	

	return(concatenated_list)    
	

def run_semantic_textual_similarity_with_S_BERT(given_chat_dialogue_sentence,segments_of_summaries_sentences_list,sentence_transformer_model):
	
	"""
	
	Run Semantic Textual Similarity with S-BERT ( Sentence BERT)
	src : https://www.sbert.net/docs/usage/semantic_textual_similarity.html

	param given_chat_dialogue_sentence :  chat dialogue_sentences
	param segments_of_summaries_sentences_list : chunks into which the summary was split
	param sentence_transformer_model : S-BERT model 

	return cosine_scores : cosine Similarity score bwtween the two inputs
	
	"""

	#Run Semantic Textual Similarity with S-BERT ( Sentence BERT)
	given_chat_embedding			 = sentence_transformer_model.encode(given_chat_dialogue_sentence, convert_to_tensor=True)
	segments_of_summaries_embeddings = sentence_transformer_model.encode(segments_of_summaries_sentences_list, convert_to_tensor=True)
	
	#Compute cosine-similarities for given chat to all chunk summaries ( 1 vecotr versus many)
	cosine_scores = util.cos_sim(given_chat_embedding, segments_of_summaries_embeddings)
	#cosine_scores -> tensor([[0.5335, 0.1335, 0.3632, 0.1483]])


	return(cosine_scores)
							


def create_order_between_segments_of_summaries(given_chat_df  ,sentence_transformer_model, chat_text_col = 'dialogue',summary_piece_text_col ="summary_piece") :

	""" 
	Function tha maps between segments of summaries  and their respective original chats

	param param given_chat_df  : given dialogue chat to analyze
	param param sentence_transformer_model : S-BERT model
	param param chat_text_col = original chat text column name
	param param summary_piece_text_col :  summary piece text column name

	return reconstructed_df : Data frame with orderes segments of summaries

	"""

	#chat_text_col = 'dialogue' ; summary_piece_text_col ="summary_piece" ;chat_id_col="id"
	#chat_id = "13728935" ; given_chat_df = mapped_subset_df[mapped_subset_df[chat_id_col] ==chat_id].copy()
	
	
	print("The chat id is :",given_chat_df['id'].unique()[0])

	given_chat_dialogue = given_chat_df[chat_text_col][0] #Original chat dialogue ; print(given_chat_dialogue)
	segments_of_summaries_sentences_list = given_chat_df[summary_piece_text_col].values.tolist()
	"""
	array(["Lucy tells Ann about Adam's unpleasant behavior in the past.",
		   "Sean's grandfather celebrates his 100th birthday tomorrow so they're organizing a party for 50 people at the weekend.",
		   "Emily had also gone out with Lucy a few times and it didn't end up well either.",
		   "Lula was the other member of this party but he's been falsely imprisoned and now people hate him."],
		  dtype=object)
	"""

	# Filter out segments of summaries ----------------
	#--------------------------------------------------
	
	#Filter out segments of summaries that got low score in semantic textual similarity
	#when compared to the original chat dialogue (based on S-BERT)
	cosine_scores_1 = run_semantic_textual_similarity_with_S_BERT(given_chat_dialogue,
																segments_of_summaries_sentences_list,
															    sentence_transformer_model)

	"""
	print("\n Orignal dialogue \n :",given_chat_dialogue )		
	print("\n segments_of_summaries_sentences_list \n : ", segments_of_summaries_sentences_list)	
	print("Orignal cosine_scores  :",cosine_scores_1[0] )		
	"""


	#Drop all cosine scores that <= max cosine scores * 0.6 (got this number of many mnaual trials)
	cosine_scores_1 = cosine_scores_1.numpy()[0]
	indexes_to_keep = np.where(cosine_scores_1>=np.max(cosine_scores_1) * 0.6)[0]
	
	# Get the elements at the specified indices using a list comprehension
	segments_of_summaries_sentences_list = [segments_of_summaries_sentences_list[i] for i in indexes_to_keep]
	
	#IF all cosine_scores are negative ( very different then just return None !)
	if ( len(segments_of_summaries_sentences_list) ==0) or (np.all(cosine_scores_1<0)):
		reconstructed_df = given_chat_df[["id","dialogue"]][:1].copy()
		reconstructed_df['summary'] =	None
		print("The chat id is :",given_chat_df['id'].unique()[0] , " No similarities were found !!!")
		return(reconstructed_df)



	# Combine chat sentences for ordering the segments of summaries ----------------
	#-------------------------------------------------------------------------------
	
	#Split a a dialogue into sentences based on new lines(\n)
	given_chat_split_into_sentences  = given_chat_dialogue.split('\n')

	#Combine chat sentences to be able to run semantic textual similarity with segments of summaries (for ordering the segments of summaries)
	given_chat_split_into_sentences  = combine_given_chat_sentences_into_sentences_divided_by_number_of_segments(
																given_chat_split_into_sentences, segments_of_summaries_sentences_list
																)

	#Run heurstic_to_order_between_segments_of_summaries
	reconstructed_summary = []
	for given_chat_dialogue_sentence in given_chat_split_into_sentences:
		
		#print(given_chat_dialogue_sentence)
		cosine_scores_2 = run_semantic_textual_similarity_with_S_BERT(given_chat_dialogue_sentence,segments_of_summaries_sentences_list,
																	  sentence_transformer_model)
		# Find the index of the maximum value
		max_index = np.argmax(cosine_scores_2.numpy())
		
		# Remove and return the element at the specified index
		#print(cosine_scores_2)
		#print("All : ",segments_of_summaries_sentences_list)
		best_match_chunk = segments_of_summaries_sentences_list.pop(max_index)
		#print("Best: " , best_match_chunk)
		
		reconstructed_summary.extend([best_match_chunk])
		
		

	#print("Orignal dialogue \n :",given_chat_dialogue )		
	#print("Reconstructed summary \n : ", reconstructed_summary)	
	
	#Preapre the required results ---
	#--------------------------------
	
	reconstructed_df = given_chat_df[["id","dialogue"]][:1].copy()
	reconstructed_df['summary'] =	" ".join(reconstructed_summary)
	
	return(reconstructed_df)
		
		
