import  sys, os, pandas as pd, numpy as np ,spacy, math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util
###########################################################################

def setup_paths(path):

	"""
	Setup the working directory
	param path : Get as input working directory path
	"""

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

	"""
	Function that extracts Parts of speech (POS) from a given sentence
	
	param text : The text to generate keys from 
	param spacy_nlp : Spacy object

	return ans_set : set of parts of speach (POS)  like :  ('night', 'NOUN'), ('parties', 'NOUN') ,('work', 'NOUN')
	"""
	
	# Process the sentence using spaCy
	doc = spacy_nlp(text)
	
	#The extraction was limited to only the "NOUN" parts of speech due to the excessive noise 
	#introduced in similarity metrics when using all parts of speech (POS).
	nouns_POS = [(token.text, token.pos_)  for token in doc if  token.pos_ == "NOUN"]
	
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
	
	# Process the text using SpaCy
	doc = spacy_nlp(text)

	#Don't include 'DATE' entities because it brought too much noise in similarity metrics
	NER_set = [(ent.text,ent.label_) for ent in doc.ents  if ent.label_ != "DATE"]

	return(set(NER_set))


def generate_identification_keys(text,spacy_nlp):

	"""
	Function that generates identification keys from a given sentence
	such as :  Named Entity Recognition (NER) and Parts Of Speech (POS)

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

	len_before = len(input_df)
	input_df['identification_keys'] = input_df[text_col].apply(lambda x: generate_identification_keys(x,spacy_nlp))
	assert(len_before == len(input_df) )


	return(input_df)
	

######################################################
# Mappings between segments of summaries
# and their respective original chats
######################################################	

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

def run_semantic_textual_similarity_with_TFIDF(given_chat_dialogue  ,summary_piece_text,tf_idf) :

	''' 
	Function that compute cosine-similarities for two senetnces based on TF-IDF vecotors

	param given_chat_dialogue :  chat dialogue
	param summary_piece_text : chunks into which the summary was split
	param tf_idf : fitted TF-IDF vectorizer

	return cosine_scores : cosine-similarities for two TF-IDF vecotors

	'''
	
	#Compute cosine-similarities for two TF-IDF vecotors
	cosine_scores = cosine_similarity( tf_idf.transform([given_chat_dialogue]), tf_idf.transform([summary_piece_text]))

	
	return(cosine_scores[0][0])	

def map_subset_of_summaries_to_each_chat(chat_df,summary_pieces_df, identification_col ="identification_keys" ,chat_id_col="id" ):

	"""
	By employing a range of heuristics, this function establishes mappings between
	segments of summaries and their respective original chats.

	param chat_df : chat conversations with missing summaries
	param summary_pieces_df : shuffled pieces of summaries for chats in chat_df
	param identification_col : identification column name
	param chat_id_col : chat id column name


	return result_df : mapped segments of summaries and their respective original chats
	
	"""
	
	# TF-IDF ---------
	#-----------------
	# Create a TF-IDF vectorizer
	vectorizer = TfidfVectorizer(min_df = 5 ,max_df =0.95,stop_words='english')

	# Fit on all dialogues in chat_df
	tf_idf = vectorizer.fit(chat_df['dialogue'])
	#-----------------
	
	#Variable for the outcome of the function 
	result_df = pd.DataFrame()

	#Heuristic to how many chunks to summary was split
	top_n = int(len(summary_pieces_df)/len(chat_df)) + 2  # top_n => 4

	# Iterating through rows
	for index, chat_row in chat_df.iterrows():
		
		#chat_row = chat_df[chat_df['id'] =='13829088'].copy()

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


		#Filter out summaries with different  PERSON entity relatively to given chat dialogue 
		relevant_summaries_df = filter_summaries_with_different_PERSON_entity_compared_to_chat(given_chat_identification_set,
																							  relevant_summaries_df,
																							  identification_col)
		if len(relevant_summaries_df) == 0:
			
			print( f"""For given chat ID : '{given_chat_id}'
					   no similarity was detected based on Named Entity Recognition (NER) and Parts of Speech (POS). Therefore, TF-IDF was applied instead.""")
			
			#Get given chat dialogue 
			given_chat_dialogue = chat_row['dialogue']

			#Taking TOP N most close segments of summaries (from all summary_pieces_df ) based on TF-IDF
			relevant_summaries_df = summary_pieces_df.copy()
			relevant_summaries_df['similarity'] = relevant_summaries_df.apply(lambda row: run_semantic_textual_similarity_with_TFIDF(
																			  given_chat_dialogue = given_chat_dialogue,
																			  summary_piece_text = row["summary_piece"],
																			  tf_idf = tf_idf ), axis=1)# nice .take top 3

			#Extract top N most similar summary chunks
			relevant_summaries_df = relevant_summaries_df.sort_values(by='similarity',ascending=False)[:top_n].copy()
			relevant_summaries_df[chat_id_col] = given_chat_id #Set chat ID

			#Filter out summaries with different  PERSON entity relatively to given chat dialogue 
			relevant_summaries_df = filter_summaries_with_different_PERSON_entity_compared_to_chat(given_chat_identification_set,
																							 	   relevant_summaries_df,
																							 	   identification_col)		
		#----------------------------------------------------------------------------------------------

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
	relevant_summaries_df.apply(\
						 lambda row: [item for item in row[identification_col].difference(given_chat_identification_set) if item[1]=='PERSON' ]\
						 ,axis=1)

	#Drop all summaries that have PERSON entities that are in summary piece BUT NOT in the corresponding chat
	relevant_summaries_df = relevant_summaries_df[relevant_summaries_df['PERSON_not_in_chat'].apply(len)==0].copy()
	
	del relevant_summaries_df['PERSON_not_in_chat']
	return(relevant_summaries_df)	

######################################################
# Create order between segments of summaries
# to semantic contents of the chats/dialogue
######################################################	
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
	
	return(set(NER_list))


def combine_given_chat_sentences_into_sentences_divided_by_number_of_segments(given_chat_split_into_sentences,segments_of_summaries_sentences_list):
	
	"""
	Goal of this funcion is to combain each 'N' lists cells from 'given_chat_split_into_sentences' to achieve the same length  
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
	Function that organizes summary segments in accordance with their respective original chat order.

	param param given_chat_df  : given dialogue chat to analyze
	param param sentence_transformer_model : S-BERT model
	param param chat_text_col = original chat text column name
	param param summary_piece_text_col :  summary piece text column name

	return reconstructed_df : Data frame with orderes segments of summaries

	"""

	given_chat_dialogue = given_chat_df[chat_text_col][0] #Original chat dialogue ; print(given_chat_dialogue)
	segments_of_summaries_sentences_list = given_chat_df[summary_piece_text_col].values.tolist()

	# Filter out segments of summaries ----------------
	#--------------------------------------------------
	
	#Filter out segments of summaries that got low score in semantic textual similarity
	#when compared to the original chat dialogue (based on S-BERT)
	cosine_scores_1 = run_semantic_textual_similarity_with_S_BERT(given_chat_dialogue,
																  segments_of_summaries_sentences_list,
															      sentence_transformer_model)

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

	# Divide the original  dialogue into sentences, ensuring that the number  --------
	# of sentencesmatches the desired number of summary chunks ---------------------
	#-------------------------------------------------------------------------------

	#Split a a dialogue into sentences based on new lines(\n)
	given_chat_split_into_sentences  = given_chat_dialogue.split('\n')

	#Combine chat sentences to be able to run semantic textual similarity with segments of summaries (for ordering the segments of summaries)
	given_chat_split_into_sentences  = combine_given_chat_sentences_into_sentences_divided_by_number_of_segments(
																given_chat_split_into_sentences, segments_of_summaries_sentences_list
																)

	"""
	Execute semantic textual similarity between the sentences of the initial dialogue and the segments of summaries.
	Choose the segment with the highest cosine similarity score. Proceed to the second sentence and find
	the most similar segment (excluding the one already selected). 
	Repeat this process until all dialogue sentences are mapped to the most similar segments of summaries.
	"""
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
		
		
	#Preapre the required results ---
	#--------------------------------
	
	reconstructed_df = given_chat_df[["id","dialogue"]][:1].copy()
	reconstructed_df['summary'] =	" ".join(reconstructed_summary)
	
	return(reconstructed_df)


#######################################
# Prepare output in required format
#######################################

def prepare_output(reconstructed_df ,chat_df,chat_id_col="id",summary_col="summary"):

	""" 
	Prepare output in required format

	param param reconstructed_df  : data frane with reconstructed summaries
	param param chat_df : chat conversations with missing summaries
	param param chat_id_col = chat id column name
	param param summary_col :   reconstructed summary column name

	return reconstructed_df : Data frame with orderes segments of summaries

	"""

	assert(len(reconstructed_df)==len(chat_df)),\
		"The returned file must contain the same number of rows as the input(chat_df)."
	
	assert(reconstructed_df[chat_id_col].nunique()==chat_df[chat_id_col].nunique()),\
		"The returned file must contain the unique chat ids as the input(chat_df)."

	return (reconstructed_df[[chat_id_col,summary_col]])
		
	
		
		
