
import os,spacy,sys, math ,imp ,numpy as np
from sentence_transformers import SentenceTransformer, util
###########################################################################
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

def main() :
	
	#Set wokring directory paths-----
	#---------------------------------
	path ="/mnt/d/Shared_vm/Home_assessment_Interviews/Cellebrite_2023"
	setup_paths(path)

	import Aux_Cellebrite as Aux_Cell
	imp.reload(Aux_Cell)


	#Get Data 
	(chat_df,summary_pieces_df,id_col) = Aux_Cell.get_data() 

	# Load the SpaCy NLP model
	spacy_nlp = spacy.load("en_core_web_sm")
	

	#Generte identification keys------
	#---------------------------------
	chat_df	  = Aux_Cell.run_generate_identification_keys(input_df = chat_df.copy(),text_col = "dialogue",spacy_nlp=spacy_nlp )	
	summary_pieces_df = Aux_Cell.run_generate_identification_keys(input_df = summary_pieces_df.copy(),text_col = "summary_piece",spacy_nlp=spacy_nlp )


	#Establishes mappings between segments of summaries ----
	#-------------------------------------------------------

	#Establishes mappings between segments of summaries and their respective original chats.
	mapped_subset_df = Aux_Cell.map_subset_of_summaries_to_each_chat(chat_df,summary_pieces_df, identification_col ="identification_keys" ,chat_id_col="id" )
	
	#Sanity
	assert(np.any(mapped_subset_df["similarity"].isnull())==False),\
	"There should be no instances of similarity with null values in mapped_subset_df."
	

	#Load the S-BERT model
	#source https://www.sbert.net/docs/pretrained_models.html
	model = SentenceTransformer('all-MiniLM-L6-v2')

	#Organizes summary segments in accordance with their respective original chat order ----
	#---------------------------------------------------------------------------------------

	reconstructed_df = mapped_subset_df.groupby(by=['id'],as_index=False).\
										apply(Aux_Cell.create_order_between_segments_of_summaries,\
										sentence_transformer_model = model,\
										chat_text_col ='dialogue',\
										summary_piece_text_col ="summary_piece")\
										.reset_index(drop=True)  

	#Reutrn results ---
	#------------------
	reconstructed_to_submit_df = Aux_Cell.prepare_output(reconstructed_df,chat_df)
	reconstructed_to_submit_df.to_csv("reconstructed_to_submit_df.csv",index=False)


