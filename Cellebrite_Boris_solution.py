
import spacy, math ,imp ,numpy as np
from sentence_transformers import SentenceTransformer, util
###########################################################################



def main() :

	import Aux_Cellebrite as Aux_Cell
	imp.reload(Aux_Cell)


	#Set wokring directory paths-----
	#---------------------------------
	path ="/mnt/d/Shared_vm/Home_assessment_Interviews/Cellebrite_2023"
	Aux_Cell.setup_paths(path)

	#Get Data 
	(chat_df,summary_pieces_df,id_col) = Aux_Cell.get_data() 

	# Load the SpaCy NLP model
	spacy_nlp = spacy.load("en_core_web_sm")
	

	#Generte identification keys------
	#---------------------------------
	chat_df	  = Aux_Cell.run_generate_identification_keys(input_df = chat_df.copy(),text_col = "dialogue",spacy_nlp=spacy_nlp )	
	summary_pieces_df = Aux_Cell.run_generate_identification_keys(input_df = summary_pieces_df.copy(),text_col = "summary_piece",spacy_nlp=spacy_nlp )
	#chat_df_bck = chat_df.copy(); summary_pieces_df_bck = summary_pieces_df.copy();


	#Establishes mappings between segments of summaries ----
	#-------------------------------------------------------

	#Establishes mappings between segments of summaries and their respective original chats.
	mapped_subset_df = Aux_Cell.map_subset_of_summaries_to_each_chat(chat_df,summary_pieces_df, identification_col ="identification_keys" ,chat_id_col="id" )
	#mapped_subset_df_bck = mapped_subset_df.copy()
	#np.sum(mapped_subset_df.similarity.isnull())
	
	#Sanity
	assert(np.any(mapped_subset_df["similarity"].isnull())==False),\
	"There should be no instances of similarity with null values in mapped_subset_df."
	

	#Load the S-BERT model
	#source https://www.sbert.net/docs/pretrained_models.html
	model = SentenceTransformer('all-MiniLM-L6-v2')
	#model = SentenceTransformer('all-mpnet-base-v2')
	
	#Organizes summary segments in accordance with their respective original chat order ----
	#---------------------------------------------------------------------------------------

	reconstructed_df = mapped_subset_df.groupby(by=['id'],as_index=False).\
								      apply(Aux_Cell.create_order_between_segments_of_summaries,
													 sentence_transformer_model = model,
								      				 chat_text_col ='dialogue',
								      				 summary_piece_text_col ="summary_piece")\
								      				 .reset_index(drop=True)  

	#Reutrn results ---
	#------------------
	reconstructed_to_submit_df = Aux_Cell.prepare_output(reconstructed_df,chat_df)
	#reconstructed_df.to_csv("reconstructed_df_1_1.csv",index=False)
	#reconstructed_to_submit_df.to_csv("reconstructed_to_submit_df.csv",index=False)




##########
#Playing
##########
"""


def play_delete_TFIDF(given_chat_df   ,tf_idf, chat_text_col = 'dialogue',summary_piece_text_col ="summary_piece") :

	''' 
	Execute Semantic Textual Similarity between Chat to it's segment of summaries.

	Boris : tried this and didn't provide good results

	'''


	#chat_id = "13611427" ; given_chat_df = result_df[result_df[chat_id_col] ==chat_id].copy()
	#chat_text_col = 'dialogue' ; summary_piece_text_col ="summary_piece"
	
	print("The chat id is :",given_chat_df['id'].unique()[0])

	given_chat_dialogue = given_chat_df[chat_text_col][0] #Original chat dialogue ; print(given_chat_dialogue)
	segments_of_summaries_sentences_list = given_chat_df[summary_piece_text_col].values.tolist()
	'''
	array(["Lucy tells Ann about Adam's unpleasant behavior in the past.",
		   "Sean's grandfather celebrates his 100th birthday tomorrow so they're organizing a party for 50 people at the weekend.",
		   "Emily had also gone out with Lucy a few times and it didn't end up well either.",
		   "Lula was the other member of this party but he's been falsely imprisoned and now people hate him."],
		  dtype=object)
	'''

	#Run Semantic Textual Similarity with S-BERT ( Sentence BERT)
	given_chat_embedding			 = tf_idf.transform([given_chat_dialogue])
	segments_of_summaries_embeddings = tf_idf.transform(segments_of_summaries_sentences_list)
	#src : https://www.sbert.net/docs/usage/semantic_textual_similarity.html

	#Compute cosine-similarities for given chat to all chunk summaries ( 1 vecotr versus many)
	cosine_scores = cosine_similarity(given_chat_embedding, segments_of_summaries_embeddings)

	#cosine_scores -> tensor([[0.5335, 0.1335, 0.3632, 0.1483]])

	print("\n Orignal dialogue \n :",given_chat_dialogue )		
	print("\n segments_of_summaries_sentences_list \n : ", segments_of_summaries_sentences_list)	
	print("Orignal cosine_scores  :",cosine_scores[0] )		
		

def play_delete_SBERT(given_chat_df  ,sentence_transformer_model, chat_text_col = 'dialogue',summary_piece_text_col ="summary_piece") :

	''' 
	Execute Semantic Textual Similarity between Chat to its segment of summaries.

	'''

	#chat_id = "13611427" ; given_chat_df = result_df[result_df[chat_id_col] ==chat_id].copy()
	#chat_text_col = 'dialogue' ; summary_piece_text_col ="summary_piece"
	
	print("The chat id is :",given_chat_df['id'].unique()[0])

	given_chat_dialogue = given_chat_df[chat_text_col][0] #Original chat dialogue ; print(given_chat_dialogue)
	segments_of_summaries_sentences_list = given_chat_df[summary_piece_text_col].values.tolist()
	'''
	array(["Lucy tells Ann about Adam's unpleasant behavior in the past.",
		   "Sean's grandfather celebrates his 100th birthday tomorrow so they're organizing a party for 50 people at the weekend.",
		   "Emily had also gone out with Lucy a few times and it didn't end up well either.",
		   "Lula was the other member of this party but he's been falsely imprisoned and now people hate him."],
		  dtype=object)
	'''

	#Run Semantic Textual Similarity with S-BERT ( Sentence BERT)
	given_chat_embedding			 = sentence_transformer_model.encode(given_chat_dialogue, convert_to_tensor=True)
	segments_of_summaries_embeddings = sentence_transformer_model.encode(segments_of_summaries_sentences_list, convert_to_tensor=True)
	#src : https://www.sbert.net/docs/usage/semantic_textual_similarity.html

	#Compute cosine-similarities for given chat to all chunk summaries ( 1 vecotr versus many)
	cosine_scores = util.cos_sim(given_chat_embedding, segments_of_summaries_embeddings)
	#cosine_scores -> tensor([[0.5335, 0.1335, 0.3632, 0.1483]])

	print("\n Orignal dialogue \n :",given_chat_dialogue )		
	print("\n segments_of_summaries_sentences_list \n : ", segments_of_summaries_sentences_list)	
	print("Orignal cosine_scores  :",cosine_scores[0] )		
		





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(min_df = 5 ,max_df =0.95,stop_words='english')

# Fit and transform the documents
tf_idf = vectorizer.fit(chat_df['dialogue'])



		

#source https://www.sbert.net/docs/pretrained_models.html
model_1 = SentenceTransformer('all-MiniLM-L6-v2')
#model_2 = SentenceTransformer('all-mpnet-base-v2')

chat_id = "13829778"
given_chat_df = result_df[result_df[chat_id_col] ==chat_id].copy()
create_order_between_segments_of_summaries(given_chat_df,sentence_transformer_model = model_1, 
										   chat_text_col ='dialogue',
										   summary_piece_text_col ="summary_piece")


play_delete_SBERT(given_chat_df,sentence_transformer_model = model_1, 
										   chat_text_col ='dialogue',
										   summary_piece_text_col ="summary_piece")


play_delete_TFIDF(given_chat_df ,tf_idf ,
				    chat_text_col ='dialogue',
					summary_piece_text_col ="summary_piece")

"""