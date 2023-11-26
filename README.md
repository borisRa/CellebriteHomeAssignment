
# Cellebrite: home challenge

---
The submision has three files :

		 - Cellebrite_Boris_solution.py
		 - Aux_Cellebrite.py
		 - reconstructed_file.csv


---
#### Solution methodology
---

 1. Loading the data into ‘chat_df’, ‘summary_pieces_df’ data frames.
 2. Producing identification keys from the "dialogue" text column in the 'chat_df' and the "summary_piece" text column in the 'summary_pieces_df' (‘run_generate_identification_keys()’).
**_The motivation was to establish a representative footprint linked to each text!_**
	 - SpaCy is used to extract Named Entity Recognition (**NER**) from a given sentence. All entities with 'DATE' label weren’t included because it brought too much noise in similarity metrics.
	 - SpaCy is used to extract Parts of speech (**POS**) , the extraction was limited to only the "NOUN" parts of speech due to the excessive noise introduced in similarity metrics when using all parts of speech (POS).
	 - Example to  set of keys  output : ('night', 'TIME'), ('night', 'NOUN'), ('two', CARDINAL'),('parties', 'NOUN'), ('Joanna', 'PERSON'), ('Daniel', 'PERSON')

 3. Utilizing a variety of heuristics to establish mappings between segments of summaries and their corresponding original chats (‘map_subset_of_summaries_to_each_chat()‘).

	 - Heuristic estimation of the number of chunks the summary was shuffled into. The formula "top_n = total subset of summaries / total chat dialogue + 2" was used.
	 - Per each identification key set in ‘chat_df’ and 'summary_pieces_df' **Jaccard similarity metric** was calculated:
		 - Jaccard similarity metric  : ∣A∪B∣/∣A∩B∣​
	- Top N most similar summary chunks were extracted.
	- Filter out all summary chunks that have a ‘PERSON’ entity different from given chat dialogue. For instance, exclude summary chunks that reference someone named "Moshe" but were not mentioned in the original dialogue.
	
	3.1. If following the filtration process, there are no remaining summary chunks for given chat, then :
	-  Apply the TF-IDF approach to compute vector representations for the provided chat dialogue and summary segments. Subsequently, utilize cosine similarity to extract the Top N most similar summary chunks.
	- Filter out all summary chunks that have a ‘PERSON’ entity different from given chat dialogue (as above).
		
4. Arranging summary segments based on their corresponding original chat order heavily relies on the utilization of S-BERT. S-BERT, a specialized variant of BERT (Bidirectional Encoder Representations from Transformers), is fine-tuned specifically for sentence embeddings in this context (‘create_order_between_segments_of_summaries()’).
	-	Filter out segments of summaries that got to low score in semantic textual similarity when compared to the original chat dialogue. Measured through the cosine similarity metric applied to the embeddings of two sentences.
	-	Divide the original dialogue into sentences, ensuring that the number of sentences matches the desired number of summary chunks.
	-	Execute semantic textual similarity between the sentences of the initial dialogue and the segments of summaries. Choose the segment with the highest cosine.similarity score. Proceed to the second sentence and find the most similar segment (excluding the one already selected). Repeat this process until all dialogue sentences are mapped to the most similar segments of summaries.
	-	If, following the application of all heuristics and filtration, no similar segments are identified, set "None" in the "summary" column. In the provided dataset, there were two IDs (13728935 and 13812859) for which no summaries were found.

5. Return the reconstructed data frame in the specified format: "id" representing the dialogue ID and "summary" representing a summary reconstructed from the shuffled pieces.




##### Thank you 😊,
##### Boris Rabinovich 
