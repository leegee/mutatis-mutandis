# `EEBO_*`

 Experiment: clean the EEBO-TCP Phase I corpus and slice, train embeddings with fastText as Word2Vec doesn't handle sub-words for spelling variants.

 Initially check vectors' with nearest neighbours.
 
    eebo_parse_tei.py
    eebo_parse_dates.py
    eebo_slice.py
    eebo_train_embeddings.py
    eebo_extract_neighbourhoods.py