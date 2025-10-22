
def caption_attack_priority(original_caption: str, image: np.ndarray, orginal_label, api) -> str:
    
    image_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)

    if orginal_label == False:
        best_sim = -1
    else:
        best_sim = 1
    
    for index, row in nounlist.iterrows():
        #print(best_sim)
        noun = row['nouns']
        text_features = row['encoded']
        sim = calc_similarity(image_features.to("cpu"), text_features.to("cpu"))
        if (sim < best_sim and orginal_label == True) or (sim > best_sim and orginal_label == False):
            best_sim = sim
            best_noun = noun
    
    
    #get word priority:
    word_priority_list = []
    doc = nlp(original_caption)
    for tok, word in zip(doc,original_caption.split(" ")):
        if tok.tag_ in NOUN_TAGS:
            word_features = encode_text(word)
            sim = calc_similarity(image_features.to("cpu"), word_features.to("cpu"))
            word_priority_list.append((word, sim))


    word_priority_list.sort(key=lambda x: x[1], reverse= not orginal_label)
    
    #print(f"Best noun to add: {best_noun} with similarity {best_sim}")
    final_caption = original_caption
    for nr_replacements in range(1,len(word_priority_list)):
        for i in range(nr_replacements):
            final_caption = final_caption.replace(word_priority_list[i][0], best_noun)
        new_score = api.score(image, final_caption)
        if (new_score <= 0.5 and orginal_label == True) or (new_score > 0.5 and orginal_label == False):
            break
    final_score = new_score
    return final_caption, final_score