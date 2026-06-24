import pandas as pd
import re

# 1. Load the dataset
df = pd.read_csv('captions.csv')

def categorize_image(caption):
    c = str(caption).lower()
    
    # Helper to find whole words using regex
    def has_w(words):
        return any(re.search(r'\b' + w + r'\b', c) for w in words)
        
    # Refined keyword lists based on actual CSV contents
    adult_words = ['man', 'men', 'woman', 'women', 'mother', 'father', 'adult', 'adults', 'parent', 'parents', 'guy', 'guys', 'lady', 'ladies']
    child_words = ['child', 'children', 'boy', 'boys', 'girl', 'girls', 'baby', 'babies', 'kid', 'kids', 'toddler', 'infant', 'son', 'daughter']
    person_words = adult_words + child_words + ['person', 'people', 'human', 'family', 'player', 'players']
    
    animal_words = ['dog', 'dogs', 'cat', 'cats', 'bird', 'birds', 'horse', 'horses', 'elephant', 'elephants', 'animal', 'animals', 'puppy', 'puppies', 'kitten', 'kittens', 'bear', 'bears', 'cow', 'cows', 'sheep', 'pup', 'pet', 'fish', 'hummingbird', 'duck', 'swan']
    group_words = ['group', 'crowd', 'several', 'many', 'three', 'four', 'five', 'bunch', 'team', 'family', 'audience']
    tech_words = ['computer', 'laptop', 'phone', 'cell', 'television', 'tv', 'screen', 'keyboard', 'video', 'camera', 'wii', 'nintendo', 'radio', 'monitor']
    food_words = ['food', 'eating', 'pizza', 'cake', 'plate', 'bowl', 'kitchen', 'restaurant', 'sandwich', 'fruit', 'coffee', 'cup', 'salad', 'drink', 'meal', 'chicken', 'vegetable', 'beer', 'wine', 'tea', 'pan', 'banana', 'pumpkin', 'apple', 'orange', 'donut', 'watermelon', 'cook', 'cooked', 'cooking']
    sports_words = ['baseball', 'tennis', 'skateboard', 'snowboard', 'skiing', 'ski', 'playing', 'ball', 'frisbee', 'surf', 'surfing', 'kite', 'game', 'bat', 'racket', 'sports', 'lebron', 'golf', 'soldier', 'target']
    outdoor_words = ['beach', 'road', 'street', 'building', 'buildings', 'tree', 'trees', 'grass', 'mountain', 'sand', 'water', 'ocean', 'park', 'curb', 'snow', 'vehicle', 'car', 'cars', 'plane', 'airplane', 'train', 'bus', 'truck', 'helicopter', 'wall', 'outside', 'bike', 'bicycle', 'motorcycle', 'fire', 'flower', 'caution', 'sign', 'house']
    indoor_words = ['desk', 'table', 'book', 'books', 'couch', 'bed', 'room', 'chair', 'chairs', 'glass', 'bottle', 'window', 'rug', 'toilet', 'stove', 'shelf', 'sink', 'lamp', 'basket', 'vase', 'box', 'bathroom', 'mixer', 'paper', 'cutting', 'toy', 'toys', 'tin', 'comic']
    
    has_adult = has_w(adult_words)
    has_child = has_w(child_words)
    has_person = has_w(person_words)
    has_animal = has_w(animal_words)
    
    # Calculate exact distinct person mentions to safely detect dyads (Pairs)
    person_mentions = 0
    for w in ['man', 'woman', 'boy', 'girl', 'person', 'guy', 'lady', 'child', 'baby']:
        person_mentions += len(re.findall(r'\b' + w + r'\b', c))
        
    has_multiple_people = (has_w(['two', 'couple', 'both', 'together']) and has_person) or (person_mentions >= 2) or has_w(['men', 'women', 'boys', 'girls', 'people'])

    # 2. Rule-Based Waterfall (Order dictates prioritization!)
    if has_adult and has_child:
        return 'Adult-Child Interaction'
    if has_w(group_words):
        return 'Group / Crowd'
    if has_multiple_people:
        return 'Social Interaction (Pairs)'
    if has_person and has_animal:
        return 'Human-Animal Interaction'
    if has_w(sports_words):
        return 'Sports / Physical Activity'
    if has_w(tech_words):
        return 'Technology / Screen'
    if has_w(food_words):
        return 'Food / Eating'
    if has_animal:
        return 'Animal Focus'
    if has_person:
        return 'Single Person'
    if has_w(indoor_words):
        return 'Indoor Objects / Scenes'
    if has_w(outdoor_words):
        return 'Outdoor / Vehicles'
        
    return 'Other'

# 3. Apply categorization
df['category'] = df['caption'].apply(categorize_image)

# 4. Save the processed data
output_file = 'captions_categorized.csv'
df.to_csv(output_file, index=False)

# 5. Output distributions
print(df['category'].value_counts())
print(f"Total processed: {len(df)}")