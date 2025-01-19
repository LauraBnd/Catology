
BREED_MAPPING = {
    "Bengal": "BEN",
    "Birman": "BIRM",
    "British Shorthair": "BRI",
    "Chartreux": "CHA",
    "European Shorthair": "EUR",
    "Maine Coon": "MCO",
    "Persian": "PER",
    "Ragdoll": "RAG",
    "Savannah": "SAV",
    "Sphynx": "SPH",
    "Siamese": "SIA",
    "Turkish Angora": "TUV"
}

DESCRIPTIONS = {
    "Time Spent": {
        1: "doesn't spend that much time together with their owner",
        2: "doesn't spend that much time together with their owner",
        3: "spends some time together with their owner",
        4: "likes to spend time together with their owner",
        5: "loves to spend time together with their owner"
    },
    "Shy": {
        1: "not shy at all",
        2: "a little bit shy",
        3: "depends on their mood",
        4: "quite shy",
        5: "really shy and introverted"
    },
    "Calm": {
        1: "very hyperactive",
        2: "mostly energetic but sometimes calm",
        3: "balanced between calm and energetic",
        4: "quite calm",
        5: "extremely calm and laid-back"
    },
    "Fearful": {
        1: "very brave and fearless",
        2: "rarely fearful",
        3: "gets scared in certain situations",
        4: "often fearful",
        5: "extremely fearful and cautious"
    },
    "Intelligent": {
        1: "not very intelligent",
        2: "average intelligence",
        3: "moderately intelligent",
        4: "very intelligent",
        5: "exceptionally intelligent and clever"
    },
    "Affectionate": {
        1: "not very affectionate",
        2: "shows some affection occasionally",
        3: "moderately affectionate",
        4: "quite affectionate",
        5: "extremely affectionate and loving"
    },
    "Friendly": {
        1: "not friendly at all",
        2: "rarely friendly",
        3: "somewhat friendly",
        4: "quite friendly",
        5: "extremely friendly and sociable"
    },
    "Independent": {
        1: "not independent at all, requires a lot of attention",
        2: "slightly dependent",
        3: "balanced between independence and dependence",
        4: "quite independent",
        5: "highly independent and self-sufficient"
    },
    "Dominant": {
        1: "not dominant at all",
        2: "shows slight dominance",
        3: "moderately dominant",
        4: "quite dominant",
        5: "extremely dominant"
    },
    "Aggressive": {
        1: "not aggressive at all",
        2: "rarely aggressive",
        3: "occasionally aggressive",
        4: "frequently aggressive",
        5: "very aggressive"
    },
    "Predictable": {
        1: "very unpredictable",
        2: "somewhat unpredictable",
        3: "balanced between predictability and unpredictability",
        4: "fairly predictable",
        5: "extremely predictable and reliable"
    },
    "Distracted": {
        1: "very focused",
        2: "rarely distracted",
        3: "sometimes distracted",
        4: "frequently distracted",
        5: "easily distracted"
    },
    "Vocal": {
        1: "very quiet",
        2: "rarely vocal",
        3: "moderately vocal",
        4: "quite vocal",
        5: "extremely vocal and expressive"
    },
    "Hair": {
        0: "The Sphynx cats are not entirely hairless but covered with fine, downy hair that is said to be like peach skin. This cat has no whiskers or eyelashes. The skin is wrinkled on parts of the head, body, and legs but should be taut everywhere else.",
        1: "very short hair",
        2: "short hair",
        3: "medium hair",
        4: "long hair",
        5: "very long hair"
    },
    "Pointy Ears": {
        1: "Maine Coon ears resemble a wild Lynx cat, they are big and pointy. They have tufts of fur that come to a point on the tip of the ears. It can also be referred to as “lynx tipping.” Their ears were made to keep them warm and help them sustain the harsh winters of Maine. The fur in the ears of the Maine Coon is lengthier and more prominent."
    },
    "Pattern": {
        1: "Their coat has one of the most popular and most recognizable pattern variety, at times closely resembling baby leopards. The spots are usually small to medium-sized patterns that are scattered all over the cat's coat, with large, dark spots on a light background being the most highly prized variation.",
        2: "Siamese cats are visually pretty distinct. While there are four generally recognized coat patterns, all Siamese cats have light-colored bodies with darker “tips” on their tails, paws, noses, and/or ears.",
    },
    "Gray coat": {
        1: "They Chartreux breed is known for it's blue (silver-grey) water-resistant short hair double coats which are often slightly thick in texture."
    },
    "Limp Body": {
        1: "When held, Ragdolls often relax their muscles completely, which gives them a 'floppy' or 'limp' appearance. This is a physical manifestation of their calm demeanor."
    },
    "Size": {
        1: "very small",
        2: "small",
        3: "medium",
        4: "large",
        5: "very large"
    }
}


def get_description(value, attribute):
    if attribute in DESCRIPTIONS:
        return DESCRIPTIONS[attribute].get(value, "unknown")
    return "unknown"

def get_breed_shortcut(breed_name):

    breed_name = breed_name.lower()
    normalized_mapping = {k.lower(): v for k, v in BREED_MAPPING.items()}
    return normalized_mapping.get(breed_name, None)



def generate_breed_description(df, breed_name):

    shortened_breed = get_breed_shortcut(breed_name)
    if not shortened_breed:
        return f"No data found for breed: {breed_name}"

    breed_data = df[df['Breed'] == shortened_breed]

    if breed_data.empty:
        return f"No data found for breed: {breed_name}"

    attribute_columns = ['Time Spent', 'Shy', 'Calm', 'Fearful', 'Intelligent', 'Affectionate',
                         'Friendly', 'Independent', 'Dominant', 'Aggressive', 'Predictable',
                         'Distracted', 'Vocal', 'Hair', 'Pointy Ears', 'Pattern',
                         'Gray coat', 'Limp Body', 'Size']

    averages = breed_data[attribute_columns].mean()

    description = [f"{breed_name} description:"]
    for attribute, avg_value in averages.items():
        avg_value_rounded = round(avg_value)
        desc = get_description(avg_value_rounded, attribute)


        if desc != "unknown":
            description.append(f"- {attribute}: {desc}")

    return "\n".join(description)
