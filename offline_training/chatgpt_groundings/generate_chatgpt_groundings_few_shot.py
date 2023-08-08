import os
import json
import openai
import random
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are playing a role-playing video game where you will need to read textual descriptions to figure out the attributes of a character.

This is a list of characters and their corresponding IDs:
airplane: 2
mage: 3
dog: 4
bird: 5
fish: 6
scientist: 7
thief: 8
ship: 9
ball: 10
robot: 11
queen: 12
sword: 13

This is a list of movement types and their corresponding IDs:
chasing: 0
fleeing: 1
stationary: 2

This is a list of role types and their corresponding IDs:
dangerous enemy: 0
secret message: 1
essential objective: 2

Now, read a description and tell me which character is being mentioned and what are its movement type and role type. Your answer should follow this format:
```
Answer: Character ID, movement type ID, role type ID
```

Here are a few examples:

Description: the plane that's flying near where you are is the critical objective.
Answer: 2, 0, 2

Description: the escaping humanoid is an important goal.
Answer: 11, 1, 2

Description: the mage is inching near you is a lethal opponent.
Answer: 3, 0, 0

Description: the classified document is the hound coming your way.
Answer: 4, 0, 1

Description: the important goal is the orb which is creeping close to you.
Answer: 10, 0, 2

Now provide the answer for the following description. Follow the format of the previous answers:

Description: %s
"""

"""
with open("chatgpt_grounding_for_text_all.json") as f:
    old_data = json.load(f)

if os.path.isfile("chatgpt_grounding_few_shot.json"):
    with open("chatgpt_grounding_few_shot.json") as f:
        new_data = json.load(f)
else:
    new_data = {}

print(len(old_data))

for i, (k, v) in enumerate(old_data.items()):

    #k = random.choice(all_descriptions)
    #v = old_data[k]

    print(i)

    if k in new_data:
        continue

    prompt = PROMPT_TEMPLATE % k
    completion = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": prompt },
        ]
    )["choices"][0]["message"]["content"]
    try:
        ids = list(map(int, completion.replace("Answer: ", "").split(", ")))
    except:
        # try to parse integers
        words = completion.split()
        ids = []
        for w in words:
            try:
                num = int(w)
                ids.append(num)
            except:
                pass
        ids = ids[:3]
        while len(ids) < 3:
            if len(ids) == 0:
                ids.append(2)
            else:
                ids.append(0)

    if ids[0] < 2 or ids[0] > 13:
        ids[0] = 2

    new_data[k] = ids

    if (i + 1) % 50 == 0:
        with open("chatgpt_grounding_few_shot.json", "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"save at {i}")


print(len(old_data), len(new_data))

with open("chatgpt_grounding_few_shot.json", "w") as f:
    json.dump(new_data, f, indent=2)
"""

"""
with open("chatgpt_grounding_few_shot.json") as f:
    new_data = json.load(f)

for k, v in new_data.items():
    if v[0] < 2 or v[0] > 13:
        v[0] = 2
    assert 0 <= v[1] < 3
    assert 0 <= v[2] < 3

with open("chatgpt_grounding_few_shot.json", "w") as f:
    json.dump(new_data, f, indent=2)
"""



