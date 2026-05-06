"""
TILM2 corpus generator v4 — scene-plan generator for learnable prose.

This generator is built around persistent scene state instead of isolated
sentence templates. Each paragraph has:

  1. A world state: time, weather, terrain, water, light, optional person/fire
  2. A scene plan: a sequence of abstract sentence roles
  3. A realization layer: each role has several valid phrasings

The goal is to teach the model recurring structure with controlled variation:
consistent semantics, repeated grammar, and simple multi-sentence discourse.

Usage:
    python3 corpus_gen4.py --words 60000 --output corpus/thematic_v4.txt
"""

import argparse
import os
import random


TIMES    = ["dawn", "morning", "day", "day", "night"]
WEATHERS = ["clear", "clear", "cold", "mist", "wind", "rain"]
LANDS    = ["field", "hill", "road", "path", "valley", "shore", "wood"]
WATERS   = ["river", "lake", "sea", None]
SURFACES = ["field", "road", "path", "shore", "hill", "water"]

# Daytime gets warm/soft; dawn/morning stay cold/grey
LIGHT_COLORS_COLD = ["pale", "grey", "thin", "cold", "soft"]
LIGHT_COLORS_WARM = ["soft", "warm", "clear", "thin"]

SKY_COLORS  = ["pale", "grey", "clear", "dark", "cold"]
LAND_STATES = ["still", "cold", "grey", "bare", "dark"]
WATER_STATES = {
    "river": {
        "day":   ["swift", "deep", "low", "clear", "still", "warm"],
        "night": ["cold",  "slow", "deep", "low",   "dark"],
        "other": ["slow",  "swift","deep", "low",   "clear"],
    },
    "lake": {
        "day":   ["still", "wide", "deep", "clear", "warm", "soft"],
        "night": ["still", "cold", "deep", "wide",  "dark"],
        "other": ["still", "deep", "clear","cold",  "wide"],
    },
    "sea": {
        "day":   ["wide", "deep", "clear", "warm", "still", "soft"],
        "night": ["wide", "deep", "cold",  "dark",  "still"],
        "other": ["wide", "deep", "clear", "cold",  "still"],
    },
}
PERSONS = [
    ("the man",   "he",  "his"),
    ("the woman", "she", "her"),
]
ANIMALS = ["bird", "crow", "wolf", "dog", "deer"]

ON_LANDS = {"hill", "road", "path", "shore"}
IN_LANDS = {"field", "valley", "wood"}


def _pick_two(r: random.Random, pool: list[str]) -> tuple[str, str]:
    first  = r.choice(pool)
    second = r.choice([x for x in pool if x != first])
    return first, second


def _cap(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:]


def _land_phrase(land: str) -> str:
    if land in IN_LANDS:
        return f"in the {land}"
    return f"on the {land}"


def _water_phrase(water: str | None) -> str:
    if water is None:
        return ""
    return f"by the {water}"


def _get_animal(scene: dict, r: random.Random) -> str:
    if scene["animal"] is None:
        scene["animal"] = r.choice(ANIMALS)
    return scene["animal"]


def _ground_track(r: random.Random) -> str:
    return f"by the {r.choice(['road', 'path', 'shore'])}"


def _open_place(scene: dict) -> str:
    if scene["land"] in IN_LANDS:
        return f"in the {scene['land']}"
    if scene["land"] in {"hill", "road", "path", "shore"}:
        return f"on the {scene['land']}"
    return "on the road"


def _animal_place(scene: dict, animal: str, r: random.Random) -> str:
    choices = [
        _open_place(scene),
        _ground_track(r),
    ]
    if scene["water"]:
        choices.append(f"by the {scene['water']}")
    if animal in {"bird", "crow"}:
        choices.append(f"over the {scene['land']}")
    return r.choice(choices)


def _person_place(scene: dict, r: random.Random) -> str:
    choices = [
        _open_place(scene),
        _ground_track(r),
    ]
    if scene["water"]:
        choices.append(f"by the {scene['water']}")
    return r.choice(choices)


def _standing_place(scene: dict, r: random.Random) -> str:
    choices = [_open_place(scene)]
    if scene["water"]:
        choices.append(f"by the {scene['water']}")
    return r.choice(choices)


def _simple_surface(scene: dict, r: random.Random) -> str:
    if scene["water"] and r.random() < 0.5:
        return f"by the {scene['water']}"
    return _open_place(scene)


def build_scene(r: random.Random) -> dict:
    time    = r.choice(TIMES)
    weather = r.choice(WEATHERS)
    land    = r.choice(LANDS)
    water   = r.choice(WATERS)

    # Fire is more meaningful at night and dawn
    fire_prob = 0.35 if time in {"night", "dawn"} else 0.12
    scene = {
        "time":    time,
        "weather": weather,
        "land":    land,
        "water":   water,
        "surface": land,
        "light":   r.choice(["light", "sun", "moon", "stars"] if time == "night"
                             else ["light", "sun"]),
        "fire":    r.random() < fire_prob,
        "person":  None,
        "animal":  None,
        "distance": "near",
    }

    if r.random() < 0.55:
        subject, pron, poss = r.choice(PERSONS)
        scene["person"] = {"subject": subject, "pron": pron, "poss": poss}

    if scene["person"] is None and r.random() < 0.28:
        scene["animal"] = r.choice(ANIMALS)

    return scene


# ---------------------------------------------------------------------------
# Scene description functions
# ---------------------------------------------------------------------------

def describe_time(scene: dict, r: random.Random) -> str:
    time    = scene["time"]
    weather = scene["weather"]
    if time == "dawn":
        a, b = _pick_two(r, ["pale", "cold", "grey", "still", "soft"])
        return r.choice([
            f"the dawn was {a} and {b}.",
            f"the dawn came {a} and {b}.",
        ])
    if time == "morning":
        a, b = _pick_two(r, ["pale", "still", "clear", "soft"])
        return r.choice([
            f"the morning was {a} and {b}.",
            f"the morning came {a} and {b}.",
        ])
    if time == "day":
        a, b = _pick_two(r, ["clear", "warm", "still", "soft"])
        return r.choice([
            f"the day was {a} and {b}.",
            f"the day was {a} and {b}.",
        ])
    # Night
    if weather == "mist":
        return r.choice([
            "the night was dark and still.",
            "the night grew cold and still.",
            "the night was still and deep.",
        ])
    return r.choice([
        "the night was dark and cold.",
        "the night was still and dark.",
        "the night grew cold and still.",
        "the night was deep and still.",
    ])


def describe_sky(scene: dict, r: random.Random) -> str:
    time = scene["time"]
    if time == "night":
        if scene["light"] == "moon":
            noun = r.choice(["hill", "field", "shore", "water", "road"])
            tone = r.choice(["pale", "cold", "soft"])
            return r.choice([
                f"the moon shone {tone} on the {noun}.",
                f"the moon rose {tone} over the {noun}.",
                f"the moon shone {tone} over the {noun}.",
                f"the moon lay {tone} on the {noun}.",
            ])
        return r.choice([
            "the stars shone pale over the hill.",
            "the stars rose one by one.",
            "the stars shone cold over the water.",
            "the stars shone still in the dark.",
            "the stars shone soft over the shore.",
            "the stars were still and clear.",
        ])

    # Warm light for day; dawn/morning avoid "grey" in sun/light phrases
    if time == "day":
        color = r.choice(LIGHT_COLORS_WARM)
    elif time == "dawn":
        color = r.choice(["pale", "thin", "cold", "soft"])
    else:
        color = r.choice(LIGHT_COLORS_COLD)
    place = scene["surface"]
    if scene["light"] == "sun":
        opts = [
            f"the sun shone {color} on the {place}.",
            f"the sun rose {color} over the {scene['land']}.",
            f"the sun fell {color} over the {scene['land']}.",
            f"the sun shone {color} over the {scene['land']}.",
        ]
        if time == "day":
            opts += [
                f"the sky was warm and clear.",
                f"the sky was clear and warm.",
                f"the sky was warm and still.",
                f"the sky was warm.",
            ]
        return r.choice(opts)
    return r.choice([
        f"the light fell {color} on the {place}.",
        f"{color} light lay on the {place}.",
        f"the light lay {color} on the {place}.",
    ])


def describe_weather(scene: dict, r: random.Random) -> str:
    weather = scene["weather"]
    land    = scene["land"]
    is_day  = scene["time"] == "day"

    if weather == "mist":
        target = r.choice([land, "shore", "field", "hill"])
        if is_day:
            return r.choice([
                f"mist lay still on the {target}.",
                f"the mist lay soft on the {target}.",
                f"the mist moved slow on the {target}.",
            ])
        return r.choice([
            f"mist lay pale on the {target}.",
            f"the mist lay still on the {target}.",
            f"the mist was cold on the {target}.",
            f"the mist moved slow on the {target}.",
        ])
    if weather == "wind":
        if is_day:
            return r.choice([
                "the wind blew soft.",
                "the wind blew warm.",
                f"the wind moved through the {r.choice(['field', 'wood', 'hill'])}.",
                f"the wind moved over the {r.choice(['field', 'hill', 'shore'])}.",
                f"the sound of the wind came through the {r.choice(['wood', 'field', 'hill'])}.",
            ])
        return r.choice([
            "the wind blew cold.",
            "the wind blew soft.",
            f"the wind moved through the {r.choice(['field', 'wood', 'hill'])}.",
            f"the wind moved over the {r.choice(['field', 'hill', 'shore'])}.",
            f"the sound of the wind came through the {r.choice(['wood', 'field', 'hill'])}.",
        ])
    if weather == "rain":
        target = scene["surface"]
        return r.choice([
            f"the rain moved over the {target}.",
            f"the rain fell on the {target}.",
            f"the rain fell on the {r.choice(['field', 'road', 'shore', 'hill'])}.",
            f"the sound of the rain came through the {r.choice(['wood', 'field', 'hill'])}.",
        ])
    if weather == "clear":
        if is_day:
            a, b = _pick_two(r, ["clear", "warm", "still", "soft"])
            return r.choice([
                f"the air was {a} and {b}.",
                f"the sky was {a} and {b}.",
                f"the air grew warm.",
                f"the sky was warm.",
                f"the sky was clear.",
                f"the sky was still.",
            ])
        else:
            a, b = _pick_two(r, ["clear", "pale", "cold", "still"])
            return r.choice([
                f"the air was {a} and {b}.",
                f"the sky was {a} and {b}.",
                f"the air grew clear.",
            ])
    if is_day:
        return r.choice([
            "the air was still and soft.",
            "the air was deep and still.",
            "the air grew still.",
        ])
    return r.choice([
        "the air was cold and still.",
        "the air grew cold.",
        "the air was still and deep.",
    ])


def describe_land(scene: dict, r: random.Random) -> str:
    land   = scene["land"]
    is_day = scene["time"] == "day"

    if land in {"field", "valley"}:
        adj_lay = r.choice(["still", "warm", "clear"] if is_day else ["dark", "grey", "still"])
        adj_was = r.choice(["warm", "still", "clear"] if is_day else ["dark", "cold", "still"])
        return r.choice([
            f"the {land} lay {adj_lay}.",
            f"the {land} was {adj_was}.",
            f"the {land} was wide and {'warm' if is_day else 'still'}.",
        ])
    if land == "wood":
        return r.choice([
            f"the wood lay {r.choice(['still', 'deep', 'warm'] if is_day else ['dark', 'grey', 'still'])}.",
            f"the wood was {r.choice(['still', 'deep', 'warm'] if is_day else ['dark', 'cold', 'still'])}.",
        ])
    if land in {"shore", "road", "path"}:
        adj = r.choice(["still", "warm", "clear"] if is_day else ["dark", "still", "cold"])
        adj_was = r.choice(["warm", "still", "clear"] if is_day else ["cold", "still", "dark"])
        return r.choice([
            f"the {land} lay {adj}.",
            f"the {land} was {adj_was}.",
        ])
    # hill
    adj = r.choice(["warm", "still", "clear"] if is_day else ["cold", "grey", "dark"])
    return r.choice([
        f"the {land} was {adj}.",
        f"the {land} lay {r.choice(['still', 'warm', 'clear'] if is_day else ['dark', 'still'])}.",
    ])


def describe_water(scene: dict, r: random.Random) -> str:
    water  = scene["water"]
    is_day = scene["time"] == "day"

    if water is None:
        land = scene["land"]
        return r.choice([
            f"the {land} was still.",
            f"the {land} lay still.",
            "the air grew warm." if is_day else "the air was deep and still.",
            "the air was soft and still." if is_day else "the air was cold and still.",
        ])
    time_key = "day" if is_day else ("night" if scene["time"] == "night" else "other")

    if water == "river":
        pool = WATER_STATES["river"][time_key]
        adv  = r.choice(pool)
        prep = r.choice(["through", "by", "near"])
        target = r.choice(["the field", "the wood", "the hill", "the road"])
        a, b = _pick_two(r, pool)
        adv2 = r.choice([x for x in pool if x != adv])
        return r.choice([
            f"the river ran {adv} {prep} {target}.",
            f"the river was {a} and {b}.",
            f"the sound of the river came through the {r.choice(['wood', 'field', 'hill'])}.",
            f"the river ran {adv} and {adv2}.",
        ])

    sea_pool = WATER_STATES[water][time_key]
    state_a, state_b = _pick_two(r, sea_pool)
    if water == "sea":
        non_wide = r.choice([x for x in sea_pool if x != "wide"])
        return r.choice([
            f"the sea lay {state_a} by the {scene['land']}.",
            f"the sea was {state_a} and {state_b}.",
            f"the sea lay {state_a} by the {scene['land']}.",
            f"the sea was wide and {non_wide}.",
        ])
    near_target = r.choice(["the shore", "the hill", "the road"])
    loc_land = scene["land"] if scene["land"] in {"hill", "shore", "wood", "field"} else "shore"
    loc = f"by the {loc_land}"
    return r.choice([
        f"the {water} lay {state_a} {loc}.",
        f"the {water} was {state_a} and {state_b}.",
        f"the {water} was {state_a} near {near_target}.",
        f"the sound of the water came through the {r.choice(['wood', 'field', 'hill'])}.",
    ])


def transition(scene: dict, r: random.Random) -> str:
    land   = scene["land"]
    water  = scene["water"]
    is_day = scene["time"] == "day"

    if scene["time"] in {"dawn", "morning", "day"}:
        options = [
            f"the sky grew {r.choice(['clear', 'wide'] if is_day else ['pale', 'clear'])}.",
            f"the light lay on the {land}.",
        ]
        if is_day:
            options += [
                "the air grew warm.",
                f"the light lay warm on the {land}.",
                f"the sky was clear and wide.",
            ]
        if water:
            options.append(f"the light fell on the {water}.")
        return r.choice(options)

    dark_target = water if water and r.random() < 0.4 else land
    return r.choice([
        f"the sky grew dark.",
        f"dark lay on the {dark_target}.",
        "the night grew cold.",
        f"the dark lay on the {land}.",
    ])


# ---------------------------------------------------------------------------
# Character / animal functions
# ---------------------------------------------------------------------------

def introduce_person(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    if person is None:
        animal = _get_animal(scene, r)
        if animal in {"bird", "crow"}:
            verb = r.choice(["stood", "passed", "rose", "fell"])
            if verb == "stood":
                place = _standing_place(scene, r)
            elif verb == "fell":
                place = r.choice([
                    f"over the {scene['land']}",
                    f"through the air",
                    f"over the {scene['water']}" if scene["water"] else f"over the {scene['land']}",
                ])
            else:
                place = _animal_place(scene, animal, r)
        else:
            verb = r.choice(["stood", "passed", "walked", "moved", "lay"])
            place = _standing_place(scene, r) if verb in {"stood", "lay"} else _animal_place(scene, animal, r)
        return f"the {animal} {verb} {place}."

    verb  = r.choice(["stood", "passed", "walked", "moved"])
    if verb == "stood":
        place = _standing_place(scene, r)
    elif verb == "passed":
        place = r.choice([_ground_track(r),
                          f"by the {scene['water']}" if scene["water"] else _ground_track(r)])
    else:
        place = _person_place(scene, r)
    return f"{person['subject']} {verb} {place}."


def person_motion(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    if person is None:
        animal = _get_animal(scene, r)
        if animal in {"bird", "crow"}:
            motion = r.choice(["passed", "rose", "moved", "fell"])
            if motion == "fell":
                dest = r.choice([
                    f"over the {scene['land']}",
                    "through the air",
                    f"over the {scene['water']}" if scene["water"] else f"over the {scene['land']}",
                ])
            else:
                dest = r.choice([
                    f"over the {scene['land']}",
                    "through the air",
                    f"over the {scene['water']}" if scene["water"] else f"over the {scene['land']}",
                ])
            return f"the {animal} {motion} {dest}."
        motion = r.choice(["walked", "moved", "ran"])
        speed  = ("slow " if motion == "walked" and r.random() < 0.4
                  else "swift " if motion == "ran" and r.random() < 0.4
                  else "")
        dest = r.choice([
            f"through the {r.choice(['field', 'wood', 'hill'])}",
            _ground_track(r),
            f"by the {scene['water']}" if scene["water"] else _open_place(scene),
        ])
        return f"the {animal} {motion} {speed}{dest}."

    pron   = person["pron"]
    motion = r.choice(["walked", "moved", "ran"])
    speed  = ("slow " if motion == "walked" and r.random() < 0.4
              else "swift " if motion == "ran" and r.random() < 0.4
              else "")
    land = scene["land"]
    dest_choices = [
        f"by the {r.choice(['road', 'path', 'shore'])}",
        f"by the {scene['water']}" if scene["water"] else f"near the {land}",
    ]
    if land in IN_LANDS:
        dest_choices.append(f"through the {land}")
    dest = r.choice(dest_choices)
    return f"{pron} {motion} {speed}{dest}."


def person_perception(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    if person is None:
        animal = _get_animal(scene, r)
        return r.choice([
            f"the {animal} stood and did not move.",
            f"the {animal} was still.",
        ])

    pron   = person["pron"]
    target = r.choice([
        "the water",
        "the road",
        "the hill",
        "the fire" if scene["fire"] else "the light",
        f"the {scene['water']}" if scene["water"] else "the shore",
    ])
    return r.choice([
        f"{pron} stood near {target}.",
        f"{pron} was still near {target}.",
        f"{pron} did not speak.",
    ])


def person_state(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    is_day = scene["time"] == "day"

    if person is None:
        animal = _get_animal(scene, r)
        water  = scene["water"]
        land   = scene["land"]
        return r.choice([
            f"the {animal} was still near the {water if water else land}.",
            f"the {animal} did not move near the {water if water else land}.",
            f"the {animal} stood near the {land}.",
            f"the {animal} lay still by the {water if water else land}.",
            f"the {animal} lay near the {land}.",
        ])

    pron    = person["pron"]
    options = [
        f"{pron} was {'warm' if is_day else 'cold'}.",
        f"{pron} was still.",
        f"{pron} was still and {'warm' if is_day else 'cold'}.",
        f"{pron} did not {r.choice(['speak', 'move'])}.",
        f"{pron} stood in the {'warm' if is_day else 'cold'} air.",
        f"{pron} stood near the {scene['water'] if scene['water'] else scene['land']}.",
    ]
    if scene["fire"]:
        options.append(f"{pron} stood near the fire.")
        options.append(f"{pron} was warm near the fire.")
        options.append(f"{pron} did not move near the fire.")
    return r.choice(options)


def person_place_state(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    if person is None:
        animal = _get_animal(scene, r)
        return f"the {animal} was still."
    pron = person["pron"]
    return f"{pron} was {'by the ' + scene['water'] if scene['water'] else 'near the ' + scene['land']}."


def person_simple_action(scene: dict, r: random.Random) -> str:
    person = scene["person"]
    if person is None:
        animal = _get_animal(scene, r)
        return r.choice([
            f"the {animal} moved {_open_place(scene)}.",
            f"the {animal} passed {_animal_place(scene, animal, r)}.",
            f"the {animal} walked {_open_place(scene)}.",
            f"the {animal} did not move.",
        ])
    pron    = person["pron"]
    is_day  = scene["time"] == "day"
    options = [
        f"{pron} stood {_open_place(scene)}.",
        f"{pron} walked by the {r.choice(['road', 'path', 'shore'])}.",
        f"{pron} moved by the {r.choice(['road', 'path', 'shore'])}.",
        f"{pron} did not {r.choice(['move', 'speak'])}.",
        f"{pron} was {'warm' if is_day else 'still'}.",
    ]
    if scene["fire"]:
        options.append(f"{pron} stood near the fire.")
    return r.choice(options)


def light_on(scene: dict, r: random.Random) -> str:
    is_day  = scene["time"] == "day"
    color   = r.choice(["warm", "soft", "clear"] if is_day else ["pale", "cold", "soft"])
    person  = scene["person"]
    animal  = scene.get("animal") or _get_animal(scene, r)
    land    = scene["land"]
    water   = scene["water"]

    entity_opts = []
    if person:
        subj = person["subject"]
        entity_opts += [
            f"the light fell {color} on {subj}.",
            f"the light lay {color} on {subj}.",
        ]
    entity_opts += [
        f"the light fell {color} on the {animal}.",
        f"the light lay {color} on the {animal}.",
        f"the light fell {color} on the {land}.",
        f"the light lay {color} on the {land}.",
    ]
    if water:
        entity_opts.append(f"the light fell {color} on the {water}.")
    return r.choice(entity_opts)


def fire_sentence(scene: dict, r: random.Random) -> str:
    if not scene["fire"]:
        land   = scene["land"]
        water  = scene["water"]
        is_day = scene["time"] == "day"
        opts = [
            "the air was still." if is_day else "the air was cold and still.",
            "the air grew warm." if is_day else "the air grew cold.",
            f"the sound of the wind came through the {land}.",
        ]
        if water:
            opts.append(f"the sound of the {water} came through the {land}.")
            opts.append(f"the {water} was still.")
        return r.choice(opts)
    land    = scene["land"]
    water   = scene["water"]
    options = [
        "the fire burned low.",
        "the fire was warm in the dark.",
        "the light of the fire moved on the stone.",
        "the fire burned slow.",
        "the fire was still and warm.",
        f"the light of the fire fell on the {land}.",
        f"the fire burned low by the {r.choice(['shore', 'road', 'path'])}.",
    ]
    if water:
        options.append(f"the fire burned low by the {water}.")
    return r.choice(options)


# ---------------------------------------------------------------------------
# Plan dispatch
# ---------------------------------------------------------------------------

ROLE_FUNCS = {
    "time":         describe_time,
    "sky":          describe_sky,
    "weather":      describe_weather,
    "land":         describe_land,
    "water":        describe_water,
    "transition":   transition,
    "intro":        introduce_person,
    "motion":       person_motion,
    "perception":   person_perception,
    "state":        person_state,
    "place_state":  person_place_state,
    "simple_action":person_simple_action,
    "fire":         fire_sentence,
    "light_on":     light_on,
}


PLANS = [
    # Pure scene descriptions
    ["time", "sky", "weather", "water"],
    ["time", "weather", "water", "transition"],
    ["time", "sky", "water", "transition"],
    ["time", "sky", "weather", "transition"],
    ["time", "sky", "land", "transition"],
    # Character with state
    ["time", "sky", "intro", "state"],
    ["time", "water", "intro", "state"],
    ["time", "sky", "intro", "simple_action", "state"],
    ["time", "weather", "intro", "state"],
    ["time", "weather", "water", "intro", "state"],
    # Character with movement
    ["time", "weather", "intro", "motion", "state"],
    ["time", "sky", "land", "intro", "motion"],
    ["time", "water", "intro", "motion", "state"],
    # Fire scenes
    ["time", "sky", "fire", "intro", "state"],
    ["time", "fire", "intro", "state"],
    ["time", "sky", "fire", "water", "transition"],
    ["time", "sky", "fire", "intro", "motion"],
    # Light on entity
    ["time", "sky", "intro", "light_on", "state"],
    ["time", "sky", "light_on", "intro", "state"],
    ["time", "weather", "intro", "light_on", "state"],
]


def realize_plan(scene: dict, plan: list[str], r: random.Random) -> list[str]:
    sentences = []
    for role in plan:
        sentence = ROLE_FUNCS[role](scene, r)
        sentences.append(_cap(sentence))
    return sentences


def generate_paragraph(r: random.Random) -> str:
    scene = build_scene(r)
    plan  = r.choice(PLANS)
    return " ".join(realize_plan(scene, plan, r))


def generate_corpus(n_words: int, seed: int = 42) -> str:
    r          = random.Random(seed)
    paragraphs = []
    word_count = 0

    while word_count < n_words:
        para = generate_paragraph(r)
        paragraphs.append(para)
        word_count += len(para.split())

    return "\n\n".join(paragraphs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene-based thematic corpus for TILM2"
    )
    parser.add_argument("--words",  type=int, default=60_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default="corpus/thematic_v4.txt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Generating ~{args.words:,} words...")
    text   = generate_corpus(args.words, seed=args.seed)
    actual = len(text.split())

    with open(args.output, "w") as f:
        f.write(text)

    print(f"Written {actual:,} words to {args.output}")
    print("\nSample:\n")
    for para in text.split("\n\n")[:8]:
        print(" ", para)
        print()


if __name__ == "__main__":
    main()
