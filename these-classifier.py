#!/usr/bin/env python3

#### PYTHON IMPORTS ################################################################################
import os
import re
import sys


#### PACKAGE IMPORTS ###############################################################################
from sentence_transformers import SentenceTransformer, util


#### GLOBALS #######################################################################################
RE_PUNCT = re.compile(r"[\.,:;?!]")
RE_WHITESPACE = re.compile(r"[\n\r\t\v\f]") # everything but regular spaces
RE_DUPLICATE_SPACES = re.compile(r"[\s]+")

TYPE_DESCRIPTIONS = [
    # Slip
    "Typos and misspellings may occur in code comments, documentation (and other development artifacts), or when typing the name of a variable, function, or class. Examples include misspelling a variable name, writing down the wrong number/name/word during requirements elicitation, referencing the wrong function in a code comment, and inconsistent whitespace (that does not result in a syntax error). Any error in coding language syntax that impacts the executability of the code. Note that Logical Errors (e.g. += instead of +) are not Syntax Errors. Examples include mixing tabs and spaces (e.g. Python), unmatched brackets/braces/parenthesis/quotes, and missing semicolons (e.g. Java). Errors resulting from overlooking (internally and externally) documented information, such as project descriptions, stakeholder requirements, API/library/tool/framework documentation, coding standards, programming language specifications, bug/issue reports, and looking at the wrong version of documentation or documentation for the wrong project/software. Errors resulting from multitasking. Attention failures while using computer peripherals, such as mice, keyboard, and cables. Examples include copy/paste errors, clicking the wrong button, using the wrong keyboard shortcut, and incorrectly plugging in cables. Errors resulting from lack of attention during formal/informal code review. Errors resulting from overlooking existing functionality, such as reimplementing variables, functions, and classes that already exist, or reimplementing functionality that already exists in a standard library. Only use this category if you believe your error to be the result of a lack of attention, but no other slip category fits.",
    # Lapse
    "Forgetting to finish a development task. Examples include forgetting to implement a required feature, forgetting to finish a user story, and forgetting to deploy a security patch. Forgetting to fix a defect that you encountered, but chose not to fix right away. Forgetting to remove debug log files, dead code, informal test code, commented out code, test databases, backdoors, etc. Examples include leaving unnecessary code in the comments, and leaving notes in internal development documentation. Forgetting to git-pull (or equivalent in other version control systems), or using an outdated version of a library. Forgetting to import a necessary library, class, variable, or function, or forgetting to access a property, attribute, or argument. Forgetting to push code, or forgetting to backup/save data or documentation. Errors resulting from forgetting details from previous development discussions. Only use this category if you believe your error to be the result of a memory failure, but no other lapse category fits.",
    # Mistake
    "A code logic error is one in which the code executes (i.e. actually runs), but produces an incorrect output/behavior due to incorrect logic. Examples include using incorrect operators (e.g. += instead of +), erroneous if/else statements, incorrect variable initializations, problems with variable scope, and omission of necessary logic. Errors resulting from incomplete knowledge of the software system's target domain (e.g. banking, astrophysics). Errors resulting from an incorrect assumption about system requirements, stakeholder expectations, project environments (e.g. coding languages and frameworks), library functionality, and program inputs. Errors resulting from inadequate communication between development team members. Errors resulting from inadequate communication with project stakeholders or third-party contractors. Misunderstood problem-solving methods/techniques result in analyzing the problem incorrectly and choosing the wrong solution. For example, choosing to implement a database system in Python rather than using SQL. Overconfidence in a solution choice also falls under this category. Errors resulting from a lack of time management, such as failing to allocate enough time for the implementation of a feature or procrastination. Failure to implement necessary test cases, failure to consider necessary test inputs, or failure to implement a certain type of testing (e.g. unit, penetration, integration) when it is necessary. Errors in configuration of libraries/frameworks/environments or errors related to missing configuration options. Examples include misconfigured IDEs, improper directory structure for a specific programming language, and missing SSH keys. Errors resulting from misunderstood code due to poor documentation or unnecessary complexity. Examples include too many nested if/else statements or for-loops and poorly named variables/functions/classes/files Errors related to internationalization and/or string/character encoding. Errors resulting from inadequate experience/unfamiliarity with a language, library, framework, or tool. Errors resulting from not having sufficient access to necessary tooling. Examples include not having access to a specific operating system, library, framework, hardware device, or not having the necessary permissions to complete a development task. Errors resulting from working out of order, such as implementing dependent features in the wrong order, implementing code before the design is stabilized, releasing code that is not ready to be released, or skipping a workflow step. Only use this category if you believe your error to be the result of a planning failure, but no other mistake category fits."
]
TYPE_LABELS = [
    "Slip",
    "Lapse",
    "Mistake"
]

CATEGORY_DESCRIPTIONS = [
   # S01
    "Typos and misspellings may occur in code comments, documentation (and other development artifacts), or when typing the name of a variable, function, or class. Examples include misspelling a variable name, writing down the wrong number/name/word during requirements elicitation, referencing the wrong function in a code comment, and inconsistent whitespace (that does not result in a syntax error).",
    # S02
    "Any error in coding language syntax that impacts the executability of the code. Note that Logical Errors (e.g. += instead of +) are not Syntax Errors. Examples include mixing tabs and spaces (e.g. Python), unmatched brackets/braces/parenthesis/quotes, and missing semicolons (e.g. Java).",
    # S03
    "Errors resulting from overlooking (internally and externally) documented information, such as project descriptions, stakeholder requirements, API/library/tool/framework documentation, coding standards, programming language specifications, bug/issue reports, and looking at the wrong version of documentation or documentation for the wrong project/software.",
    # S04
    "Errors resulting from multitasking.",
    # S05
    "Attention failures while using computer peripherals, such as mice, keyboard, and cables. Examples include copy/paste errors, clicking the wrong button, using the wrong keyboard shortcut, and incorrectly plugging in cables.",
    # S06
    "Errors resulting from lack of attention during formal/informal code review.",
    # S07
    "Errors resulting from overlooking existing functionality, such as reimplementing variables, functions, and classes that already exist, or reimplementing functionality that already exists in a standard library.",
    # S08
    "Only use this category if you believe your error to be the result of a lack of attention, but no other slip category fits.",
    # L01
    "Forgetting to finish a development task. Examples include forgetting to implement a required feature, forgetting to finish a user story, and forgetting to deploy a security patch.",
    # L02
    "Forgetting to fix a defect that you encountered, but chose not to fix right away.",
    # L03
    "Forgetting to remove debug log files, dead code, informal test code, commented out code, test databases, backdoors, etc. Examples include leaving unnecessary code in the comments, and leaving notes in internal development documentation.",
    # L04
    "Forgetting to git-pull (or equivalent in other version control systems), or using an outdated version of a library.",
    # L05
    "Forgetting to import a necessary library, class, variable, or function, or forgetting to access a property, attribute, or argument.",
    # L06
    "Forgetting to push code, or forgetting to backup/save data or documentation.",
    # L07
    "Errors resulting from forgetting details from previous development discussions.",
    # L08
    "Only use this category if you believe your error to be the result of a memory failure, but no other lapse category fits.",
    # M01
    "A code logic error is one in which the code executes (i.e. actually runs), but produces an incorrect output/behavior due to incorrect logic. Examples include using incorrect operators (e.g. += instead of +), erroneous if/else statements, incorrect variable initializations, problems with variable scope, and omission of necessary logic. ",
    # M02
    "Errors resulting from incomplete knowledge of the software system's target domain (e.g. banking, astrophysics).",
    # M03
    "Errors resulting from an incorrect assumption about system requirements, stakeholder expectations, project environments (e.g. coding languages and frameworks), library functionality, and program inputs.",
    # M04
    "Errors resulting from inadequate communication between development team members.",
    # M05
    "Errors resulting from inadequate communication with project stakeholders or third-party contractors.",
    # M06
    "Misunderstood problem-solving methods/techniques result in analyzing the problem incorrectly and choosing the wrong solution. For example, choosing to implement a database system in Python rather than using SQL. Overconfidence in a solution choice also falls under this category.",
    # M07
    "Errors resulting from a lack of time management, such as failing to allocate enough time for the implementation of a feature or procrastination.",
    # M08
    "Failure to implement necessary test cases, failure to consider necessary test inputs, or failure to implement a certain type of testing (e.g. unit, penetration, integration) when it is necessary.",
    # M09
    "Errors in configuration of libraries/frameworks/environments or errors related to missing configuration options. Examples include misconfigured IDEs, improper directory structure for a specific programming language, and missing SSH keys.",
    # M10
    "Errors resulting from misunderstood code due to poor documentation or unnecessary complexity. Examples include too many nested if/else statements or for-loops and poorly named variables/functions/classes/files.",
    # M11
    "Errors related to internationalization and/or string/character encoding.",
    # M12
    "Errors resulting from inadequate experience/unfamiliarity with a language, library, framework, or tool.",
    # M13
    "Errors resulting from not having sufficient access to necessary tooling. Examples include not having access to a specific operating system, library, framework, hardware device, or not having the necessary permissions to complete a development task.",
    # M14
    "Errors resulting from working out of order, such as implementing dependent features in the wrong order, implementing code before the design is stabilized, releasing code that is not ready to be released, or skipping a workflow step.",
    # M15
    "Only use this category if you believe your error to be the result of a planning failure, but no other mistake category fits."
]
CATEGORY_LABELS = [
    "S01: Typos & Mispellings",
    "S02: Syntax Errors",
    "S03: Overlooking Documented Information",
    "S04: Multitasking Errors",
    "S05: Hardware Interaction Errors",
    "S06: Overlooking Proposed Code Changes",
    "S07: Overlooking Existing Functionality",
    "S08: General Attentional Failure",
    "L01: Forgetting to Finish a Development Task",
    "L02: Forgetting to Fix a Defect",
    "L03: Forgetting to Remove Development Artifacts",
    "L04: Working With Outdated Source Code",
    "L05: Forgetting an Import Statement",
    "L06: Forgetting to Save Work",
    "L07: Forgetting Previous Development Discussion",
    "L08: General Memory Failure",
    "M01: Code Logic Errors",
    "M02: Incomplete Domain Knowledge",
    "M03: Wrong Assumption Errors",
    "M04: Internal Communication Errors",
    "M05: External Communication Errors",
    "M06: Solution Choice Errors",
    "M07: Time Management Errors",
    "M08: Inadequate Testing",
    "M09: Incorrect/Insufficient Configuration",
    "M10: Code Complexity Errors",
    "M11: Internationalization/String Encoding Errors",
    "M12: Inadequate Experience Errors",
    "M13: Insufficient Tooling Access Errors",
    "M14: Workflow Order Errors",
    "M15: General Planning Failure"
]


#### CLASSES #######################################################################################


#### FUNCTIONS #####################################################################################
def _maxCosine(cosine_scores):
    max_index = -999
    max_value = -999

    for i in range(0, len(cosine_scores)):
        if cosine_scores[i] >= max_value:
            max_value = cosine_scores[i]
            max_index = i

    return max_index


def _computeEmbeddings(model, descriptions, start=0, stop=None):
    if stop is None:
        stop = len(descriptions)
    embeddings = [
        model.encode(descr, convert_to_tensor=True) for descr in descriptions[start:stop]
    ]
    return embeddings


def _computeCosineScores(text_emb, target_embs):
    return [ util.cos_sim(text_emb, target_emb) for target_emb in target_embs ]


def _makePrediction(cosine_scores, labels):
    max_index = _maxCosine(cosine_scores)
    label = labels[max_index]
    return label, max_index


def _predict(model, text, start, stop, cleantext):
    #### Step 0: Setup Labels
    type_labels = TYPE_LABELS
    category_labels = CATEGORY_LABELS[start:stop]

    type_descriptions = list()
    category_descriptions = list()
    #### Step 1: Clean Text
    if cleantext == "clean":
        text = _cleanText(text)
        type_descriptions = [ _cleanText(descr) for descr in TYPE_DESCRIPTIONS ]
        category_descriptions = [ _cleanText(descr) for descr in CATEGORY_DESCRIPTIONS ]
    else:
        type_descriptions = TYPE_DESCRIPTIONS.copy()
        category_descriptions = CATEGORY_DESCRIPTIONS.copy()

    #### Step 2: Compute Embeddings
    text_embedding = model.encode(text, convert_to_tensor=True)
    type_embeddings = _computeEmbeddings(model, type_descriptions)
    category_embeddings = _computeEmbeddings(model, category_descriptions, start, stop)

    #### Step 3: Compute Cosine Similarities
    type_cosine_scores = _computeCosineScores(text_embedding, type_embeddings)
    category_cosine_scores = _computeCosineScores(text_embedding, category_embeddings)

    #### Step 4: Make Predictions
    type_label, type_index = _makePrediction(type_cosine_scores, type_labels)
    category_label, category_index = _makePrediction(category_cosine_scores, category_labels)

    #### Step 5: Results
    results = [
        [type_label, type_cosine_scores[type_index]],            # Type w/ Cosine
        [category_label, category_cosine_scores[category_index]] # Category w/ Cosine
    ]

    return results


def _cleanText(text):
    clean_text = RE_PUNCT.sub(" ", text.lower())
    clean_text = RE_WHITESPACE.sub(" ", clean_text)
    clean_text = RE_DUPLICATE_SPACES.sub(" ", clean_text)
    return clean_text


def _makeModels(cleantext, metric):
    best_precision_dirty = [
        ["paraphrase-multilingual-mpnet-base-v2", 128],  # Slips: 0.518
        ["distiluse-base-multilingual-cased-v1", 128],   # Lapses: 0.236
        ["paraphrase-multilingual-mpnet-base-v2", 128]   # Mistakes: 0.825
    ]
    best_recall_dirty = [
        ["all-mpnet-base-v2", 384],                      # Slips: 0.711
        ["jhgan/ko-sroberta-multitask", 512],            # Lapses: 0.915
        ["nikcheerla/nooks-amd-detection-realtime", 512] # Mistakes: 0.603
    ]
    best_f1_dirty = [
        ["paraphrase-multilingual-mpnet-base-v2", 128],  # Slips: 0.484
        ["distiluse-base-multilingual-cased-v1", 128],   # Lapses: 0.341
        ["nikcheerla/nooks-amd-detection-realtime", 512] # Mistakes: 0.655
    ]
    best_precision_clean = [
        ["paraphrase-MiniLM-L3-v2", 128],                # Slips: 0.531
        ["distiluse-base-multilingual-cased-v1", 128],   # Lapses: 0.224
        ["nikcheerla/nooks-amd-detection-v2-full", 512]  # Mistakes: 0.816
    ]
    best_recall_clean = [
        ["multi-qa-distilbert-cos-v1", 512],             # Slips: 0.701
        ["nikcheerla/nooks-amd-detection-v2-full", 512], # Lapses: 0.851
        ["paraphrase-albert-small-v2", 256]              # Mistakes: 0.362
    ]
    best_f1_clean = [
        ["all-mpnet-base-v2", 384],                      # Slips: 0.466
        ["distiluse-base-multilingual-cased-v2", 128],   # Lapses: 0.332 
        ["paraphrase-albert-small-v2", 256]              # Mistakes: 0.474
    ]

    models_dict = {
        "dirty": {
            "precision": best_precision_dirty,
            "recall": best_recall_dirty,
            "f1": best_f1_dirty
        },
        "clean": {
            "precision": best_precision_clean,
            "recall": best_recall_clean,
            "f1": best_f1_clean
        }
    }

    best_models = models_dict[cleantext][metric]
    
    slips_model = SentenceTransformer(best_models[0][0])
    slips_model.seq_length = best_models[0][1]
    lapses_model = SentenceTransformer(best_models[1][0])
    lapses_model.seq_length = best_models[1][1]
    mistakes_model = SentenceTransformer(best_models[2][0])
    mistakes_model.seq_length = best_models[2][1]

    return slips_model, lapses_model, mistakes_model


def _readFile(text_file):
    if os.path.exists(text_file):
        with open(text_file, "r") as f:
            data = f.read().replace("\n", " ")
        return data
    else:
        return "NO_TEXT_TO_CLASSIFY"


def predict(cleantext, metric, text_file):
    # Models
    slips_model, lapses_model, mistakes_model = _makeModels(cleantext, metric)

    text = _readFile(text_file)

    # Predictions
    slip_predictions = _predict(slips_model, text, 0, 8, cleantext)
    lapse_predictions = _predict(lapses_model, text, 8, 16, cleantext)
    mistake_predictions = _predict(mistakes_model, text, 16, 31, cleantext)

    # Type Prediction Logic
    type_preds = [ slip_predictions[0][0], lapse_predictions[0][0], mistake_predictions[0][0] ]
    counts = [ type_preds.count("Slip"), type_preds.count("Lapse"), type_preds.count("Mistake") ]
    # If Slips greater than Lapses and Mistakes
    if counts[0] > counts[1] and counts[0] > counts[2]:
        print("{}, {}".format(slip_predictions[0][0], slip_predictions[1][0]))
    # If Lapses greater than Slips and Mistakes
    elif counts[1] > counts[0] and counts[1] > counts[2]:
        print("{}, {}".format(lapse_predictions[0][0], lapse_predictions[1][0]))
    # If Mistakes greater than Slips and Lapses
    elif counts[2] > counts[0] and counts[2] > counts[1]:
        print("{}, {}".format(mistake_predictions[0][0], mistake_predictions[1][0]))
    # If they all occur equally, take the one with highest cosine similarity
    else:
        type_pred, max_index = _makePrediction(counts, TYPE_LABELS)
        if type_pred == "Slip":
            print("{}, {}".format(type_pred, slip_predictions[1][0]))
        elif type_pred == "Lapse":
            print("{}, {}".format(type_pred, lapse_predictions[1][0]))
        elif type_pred == "Mistake":
            print("{}, {}".format(type_pred, mistake_predictions[1][0]))


#### MAIN ##########################################################################################
def main():
    args = sys.argv[1:]
    #print(args)
    cleantext = args[0] # Whether or not to clean up text; "clean" or "dirty"
    metric = args[1] # What metric to use for "best"; "precision" or "recall" or "f1"
    #text = args[2] # Text to classify
    text = args[2] # Filename with to classify
    predict(cleantext, metric, text)


if __name__ == "__main__":
    main()
