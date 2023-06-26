#!/usr/bin/env python3

#### PYTHON IMPORTS ################################################################################
import sys


#### PACKAGE IMPORTS ###############################################################################
from sentence_transformers import SentenceTransformer, util


#### GLOBALS #######################################################################################
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
CATEGORY_DESCRIPTIONS = [
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
def predictMistakes(text):
    # Setup best mistakes model
    model = SentenceTransformers("paraphrase-multilingual-mpnet-base-v2")
    model.max_seq_length = 128

    # Setup data and labels
    labels = ["NOPE", "NOPE", "MISTAKE"]




#### MAIN ##########################################################################################
def main():
    args = sys.argv[1:]


if __name__ == "__main__":
    main()
