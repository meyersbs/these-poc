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
    "Typos and misspellings may occur in code comments, documentation (and other development artifacts), or when typing the name of a variable, function, or class. Examples include misspelling a variable name, writing down the wrong number/name/word during requirements elicitation, referencing the wrong function in a code comment, and inconsistent whitespace (that does not result in a syntax error). Any error in coding language syntax that impacts the executability of the code. Note that Logical Errors (e.g. += instead of +) are not Syntax Errors. Examples include mixing tabs and spaces (e.g. Python), unmatched brackets/braces/parenthesis/quotes, and missing semicolons (e.g. Java). Inappropriate Whitespace Style. The source code contains whitespace that is inconsistent across the code or does not follow expected standards for the product. Errors resulting from overlooking (internally and externally) documented information, such as project descriptions, stakeholder requirements, API/library/tool/framework documentation, coding standards, programming language specifications, bug/issue reports, and looking at the wrong version of documentation or documentation for the wrong project/software. Use of Obsolete Function. The code uses deprecated or obsolete functions, which suggests that the code has not been actively reviewed or maintained. Inconsistency Between Implementation and Documented Design. The implementation of the product is not consistent with the design as described within the relevant documentation. Errors resulting from multitasking, i.e. working on multiple software engineering tasks at the same time. Attention failures while using computer peripherals, such as mice, keyboard, and cables. Examples include copy/paste errors, clicking the wrong button, using the wrong keyboard shortcut, and incorrectly plugging in cables. Errors resulting from lack of attention during formal/informal code review. Examples include overlooking incorrect logic, or skipping files, functions, or classes during a review. Errors resulting from overlooking existing functionality, such as reimplementing or duplicating variables, functions, and classes that already exist, or reimplementing functionality that already exists in a standard library. Other examples include deleting necessary variables, functions, and classes. Use of Redundant Code. The product has multiple functions, methods, procedures, macros, etc. that contain the same code. Only use this category if you believe your error to be the result of a lack of attention, but no other slip category fits. Clerical Errors. Result from carelessness while performing mechanical transcriptions from one format or from one medium to another. Requirement examples include carelessness while documenting specifications from elicited user needs. Term Substitution Errors. After introducing a term correctly, the requirement author substitutes a different term that refers to a different concept.",
    # Lapse
    "Forgetting to finish a development task. Examples include forgetting to implement a required feature, forgetting to finish a user story, and forgetting to deploy a security patch. Missing Synchronization. The product utilizes a shared resource in a concurrent manner but does not attempt to synchronize access to the resource. Missing Cryptographic Step. The product does not implement a required step in a cryptographic algorithm, resulting in weaker encryption than advertised by the algorithm. Forgetting to fix a defect that you encountered, but chose not to fix right away. Forgetting to remove debug log files, dead code, informal test code, commented out code, test databases, backdoors, etc. Examples include leaving unnecessary code in the comments, and leaving notes in internal development documentation. Improper Output Neutralization for Logs. The product does not neutralize or incorrectly neutralizes output that is written to logs. Active Debug Code. The product is deployed to unauthorized actors with debugging code still enabled or active, which can create unintended entry points or expose sensitive information. Dead Code. The product contains dead code, which can never be executed. Irrelevant Code. The product contains code that is not essential for execution, i.e. makes no state changes and has no side effects that alter data or control flow, such that removal of the code would have no impact to functionality or correctness. Forgetting to git-pull (or equivalent in other version control systems), or using an outdated version of a library. Forgetting to import a necessary library, class, variable, or function, or forgetting to access a property, attribute, or argument. Examples include forgetting to import python's sys library, forgetting to include a header file in C, or forgetting to pass an argument to a function. Forgetting to push code, or forgetting to backup/save data or documentation. Errors resulting from forgetting details from previous development discussions. Only use this category if you believe your error to be the result of a memory failure, but no other lapse category fits. Loss of Information from Stakeholders. Result from a requirement author forgetting, discarding, or failing to store information or documents provided by stakeholders, e.g. some important user need. Accidentally Overlooking Requirements. Occur when the stakeholders who are the source of requirements assume that some requirements are obvious and fail to verbalize them. Multiple Terms for the Same Concept. Occur when requirement authors fail to realize they have already defined a term for a concept and so introduce a new term at a later time.",
    # Mistake
    "A code logic error is one in which the code executes (i.e. actually runs), but produces an incorrect output/behavior due to incorrect logic. Examples include using incorrect operators (e.g. += instead of +), erroneous if/else statements, incorrect variable initializations, problems with variable scope, and omission of necessary logic.  Improper Handling of Insufficient Permissions or Privileges. The product does not handle or incorrectly handles when it has insufficient privileges to access resources or functionality as specified by their permissions. This may cause it to follow unexpected code paths that may leave the product in an invalid state. Missing Default Case in Multiple Condition Expression. The code does not have a default case in an expression with multiple conditions, such as a switch statement. Function Call with Incorrectly Specified Arguments. The product calls a function, procedure, or routine with arguments that are not correctly specified, leading to always-incorrect behavior and resultant weaknesses. Use of Incorrect Operator. The product accidentally uses the wrong operator, which changes the logic in security-relevant ways. Omitted Break Statement in Switch. The product omits a break statement within a switch or similar construct, causing code associated with multiple conditions to execute. This can cause problems when the programmer only intended to execute code associated with one condition. Operator Precedence Logic Error. The product uses an expression in which operator precedence causes incorrect logic to be used. Loop with Unreachable Exit Condition ('Infinite Loop'). The product contains an iteration or loop with an exit condition that cannot be reached, i.e., an infinite loop. Comparison Using Wrong Factors. The code performs a comparison between two entities, but the comparison examines the wrong factors or characteristics of the entities, which can lead to incorrect results and resultant weaknesses. Time-of-check Time-of-use (TOCTOU) Race Condition. The product checks the state of a resource before using that resource, but the resource's state can change between the check and the use in a way that invalidates the results of the check. This can cause the product to perform invalid actions when the resource is in an unexpected state. Wrap-around Error. Wrap around errors occur whenever a value is incremented past the maximum value for its type and therefore 'wraps around' to a very small, negative, or undefined value. Divide by Zero. The product divides a value by zero. Errors resulting from incomplete knowledge of the software system's target domain (e.g. banking, astrophysics). Examples include planning/designing a system without understanding the nuances of the domain. Errors resulting from an incorrect assumption about system requirements, stakeholder expectations, project environments (e.g. coding languages and frameworks), library functionality, and program inputs. Improper Input Validation. The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly. Errors resulting from inadequate communication between development team members. Examples include misunderstanding development discussion, misinterpretting or providing ambiguous instructions, communicating using the wrong medium (e.g. oral vs. written), or communicating ineffectively (e.g. too formal/informal, too much unecessarilly complex language, hostile language/body language). Errors resulting from inadequate communication with project stakeholders or third-party contractors. Examples include providing ambiguous or unclear directions to third-parties or users, or misinterpretting stakeholder feedback, communicating using the wrong medium (e.g. oral vs. written), or communicating ineffectively (e.g. too formal/informal, too much unecessarilly complex language, hostile language/body language). Misunderstood problem-solving methods/techniques result in analyzing the problem incorrectly and choosing the wrong solution. For example, choosing to implement a database system in Python rather than using SQL, or choosing the wrong software design pattern. Overconfidence in a solution choice also falls under this category. Use of Single-factor Authentication. The use of single-factor authentication can lead to unnecessary risk of compromise when compared with the benefits of a dual-factor authentication scheme. Use of Client-Side Authentication. A client/server product performs authentication within client code but not in server code, allowing server-side authentication to be bypassed via a modified client that omits the authentication check. Use of Hard-coded, Security-relevant Constants. The product uses hard-coded constants instead of symbolic names for security-critical values, which increases the likelihood of mistakes during code maintenance or security policy change. Use of Unmaintained Third Party Components. The product relies on third-party components that are not actively supported or maintained by the original developer or a trusted proxy for the original developer. Use of Hard-coded Credentials. The product contains hard-coded credentials, such as a password or cryptographic key, which it uses for its own inbound authentication, outbound communication to external components, or encryption of internal data. Errors resulting from a lack of time management, such as failing to allocate enough time for the implementation of a feature, procrastinating a development task, or predicting the time required for a task incorrectly. Failure to implement necessary test cases, failure to consider necessary test inputs, failure to implement a certain type of testing (e.g. unit, penetration, integration) when it is necessary, or failure to consider edge cases or unexpected inputs. Improper Check for Unusual or Exceptional Conditions. The product does not check or incorrectly checks for unusual or exceptional conditions that are not expected to occur frequently during day to day operation of the product. Errors in configuration of libraries/frameworks/environments or errors related to missing configuration options. Examples include misconfigured IDEs or text editors, improper directory structure for a specific programming language, missing SSH keys, missing or incorrectly named database fields or tables, missing or incorrectly named/formatted configuration files, or not installing a required library. Incorrect Privilege Assignment. A product incorrectly assigns a privilege to a particular actor, creating an unintended sphere of control for that actor. Insufficient Granularity of Access Control. The product implements access controls via a policy or other feature with the intention to disable or restrict accesses (reads and/or writes) to assets in a system from untrusted agents. However, implemented access controls lack required granularity, which renders the control policy too broad because it allows accesses from unauthorized agents to the security-sensitive assets. Use of Default Credentials. The product uses default credentials (such as passwords or cryptographic keys) for potentially critical functionality. Errors resulting from misunderstood code due to poor documentation or unnecessary complexity. Examples include too many nested if/else statements or for-loops and poorly named variables/functions/classes/files. Inconsistent Naming Conventions for Identifiers. The product's code, documentation, or other artifacts do not consistently use the same naming conventions for variables, callables, groups of related callables, I/O capabilities, data types, file names, or similar types of elements. Use of Same Variable for Multiple Purposes. The code contains a callable, block, or other code element in which the same variable is used to control more than one unique task or store more than one instance of data. Class with Excessively Deep Inheritance. A class has an inheritance level that is too high, i.e., it has a large number of parent classes. Source Code File with Excessive Number of Lines of Code. A source code file has too many lines of code. Excessively Deep Nesting. The code contains a callable or other code grouping in which the nesting / branching is too deep. Errors related to internationalization and/or string/character encoding. Examples include using ASCII intead of unicode, using UTF8 when UTF16 was necessary, failure to design the system with internationalization in mind, or failing to verify the character length of user input. Improper Handling of Unicode Encoding. The product does not properly handle when an input contains Unicode encoding. Improper Handling of Alternate Encoding. The product does not properly handle when an input uses an alternate encoding that is valid for the control sphere to which the input is being sent. Errors resulting from inadequate experience/unfamiliarity with a language, library, framework, or tool. Errors resulting from not having sufficient access to necessary tooling. Examples include not having access to a specific operating system, library, framework, hardware device, or not having the necessary permissions to complete a development task. Errors resulting from working out of order, such as implementing dependent features in the wrong order, implementing code before the design is stabilized, releasing code that is not ready to be released, or skipping a workflow step. Only use this category if you believe your error to be the result of a planning failure, but no other mistake category fits. Application. Arise from a misunderstanding of the application or problem domain or a misunderstanding of some aspect of overall system functionality. Environment. Result from lack of knowledge about the available infrastructure (e.g., tools, templates) that supports the elicitation, understanding, or documentation of software requirements. Solution Choice. These errors occur in the process of finding a solution for a stated and well-understood problem. If RE analysts do not understand the correct use of problem-solving methods and techniques, they might end up analyzing the problem incorrectly, and choose the wrong solution. Syntax. Occur when a requirement author misunderstands the grammatical rules of natural language or the rules, symbols, or standards in a formal specification language like UML. Information Management. Result from a lack of knowledge about standard requirements engineering or documentation practices and procedures within the organization. Wrong Assumptions.  Occur when the requirements author has a mistaken assumption about system features or stakeholder opinions. Mistaken Belief that it is Impossible to Specify Non-Functional Requirements. The requirements engineer(s) may believe that non-functional requirements cannot be captured and therefore omit this process from their elicitation and development plans. Lack of Clear Distinction Between Client and Users. If requirements engineering practitioners fail to distinguish between clients and end users, or do not realize that the clients are distinct from end users, they may fail to gather and analyze the end users' requirements. Lack of Awareness of Requirement Sources. Requirements gathering person is not aware of all stakeholders which he/she should contact in order to gather the complete set of user needs (including end users, customers, clients, and decision-makers). Innapropriate Communication Based on Incomplete or Faulty Understanding  of Rules. Without proper understanding of developer roles, communication gaps may arise, either by failing to communicate at all or by ineffective communication. The management structure of project team resources is lacking. Inadequate Requirements Process. Occur when the requirement authors do not fully understand all of the requirements engineering steps necessary to ensure the software is complete and neglect to incorporate one or more essential steps into the plan."
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
    "Any error in coding language syntax that impacts the executability of the code. Note that Logical Errors (e.g. += instead of +) are not Syntax Errors. Examples include mixing tabs and spaces (e.g. Python), unmatched brackets/braces/parenthesis/quotes, and missing semicolons (e.g. Java). Inappropriate Whitespace Style. The source code contains whitespace that is inconsistent across the code or does not follow expected standards for the product.",
    # S03
    "Errors resulting from overlooking (internally and externally) documented information, such as project descriptions, stakeholder requirements, API/library/tool/framework documentation, coding standards, programming language specifications, bug/issue reports, and looking at the wrong version of documentation or documentation for the wrong project/software. Use of Obsolete Function. The code uses deprecated or obsolete functions, which suggests that the code has not been actively reviewed or maintained. Inconsistency Between Implementation and Documented Design. The implementation of the product is not consistent with the design as described within the relevant documentation.",
    # S04
    "Errors resulting from multitasking, i.e. working on multiple software engineering tasks at the same time.",
    # S05
    "Attention failures while using computer peripherals, such as mice, keyboard, and cables. Examples include copy/paste errors, clicking the wrong button, using the wrong keyboard shortcut, and incorrectly plugging in cables.",
    # S06
    "Errors resulting from lack of attention during formal/informal code review. Examples include overlooking incorrect logic, or skipping files, functions, or classes during a review.",
    # S07
    "Errors resulting from overlooking existing functionality, such as reimplementing or duplicating variables, functions, and classes that already exist, or reimplementing functionality that already exists in a standard library. Other examples include deleting necessary variables, functions, and classes. Use of Redundant Code. The product has multiple functions, methods, procedures, macros, etc. that contain the same code.",
    # S08
    "Only use this category if you believe your error to be the result of a lack of attention, but no other slip category fits.",
    # L01
    "Forgetting to finish a development task. Examples include forgetting to implement a required feature, forgetting to finish a user story, and forgetting to deploy a security patch. Missing Synchronization. The product utilizes a shared resource in a concurrent manner but does not attempt to synchronize access to the resource. Missing Cryptographic Step. The product does not implement a required step in a cryptographic algorithm, resulting in weaker encryption than advertised by the algorithm.",
    # L02
    "Forgetting to fix a defect that you encountered, but chose not to fix right away.",
    # L03
    "Forgetting to remove debug log files, dead code, informal test code, commented out code, test databases, backdoors, etc. Examples include leaving unnecessary code in the comments, and leaving notes in internal development documentation. Improper Output Neutralization for Logs. The product does not neutralize or incorrectly neutralizes output that is written to logs. Active Debug Code. The product is deployed to unauthorized actors with debugging code still enabled or active, which can create unintended entry points or expose sensitive information. Dead Code. The product contains dead code, which can never be executed. Irrelevant Code. The product contains code that is not essential for execution, i.e. makes no state changes and has no side effects that alter data or control flow, such that removal of the code would have no impact to functionality or correctness.",
    # L04
    "Forgetting to git-pull (or equivalent in other version control systems), or using an outdated version of a library.",
    # L05
    "Forgetting to import a necessary library, class, variable, or function, or forgetting to access a property, attribute, or argument. Examples include forgetting to import python's sys library, forgetting to include a header file in C, or forgetting to pass an argument to a function.",
    # L06
    "Forgetting to push code, or forgetting to backup/save data or documentation.",
    # L07
    "Errors resulting from forgetting details from previous development discussions.",
    # L08
    "Only use this category if you believe your error to be the result of a memory failure, but no other lapse category fits.",
    # M01
    "A code logic error is one in which the code executes (i.e. actually runs), but produces an incorrect output/behavior due to incorrect logic. Examples include using incorrect operators (e.g. += instead of +), erroneous if/else statements, incorrect variable initializations, problems with variable scope, and omission of necessary logic. Improper Handling of Insufficient Permissions or Privileges. The product does not handle or incorrectly handles when it has insufficient privileges to access resources or functionality as specified by their permissions. This may cause it to follow unexpected code paths that may leave the product in an invalid state. Missing Default Case in Multiple Condition Expression. The code does not have a default case in an expression with multiple conditions, such as a switch statement. Function Call with Incorrectly Specified Arguments. The product calls a function, procedure, or routine with arguments that are not correctly specified, leading to always-incorrect behavior and resultant weaknesses. Use of Incorrect Operator. The product accidentally uses the wrong operator, which changes the logic in security-relevant ways. Omitted Break Statement in Switch. The product omits a break statement within a switch or similar construct, causing code associated with multiple conditions to execute. This can cause problems when the programmer only intended to execute code associated with one condition. Operator Precedence Logic Error. The product uses an expression in which operator precedence causes incorrect logic to be used. Loop with Unreachable Exit Condition ('Infinite Loop'). The product contains an iteration or loop with an exit condition that cannot be reached, i.e., an infinite loop. Comparison Using Wrong Factors. The code performs a comparison between two entities, but the comparison examines the wrong factors or characteristics of the entities, which can lead to incorrect results and resultant weaknesses. Time-of-check Time-of-use (TOCTOU) Race Condition. The product checks the state of a resource before using that resource, but the resource's state can change between the check and the use in a way that invalidates the results of the check. This can cause the product to perform invalid actions when the resource is in an unexpected state. Wrap-around Error. Wrap around errors occur whenever a value is incremented past the maximum value for its type and therefore 'wraps around' to a very small, negative, or undefined value. Divide by Zero. The product divides a value by zero.",
    # M02
    "Errors resulting from incomplete knowledge of the software system's target domain (e.g. banking, astrophysics). Examples include planning/designing a system without understanding the nuances of the domain.",
    # M03
    "Errors resulting from an incorrect assumption about system requirements, stakeholder expectations, project environments (e.g. coding languages and frameworks), library functionality, and program inputs. Improper Input Validation. The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly.",
    # M04
    "Errors resulting from inadequate communication between development team members. Examples include misunderstanding development discussion, misinterpretting or providing ambiguous instructions, communicating using the wrong medium (e.g. oral vs. written), or communicating ineffectively (e.g. too formal/informal, too much unecessarilly complex language, hostile language/body language).",
    # M05
    "Errors resulting from inadequate communication with project stakeholders or third-party contractors. Examples include providing ambiguous or unclear directions to third-parties or users, or misinterpretting stakeholder feedback, communicating using the wrong medium (e.g. oral vs. written), or communicating ineffectively (e.g. too formal/informal, too much unecessarilly complex language, hostile language/body language).",
    # M06
    "Misunderstood problem-solving methods/techniques result in analyzing the problem incorrectly and choosing the wrong solution. For example, choosing to implement a database system in Python rather than using SQL, or choosing the wrong software design pattern. Overconfidence in a solution choice also falls under this category. Use of Single-factor Authentication. The use of single-factor authentication can lead to unnecessary risk of compromise when compared with the benefits of a dual-factor authentication scheme. Use of Client-Side Authentication. A client/server product performs authentication within client code but not in server code, allowing server-side authentication to be bypassed via a modified client that omits the authentication check. Use of Hard-coded, Security-relevant Constants. The product uses hard-coded constants instead of symbolic names for security-critical values, which increases the likelihood of mistakes during code maintenance or security policy change. Use of Unmaintained Third Party Components. The product relies on third-party components that are not actively supported or maintained by the original developer or a trusted proxy for the original developer. Use of Hard-coded Credentials. The product contains hard-coded credentials, such as a password or cryptographic key, which it uses for its own inbound authentication, outbound communication to external components, or encryption of internal data.",
    # M07
    "Errors resulting from a lack of time management, such as failing to allocate enough time for the implementation of a feature, procrastinating a development task, or predicting the time required for a task incorrectly.",
    # M08
    "Failure to implement necessary test cases, failure to consider necessary test inputs, failure to implement a certain type of testing (e.g. unit, penetration, integration) when it is necessary, or failure to consider edge cases or unexpected inputs. Improper Check for Unusual or Exceptional Conditions. The product does not check or incorrectly checks for unusual or exceptional conditions that are not expected to occur frequently during day to day operation of the product.",
    # M09
    "Errors in configuration of libraries/frameworks/environments or errors related to missing configuration options. Examples include misconfigured IDEs or text editors, improper directory structure for a specific programming language, missing SSH keys, missing or incorrectly named database fields or tables, missing or incorrectly named/formatted configuration files, or not installing a required library. Incorrect Privilege Assignment. A product incorrectly assigns a privilege to a particular actor, creating an unintended sphere of control for that actor. Insufficient Granularity of Access Control. The product implements access controls via a policy or other feature with the intention to disable or restrict accesses (reads and/or writes) to assets in a system from untrusted agents. However, implemented access controls lack required granularity, which renders the control policy too broad because it allows accesses from unauthorized agents to the security-sensitive assets. Use of Default Credentials. The product uses default credentials (such as passwords or cryptographic keys) for potentially critical functionality.",
    # M10
    "Errors resulting from misunderstood code due to poor documentation or unnecessary complexity. Examples include too many nested if/else statements or for-loops and poorly named variables/functions/classes/files. CWE-1099: Inconsistent Naming Conventions for Identifiers. The product's code, documentation, or other artifacts do not consistently use the same naming conventions for variables, callables, groups of related callables, I/O capabilities, data types, file names, or similar types of elements. Use of Same Variable for Multiple Purposes. The code contains a callable, block, or other code element in which the same variable is used to control more than one unique task or store more than one instance of data. Class with Excessively Deep Inheritance. A class has an inheritance level that is too high, i.e., it has a large number of parent classes. Source Code File with Excessive Number of Lines of Code. A source code file has too many lines of code. Excessively Deep Nesting. The code contains a callable or other code grouping in which the nesting / branching is too deep.",
    # M11
    "Errors related to internationalization and/or string/character encoding. Examples include using ASCII intead of unicode, using UTF8 when UTF16 was necessary, failure to design the system with internationalization in mind, or failing to verify the character length of user input. Improper Handling of Unicode Encoding. The product does not properly handle when an input contains Unicode encoding. Improper Handling of Alternate Encoding. The product does not properly handle when an input uses an alternate encoding that is valid for the control sphere to which the input is being sent.",
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
    """
    Find the index of the max value in a list.
    """
    max_index = -999
    max_value = -999

    for i in range(0, len(cosine_scores)):
        if cosine_scores[i] >= max_value:
            max_value = cosine_scores[i]
            max_index = i

    return max_index


def _computeEmbeddings(model, descriptions, start=0, stop=None):
    """
    Compute BERT embeddings for the given descriptions using the given model.
    """
    if stop is None:
        stop = len(descriptions)
    embeddings = [
        model.encode(descr, convert_to_tensor=True) for descr in descriptions[start:stop]
    ]
    return embeddings


def _computeCosineScores(text_emb, target_embs):
    """
    Compute cosine similarities between the text embedding and the target embeddings.
    """
    return [ util.cos_sim(text_emb, target_emb) for target_emb in target_embs ]


def _makePrediction(cosine_scores, labels):
    """
    Given a list of cosine similarity scores, make a prediction.
    """
    max_index = _maxCosine(cosine_scores)
    label = labels[max_index]
    return label, max_index


def _predict(model, text, start, stop, cleantext):
    """
    Helper function for predict(...). Does the heavy lifting.
    """
    #### Step 0: Setup Labels
    type_labels = TYPE_LABELS
    category_labels = CATEGORY_LABELS[start:stop]

    type_descriptions = list()
    category_descriptions = list()
    #### Step 1: Clean Text
    if cleantext:
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
    """
    Lowercase, remove punctuation, remove non-space whitespace, remove duplicate spaces.
    """
    clean_text = RE_PUNCT.sub(" ", text.lower())
    clean_text = RE_WHITESPACE.sub(" ", clean_text)
    clean_text = RE_DUPLICATE_SPACES.sub(" ", clean_text)
    return clean_text


def _makeModels():
    """
    Make models.
    """    
    slips_model = SentenceTransformer("all-distilroberta-v1")
    slips_model.seq_length = 512
    lapses_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    lapses_model.seq_length = 128
    mistakes_model = SentenceTransformer("paraphrase-albert-small-v2")
    mistakes_model.seq_length = 256

    return slips_model, lapses_model, mistakes_model


def _readFile(text_file):
    """
    Read in the file with text to classify.
    """
    with open(text_file, "r") as f:
        data = f.read().replace("\n", " ")

    return data


def predict(text_file):
    """
    Make prediction based on GitHub Issue description and comment text.

    Prediction uses an ensemble of three "classifiers"; one each for the "best" models based on the
    "training" data. If the three classifiers return a majority class, then that is the
    prediction given.

    Example:
      (1) Slip model predicts "Slip"
      (2) Lapse model predicts "Lapse"
      (3) Mistake model predicts "Slip"
      Predicted Class: Slip

    If there is no majority class, the class with the highest cosine similarity to the GitHub text
    is chosen.

    Along with the human error type, a category prediction is also given.

    Example Output: Slip, S02: Syntax Errors
    """
    # Models
    slips_model, lapses_model, mistakes_model = _makeModels()

    text = _readFile(text_file)

    # Predictions
    slip_predictions = _predict(slips_model, text, 0, 8, True)
    lapse_predictions = _predict(lapses_model, text, 8, 16, True)
    mistake_predictions = _predict(mistakes_model, text, 16, 31, True)

    # Type Prediction Logic
    type_preds = [ slip_predictions[0][0], lapse_predictions[0][0], mistake_predictions[0][0] ]
    counts = [ type_preds.count("Slip"), type_preds.count("Lapse"), type_preds.count("Mistake") ]
    # If Slips greater than Lapses and Mistakes
    if counts[0] > counts[1] and counts[0] > counts[2]:
        print("{}, {}".format("Slip", slip_predictions[1][0]))
    # If Lapses greater than Slips and Mistakes
    elif counts[1] > counts[0] and counts[1] > counts[2]:
        print("{}, {}".format("Lapse", lapse_predictions[1][0]))
    # If Mistakes greater than Slips and Lapses
    elif counts[2] > counts[0] and counts[2] > counts[1]:
        print("{}, {}".format("Mistake", mistake_predictions[1][0]))
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
    # Filename with text to classify
    text = args[0]
    assert os.path.exists(text), "Argument 'text' must be a valid filename"
    predict(text)


if __name__ == "__main__":
    main()
